#!/bin/bash
#SBATCH -J masslik_gpu
#SBATCH -A mit_general
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1        # override at submit: sbatch -G h200:1 ...
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH -t 04:00:00
#SBATCH -o /home/david_w/orcd/pool/zmass/logs/%x_%j.out
#SBATCH -e /home/david_w/orcd/pool/zmass/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Validation + benchmark of the unbinned mass fit on an Engaging GPU.
#
#   sbatch masslik_fit_gpu.sbatch [gun|bto] [--rabbit]
#
# Stage 0  bench_masslik.py: NLL / grad / Hessian timings (GPU-vs-CPU table)
# Stage 1  cf_masslik_fit.py --model families, k_rad = 0 and 1  (reference
#          implementation; the numbers to reproduce are in
#          ~/public_html/cvh/260904_masslikfit/masslikfit_summary.txt on submit)
# Stage 2  (--rabbit) the same likelihood through rabbit: build the datacard
#          with tests/make_unbinned_mass_tensor.py, then rabbit_fit.py -t 0
#          --unblind with the Hessian.
#
# --mem: the gun pairs cache is 2.4 GB on disk and the (n, 448) float32 blocks
# are held in host memory before being chunked onto the device; 128 G is
# comfortable for 300 k candidates.  -G l40s:1 because mit_normal_gpu caps a
# user at 2 GPUs / 6 h; swap to `-G h200:1` when h200 nodes are idle, or move
# to `-p mit_preemptable` (4 GPUs / 48 h, PreemptMode=REQUEUE) for long fits.
# ---------------------------------------------------------------------------
set -u
SAMPLE=${1:-gun}
DO_RABBIT=${2:-}

ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
cd $ZMASS/resolution

case $SAMPLE in
  gun) P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
       K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
       T=jpsigun_260903x ;;
  bto) P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
       K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
       T=btojpsix_v3_260903x ;;
  *)   echo "unknown sample $SAMPLE"; exit 1 ;;
esac

echo "=== node $(hostname)  job $SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

# --threads is the TF intra/inter-op count; on GPU the heavy kernels are on the
# device, so it only sizes the host-side ops -- keep it at the allocated cores.
THREADS=${SLURM_CPUS_PER_TASK:-16}

# STAGES=... selects which stages run (default all): e.g.
#   sbatch --export=ALL,STAGES=2 masslik_fit_gpu.sbatch gun --rabbit
STAGES=${STAGES:-0 1 2}
has () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ------------------------------------------------------------------ stage 0
# NLL / gradient / Hessian timings of the identical graph, for the GPU-vs-CPU
# speedup table.  Also the cheapest numerical cross-check: the three values
# must match the submit-CPU run to float64 round-off.
has 0 && BENCH_SITE=engaging_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d ' ') \
python $ZMASS/engaging/bench_masslik.py \
    --pairs-cache "$P" --kernel-cache "$K" --model families --krad 0 \
    --threads $THREADS --repeat 3 \
    --json-out $ZMASS/logs/bench_${T}_gpu.json

# ------------------------------------------------------------------ stage 1
has 1 && for KRAD in 0 1; do
  nm=${T}_fam_krad${KRAD}_gpu
  echo "=== $nm"
  /usr/bin/time -v python cf_masslik_fit.py \
      --pairs-cache "$P" --kernel-cache "$K" \
      --model families --krad $KRAD \
      --threads $THREADS --no-plots \
      --outpath $ZMASS/fitresults \
      --tag "$nm" --out runs/masslikfit_${nm}.npz \
      2>&1 | grep -vE '^\s+(Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Average)'
done

# ------------------------------------------------------------------ stage 2
if [ "$DO_RABBIT" = "--rabbit" ] && has 2; then
  # CARD= overrides the datacard (e.g. a smaller --chunk build).  The default
  # 32768-chunk card needs more than an L40S's 46 GB for the 5x5 Hessian:
  # rabbit builds it with tf.vectorized_map (pfor) over the parameters, so peak
  # device memory is ~npar x chunk x ntgrid x 16 B.  Use -G h200:1 (141 GB), or
  # rebuild the card with --chunk 8192.
  CARD=${CARD:-$ZMASS/cards/${T}_families.hdf5}
  mkdir -p $ZMASS/cards $ZMASS/fitresults
  if [ ! -s "$CARD" ]; then
    echo "=== building rabbit datacard $CARD"
    python $ZMASS/rabbit/tests/make_unbinned_mass_tensor.py \
        --pairs-cache "$P" --kernel-cache "$K" \
        --model families -o "$CARD"
  fi
  echo "=== rabbit_fit.py on $CARD"
  /usr/bin/time -v python $ZMASS/rabbit/bin/rabbit_fit.py "$CARD" \
      -o $ZMASS/fitresults --postfix ${T}_gpu \
      -t 0 --unblind --paramModel UnbinnedParams \
      2>&1 | grep -vE '^\s+(Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Average)'
fi

echo "=== done $(date)"

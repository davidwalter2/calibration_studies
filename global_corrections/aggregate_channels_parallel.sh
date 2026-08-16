#!/bin/bash
# Parallel per-channel aggregation of the CVH grads into per-channel info
# npz (grad + hess on the floated params). Each channel's files are split
# into chunks aggregated concurrently; the chunk npzs are summed into the
# channel npz. Hessians/gradients are additive, so chunking is exact.
set -uo pipefail
GC=/work/submit/david_w/ZMass/calibration_studies/global_corrections
GRPFILE=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
VENV=/work/submit/david_w/ZMass/mfs/.venv
WORK=${1:?output dir for npz}
NCHUNK=${2:-8}
mkdir -p "$WORK"
source $VENV/bin/activate

declare -A CHAN
CHAN[Jpsi]="/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsi_calib2016_grads50_globalmat_260717_357acafa052"
CHAN[Ks]="/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_calib2016_grads50_260719_ae2f4c33457"
CHAN[cosmics]="/ceph/submit/data/user/d/david_w/ZMass/cvh/cosmics_calib2016_grads50_260718_357acafa052"

for ch in "${!CHAN[@]}"; do
  dir=${CHAN[$ch]}
  # collect all stream files, split into NCHUNK filelists
  ls $dir/task_*/globalcor_*.root 2>/dev/null | sort > "$WORK/${ch}_all.txt"
  nf=$(wc -l < "$WORK/${ch}_all.txt")
  echo "$(date) $ch: $nf files -> $NCHUNK chunks"
  split -n l/$NCHUNK -d --additional-suffix=.txt "$WORK/${ch}_all.txt" "$WORK/${ch}_chunk"
  for cf in "$WORK/${ch}_chunk"*.txt; do
    out="${cf%.txt}.npz"
    python $GC/fit_global_grads.py -i "$cf" --groups "$GRPFILE" \
      --save-info "$out" > "${cf%.txt}.log" 2>&1 &
  done
done
echo "$(date) launched all chunk jobs, waiting..."
wait
echo "$(date) all chunks done; summing"

# sum chunk npzs per channel
python - "$WORK" "${!CHAN[@]}" << 'PYEOF'
import sys, glob, numpy as np
work = sys.argv[1]; chans = sys.argv[2:]
for ch in chans:
    parts = sorted(glob.glob(f"{work}/{ch}_chunk*.npz"))
    if not parts:
        print(f"{ch}: no chunk npz"); continue
    g = None; H = None; nc = 0; meta = None
    for p in parts:
        d = np.load(p, allow_pickle=True)
        g = d["grad"].copy() if g is None else g + d["grad"]
        H = d["hess"].copy() if H is None else H + d["hess"]
        nc += int(d["ncand"])
        meta = d
    np.savez(f"{work}/{ch}.npz", grad=g, hess=H,
             fitidx=meta["fitidx"], parmtype=meta["parmtype"],
             names=meta["names"], priors=meta["priors"],
             ncand=nc, nfiles=sum(int(np.load(p, allow_pickle=True)["nfiles"]) for p in parts))
    print(f"{ch}: summed {len(parts)} chunks -> {nc:,} candidates -> {work}/{ch}.npz")
PYEOF
echo "$(date) AGGREGATION_DONE"

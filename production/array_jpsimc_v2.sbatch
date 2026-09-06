#!/bin/bash
# THE 4-THREAD VARIANT of array_jpsimc.sbatch, for the J/psi re-production
# (`jpsimc_20M_260906_v2`) out of CMSSW_15_0_19_patch2_dev2.
#
# DO NOT SUBMIT THIS WHILE THE 260905 ARRAYS ARE IN squeue. It is here so the
# decision can be taken later; the current arrays own the fair share and this
# would only queue behind them.
#
# WHAT CHANGES, AND WHY (measured 2026-09-06, see condor_dymc_v2/STATE_dy_v2.md):
#   * `--cpus-per-task=4` -- the CVH refit is an `edm::stream::EDProducer<>`
#     whose total CPU is FLAT in the thread count (J/psi: 2028 s at 1 thread,
#     2024 s at 8) and whose only serial cost is a ~36 s geometry/field/G4
#     head. `wall(N) = 36.0 + 2003.9/N` fits the measured 1/2/4/8-thread arms
#     to 1.3 %. On a real 12 185-event chunk with the new exports that is
#     3.40 h -> 0.86 h, a 3.97x speedup.
#   * `--mem=5G` instead of 6G. Memory is set by the single-task TAIL, not the
#     core count: ~1.3 GB shared plus ~76 MB per stream, against a measured
#     1-thread MaxRSS of median 1.9 / p99 3.1 / max 3.56 GB over the 1616
#     tasks of jpsimc_20M_260905. 3.56 + 3 x 0.076 = 3.79, x1.3 = 4.9 GB.
#     So this asks for LESS memory per task than today AND four times the
#     cores: 1.25 GB/core against 6.
#   * `--time=4:00:00` instead of 12 h. The 12 h request was ~5x the true max
#     and actively hurt throughput -- slurm's backfill scheduler can only slot
#     a job into a gap it declares it fits in. At 4 threads the longest chunk
#     is ~1 h, so 4 h is 4x headroom and backfills readily.
#   * the dev2 area, which carries the new exports and `fab515e`.
#
# THE OUTPUT IS N FILES, NOT ONE. `numberOfStreams` follows `numberOfThreads`
# and the maker writes `globalcor_<stream>.root` per stream, so a task produces
# four. The completion check below counts them against the configuration that
# was actually passed. Verified bit-identical to the 1-thread output, candidate
# by candidate, over 1991 J/psi candidates and all 228 branches.
#
# One array task = one CHUNK of one UL16 J/psi MC ALCARECO file -> one CVH
# two-track refit output directory.
#
# WHY CHUNKS AND NOT WHOLE FILES: the repacked ALCARECO files hold ~53k events
# each, which at ~0.7 s/candidate is 10+ h of Geant4e in a single task. Each
# file is therefore split into NCHUNK equal event ranges via the driver's
# skipEvents/nEvents, giving ~2-3 h tasks that fit comfortably inside the wall
# limit and fail in small pieces.
#
# WHY THE INPUT READABILITY IS ASSERTED (2026-09-05): the submit login nodes'
# CephFS client intermittently returns EPERM for the whole /ceph/submit tree
# (submit82 is evicted right now). cmsRun with skipBadFiles=True then SILENTLY
# SKIPS the input and writes a valid, EMPTY output. Compute nodes see ceph
# fine, but a node can be evicted mid-flight, so the check stays.
#
# env (exported by submit_jpsimc20M.sh):
#   CHUNKLIST  absolute path to the chunk list, one "<path> <skip> <nev>" per line
#   OUTBASE    absolute output base directory
#   IDXOFFSET  added to SLURM_ARRAY_TASK_ID (MaxArraySize is 1001, we need ~1640)
#   EXTRA      extra cmsRun VarParsing args (the physics configuration)
#SBATCH --job-name=jpsimc_v2
#SBATCH --partition=submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=4:00:00
set -uo pipefail
: "${CHUNKLIST:?}" ; : "${OUTBASE:?}"
export CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh

IDX=$(( SLURM_ARRAY_TASK_ID + ${IDXOFFSET:-0} ))
OUTDIR="$OUTBASE/task_$(printf '%04d' "$IDX")"

# Resume on the sentinel ONLY. cmsRun creates its output at START, so a killed
# task leaves a non-empty but TRUNCATED .root that an `-s` test would accept
# forever.
if [[ -f "$OUTDIR/.complete" ]]; then echo "[skip] task $IDX already complete"; exit 0; fi

LINE=$(sed -n "$((IDX + 1))p" "$CHUNKLIST")
[[ -n "$LINE" ]] || { echo "[FATAL] empty chunk line $((IDX+1)) of $CHUNKLIST" >&2; exit 2; }
read -r INPUT SKIP NEV _rest <<< "$LINE"
[[ -n "$INPUT" && -n "$SKIP" && -n "$NEV" ]] || { echo "[FATAL] malformed chunk line: $LINE" >&2; exit 2; }
[[ -r "$INPUT" ]] || { echo "[FATAL] input not readable on $(hostname): $INPUT" >&2; exit 3; }

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.root "$OUTDIR/.complete"

echo ">>> host=$(hostname) task=$IDX skip=$SKIP nev=$NEV $(date)"
echo ">>> input=$INPUT"
# shellcheck disable=SC2086
"$RUN_ONE" "$CFG" "$INPUT" "$OUTDIR" skipEvents="$SKIP" nEvents="$NEV" ${EXTRA:-} \
    > "$OUTDIR/local.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then echo "[FATAL] cmsRun rc=$rc (see $OUTDIR/local.log)" >&2; exit $rc; fi

shopt -s nullglob
# ONE OUTPUT FILE PER EDM STREAM. `numberOfStreams` follows `numberOfThreads`
# and the maker writes `globalcor_<stream>.root`
# (ResidualGlobalCorrectionMakerBase.cc:451), so a multithreaded task produces N
# files and the old `>= 1 && -s out[0]` test would have accepted a task that
# wrote ONE of them -- i.e. 1/N of the candidates, silently, with a `.complete`
# sentinel on top. N is read back out of the configuration that was actually
# passed, so a single-threaded task behaves exactly as before.
NSTREAM=$(grep -o 'numberOfThreads=[0-9]*' <<< "${EXTRA:-}" | tail -1 | grep -o '[0-9]*')
NSTREAM=${NSTREAM:-1}
out=("$OUTDIR"/globalcor_*.root)
[[ ${#out[@]} -eq $NSTREAM ]] || {
  echo "[FATAL] expected $NSTREAM stream file(s) in $OUTDIR, found ${#out[@]}" >&2; exit 4; }
for _f in "${out[@]}"; do
  [[ -s "$_f" ]] || { echo "[FATAL] empty output $_f" >&2; exit 4; }
done
# Guard against the silent-empty-input failure: the maker always prints a fit
# summary, and a run that read nothing prints attempted=0.
if grep -q "fit summary  attempted=0 " "$OUTDIR/local.log"; then
  echo "[FATAL] attempted=0 candidates -- input was skipped (ceph EPERM?)" >&2; exit 5
fi
touch "$OUTDIR/.complete"
echo "[done] task $IDX $(date)"

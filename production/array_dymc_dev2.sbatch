#!/bin/bash
# THE RECOVERY VARIANT of array_dymc.sbatch: identical in every respect except
# the CMSSW area it runs from.
#
# WHY IT EXISTS (2026-09-06). 62+ tasks of dymc_8p5M_260905 died with SIGABRT
# in the two-track maker's factored-Hessian export: a Z candidate whose two
# MiniAOD legs carry only ten valid-hit-equivalents between them leaves
# `ndof == 0`, `nrank = min(ndof, nparsfinal) = 0`, and the truncation-gap
# report then reads `eigvals(nparsfinal)` -- one past the end of the
# eigenvalue vector -- which aborts inside Eigen and takes the WHOLE TASK with
# it (a crashed cmsRun output has no keys at all, so nothing of the task
# survives). The fix lives on `cvh-exports-260906` in
# CMSSW_15_0_19_patch2_dev2 and must be cherry-picked to
# WmassNanoProd_15_0_19_patch2_dev once the productions have drained.
#
# POOLABILITY: dev2 is ~11 commits ahead of the production commit 157775e with
# new EXPORTS only (physics bit-identical -- gated by the three
# smoke_exports_260906.sh smokes). Two consequences for the tasks driven by
# this script:
#   * `exportHitResBlocks=False` is REQUIRED and is added by resume_dy_dev2.sh.
#     It defaults to True on the branch and would register the parmtype-8/9 hit
#     blocks in `dVs`, changing `resinfv`/`reseigidx`/`resinfvarv` content.
#   * They carry three extra always-on branches the rest of the production set
#     does not have -- `Jpsi_covrefmom`, `Jpsigenpre_*`, `Mu*_maxfracloss`.
#     Harmless (nothing pools on them), but they exist.
#
#SBATCH --job-name=dymc8p5M
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
set -uo pipefail
: "${CHUNKLIST:?}" ; : "${OUTBASE:?}"
export CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
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

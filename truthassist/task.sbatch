#!/bin/bash
# Truth-assisted corrections study: one task per line of $TASKS
# ("<mode> <inputType> <input file> <task dir>"), runCvhTruthAssisted.py in
# the CMSSW_15_0_19_patch2 area (el9-native). NEVER rebuild that area while
# this array runs: a relinked .so kills the running tasks.
#SBATCH --job-name=cvhtruth
#SBATCH --partition=submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=10:00:00
set -uo pipefail
: "${TASKS:?}"; : "${SLURM_ARRAY_TASK_ID:?}"
read -r MODE ITYPE INFILE TDIR <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")"
[[ -n "$TDIR" ]] || { echo "no task line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
mkdir -p "$TDIR" && cd "$TDIR" || exit 1
rm -f .complete globalcor_truth_*.root
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2/src && eval "$(scramv1 runtime -sh)" && cd "$TDIR" || exit 1
echo "host $(hostname) mode=$MODE inputType=$ITYPE input=$INFILE start $(date)"
/usr/bin/time -v cmsRun "$CMSSW_BASE/src/Analysis/HitAnalyzer/test/runCvhTruthAssisted.py" \
    mode="$MODE" inputType="$ITYPE" input="$INFILE" nEvents=-1 numberOfThreads=4 \
    > local.log 2> time.log
rc=$?
echo "cmsRun exit $rc $(date)"
[[ $rc -eq 0 ]] || exit $rc
n=$(python3 -c "
import ROOT, glob
t = 0
for f in glob.glob('globalcor_truth_*.root'):
    r = ROOT.TFile.Open(f); t += r.Get('tree').GetEntries()
print(t)")
echo "entries $n"
[[ "$n" -gt 0 ]] || { echo "EMPTY output" >&2; exit 2; }
printf 'entries=%s\nstreams=4\nmode=%s\ninput=%s\n' "$n" "$MODE" "$INFILE" > .complete

#!/bin/bash
# Generate a toy clean-propagation sim by SPLITTING it across seeded jobs.
#
# WHY. The sim is single-threaded (the ToyStateNtuplizer watcher owns its output
# file, so Geant4 MT is off by design) and runs at 3.10 ms/event with ~13 s of
# fixed startup: a 100k sample is 5.3 min of wall clock on one core of a 768-core
# machine. The initial state is fixed and only the Geant4 seed varies between
# events, so N jobs at different seeds are N chunks of the SAME sample --
# `toy_loader.load_toy_sim` takes a glob and concatenates them in sorted order.
#
# Measured scaling (2026-08-15). The EVENT LOOP scales essentially perfectly:
# 3.02 ms/event for one job alone, 3.15 at 16-way, 3.11 at 32-way. What does not
# scale is STARTUP -- every job parses 278 geometry XMLs and pulls the same
# conditions -- so the useful width is bounded by the herd, not by the cores:
#
#     32 000 events   serial (projected from 20k)   112 s
#                     16 jobs, no stagger            24.2 s   <- measured, best
#                     32 jobs, 0.7 s stagger         44.5 s   <- measured
#                     32 jobs, no stagger           200.4 s   <- measured, herd
#
#     100 000 events  serial                        317 s     <- measured
#                     16 jobs                        ~33 s    (projected)
#
# So: 16 concurrent jobs per configuration, and add width by running more
# CONFIGURATIONS in parallel private areas rather than more jobs per config.
#
# WHAT THIS DOES NOT DO: run several CONFIGURATIONS at once. gen_toy_config.py
# rewrites data/tracker.xml, test/toyPlanes_pt3.py AND the watcher radii inside
# runToyGeomCheck.py itself, all shared, so two configurations in one area would
# corrupt each other. Concurrent configurations need one private area each --
# the convention the pT-scan work used with `toyscan_pt/`. Within a
# configuration, this script saturates as wide as you ask.
#
# usage:
#   ./run_toy_sim_split.sh <tag> [nevents] [njobs] [outdir]
#   ./run_toy_sim_split.sh K1 100000 32
#
# then offline:
#   python cleanprop/toy_closure.py --sim '<outdir>/hs<tag>_*.root' --model ...
set -uo pipefail

TAG=${1:?usage: $0 <tag> [nevents] [njobs] [outdir]}
NEV=${2:-100000}
NJOB=${3:-32}
OUT=${4:-${SCRATCH:-/tmp}}
TESTDIR=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test
LOCK=$TESTDIR/cmsswlock.sh

PER=$(( (NEV + NJOB - 1) / NJOB ))
echo "[split] tag=$TAG  $NEV events over $NJOB jobs = $PER each  -> $OUT/hs${TAG}_*.root"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
eval "$(scramv1 runtime -sh)"
cd "$TESTDIR"
mkdir -p "$OUT"
rm -f "$OUT"/hs${TAG}_*.root

# STAGGER the launches. The event loop scales perfectly -- 32 concurrent jobs
# ran at 3.11 ms/event against 3.02 for one job alone -- but STARTUP does not:
# every job parses 278 geometry XMLs and pulls the same conditions, and 32 of
# them starting in the same instant spread their first event over 27 s and took
# the whole 32x1000 campaign to 200 s. A small gap between launches removes the
# thundering herd for a cost of STAGGER x NJOB seconds.
# Adaptive: 16 at once caused no herd (32k events in 22 s with no stagger at
# all), 32 at once did. So stagger only above 16, where it pays for itself.
if [[ -z "${STAGGER:-}" ]]; then
    if [[ "$NJOB" -gt 16 ]]; then STAGGER=0.7; else STAGGER=0; fi
fi

s=$(date +%s.%N)
for i in $(seq 1 "$NJOB"); do
    # seed MUST be non-zero and distinct: seed=0 is the historical default and
    # every job would then produce the identical sample.
    printf -v n "%04d" "$i"
    "$LOCK" run cmsRun runToyGeomCheck.py \
        events="$PER" seed="$i" output="$OUT/hs${TAG}_${n}.root" \
        > "$OUT/hs${TAG}_${n}.simlog" 2>&1 &
    [[ "$STAGGER" != 0 ]] && sleep "$STAGGER"
done
wait
e=$(date +%s.%N)

nok=$(ls -1 "$OUT"/hs${TAG}_*.root 2>/dev/null | wc -l)
nev=$(grep -h -c "Begin processing" "$OUT"/hs${TAG}_*.simlog 2>/dev/null | paste -sd+ | bc)
# Count the logs that DO carry the stepper line. Not `grep -L ... && fail`:
# -L changes what grep PRINTS, not its exit status, which is still "0 if a line
# was selected" -- so that form fires when every file is fine and stays quiet
# when none of them are, i.e. exactly backwards. (Cost: one spurious FAILED.)
ntight=$(grep -l "TIGHT stepper" "$OUT"/hs${TAG}_*.simlog 2>/dev/null | wc -l)
echo "[split] $nok/$NJOB files, $nev events total, $ntight/$NJOB tight-stepper, $(echo "$e - $s" | bc) s wall"
[[ "$nok" -eq "$NJOB" ]] || { echo "[split] FAILED: $((NJOB-nok)) job(s) produced no file"; exit 1; }
[[ "$ntight" -eq "$NJOB" ]] || { echo "[split] FAILED: $((NJOB-ntight)) job(s) did not use the tight stepper"; exit 1; }
[[ "$nev" -eq "$((PER*NJOB))" ]] || { echo "[split] FAILED: $nev events, expected $((PER*NJOB))"; exit 1; }
echo "[split] ok -- read back with --sim '$OUT/hs${TAG}_*.root'"

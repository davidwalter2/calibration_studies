#!/bin/bash
# Step 4: does correcting the hit covariance move the fit TOWARD gen truth?
#
# The pull study measures two things that a single scale cannot both fix,
# because the pixel and strip pulls are wrong in OPPOSITE directions:
#   core-matching   makes the 68 % width of the pull 1 (what a likelihood
#                   should match if the residual were Gaussian);
#   variance-matching makes the second moment right (what an optimal LINEAR
#                   weighting wants, and what a second-moment estimator such
#                   as parmtype 8/9 would converge to).
# Those two prescriptions DISAGREE here, which is the whole point, so both
# are run and truth decides.
#
# Everything is paired: the same tracks, one input changed, compared with
# hitres_nulltest.py-style key matching. A per-track shift quoted without a
# truth comparison repeats the mistake of NOTES_CGFFIT sections 86-87.
#
# usage: ./run_hitres_covscale.sh <species> <filelist> <basetag> \
#            "<pix_core> <strip_core>" "<pix_var> <strip_var>" [nfiles] [npar]
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
SP=$1; LIST=$2; BASE=$3; CORE=$4; VAR=$5; NF=${6:-40}; NPAR=${7:-40}
LOG=${LOG:-$RES/runs/260829_hitres}
cd "$RES"
read -r PC SC <<< "$CORE"
read -r PV SV <<< "$VAR"
pids=()
for arm in "core:$PC:$SC" "var:$PV:$SV"; do
  IFS=: read -r name p s <<< "$arm"
  echo "[covscale] $BASE $name: pixel x$p  strip x$s"
  EXTRA="hitCovScalePixel=$p hitCovScaleStrip=$s" STAGGER=3 \
    ./run_local_hitres.sh "$SP" "$LIST" "${BASE}_cs${name}" "$NPAR" 0 $((NF-1)) \
      > "$LOG/${BASE}_cs${name}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[covscale] done"

#!/bin/bash
# FINITE-DIFFERENCE DRIVER for the two-track maker's VARIANCE (log-det)
# gradient (`exportVarianceGrads`).
#
#   usage: fd_variance_260906.sh <CMSSW_AREA> <OUTROOT> [nev]
#
# The claim under test is that the parmtype-p column of `gradv` is
#
#     d/dtheta_p [ r^T R r + ln|V| + ln|C| ]
#
# so the test is: move theta_p by +-delta IN THE PROPAGATOR, re-fit, and
# compare the central difference of `objval` (the same objective, exported in
# double under `exportObjective`) with the analytic column of the UNPERTURBED
# run.  Two independent knobs, one per family:
#
#   parmtype 15   the material group's own k_g, injected through the `k_init`
#                 column of a COPY of materialGroups50.txt.  This scales the
#                 group's steps' mean loss AND their MS covariance and
#                 ionization variance -- all three carry the same exp(k_g) --
#                 so the FD is against the FULL parmtype-15 column, mean part
#                 included.
#   parmtype 10   CVH_MS_SCALE, which multiplies `DD` inside
#                 `PropagateErrorMSC` and therefore every step's MS covariance
#                 (and, coherently, the `dQMS` that is registered as the
#                 family's dV).  It touches nothing else: the (0,0) element is
#                 overwritten by the ionization variance AFTER the scaling.
#                 It is a COMMON log scale over all parmtype-10 globals, so
#                 the FD is against sum_idx gradv[parmtype-10 column].
#
# Deltas 1e-3 and 1e-4: the residual must be flat between them, otherwise it
# is finite-difference truncation rather than a real modelling gap.
set -uo pipefail

AREA=${1:?usage: fd_variance_260906.sh <CMSSW_AREA> <OUTROOT> [nev]}
OUTROOT=${2:?usage: fd_variance_260906.sh <CMSSW_AREA> <OUTROOT> [nev]}
NEV=${3:-60}
FDGROUPS=${FD_GROUPS:-"16 6"}          # tib_support (89 % occupancy), bpix_support
DELTAS=${FD_DELTAS:-"1e-3 1e-4"}

STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
IN_GUN_TT=$STAGE/jpsigun_task0000_step2.root
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
TESTDIR=$AREA/src/Analysis/HitAnalyzer/test
GROUPFILE=$AREA/src/Analysis/HitAnalyzer/data/materialGroups50.txt
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

mkdir -p "$OUTROOT/groupfiles"

# k_init is field 10 of a RULE line (RULE id name regex rmin rmax zmin zmax
# zside k_init prior_sigma).  Write it for EVERY rule whose groupId matches --
# `bpix_support` has two.
mk_groupfile() {
  local gid=$1 val=$2 dst=$3
  awk -v gid="$gid" -v val="$val" 'BEGIN{FS=OFS="\t"}
       $1=="RULE" && $2==gid {$10=val}
       {print}' "$GROUPFILE" > "$dst"
  local n
  n=$(awk -v gid="$gid" -v val="$val" 'BEGIN{FS="\t"} $1=="RULE" && $2==gid && $10==val {c++} END{print c+0}' "$dst")
  [[ "$n" -ge 1 ]] || { echo "[FATAL] k_init patch missed for group $gid" >&2; exit 2; }
}

run_one() {
  local out=$1; shift
  mkdir -p "$out"
  rm -f "$out"/*.root
  ( cd "$out" \
    && export CVH_MS_SCALE="${FD_MS_SCALE:-1.0}" \
    && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$out" \
    && "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
       cmsRun "$TESTDIR/runCvhJpsiGenMC.py" input="$IN_GUN_TT" \
       nEvents=$NEV numberOfThreads=1 doRes=True \
       exportCfExponents=True exportStepRecords=False \
       fillJac=True fillGrads=True fitFromGenParms=False CgfQoPMode=0 \
       scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
       doTrigger=False applyHltFilter=False $GUNFIELD \
       exportMaterialNoise=True exportVarianceGrads=True exportObjective=True \
       ${FD_EXTRA:-} "$@" ) > "$out/cmsrun.log" 2>&1
  local rc=$?
  local nent
  nent=$( ( source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
            cd "$AREA/src" && eval "$(scramv1 runtime -sh)" 2>/dev/null
            python3 - "$out" <<'PY' 2>/dev/null
import glob, sys
try:
    import uproot
    fs = glob.glob(sys.argv[1] + "/*.root")
    print(uproot.open(fs[0])["tree"].num_entries if fs else -1)
except Exception:
    print("?")
PY
          ) )
  echo "    rc=$rc entries=$nent  $out"
  [[ "$nent" == "0" ]] && { echo "    [FATAL] zero candidates"; return 5; }
  return $rc
}

echo ">>> nominal (k = 0, MS scale 1), families 10,11,15"
run_one "$OUTROOT/nom" varianceGradFamilies=10,11,15

for g in $FDGROUPS; do
  for d in $DELTAS; do
    for sgn in p m; do
      v=$( [[ $sgn == p ]] && echo "$d" || echo "-$d" )
      gf="$OUTROOT/groupfiles/g${g}_${sgn}${d}.txt"
      mk_groupfile "$g" "$v" "$gf"
      echo ">>> parmtype-15 FD: group $g  k = $v"
      run_one "$OUTROOT/mat_g${g}_${sgn}${d}" varianceGradFamilies=10,11,15 \
              materialGroupsFile="$gf"
    done
  done
done

for d in $DELTAS; do
  for sgn in p m; do
    v=$( [[ $sgn == p ]] && echo "$d" || echo "-$d" )
    s=$(python3 -c "import math,sys; print(repr(math.exp(float(sys.argv[1]))))" "$v")
    echo ">>> parmtype-10 FD: CVH_MS_SCALE = $s  (delta = $v)"
    FD_MS_SCALE=$s run_one "$OUTROOT/ms_${sgn}${d}" varianceGradFamilies=10,11,15
  done
done
echo ">>> done"

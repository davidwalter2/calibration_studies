#!/bin/bash
# FINITE-DIFFERENCE TEST of the two-track maker's exported global derivatives
# under both Gauss-Newton linearisation points (localUpdate False / True).
#
#   usage: fd_localupdate_260924.sh <CMSSW_AREA> <OUTROOT> [nev]
#
# For each mode, a nominal run plus +-delta runs of one global parameter at a
# time, moved IN THE PROPAGATOR and followed by the full refit:
#
#   parmtype 15   material group k_g via the k_init column of a copy of
#                 materialGroups50.txt
#   parmtype 14   scalar-potential field mode via injectFieldModes
#
# check_localupdate_grads_260924.py then compares the central differences of
# objval, Jpsi_mass and Mu{plus,minus}_refParms with gradv, Jpsi_jacMass and
# Mu{plus,minus}_jacRef of the nominal run, per candidate.  Two deltas per
# parameter: the residual must be flat between them, otherwise it is
# finite-difference truncation rather than a derivative error.
#
# All runs are independent and single-threaded; they run in parallel.
set -uo pipefail

AREA=${1:?usage: fd_localupdate_260924.sh <CMSSW_AREA> <OUTROOT> [nev]}
OUTROOT=${2:?usage: fd_localupdate_260924.sh <CMSSW_AREA> <OUTROOT> [nev]}
NEV=${3:-300}
MATGROUP=${FD_MATGROUP:-16}                       # tib_support
MATDELTAS=${FD_MATDELTAS:-"1e-3 1e-4"}
# mode 0 = (l=1, m=0) uniform Bz, coefficient 1219.5 <-> 3.8 T, so 0.12 is a
# relative 1e-4; mode 3 = (l=2, m=0), the axial gradient
FIELDRUNS=${FD_FIELDRUNS:-"0:0.12 0:0.012 3:0.5 3:0.05"}
VARFAM=${FD_VARFAM:-15}                           # exportVarianceGrads families
MODES=${FD_MODES:-"False True"}

IN_GUN_TT=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs/jpsigun_task0000_step2.root
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
TESTDIR=$AREA/src/Analysis/HitAnalyzer/test
GROUPFILE=$AREA/src/Analysis/HitAnalyzer/data/materialGroups50.txt
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

mkdir -p "$OUTROOT/groupfiles"

# k_init is field 10 of a RULE line; patch it on every rule of the group
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
    && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$out" \
    && "$TESTDIR/cmsswlock.sh" run \
       cmsRun "$TESTDIR/runCvhJpsiGenMC.py" input="$IN_GUN_TT" \
       nEvents=$NEV numberOfThreads=1 doRes=True \
       exportCfExponents=True exportStepRecords=False \
       fillJac=True fillGrads=True fitFromGenParms=False CgfQoPMode=0 \
       scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
       doTrigger=False applyHltFilter=False $GUNFIELD \
       exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=$VARFAM \
       exportObjective=True "$@" ) > "$out/cmsrun.log" 2>&1
  echo "    rc=$? $out"
}

for lu in $MODES; do
  base="$OUTROOT/lu$lu"
  run_one "$base/nom" localUpdate=$lu &
  for d in $MATDELTAS; do
    for sgn in p m; do
      v=$( [[ $sgn == p ]] && echo "$d" || echo "-$d" )
      gf="$OUTROOT/groupfiles/g${MATGROUP}_${sgn}${d}.txt"
      [[ -f "$gf" ]] || mk_groupfile "$MATGROUP" "$v" "$gf"
      run_one "$base/mat_g${MATGROUP}_${sgn}${d}" localUpdate=$lu materialGroupsFile="$gf" &
    done
  done
  for fr in $FIELDRUNS; do
    m=${fr%%:*}; d=${fr##*:}
    for sgn in p m; do
      v=$( [[ $sgn == p ]] && echo "$d" || echo "-$d" )
      run_one "$base/field_m${m}_${sgn}${d}" localUpdate=$lu \
              injectFieldModes=$m injectFieldModeValues=$v &
    done
  done
done
wait
echo ">>> done"

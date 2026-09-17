#!/bin/bash
# THE CONCATENATED-TAU-GRID A/B.
#
# The four resolution-CF functionals of the two-track candidate (mass, vertex
# DCA, the two whitened beam-line pulls) used to cost one
# `cvhcf::trackExponents` pass EACH.  They now share ONE pass on the
# concatenated argument list { w_{b,k} tau_j } -- exact, not an approximation,
# because every exponent primitive reads the weight only through `w tau`.
#
# This script produces the OLD and the NEW build's output on the SAME inputs,
# for the numerical gate (`cmp_taugrid.py`), and -- with `timing` -- runs the
# interleaved pinned timing A/B.
#
#   ./run_taugrid_ab.sh gates            200 gun + 400 DY events, both builds
#   ./run_taugrid_ab.sh timing [nev] [cpu] [reps]
#
# OLD = the build the tail study runs from; NEW = the taugrid build area.
#
# NOTE (2026-09-13, after `f7fbef244c6` landed in dev2): dev2 now CARRIES the
# change, so $OLDA and $NEWA are the same code and a rerun measures nothing.
# To repeat the A/B, point OLDA at a build of dbdedfde3c1.
set -uo pipefail
OLDA=${OLDA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
NEWA=${NEWA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_taugrid}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
DYLIST=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
GUNLIST=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_jpsigun_ul16.txt
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/taugrid}
DYIN=$(sed -n 1p $DYLIST)
GUNIN=$(sed -n 1p $GUNLIST)

# `run_prod_dy.sh`'s configuration with the beam rows and their two residuals
# ON, so all four functionals are exercised.
DYBASE="numberOfThreads=1 doRes=True exportCfExponents=True \
exportStepRecords=False exportCfGroupExponents=True \
exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
fillJac=True fillGrads=False fillGradsFactored=True \
fitFromGenParms=False doSimHits=False doGen=True requireGen=False \
genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
scalarPot3DInitFile=$INIT"
# the gate configuration is the base plus all four functionals; the timing case
# appends its OWN switches, so they must not be in the base (VarParsing refuses
# a repeated assignment).
DYARGS="$DYBASE exportVtxResidual=True bsConstraint=True exportBsResidual=True"

# `run_prod.sh`'s J/psi-gun configuration, verbatim.  The gun driver has no
# `exportBsResidual` knob, so the gun gates the MASS and VERTEX functionals and
# the DY leg gates all four.
GUNARGS="numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
doTrigger=False applyHltFilter=False useIdealGeometry=True useDefaultField=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0 \
exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False"

one() {  # one <area> <cfg> <input> <outdir> <nev> <args...>
  local area=$1 cfg=$2 input=$3 outdir=$4 nev=$5; shift 5
  mkdir -p "$outdir"; rm -f "$outdir"/globalcor_*.root
  # shellcheck disable=SC2086
  CMSSW_AREA=$area $RUN_ONE "$cfg" "$input" "$outdir" nEvents=$nev "$@" \
      > "$outdir/local.log" 2>&1
  echo "$outdir rc=$? bytes=$(stat -c%s $outdir/globalcor_0.root 2>/dev/null)"
}

case ${1:-gates} in
gates|gates_old|gates_new)
  NGUN=${2:-200}; NDY=${3:-400}
  if [[ ${1:-gates} != gates_new ]]; then
    # shellcheck disable=SC2086
    one $OLDA $OLDA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py   "$GUNIN" $OUT/old_gun $NGUN $GUNARGS &
    # shellcheck disable=SC2086
    one $OLDA $OLDA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py "$DYIN" $OUT/old_dy  $NDY  $DYARGS &
    wait
  fi
  if [[ ${1:-gates} != gates_old ]]; then
    # shellcheck disable=SC2086
    one $NEWA $NEWA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py   "$GUNIN" $OUT/new_gun $NGUN $GUNARGS &
    # shellcheck disable=SC2086
    one $NEWA $NEWA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py "$DYIN" $OUT/new_dy  $NDY  $DYARGS &
    wait
  fi
  echo GATESDONE ;;
timing)
  # INTERLEAVED and PINNED.  The node load drifts (section 14's standing note:
  # a sequential A/B is only valid if the load is stable, and the way to test
  # that is to REPEAT THE FIRST CONFIGURATION AT THE END), so old and new
  # alternate on ONE cpu and `m4` is run again last.
  #
  # Three functional counts, so "the cost of the four relative to one" is
  # measured and not inferred:
  #   m1  mass only                         exportVtxResidual=False, bs off
  #   m2  mass + vertex DCA                 bs off       (the old +55/59 % base)
  #   m4  + the two whitened beam pulls     rows + residual (the full export)
  NEV=${2:-400}; CPU=${3:-3}
  CFGS=${CFGS:-"m4 m2 m1 m4"}
  echo "host $(cat /proc/sys/kernel/hostname)  cpu $CPU  nevents $NEV  seq: $CFGS"
  seq=0
  for cfgname in $CFGS; do
    seq=$((seq+1))
    case $cfgname in
      m4) X="exportVtxResidual=True bsConstraint=True exportBsResidual=True" ;;
      m2) X="exportVtxResidual=True bsConstraint=False" ;;
      m1) X="exportVtxResidual=False bsConstraint=False" ;;
    esac
    for build in old new; do
      [[ $build == old ]] && A=$OLDA || A=$NEWA
      d=$OUT/timing/s${seq}_${cfgname}_${build}; mkdir -p $d; rm -f $d/globalcor_*.root
      t0=$(date +%s)
      # shellcheck disable=SC2086
      CMSSW_AREA=$A taskset -c $CPU $RUN_ONE \
          "$A/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py" \
          "$DYIN" "$d" nEvents=$NEV $DYBASE $X > $d/local.log 2>&1
      t1=$(date +%s)
      echo "TIMING seq$seq $cfgname $build wall $((t1-t0)) end $t1 bytes $(stat -c%s $d/globalcor_0.root 2>/dev/null)"
    done
  done
  echo TIMINGDONE ;;
single)
  # THE n = 1 PATH.  The SINGLE-track maker is the other consumer of
  # `cvhcf::trackExponents`, and it calls the single-functional signature --
  # which is now `trackExponentsImpl(&in, 1, &out)`.  A short gen-closure run
  # on both builds gates that the delegation is bit-identical.
  NEV=${2:-200}
  SLIST=${SLIST:-/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_btojpsix_v3_chunk50.txt}
  SIN=$(sed -n 1p $SLIST)
  SARGS="numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
scalarPot3DInitFile=$INIT"
  for build in old new; do
    [[ $build == old ]] && A=$OLDA || A=$NEWA
    d=$OUT/single_$build; mkdir -p $d; rm -f $d/globalcor_*.root
    # shellcheck disable=SC2086
    CMSSW_AREA=$A $RUN_ONE "$A/src/Analysis/HitAnalyzer/test/runCvhResClosure.py" \
        "$SIN" "$d" nEvents=$NEV $SARGS > $d/local.log 2>&1
    echo "single_$build rc=$? bytes=$(ls -s $d/globalcor_*.root 2>/dev/null | head -1)"
  done
  echo SINGLEDONE ;;
cfblock)
  # HOW BIG IS THE CF BLOCK AT ALL?  `exportCfExponents=False` keeps every
  # functional'"'"'s influence loop, solve and variance share but makes NO
  # `cvhcf::trackExponents` call and writes no exponent branch, so
  #     cfon - cfoff  =  the evaluator + its export
  # and the rest of the beam functionals'"'"' cost is the two extra sparse
  # solves, the two extra per-block influence loops and their own branches.
  # The `cfoff` pair is a NULL: this change cannot touch it, so the difference
  # between the two builds there IS the node drift.
  NEV=${2:-400}; CPU=${3:-5}
  DYNOCF=${DYBASE/exportCfExponents=True/exportCfExponents=False}
  X="exportVtxResidual=True bsConstraint=True exportBsResidual=True"
  echo "host $(cat /proc/sys/kernel/hostname)  cpu $CPU  nevents $NEV  (cf block)"
  for tag in cfon_old cfoff_old cfon_new cfoff_new; do
    case $tag in
      cfon_old)  A=$OLDA; B="$DYBASE" ;;
      cfoff_old) A=$OLDA; B="$DYNOCF" ;;
      cfon_new)  A=$NEWA; B="$DYBASE" ;;
      cfoff_new) A=$NEWA; B="$DYNOCF" ;;
    esac
    d=$OUT/timing/cf_$tag; mkdir -p $d; rm -f $d/globalcor_*.root
    t0=$(date +%s)
    # shellcheck disable=SC2086
    CMSSW_AREA=$A taskset -c $CPU $RUN_ONE \
        "$A/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py" \
        "$DYIN" "$d" nEvents=$NEV $B $X > $d/local.log 2>&1
    t1=$(date +%s)
    echo "CFBLOCK $tag wall $((t1-t0)) end $t1 bytes $(stat -c%s $d/globalcor_0.root 2>/dev/null)"
  done
  echo CFBLOCKDONE ;;
*) echo "usage: $0 gates|timing"; exit 2 ;;
esac

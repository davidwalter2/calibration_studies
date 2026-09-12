#!/bin/bash
# THE EXPORT-CHANGE SMOKE.  Three cmsRun jobs that between them touch every
# code path the export changes reach, run against ONE CMSSW area and written
# into ONE output directory, so two areas can be compared branch by branch
# with `root_bitcompare.py`.
#
#   usage: smoke_exports_260906.sh <CMSSW_AREA> <OUTROOT> [gun_tt|gun_st|data_tt ...]
#
# The three:
#
#   gun_tt   the TWO-TRACK maker on one file of the 260905d J/psi gun
#            (`resolution_simprod_jpsigun_ul16/task_0000`), in the SAME
#            configuration `runs/stepdamp260905/slurm/submit_prod.sh` used --
#            doRes, the CF exponents, gen present.  This is the smoke for the
#            per-group exponents, the hit-class blocks, the per-leg covariance
#            and the pre-FSR mass.
#   gun_st   the SINGLE-TRACK maker on one file of the low-pT muon gun, same
#            configuration.  The q/p functional's half of the per-group split
#            (`cfqop_*`), and the parmtype-8/9 blocks in the maker that ALREADY
#            registers them.
#   data_tt  the two-track maker on 48 events of the 15_0-native 2016F
#            Charmonium ALCARECO (`...199B282B...`), doRes on.  Real data, real
#            geometry, real field: the path a production runs, and the one that
#            has no gen at all.
#
# INPUTS ARE READ FROM /work, NOT /ceph.  The login node's CephFS client is
# evicted often enough that a smoke reading /ceph can SILENTLY produce a valid
# empty output (see refit.sbatch's header).  `--stage` copies the three inputs
# once; after that the smoke has no ceph dependence at all.
set -uo pipefail

AREA=${1:?usage: smoke_exports_260906.sh <CMSSW_AREA> <OUTROOT> [which...]}
OUTROOT=${2:?usage: smoke_exports_260906.sh <CMSSW_AREA> <OUTROOT> [which...]}
shift 2
WHICH=("$@")
[[ ${#WHICH[@]} -eq 0 ]] && WHICH=(gun_tt gun_st data_tt)

STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
IN_GUN_TT=$STAGE/jpsigun_task0000_step2.root
IN_GUN_ST=$STAGE/mugun_task0000_step2.root
IN_DATA=$STAGE/alcareco_199B282B.root
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
TESTDIR=$AREA/src/Analysis/HitAnalyzer/test

# `doGen` and `requireGen` are hard-coded True in the MC drivers -- it is a
# gen-closure driver -- so the data smoke runs off a COPY with those two lines
# flipped.  WITHOUT the `requireGen` flip the maker rejects every candidate and
# writes a VALID, EMPTY tree, which a bit-comparison then passes vacuously --
# which is why the entry count is asserted below.  The copy is made from the
# area under test, so a driver change is exercised too.
mk_data_cfg() {
  local dst=$1
  sed -e 's/^    doGen=cms.bool(True),$/    doGen=cms.bool(False),/' \
      -e 's/^    requireGen=cms.bool(True),$/    requireGen=cms.bool(False),/' \
      "$TESTDIR/runCvhJpsiGenMC.py" > "$dst"
  grep -q "doGen=cms.bool(False)" "$dst" || { echo "[FATAL] doGen patch missed" >&2; exit 2; }
  grep -q "requireGen=cms.bool(False)" "$dst" || { echo "[FATAL] requireGen patch missed" >&2; exit 2; }
}

# Per-smoke option sets, each a FAITHFUL copy of the production it stands in
# for -- gun_tt/gun_st from `runs/stepdamp260905/slurm/{refit.sbatch,
# submit_prod.sh}` (fillGrads, the dense Hessian), data_tt from
# `production/config_jpsimc20M.sh` (fillGradsFactored, the B^T B one), so both
# Hessian export paths are exercised.  One thread everywhere: the Geant4e
# propagator has one master per job.
INIT_OPT="scalarPot3DInitFile=$INIT"
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"
# Extra cmsRun options appended to EVERY smoke, e.g.
#   EXTRA_OPTS="exportCfGroupExponents=True" smoke_exports_260906.sh ...
EXTRA_OPTS=${EXTRA_OPTS:-}

run_one() {
  local tag=$1 cfg=$2 input=$3 outsub=$4; shift 4
  local out="$OUTROOT/$outsub"
  mkdir -p "$out"
  rm -f "$out"/*.root
  echo ">>> [$tag] area=$AREA"
  ( cd "$out" \
    && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$out" \
    && "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
       cmsRun "$cfg" input="$input" "$@" ) > "$out/cmsrun.log" 2>&1
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
  echo "    rc=$rc entries=$nent  $(ls -la "$out"/*.root 2>/dev/null | awk '{print $5, $9}')"
  # AN EMPTY TREE IS A FAILED SMOKE, not a passing one: every branch of an
  # empty tree is trivially bit-identical to every branch of another empty
  # tree, so a vacuous comparison looks exactly like a clean one.
  if [[ "$nent" == "0" ]]; then
    echo "    [FATAL] $tag produced ZERO candidates -- the comparison would be vacuous" >&2
    return 5
  fi
  return $rc
}

mkdir -p "$OUTROOT"
rcall=0
for w in "${WHICH[@]}"; do
  case $w in
    gun_tt)
      run_one gun_tt "$TESTDIR/runCvhJpsiGenMC.py" "$IN_GUN_TT" gun_tt \
        nEvents=${NEV_GUN_TT:-60} numberOfThreads=1 doRes=True \
        exportCfExponents=True exportStepRecords=True \
        fillJac=True fillGrads=True fitFromGenParms=False CgfQoPMode=0 \
        $INIT_OPT trackSrc=generalTracks useLegacyPairLoop=True \
        doTrigger=False applyHltFilter=False $GUNFIELD $EXTRA_OPTS || rcall=1 ;;
    gun_st)
      run_one gun_st "$TESTDIR/runCvhResClosure.py" "$IN_GUN_ST" gun_st \
        nEvents=${NEV_GUN_ST:-60} numberOfThreads=1 doRes=True \
        exportCfExponents=True exportStepRecords=True \
        fillGrads=True fitFromGenParms=False \
        $INIT_OPT trackSrc=generalTracks $GUNFIELD $EXTRA_OPTS || rcall=1 ;;
    data_tt)
      mk_data_cfg "$OUTROOT/runCvhJpsiGenMC_nogen.py"
      run_one data_tt "$OUTROOT/runCvhJpsiGenMC_nogen.py" "$IN_DATA" data_tt \
        nEvents=${NEV_DATA:-48} numberOfThreads=1 doRes=True \
        exportCfExponents=True exportStepRecords=True \
        fillJac=True fillGrads=False fillGradsFactored=True \
        fitFromGenParms=False CgfQoPMode=0 $INIT_OPT \
        trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True doTrigger=True \
        applyHltFilter=False globalTag=auto:run2_data $EXTRA_OPTS || rcall=1 ;;
    *) echo "[FATAL] unknown smoke '$w'" >&2; exit 2 ;;
  esac
done
exit $rcall

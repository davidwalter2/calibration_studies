#!/bin/bash
# Thread scan for the CVH refit in CMSSW_15: throughput and memory vs
# numberOfThreads, on ONE chunk of a production, run one configuration at a
# time on a quiet node.
#
# WHY THE MEASUREMENT IS SHAPED THIS WAY
#   * `numberOfStreams` FOLLOWS `numberOfThreads` in both drivers, so N is one
#     knob, and the maker writes ONE OUTPUT FILE PER STREAM
#     (`<outprefix>_<stream>.root`, ResidualGlobalCorrectionMakerBase.cc).
#     An N-thread task therefore produces N .root files, which the slurm/condor
#     wrapper's `*.root` glob already tolerates but which every downstream
#     reader must expect.
#   * Neither driver loads the `Timing` or `SimpleMemoryCheck` services, and
#     adding them would change the configuration under test, so the numbers
#     come from `/usr/bin/time -v` (wall, user+sys CPU, peak RSS = VmHWM) plus
#     a 2 s RSS poller for the shape of the ramp. VmHWM is the process high
#     water mark: threads share pages, so this is exactly the per-TASK figure a
#     batch system's memory request has to cover.
#   * One configuration at a time. The whole point is the wall time, and two
#     concurrent arms would contend for the same cores and the same ceph reads.
#
# usage: run_scan.sh <dy|jpsi> <nevents> <outbase> [threads...]
set -uo pipefail

CHANNEL=$1 ; NEV=$2 ; OUTBASE=$3 ; shift 3
THREADS=("$@") ; [[ ${#THREADS[@]} -gt 0 ]] || THREADS=(1 2 4 8)

PROD=/work/submit/david_w/ZMass/calibration_studies/production
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt

# The re-production switches (PRODUCTIONS.md §3) on top of the running
# production's configuration, with numberOfThreads left OUT -- it is the scan
# variable and is appended per arm.
NEWEXPORTS="exportCfGroupExponents=True exportMaterialNoise=True \
 exportVarianceGrads=True varianceGradFamilies=15"

case "$CHANNEL" in
  dy)
    CFG=$AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py
    CHUNKS=$PROD/chunks_dymc_8p5M_260905.txt
    EXTRA="doRes=True exportCfExponents=True exportStepRecords=False \
 $NEWEXPORTS \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False doSimHits=False \
 doGen=True requireGen=False \
 genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
 doTrigger=False applyHltFilter=False \
 massMin=60 massMax=120 \
 useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
 doMassConstraint=False \
 CgfQoPMode=0 tightG4eStepper=True \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT" ;;
  jpsi)
    CFG=$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
    CHUNKS=$PROD/chunks_jpsimc_20M_260905.txt
    EXTRA="doRes=True exportCfExponents=True exportStepRecords=False \
 $NEWEXPORTS \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False \
 trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True \
 doTrigger=True applyHltFilter=False doSimHits=False \
 useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
 doMassConstraint=False \
 CgfQoPMode=0 \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT" ;;
  *) echo "unknown channel $CHANNEL" >&2 ; exit 2 ;;
esac

read -r INPUT _SKIP _NEV _rest < <(sed -n '1p' "$CHUNKS")
[[ -r "$INPUT" ]] || { echo "[FATAL] input unreadable on $(hostname): $INPUT" >&2; exit 3; }

mkdir -p "$OUTBASE"
echo "### thread scan  channel=$CHANNEL  host=$(hostname)  nev=$NEV  $(date)"
echo "### input=$INPUT"
echo "### cfg=$CFG"

for N in "${THREADS[@]}"; do
  OUT="$OUTBASE/t$N"
  rm -rf "$OUT" ; mkdir -p "$OUT"
  echo "=== threads=$N  start $(date +%s)  $(date)"
  ( cd "$OUT" && \
    source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && \
    cd "$AREA/src" && eval $(scramv1 runtime -sh) && cd "$OUT" && \
    ( /usr/bin/time -v cmsRun "$CFG" input="$INPUT" skipEvents=0 nEvents="$NEV" \
        numberOfThreads="$N" $EXTRA > cmsrun.log 2> time.log ) ; \
    echo "rc=$?" >> time.log )
  # RSS from the kernel high-water mark, echoed by /usr/bin/time -v as
  # "Maximum resident set size (kbytes)".
  echo "=== threads=$N  end   $(date +%s)  $(date)"
  grep -E "Elapsed \(wall|User time|System time|Maximum resident|Percent of CPU|rc=" "$OUT/time.log"
  ls -la "$OUT"/*.root 2>/dev/null | awk '{print "    out:",$NF,$5}'
done
echo "### scan done $(date)"

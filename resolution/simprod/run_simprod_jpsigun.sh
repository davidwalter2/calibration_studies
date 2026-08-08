#!/bin/bash
# Z-MOMENTUM single-muon private production, for the single-track PDF test at
# the momenta the Z mass measurement actually uses.
#
# Why this sample exists. Every track-level / candidate-level closure so far
# runs on B->J/psi+X or a J/psi gun, i.e. muons at p ~ 3-15 GeV. The radiative
# (brems + pair) mean-vs-mode bias measured on 2026-08-06 is
#   8.2e-6 at pT=10, 1.11e-5 at pT=40, 1.37e-5 at pT=100
# -- i.e. AT the 1e-5 Z-mass target and pT-DEPENDENT, so it does not cancel
# when a J/psi-derived calibration is extrapolated to Z muons. There is
# currently no single-track PDF test at those momenta. This makes one.
#
# Identical recipe to the rung-E J/psi gun (run_simprod.sh): CMSSW_15_0 so the
# simulation Geant4 matches the CVH refit propagator, auto:run2_design GT,
# ideal geometry, DB grid field, NoPileUp, PSimHits kept. The ONLY change is
# the generator: a flat-pT muon gun over the Z-muon range instead of J/psi.
#
# usage: ./run_simprod_mugun.sh [nparallel] [task_from] [task_to] [nevents] [ptmin] [ptmax]
#   env: OUTROOT, PTMIN/PTMAX (the J/psi pT, not the muon pT)
set -euo pipefail
NPAR=${1:-12}
FROM=${2:-0}
TO=${3:-11}
NEVT=${4:-4000}
# J/psi pT, not muon pT. 5-30 is the gun's own default and spans the
# TkAlJpsiMuMu ALCARECO range the calibration actually uses.
PTMIN=${5:-5}
PTMAX=${6:-30}
# BOTH CHARGES by default (2026-08-07). The original sample was mu- only
# (ParticleID=13, AddAntiParticle=False), which made q+ = 0 and left the
# charge parity of any bias UNTESTABLE -- and charge parity is the natural
# discriminator between a curvature/field-like effect (charge-ODD) and a
# material/energy-loss-like one (charge-EVEN). This blocked the diagnosis of
# the +-0.07 pull-unit (dp/p ~ 7e-4) PHI-dependent bias that survives
# matching the SIM and refit field models.
# Listing both IDs makes Pythia8PtGun emit one particle PER ID per event,
# each with INDEPENDENTLY sampled pt/eta/phi -- preferable to
# AddAntiParticle=True, which would mirror the momentum (eta -> -eta,
# phi -> phi+pi) and correlate the two charges' kinematics.
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
SIMPROD=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16}

run_task() {
  local idx=$1
  local outdir="$OUTROOT/task_$(printf '%04d' "$idx")"
  # -s on step2.root is SAFE here (unlike the trackres resume): step2 is written
  # to step2.tmp.root and renamed only on success, so a killed task never leaves
  # a non-empty truncated step2.root. Checked before the stagger sleep so a
  # resume pass skips instantly.
  [[ -s "$outdir/step2.root" ]] && { echo "[skip] task $idx"; return 0; }
  # STAGGERED START. xargs -P fires all NPAR jobs at once; at 60+ that wedged
  # every one in futex_do_wait inside the NSS/sssd lookup (0 s CPU in 40 min)
  # while a single job ran at 90%. Spreading the starts fixes it; measured 40-way
  # at 95-98% CPU per job.
  sleep $(( (idx % ${NPAR:-8}) * ${STAGGER:-3} ))
  mkdir -p "$outdir"
  cd "$outdir"
  {
    cat "$SIMPROD/step1_gensim.py"
    echo ""
    echo "# --- J/psi -> mu mu gun: step1_gensim.py ALREADY is this gun"
    echo "# ParticleID stays 443 and the jpsiDecay block stays ENABLED --"
    echo "# step1_gensim.py was generated from simprod_JpsiGun_cfi.py and is"
    echo "# already a flat-pT J/psi -> mu mu gun. The muon samples were made by"
    echo "# OVERRIDING it; here we simply do not."
    echo "process.generator.PGunParameters.MinPt = cms.double($PTMIN)"
    echo "process.generator.PGunParameters.MaxPt = cms.double($PTMAX)"
    echo "process.generator.psethack = cms.string('Jpsi->mumu flat pT gun, pT $PTMIN-$PTMAX')"
    echo ""
    echo "process.RandomNumberGeneratorService.generator.initialSeed = $((91300 + idx))"
    echo "process.RandomNumberGeneratorService.g4SimHits.initialSeed = $((94600 + idx))"
    echo "process.RandomNumberGeneratorService.VtxSmeared.initialSeed = $((97200 + idx))"
    echo "process.source.firstLuminosityBlock = cms.untracked.uint32($((idx + 1)))"
    echo "process.maxEvents.input = cms.untracked.int32($NEVT)"
  } > step1_task.py
  {
    cat "$SIMPROD/step2_digireco.py"
    echo ""
    echo "process.maxEvents.input = cms.untracked.int32(-1)"
  } > step2_task.py
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  pushd "$CMSSW_AREA/src" > /dev/null
  eval "$(scramv1 runtime -sh)"
  popd > /dev/null
  echo "[run ] task $idx step1"
  cmsRun step1_task.py > step1.log 2>&1 || { echo "[FAIL] task $idx step1"; return 1; }
  echo "[run ] task $idx step2"
  # Write step2 to a temp name and rename on success. cmsRun creates its output
  # file at START, so a crashed/running task leaves a NON-EMPTY step2.root that
  # the `-s` skip test above would accept -- a resumed run would then silently
  # keep a truncated sample. (Same trap as the clean-prop sim outputs.)
  sed -i "s|fileName = cms.untracked.string('file:step2.root')|fileName = cms.untracked.string('file:step2.tmp.root')|" step2_task.py
  cmsRun step2_task.py > step2.log 2>&1 || { echo "[FAIL] task $idx step2"; rm -f step2.tmp.root; return 1; }
  mv -f step2.tmp.root step2.root
  rm -f step1.root   # keep only the RECO+simhits output
  echo "[done] task $idx"
}
export -f run_task
export CMSSW_AREA SIMPROD OUTROOT NEVT PTMIN PTMAX SIMPROD_GT NPAR STAGGER
seq "$FROM" "$TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "all mugun simprod tasks finished -> $OUTROOT"

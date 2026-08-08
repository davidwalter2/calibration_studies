#!/bin/bash
# Kaon / pion / proton flat-pT gun samples, GEN-SIM-DIGI-RECO.
#
# Why these exist. Until now only MUONS had a full gun sample; the hadron
# work (2026-08-06 pT/eta scan) was all `cleanprop`, i.e. single-track
# Geant4 propagation with NO digitisation, NO reconstruction and NO fit.
# That tests the propagation and energy-loss MODEL but leaves the FIT
# untested on hadrons -- which matters for two channels the analysis needs:
#   * the kaon in B+- -> J/psi K+-, the channel that breaks the A-epsilon
#     degeneracy (kaons have different dE/dx than muons at the same pT);
#   * protons from Lambda0 -> p pi.
# Pions are included as the third species and as the decay-in-flight
# reference for the kink finder.
#
# pT 2-20 GeV, not the muon sample's 20-60: these species are used at the
# momenta where they actually occur (the cleanprop scan points were K 3/10,
# pi 3/10, p 2/3/5 GeV). Hadrons also undergo NUCLEAR INTERACTIONS in the
# tracker, so expect a substantially lower usable-track yield than muons --
# that attrition is physics, not a failure.
#
# Conditions/geometry are inherited from run_simprod_mugun.sh, i.e. the
# release-native 150X_mcRun2_asymptotic_v1 whose UL16 pixel payloads match
# the CVH fit (see step1_gensim.py for why this matters).
#
# usage: ./run_simprod_hadrons.sh [nparallel_per_species] [ntasks] [nevents]
set -euo pipefail
NPAR=${1:-120}
NTASK=${2:-160}
NEVT=${3:-1000}
PTMIN=${PTMIN:-2}
PTMAX=${PTMAX:-20}
SIMPROD=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOGD=${LOGD:-/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad}

# name:pdg-id-pair  (both charges, independently sampled -- see the
# PARTICLE_IDS note in run_simprod_mugun.sh)
SPECIES=("kaon:321, -321" "pion:211, -211" "proton:2212, -2212")

pids=()
for spec in "${SPECIES[@]}"; do
  name="${spec%%:*}"
  ids="${spec#*:}"
  echo "[hadrons] launching $name  ids='$ids'  ${NTASK} tasks x ${NEVT} evt  pT ${PTMIN}-${PTMAX}"
  PARTICLE_IDS="$ids" \
  OUTROOT="$CEPH/resolution_simprod_${name}gun_ul16" \
    "$SIMPROD/run_simprod_mugun.sh" "$NPAR" 0 $((NTASK - 1)) "$NEVT" "$PTMIN" "$PTMAX" \
      > "$LOGD/simprod_${name}.log" 2>&1 &
  pids+=($!)
done

echo "[hadrons] pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

for spec in "${SPECIES[@]}"; do
  name="${spec%%:*}"
  d="$CEPH/resolution_simprod_${name}gun_ul16"
  n=$(ls "$d"/task_*/step2.root 2>/dev/null | wc -l)
  echo "[hadrons] $name: $n/$NTASK step2.root  ($(grep -c FAIL "$LOGD/simprod_${name}.log" 2>/dev/null || echo 0) FAIL)"
  ls "$d"/task_*/step2.root 2>/dev/null | sort > "$SIMPROD/filelist_${name}gun_ul16.txt"
done
exit $fail

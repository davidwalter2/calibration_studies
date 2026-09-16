#!/bin/bash
# Regenerate the clean-propagation MODEL files against the current propagator.
#
# WHY. The clean-propagation test compares a Geant4 SIM sample
# against a deterministic MODEL job that exports the per-step transport
# Jacobians and physics records. The two age differently:
#
#   * the SIM is Geant4 GROUND TRUTH. runCleanPropSim.py is stable, so every
#     existing sim sample was produced by identical code. They do NOT need
#     regenerating, and at 100-128 files each that would be hours of compute
#     for no change.
#   * the MODEL carries the exported records, so it goes stale whenever the
#     propagator's export changes. A model file that predates the radiative
#     export or the per-element Moliere sums has msmoliv stride 8 instead of
#     10; until it is regenerated the offline model silently falls back to the
#     effZ approximation of the screening term, which is precisely the thing
#     under test.
#
# Each sample has its OWN phi (0.70 / 0.20 / 0.50 / 0.10 / ...), so the table
# below is explicit rather than derived -- getting phi wrong would silently
# compare a model on one material path against a sim on another.
#
# NOTE this does NOT fix the confound that the eta=0.30 samples sit at four
# different phi, so momentum and material path cannot be separated with them.
# That needs new SIM samples at fixed (eta, phi) across pT.
#
# usage: ./regen_models.sh [nparallel]
set -euo pipefail
NPAR=${1:-12}
CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/test/runCleanPropModel.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
TGT=/work/submit/david_w/ZMass/calibration_studies/resolution/cleanprop/targets
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/model

# name : pt : eta : phi : pdg   (phi MUST match the corresponding sim sample)
JOBS=(
  "model_mu_pt3_eta0.30:3:0.30:0.70:13:targets_mu_pt3_eta0.30.txt"
  # DELIBERATELY ABSENT: model_pt10_eta0.30_phi0.20. That point is phi=0.20
  # and there is no targets_mu_pt10_eta0.30_phi0.20.txt for it. Adding it with
  # the phi=0.70 targets (targets_mu_pt3_eta0.30.txt) is exactly the confound
  # the header warns about, and the [[ -f ]] guard below does NOT catch it
  # because that file exists: the model is then built on a DIFFERENT material
  # path than the sim (20 sim planes vs 19 model legs, detids disagreeing from
  # index 0). Add it only together with its own targets file.
  "model_mu_pt40_eta0.30:40:0.30:0.50:13:targets_mu_pt40_eta0.30.txt"
  "model_mu_pt100_eta0.30:100:0.30:0.10:13:targets_mu_pt100_eta0.30.txt"
  "model_mu_pt10_eta1.00:10:1.00:0.10:13:targets_mu_pt10_eta1.00.txt"
  "model_mu_pt10_eta1.60:10:1.60:0.10:13:targets_mu_pt10_eta1.60.txt"
  "model_pi_pt3_eta0.30:3:0.30:0.70:-211:targets_pi_pt3_eta0.30.txt"
  "model_pi_pt10_eta0.30:10:0.30:0.20:-211:targets_pi_pt10_eta0.30.txt"
  "model_K_pt3_eta0.30:3:0.30:0.70:-321:targets_K_pt3_eta0.30.txt"
  "model_K_pt10_eta0.30:10:0.30:0.20:-321:targets_K_pt10_eta0.30.txt"
  "model_p_pt2_eta0.30:2:0.30:0.90:2212:targets_p_pt2_eta0.30.txt"
  "model_p_pt3_eta0.30:3:0.30:1.00:2212:targets_p_pt3_eta0.30.txt"
  "model_p_pt5_eta0.30:5:0.30:0.80:2212:targets_p_pt5_eta0.30.txt"
)

mkdir -p "$OUT" "$OUT/old"
run_one() {
  IFS=: read -r name pt eta phi pdg tgt <<< "$1"
  [[ -f "$TGT/$tgt" ]] || { echo "[skip] $name (no targets $tgt)"; return 0; }
  # keep the previous model beside the new one rather than clobbering it:
  # a stride-8 file is still readable and is the only record of what a
  # comparison made with it actually used.
  [[ -f "$OUT/$name.root" ]] && cp -n "$OUT/$name.root" "$OUT/old/$name.root" 2>/dev/null || true
  (
    cd "$AREA/src"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    eval "$(scramv1 runtime -sh)"
    cd "$OUT"
    cmsRun "$CFG" pt="$pt" eta="$eta" phi="$phi" partId="$pdg" \
        targets="$TGT/$tgt" output="$OUT/$name.tmp.root" \
        > "$OUT/$name.regen.log" 2>&1
  ) || { echo "[FAIL] $name"; rm -f "$OUT/$name.tmp.root"; return 1; }
  mv -f "$OUT/$name.tmp.root" "$OUT/$name.root"
  echo "[done] $name"
}
export -f run_one
export CFG AREA TGT OUT

printf '%s\n' "${JOBS[@]}" | xargs -P "$NPAR" -I{} bash -c 'run_one "$@"' _ {}
echo "model regeneration finished -> $OUT"

#!/bin/bash
# Drive the full clean-propagation campaign: for each (pt, eta, phi, species)
# point, run the ground-truth simulation, extract the target surfaces, run the
# one deterministic model propagation, and compare.
#
# Each stage skips if its output already exists, so the script is resumable and
# safe to re-run after a single point fails.
#
# usage: ./run_campaign.sh <points-file> [nev_total] [tag]
#
# points-file lines:  <pt> <eta> <phi> <partId> <label>
#   e.g.  40 0.30 0.20 13   pt40
#         2  0.30 0.20 -321 K2
# '#' comments and blank lines are ignored.
set -euo pipefail

POINTS=${1:?usage: run_campaign.sh <points-file> [nev_total] [tag]}
NEVTOT=${2:-200000}
TAG=${3:-$(date +%y%m%d)}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
# The two sides need different environments: cmsRun needs the CMSSW area,
# cf_propagation_test.py needs the calibration_studies venv (uproot, wums).
# Each is sourced inside its own subshell so they never collide.
ENVSH="$HERE/../../setup_env.sh"
MODELCFG="$AREA/src/Analysis/HitAnalyzer/test/runCleanPropModel.py"
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop
OUTDIR=${OUTDIR:-$HOME/public_html/calibration_studies/${TAG}_cleanprop_scan}
NTASK=${NTASK:-50}
NPAR=${NPAR:-50}
NEV=$(( NEVTOT / NTASK ))

mkdir -p "$OUTDIR" "$CEPH/model" "$HERE/targets"

while read -r PT ETA PHI PDG LABEL; do
  [[ -z "${PT:-}" || "$PT" == \#* ]] && continue
  echo "=============================================================="
  echo "point $LABEL : pt=$PT eta=$ETA phi=$PHI pdg=$PDG"

  SIMDIR="$CEPH/sim_${TAG}_pt${PT}_eta${ETA}_phi${PHI}"
  [[ "$PDG" != "13" ]] && SIMDIR="${SIMDIR}_pdg${PDG}"
  TGT="$HERE/targets/targets_${LABEL}.txt"
  MODEL="$CEPH/model/model_${LABEL}.root"

  # 1. ground truth
  if [[ "$(ls "$SIMDIR"/simstates_*.root 2>/dev/null | wc -l)" -lt "$NTASK" ]]; then
    NPAR=$NPAR "$HERE/run_cleanprop_sim.sh" "$NTASK" "$NEV" "$ETA" "$PHI" "$PT" "$TAG" "$PDG"
  else
    echo "[skip sim] $SIMDIR"
  fi

  # 2. target surfaces (modal crossed-module sequence + entry local z)
  if [[ ! -s "$TGT" ]]; then
    # a failure here must not abort the remaining points (set -e would)
    ( set +eu; source "$ENVSH" >/dev/null 2>&1; set -e
      cd "$HERE/.." && python cf_propagation_test.py --targets \
        --sim "$(ls "$SIMDIR"/simstates_*.root | head -1)" --out "$TGT" ) \
      || { echo "[TARGETS FAIL] $LABEL"; continue; }
  else
    echo "[skip targets] $TGT"
  fi

  # 3. deterministic model propagation (one event)
  if [[ ! -s "$MODEL" ]]; then
    ( set +eu; cd "$AREA/src"
      source /cvmfs/cms.cern.ch/cmsset_default.sh
      eval "$(scramv1 runtime -sh)"; set -e
      cmsRun "$MODELCFG" pt="$PT" eta="$ETA" phi="$PHI" partId="$PDG" \
          targets="$TGT" output="$MODEL" > "$OUTDIR/model_${LABEL}.log" 2>&1
    ) && echo "[model ok] $LABEL" || { echo "[MODEL FAIL] $LABEL"; continue; }
  else
    echo "[skip model] $MODEL"
  fi

  # 4. comparison, in transform space
  ( set +eu; source "$ENVSH" >/dev/null 2>&1; set -e
    cd "$HERE/.." && python cf_propagation_test.py --compare \
      --sim "$SIMDIR/simstates_*.root" --model "$MODEL" \
      --functionals qop locx dxdz --label "$LABEL" \
      --outpath "$OUTDIR" --postfix "_$LABEL" ) \
    > "$OUTDIR/compare_${LABEL}.log" 2>&1 \
    && echo "[compare ok] $LABEL" || echo "[COMPARE FAIL] $LABEL"
done < "$POINTS"

echo "=============================================================="
echo "campaign done -> $OUTDIR"

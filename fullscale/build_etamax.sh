#!/bin/bash
# THE SAFE BAND DEFINITION. `--eta-lead` cuts on the |eta| of the leg with the
# larger RECO pT, so when the legs have similar pT the band edge is decided by
# which one fluctuated up: corr(|eta| lead, z) = +0.0203, against +0.0025 for
# max(|eta_p|,|eta_m|) and +0.0004 for the gen definition.
# These cards repeat the eta bands and the barrel sigma/m split on the safe
# variable, which needs no truth.
#
# The measured per-leg charge-even momentum bias predicts what the fits must
# give: the band spread should fall from +67.8 MeV to -8.2 MeV, and the barrel
# sigma/m split from -36.3 MeV to +18.6 MeV (it does NOT vanish, because the
# sigma/m cut is itself a cut on the residual whatever the band variable is).
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
export PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:/work/submit/david_w/WRemnants_dev/wums:$FS/../env_tf/pypath:$FS/../resolution
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=${THREADS:-8} TF_NUM_INTRAOP_THREADS=${THREADS:-8}
P=/opt/venv/bin/python3
card () {  # tag  extra-args...
  local tag=$1; shift
  [ -f "$FS/cards/z_$tag.hdf5" ] && { echo "[card] $tag exists"; return; }
  echo "[card] $tag  $(date +%H:%M:%S)"
  $P -u $FS/make_card.py --pairs $FS/runs/zpairs_dyv2_jac_full.npz \
      --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
      --shape 5 --vpow 1.264 "$@" -o "$FS/cards/z_$tag.hdf5" \
      2>&1 | tee $FS/logs/card_$tag.log | grep -E "selection|candidates,|wrote|->|max\(" 
}
card VX_etaB --eta-max 0 0.9
card VX_etaT --eta-max 0.9 1.6
card VX_etaE --eta-max 1.6 3.0
# and the barrel sigma/m split on the SAFE band; the median sigma/m of THIS
# barrel is 0.00853, not the 0.01032 of the leading-pT barrel
card VX_etaB_slo --eta-max 0 0.9 --max-sigma-rel 0.00853
card VX_etaB_shi --eta-max 0 0.9 --min-sigma-rel 0.00853
echo "=== done $(date +%H:%M:%S)"
ls -la $FS/cards/z_VX_*.hdf5

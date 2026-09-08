#!/bin/bash
# Does the v-form residual track |eta| or sigma/m? They are nearly collinear in
# this sample (barrel sigma/m q25-q75 0.0087-0.0119, endcap 0.0136-0.0215), so
# the discriminating test is a sigma/m split AT FIXED eta: the BARREL, cut at
# its own median sigma/m = 0.01032. If m_Z moves strongly between the two
# halves the residual is a resolution effect and eta is only its proxy; if it
# does not, it is genuinely eta.
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
      2>&1 | tee $FS/logs/card_$tag.log | grep -E "selection|candidates,|wrote|->"
}
card V_etaB_slo --eta-lead 0 0.9 --max-sigma-rel 0.01032
card V_etaB_shi --eta-lead 0 0.9 --min-sigma-rel 0.01032
# and the same split on the ENDCAP, where the mismatch p-(1+f) changes sign
card V_etaE_slo --eta-lead 1.6 3.0 --max-sigma-rel 0.01658
card V_etaE_shi --eta-lead 1.6 3.0 --min-sigma-rel 0.01658

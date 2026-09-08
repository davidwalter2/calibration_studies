#!/bin/bash
# The kernel rebuilt from the SELECTED candidates' own gen record.
#
# The FSR kernel and the acceptance in use are measured on GENERATOR events in
# a gen fiducial (`pT > 5`, `|eta| < 2.4`).  The selection that defines the
# fitted sample acts on RECONSTRUCTED muons, and the difference correlates with
# the radiation: measured on this sample, `<u> = <-ln(m_post/m_pre)>` over the
# same window is 0.014476 in the barrel against the model's 0.014276, and
# 0.014160 against 0.014164 in the endcap -- a +2.0e-4 barrel excess and an
# endcap that agrees, which is the eta pattern and the sign of the -11 MeV.
#
# `zchannel/kern_from_selected.py` measures both objects on the selected
# candidates instead:
#   * the kernel, banded in `m_pre` exactly as the one in use;
#   * `A(m_pre) = P(selected | m_pre)`, which is a TOP-HAT (the production's
#     60-120 mass cut seen through `m_post ~ m_pre`) and therefore a `grid`
#     acceptance, not a Bernstein: no degree-8 polynomial fits a step
#     (chi2/ndf = 30638/184).
#
# The window truncation stays on. With the window baked into A and the kernel
# the modelled density is already concentrated inside it, so `norm_window` is
# very nearly a no-op rather than a second cut; what it still does is remove
# the mass the RESOLUTION convolution pushes over the edge, which is what it is
# for.
#
# usage: ./run_kernfix.sh [cards|fits|bands]
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
RUN=$FS/run_tf.sh
PAIRS=$FS/runs/zpairs_dyv2_jac_full.npz
CARDS=$FS/cards; LOGS=$FS/logs; RES=$FS/results
KFIX="k_hit k_ms k_ioni k_rad"
mkdir -p "$CARDS" "$LOGS" "$RES"

card () {   # card <tag> <fsr> <acc> <extra make_card args...>
  local tag=$1 fsr=$2 acc=$3; shift 3
  [ -f "$CARDS/z_$tag.hdf5" ] && { echo "[card] $tag exists"; return; }
  echo "[card] $tag  $(date +%H:%M:%S)"
  THREADS=${THREADS:-32} $RUN python3 -u "$FS/make_card.py" --pairs "$PAIRS" \
      --fsr "$fsr" --acc "$acc" --shape 5 \
      -o "$CARDS/z_$tag.hdf5" "$@" 2>&1 | tee "$LOGS/card_$tag.log"
}

fit () {    # fit <tag> <card tag> <extra fit args...>
  local tag=$1 ct=$2; shift 2
  echo "[fit] $tag  $(date +%H:%M:%S)"
  THREADS=${THREADS:-48} $RUN python3 -u "$FS/fit.py" --card "$CARDS/z_$ct.hdf5" \
      --chunk "${CHUNK:-262144}" --fix $KFIX "$@" --label "$tag" \
      -o "$RES/fit_$tag.json" 2>&1 | tee "$LOGS/fit_$tag.log"
}

KS=$Z/data/kern_selected_band3.3e-4.npz
AS=$Z/data/acc_selected_grid.json
KL=$Z/data/kern_loose_band3.3e-4.npz
AL=$Z/data/acc_loose_d8.json

for st in "${@:-cards}"; do
case $st in
  cards)
    # the A/B pair at 300 k: the same candidates, only the kernel changes
    card kf300 "$KS" "$AS" --maxn 300000 --chunk 262144
    card kl300 "$KL" "$AL" --maxn 300000 --chunk 262144
    ;;
  full)
    card kf380 "$KS" "$AS" --chunk 32768
    ;;
  bands)
    for b in B:0:0.9 T:0.9:1.6 E:1.6:3.0; do
      IFS=: read -r n lo hi <<<"$b"
      card kf_eta$n "$Z/data/kern_sel_eta$n.npz" "$Z/data/acc_sel_eta$n.json" \
          --eta-lead "$lo" "$hi" --maxn 300000 --chunk 262144
      card kl_eta$n "$KL" "$AL" --eta-lead "$lo" "$hi" --maxn 300000 --chunk 262144
    done
    ;;
  fits)
    fit kf300 kf300
    fit kl300 kl300
    ;;
  bandfits)
    for n in B T E; do
      fit kf_eta$n kf_eta$n
      fit kl_eta$n kl_eta$n
    done
    ;;
esac
done

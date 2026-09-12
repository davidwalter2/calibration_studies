#!/bin/bash
# THE KERNEL-FREE DETECTOR TEST, PER BAND.
#
# `--residual-mode` models `m_reco - m_gen` directly against the per-candidate
# resolution CF: the kernel is a DELTA and the FSR fold, the acceptance and
# K(m) are all off, so nothing of the mass model survives and what is measured
# is the detector half alone. Inclusively it closes at +0.88 +- 2.12 MeV while
# the full fit gives -11.06 +- 2.27.
#
# Run PER eta band and PER sigma/m tertile, the question is which side of the
# combination the -21/+12/+34 pattern lives on:
#   * if the kernel-free fit shows the pattern, the a / Jensen coefficients are
#     wrong as a function of sigma/m (a coefficient error proportional to
#     sigma/m gives exactly a bias linear in sigma/m) and the fix is on the
#     DETECTOR side;
#   * if it is flat per band, the pattern lives only in the COMBINATION with
#     the kernel, and the sigma/m split at fixed eta, the per-band K ladder and
#     the per-band FSR kernel rows decide between kernel-side causes.
#
# `--maxn 700000` per band: sigma(alpha) scales as the inclusive 2.12 MeV at
# 297 557, so 700 k gives ~1.4 MeV -- ample against a 55 MeV spread, and it
# keeps each card under 1 GB (submit's /work quota is the binding constraint).
# `--floor-scale 1e-7`. At the default 1e-9 the softplus positivity floor
# UNDERFLOWS and the NLL is `inf` before any minimiser runs: three candidates
# in 695 757 have a NEGATIVE density from CF ringing (li = -5.7e-5, -2.4e-5,
# -1.9e-5), all of them large-sigma (2.3-2.4 GeV), near-pure-Gaussian
# (vgf 0.91-0.94) candidates at ~4 sigma, and `s*softplus(li/s)` with
# s = 1e-9 is exactly 0 there. The scale is SET by float64 underflow, not
# chosen: `s*softplus(li/s)` is exactly 0 once li/s < -745, and it distorts
# the density by more than 1 % once li < 4.6 s. So s must satisfy
# 5.7e-5/745 = 7.7e-8 < s and keep 4.6 s below the densities that matter.
# s = 1e-7 handles li down to -7.45e-5 and distorts only below li = 4.6e-7 --
# beyond ~5.2 sigma, where nothing but the three pathologies lives.
# (s = 1e-4 is WRONG: it inflates every density below 4.6e-4, i.e. everything
# past ~4 sigma, and takes the fitted alpha to +71 +- 25 MeV.) `--floor clip` would not do either: max(li, 0) = 0 gives
# log 0 as well.
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
export PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:/work/submit/david_w/WRemnants_dev/wums:$FS/../env_tf/pypath:$FS/../resolution
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=${THREADS:-8} TF_NUM_INTRAOP_THREADS=${THREADS:-8}
P=/opt/venv/bin/python3
card () {  # tag  extra-args...
  local tag=$1; shift
  [ -f "$FS/cards/z_$tag.hdf5" ] && { echo "[card] $tag exists"; return; }
  echo "[card] $tag  $(date +%H:%M:%S)"
  $P -u $FS/make_card.py --pairs $FS/runs/zpairs_dyv2_full.npz \
      --residual-mode --maxn 700000 --floor-scale 1e-7 "$@" -o "$FS/cards/z_$tag.hdf5" \
      2>&1 | tee $FS/logs/card_$tag.log | grep -E "RESIDUAL|candidates,|->|keeps"
}
# the three eta bands, the same cuts as z_M_eta* / z_V_eta*
card residB --eta-lead 0   0.9
card residT --eta-lead 0.9 1.6
card residE --eta-lead 1.6 3.0
# the three sigma/m tertiles of the FULL sample (boundaries measured on
# z_full380_fl: 0.011036 and 0.014209)
card residSlo  --max-sigma-rel 0.011036
card residSmid --min-sigma-rel 0.011036 --max-sigma-rel 0.014209
card residShi  --min-sigma-rel 0.014209

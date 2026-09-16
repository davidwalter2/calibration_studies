#!/bin/bash
# The J/psi FSR-kernel cards.
#
# ONE change against the phase-2 reference `joint_ok_full`: the J/psi mass
# term's delta at MJPSI becomes delta (x) K with K the SAMPLE'S OWN kernel,
# measured from its gen record (`zchannel/jpsi_fsr_kernel.py mc`).  Everything
# else -- the candidates, the selection, the two corrections, the Z term, the
# quadratic term, the whitening -- is bit-identical, which is what makes the
# difference of the two fits the FSR effect and not a card difference.
#
# The two J/psi-ONLY cards are the attribution: with no Z term, `m_Z` and
# `Gamma_Z` are not in the likelihood at all, so whatever moves between
# `jpsi_nok` and `jpsi_mc` moved because of the J/psi leg.
#
# usage:  ./build_jpsi_fsr.sh [kernel] [p2] [jpsi]
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
KMC=$Z/data/jpsi_kern_mc.npz
# the exact-QED kernel, truncated to the SAME +-0.35 GeV gen acceptance the
# pairs cache has -- the untruncated one models a population this sample does
# not contain
KDATA=$Z/data/jpsi_kern_data_trunc.npz
CARDS=$FS/cards; LOGS=$FS/logs
mkdir -p "$CARDS" "$LOGS"
STAGES=${*:-kernel p2 jpsi}

# the phase-2 reference's own arguments (logs/card_joint_ok_full.log)
common=(--jpsi-pairs "$FS/runs/jpairs_v2_n600.npz"
        --jpsi-maxn 3000000
        --quad "$FS/runs/quad_jpsiv2_ok.npz" "$FS/runs/quad_dyv2.npz"
        --groups "$GRP" --whiten)
zleg=(--z-pairs "$FS/runs/zpairs_dyv2_jac_full.npz" --shape 5
      --fsr "$Z/data/kern_loose_band3.3e-4.npz"
      --acc "$Z/data/acc_loose_d8.json")

card () {   # card <name> <extra args...>
  local tag=$1; shift
  if [ -f "$CARDS/$tag.hdf5" ]; then echo "[card] $tag exists"; return; fi
  echo "[card] $tag  $(date +%H:%M:%S)"
  THREADS=${THREADS:-32} "$FS/run_tf.sh" python3 -u "$FS/make_joint_card.py" \
      "${common[@]}" "$@" -o "$CARDS/$tag.hdf5" 2>&1 | tee "$LOGS/card_$tag.log"
}

for st in $STAGES; do
case $st in
kernel)
  # the sample's own kernel; --save-dm keeps the samples for the gate/figures
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 -u "$Z/jpsi_fsr_kernel.py" mc --pairs "$FS/runs/jpairs_v2_n600.npz" \
      -o "$KMC" --report --save-dm 2>&1 | tee "$LOGS/jpsi_kern_mc.log"
  python3 -u "$Z/jpsi_fsr_kernel.py" analytic --variant exp2nll --pair e mu \
      --gamma 9.26e-5 -o "$Z/data/jpsi_kern_data.npz" --report \
      2>&1 | tee "$LOGS/jpsi_kern_data.log"
  python3 -u "$Z/jpsi_fsr_kernel.py" analytic --variant exp2nll --pair e mu \
      --gamma 9.26e-5 --truncate 0.35 -o "$Z/data/jpsi_kern_data_trunc.npz" \
      --report 2>&1 | tee "$LOGS/jpsi_kern_data_trunc.log"
  ;;
p2)
  # phase 2, the SAME code on both sides.  `joint_ok_full` (the P2XP reference)
  # was built on 2026-09-07, BEFORE `65319ab` gave every card a real positivity
  # floor, so its J/psi leg ran at rabbit's 1e-9 default -- a second difference,
  # and one that makes the NLLs incomparable.  `joint_nok` is the same card
  # rebuilt today, so `P2N` vs `P2K` differ by the kernel and nothing else.
  card joint_nok  "${zleg[@]}"
  card joint_fsrmc "${zleg[@]}" --jpsi-fsr "$KMC"
  ;;
jpsi)
  # J/psi + the 92 calibration parameters: no kernel, the sample's own, and
  # exact QED.  The third is the MODEL dependence of the kernel measured on
  # the extracted scale rather than only on the kernel's mean.
  card jpsi_nok
  card jpsi_fsrmc   --jpsi-fsr "$KMC"
  card jpsi_fsrdata --jpsi-fsr "$KDATA"
  ;;
esac
done
echo "[build_jpsi_fsr] done $(date +%H:%M:%S)"

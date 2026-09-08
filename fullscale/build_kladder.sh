#!/bin/bash
# Extend the K(m) ladder. Each added Legendre term is a 13-15 sigma improvement
# in 2*deltaNLL at 5 -> 6 -> 7, so the basis is NOT saturated and the ladder has
# to be pushed until the added term stops buying likelihood, before either the
# m_Z drift (6.2 MeV over 5->7) or the Gamma_Z swing (32 MeV) can be called a
# truncation systematic rather than an unconverged model.
#
#   ./build_kladder.sh 9 12
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
export PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:/work/submit/david_w/WRemnants_dev/wums:$FS/../env_tf/pypath:$FS/../resolution
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${THREADS:-10} TF_NUM_INTRAOP_THREADS=${THREADS:-10} TF_NUM_INTEROP_THREADS=2
P=/opt/venv/bin/python3
for N in "$@"; do
  for form in m v; do
    # the two forms were built from DIFFERENT pairs caches and must stay that
    # way: the m-form ladder off `zpairs_dyv2_full.npz`, the v-form ladder off
    # `zpairs_dyv2_jac_full.npz` (same 3 682 662 candidates, the second also
    # carrying the sparse D). Mixing them would put a new rung of the ladder on
    # a different input from the rungs it is being compared with.
    if [ "$form" = m ]; then out=$FS/cards/z_full380_fl_s$N.hdf5; extra=""
                             pairs=$FS/runs/zpairs_dyv2_full.npz
    else out=$FS/cards/z_V_s$N.hdf5; extra="--vpow 1.264"
                             pairs=$FS/runs/zpairs_dyv2_jac_full.npz; fi
    [ -f "$out" ] && { echo "[card] $(basename $out) exists"; continue; }
    echo "[card] $(basename $out)  $(date +%H:%M:%S)"
    $P -u $FS/make_card.py --pairs "$pairs" \
        --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
        --shape $N $extra -o "$out" 2>&1 | tee $FS/logs/card_$(basename $out .hdf5).log \
        | grep -E "candidates|shape|parameters|wrote|POI"
  done
done

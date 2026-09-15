#!/bin/bash
# Build the h tables and the selection-conditional kernels of the per-leg
# factorisation, including every discretisation variant quoted in the README.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
G=data/genmerged_full.npz
PL=data/perleg_full.npz
P=python3

# --- h(a_+, a_- | m): one table per band width, plus a coarse (a_+,a_-) grid
for W in 0.5 1.0 2.0; do
  [ -s data/ht_pt25_${W}gev.npz ] || \
    $P fsr_perleg.py htable --gen $G -o data/ht_pt25_${W}gev.npz --band-width $W
done
[ -s data/ht_pt25_1.0gev_thin4.npz ] || \
  $P fsr_perleg.py htable --gen $G -o data/ht_pt25_1.0gev_thin4.npz \
     --band-width 1.0 --b-thin 4

# --- the kernels.  `data` = exp2nll (x) exact pairs e,mu,tau,had
K() { [ -s "$2" ] || $P fsr_perleg.py kernel --htable "$1" -o "$2" \
        --acceptance "$3" --variant exp2nll --pair e mu tau had "${@:4}"; }
K data/ht_pt25_1.0gev.npz data/kern_perleg_data_1gev.npz data/acc_perleg_1gev.json
K data/ht_pt25_0.5gev.npz data/kern_perleg_data_0p5gev.npz data/acc_perleg_0p5gev.json
K data/ht_pt25_2.0gev.npz data/kern_perleg_data_2gev.npz data/acc_perleg_2gev.json
K data/ht_pt25_1.0gev_thin4.npz data/kern_perleg_data_thin4.npz data/acc_perleg_thin4.json
K data/ht_pt25_1.0gev.npz data/kern_perleg_data_nleg1500.npz data/acc_perleg_nleg1500.json --n-leg 1500
K data/ht_pt25_1.0gev.npz data/kern_perleg_data_nleg6000.npz data/acc_perleg_nleg6000.json --n-leg 6000
K data/ht_pt25_1.0gev.npz data/kern_perleg_data_vb6e-9.npz  data/acc_perleg_vb6e-9.json  --var-budget 6e-9
K data/ht_pt25_1.0gev.npz data/kern_perleg_data_vb6e-11.npz data/acc_perleg_vb6e-11.json --var-budget 6e-11

# --- photonic variants, for the mc -> data difference of the QED content
K data/ht_pt25_1.0gev.npz data/kern_perleg_nopair.npz data/acc_perleg_nopair.json --pair
[ -s data/kern_perleg_exp1.npz ] || $P fsr_perleg.py kernel --htable data/ht_pt25_1.0gev.npz \
     -o data/kern_perleg_exp1.npz --acceptance data/acc_perleg_exp1.json \
     --variant exp1 --pair e mu tau had

# --- the empirical per-leg D (the sample's own Photos content), same h.
# D depends on m only through beta(m), so it is measured in 10 GeV windows;
# at 1 GeV its band-to-band statistical jitter is 7 % of <u>.
E() { [ -s "$2" ] || $P fsr_perleg.py kernel --htable "$1" -o "$2" \
        --acceptance "$3" --empirical-leg $PL "${@:4}"; }
E data/ht_pt25_1.0gev.npz data/kern_perleg_emp_1gev.npz data/acc_perleg_emp_1gev.json
E data/ht_pt25_1.0gev.npz data/kern_perleg_emp1_1gev.npz data/acc_perleg_emp1_1gev.json --leg-band-width 1
E data/ht_pt25_1.0gev.npz data/kern_perleg_emp20_1gev.npz data/acc_perleg_emp20_1gev.json --leg-band-width 20
E data/ht_pt25_2.0gev.npz data/kern_perleg_emp_2gev.npz data/acc_perleg_emp_2gev.json

# --- Bernstein-8 renderings of the tabulated A(m)
for T in 1gev emp_1gev; do
  $P fsr_perleg.py accfit --acceptance data/acc_perleg_${T}.json \
       -o data/acc_perleg_${T}_b8.json --lo 55 --hi 150 --degree 8
done

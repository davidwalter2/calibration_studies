#!/bin/bash
# The tables and kernels of the ASYMMETRIC cuts and of the RESOLUTION in the
# acceptance.  One reference table at pT_ref = 10 GeV serves every cut above
# it; the h4 table adds the eta axis the pass PROBABILITY needs.
#
# The kernels are built with `--atoms`: they are the LEGACY banded form the
# published rows of the selection sections were measured on.  numpy only.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
P="python3 -u -W ignore"
G=data/genmerged_full.npz
RUN=data/photos/gen_mcMix.npz
PAIRS="--pair e mu tau had"
HT=data/ht_ref10_1.0gev.npz
H4=data/h4_ref10_1.0gev.npz
RES=data/ptres_dyv2.npz
GRID="--pt-ref 10 --anchor-cuts 25 --b-min -0.3 --b-max 4.0"

# --- 1. the tables -------------------------------------------------------
[ -s $HT ] || $P fsr_perleg.py htable   --gen $G -o $HT $GRID
[ -s $H4 ] || $P fsr_perleg.py h4table  --gen $G -o $H4 $GRID \
   --fine-step 1e-2 --fine-width 0.2 --coarse-step 4e-2
# discretisation variants of the h4 grid: a finer b grid, and 8 eta bands
[ -s data/h4_ref10_fineb.npz ] || $P fsr_perleg.py h4table --gen $G \
   -o data/h4_ref10_fineb.npz $GRID \
   --fine-step 5e-3 --fine-width 0.5 --coarse-step 4e-2
[ -s data/ptres_dyv2_eta8.npz ] || $P ptres.py measure \
   --aux ../fullscale/runs/auxgen_dyv2.npz \
   --pairs ../fullscale/runs/zpairs_dyv2_full.npz -o data/ptres_dyv2_eta8.npz \
   --eta-edges 0 0.4 0.8 1.2 1.6 1.9 2.1 2.25 2.4 > data/00_ptres_eta8.txt
[ -s data/h4_ref10_eta8.npz ] || $P fsr_perleg.py h4table --gen $G \
   -o data/h4_ref10_eta8.npz $GRID \
   --fine-step 1e-2 --fine-width 0.2 --coarse-step 4e-2 \
   --eta-edges 0 0.4 0.8 1.2 1.6 1.9 2.1 2.25 2.4

# --- 2. the MC's own conditional kernel + A(m), one per selection ---------
# measured on the SAME record and the SAME smearing draw the fit sees, so
# these rows carry no record floor of their own.
C() { t=$1; shift; [ -s data/kern_cond_$t.npz ] || $P fsr_perleg.py condker \
        --gen $G --mode true -o data/kern_cond_$t.npz \
        --acceptance data/acc_cond_$t.json "$@"; }
C 2525   --pt-cuts 25 25
C 2510   --pt-cuts 25 10
C 2525sm --pt-cuts 25 25 --smear $RES
C 2510sm --pt-cuts 25 10 --smear $RES
C 2510smg --pt-cuts 25 10 --smear $RES --smear-mode gauss

# --- 3. the model: the correlated two-leg kernel, both configurations -----
K() { t=$1; shift
  [ -s data/kern_corr_data_$t.npz ] || $P fsr_perleg.py corr --atoms --htable $HT \
     -o data/kern_corr_data_$t.npz --acceptance data/acc_corr_data_$t.json \
     $PAIRS "$@"
  [ -s data/kern_corr_mc_$t.npz ] || $P fsr_perleg.py corr --atoms --htable $HT \
     --run $RUN -o data/kern_corr_mc_$t.npz \
     --acceptance data/acc_corr_mc_$t.json "$@"; }
K 2525 --pt-cuts 25 25
K 2510 --pt-cuts 25 10
K 2525sm --pt-cuts 25 25 --h4 $H4 --resol $RES
K 2510sm --pt-cuts 25 10 --h4 $H4 --resol $RES

# --- 4. the discretisation and modelling variants of the smeared region ---
V() { t=$1; shift; [ -s data/kern_corr_mc_$t.npz ] || $P fsr_perleg.py corr --atoms \
        --htable $HT --run $RUN -o data/kern_corr_mc_$t.npz \
        --acceptance data/acc_corr_mc_$t.json --pt-cuts 25 10 "$@"; }
V 2510sm_gauss --h4 $H4     --resol $RES --resol-mode gauss
V 2510sm_fineb --h4 data/h4_ref10_fineb.npz --resol $RES
V 2510sm_eta8  --h4 data/h4_ref10_eta8.npz  --resol data/ptres_dyv2_eta8.npz

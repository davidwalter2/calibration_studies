#!/bin/bash
# The machinery-only benchmark in the CELL-INTEGRATED table representation.
#
# Both sides of the test become tables: the MC's own selection-conditional
# kernel `K_sel,MC[m_node, u_cell]`, measured on the fit's own events, and the
# two-leg model's `K(u|m) Gbar(u|m)`.  The atom form's two discretisations --
# the `m_pre` bands and the `sigma_cap` merge -- are then gone from both, and
# neither side's atom offset can leak into the difference.
#
# numpy only, ~15 min in total.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
P="python3 -u -W ignore"
G=data/genmerged_full.npz
HT=data/ht_pt25_1.0gev.npz
RUN=data/photos/gen_mcMix.npz
CUTS="--pt-cuts 25 25"

# the band sets of the cond scan: the atom form's own bands, then uniform
# widths, all with the same treatment of the high tail (140, 160)
CB="70 80 85 88 91 94 98 105 115 130"
CBX="$CB 140 160"
W4=$($P -c 'print(" ".join("%g"%x for x in list(range(60,133,4))+[140,160]))')
W2=$($P -c 'print(" ".join("%g"%x for x in list(range(60,131,2))+[140,160]))')
W1=$($P -c 'print(" ".join("%g"%x for x in list(range(60,131,1))+[140,160]))')

# --- 1. the MC's own conditional kernel as a table -----------------------
C() { t=$1; shift; [ -s data/ktab_cond_$t.npz ] || $P fsr_table.py cond \
        --gen $G -o data/ktab_cond_$t.npz -a data/atab_cond_$t.json \
        $CUTS "$@"; }
C 2525                                    # the atom form's own bands
C 2525x   --bands $CBX                    # + the high tail resolved
C 2525w4  --bands $W4
C 2525w2  --bands $W2
C 2525w1  --bands $W1
C 2525w2c1000 --bands $W2 --n-cell 1000
C 2525w2c4000 --bands $W2 --n-cell 4000
C 2525w2a --bands $W2 --half 1            # the half-sample split: the table's
C 2525w2b --bands $W2 --half 2            # own statistical noise
# the cell-count scan repeated on ONE half: a knob that moves the fit by more
# than the half-sample noise is a representation effect, one that does not is
# the noise reprojected
C 2525w2c1000a --bands $W2 --n-cell 1000 --half 1
C 2525w2c4000a --bands $W2 --n-cell 4000 --half 1
C 2525w2c8000  --bands $W2 --n-cell 8000
C 2525w2c8000a --bands $W2 --n-cell 8000 --half 1

# --- 2. the model as a table --------------------------------------------
# `mc` K = the standalone Photos histograms, `sample` K = the fit sample's own
# inclusive kernel, which removes the standalone-vs-sample floor entirely.
M() { t=$1; shift; [ -s data/ktab_corr_$t.npz ] || $P fsr_table.py corr \
        --htable $HT -o data/ktab_corr_$t.npz \
        -a data/atab_corr_$t.json $CUTS --mode multi \
        --rho data/rho_multi_mc.npz "$@"; }
M mc_multi        --run $RUN --dm-node 1.0
M mc_multi_dm05   --run $RUN --dm-node 0.5
M mc_multi_dm2    --run $RUN --dm-node 2.0
M mc_multi_c1000  --run $RUN --dm-node 1.0 --n-cell 1000
M mc_multi_c4000  --run $RUN --dm-node 1.0 --n-cell 4000
M mc_multi_dm025  --run $RUN --dm-node 0.25
M smp_multi       --sample $G --dm-node 1.0
M smp_multi_dm05  --sample $G --dm-node 0.5
M smp_multi_dm025 --sample $G --dm-node 0.25
M smp_single_dm05 --sample $G --dm-node 0.5 --mode single
# the model kernel's own statistical floor: the sample's K on each half
M smp_multi_a     --sample $G --dm-node 0.5 --sample-half 1
M smp_multi_b     --sample $G --dm-node 0.5 --sample-half 2
M mc_single_dm05  --run $RUN --dm-node 0.5 --mode single
M smp_single      --sample $G --dm-node 1.0 --mode single
M mc_single       --run $RUN --dm-node 1.0 --mode single

# rho driven by the sample's own kernel instead of the standalone's
[ -s data/ktab_corr_smp_multirs.npz ] || $P fsr_table.py corr --htable $HT \
   -o data/ktab_corr_smp_multirs.npz -a data/atab_corr_smp_multirs.json \
   $CUTS --mode multi --sample $G --rho-sample --dm-node 1.0 \
   --rho data/rho_multi_smp.npz

# --- 3. the standalone-vs-sample kernel floor, in the table representation
# the sample's own INCLUSIVE kernel on the same 1 GeV nodes as the `mc` table
[ -s data/ktab_incl_smp.npz ] || $P fsr_table.py cond --gen $G --inclusive \
   --band-width 1 --band-lo 50 --band-hi 200 \
   -o data/ktab_incl_smp.npz -a data/atab_incl_smp.json
[ -s data/ktab_mc_dm10.npz ] || $P fsr_table.py mc -o data/ktab_mc_dm10.npz \
   --dm-node 1.0

$P fsr_table.py check -i data/ktab_cond_2525.npz data/ktab_corr_mc_multi.npz \
   data/ktab_corr_smp_multi.npz data/ktab_incl_smp.npz

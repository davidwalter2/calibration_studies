#!/bin/bash
# THE CHI2/NDOF CUT, measured the way the |z_v| cut was (`STATE.md` 15.8).
#
# ONE extraction per sample with the cut OFF, then cards that differ ONLY by
# the cut value, so the likelihood and the candidates are the only things that
# move.  The gun is pure signal; the DY leg is gen-classified and fitted on
# gen SIGNAL ONLY, so the 1.0 % excess-chi2 signal population is what the
# comparison is about and not the background the cut also happens to take.
#
#   ./run_chi2.sh extract|extract-dy|cards|cards-dy|fits|report|dytable
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
ROOT=${ROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/chi2cut}
R=${R:-$ROOT/runs}
PROD=${PROD:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod_vtxon}
DYP=${DYP:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/bkg/dy_vtxon_gen}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
PY=${PY:-/work/submit/david_w/ZMass/mfs/.venv/bin/python3}
J=${J:-24}
MAXC=${MAXC:-130}          # per file; 161 files -> ~21 k gun candidates
MZ=91.1876
mkdir -p $R/cards $R/fits $HERE/logs_chi2
cd $HERE || exit 9

# the four arms: the cut off, and at 3 (the standard), 5 and 10
ARMS=${ARMS:-"off:0 c3:3 c5:5 c10:10"}

case ${1:-} in
extract)
  # NO chi2 cut and NO |z_v| cut at extraction: the CARDS apply both, so one
  # extraction serves every arm and nothing else can differ between them.
  for F in vtx mass; do
    $PY -u extract_vtx.py --files "$PROD/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J \
      --max-abs-vtxz 0 --max-chi2-ndof 0 --max-cands $MAXC \
      -o $R/gun_$F.npz 2>&1 | tee logs_chi2/extract_gun_$F.log
  done ;;
extract-dy)
  for F in vtx mass; do
    $PY -u extract_vtx.py --files "$DYP/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J --require-complete \
      --max-abs-vtxz 0 --max-chi2-ndof 0 \
      -o $R/dy_$F.npz 2>&1 | tee logs_chi2/extract_dy_$F.log
  done ;;
genmask)
  # the gen-SIGNAL mask of the DY extraction (`genbkg.classify`), written once
  # and applied by every DY card, so every DY arm sits on the same truth.
  for F in vtx mass; do
    $PY -u gensig_mask.py --npz $R/dy_$F.npz -o $R/dy_${F}_gensig.npz \
      2>&1 | tee logs_chi2/genmask_dy_$F.log
  done ;;
cards)
  for A in $ARMS; do
    NM=${A%%:*}; Q=${A##*:}
    # MASS: the standard selection except for the cut under study; alpha free
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --mass-npz $R/gun_mass.npz --arm cf \
      --max-chi2-ndof $Q --alpha --mass-corrections \
      -o $R/cards/gun_mass_$NM.hdf5 2>&1 | tee logs_chi2/card_gun_mass_$NM.log
    # VERTEX: |z_v| < 5 with the truncated normalisation
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --vtx-npz $R/gun_vtx.npz --arm cf \
      --max-chi2-ndof $Q \
      -o $R/cards/gun_vtx_$NM.hdf5 2>&1 | tee logs_chi2/card_gun_vtx_$NM.log
  done
  # (c) of the vertex three-way: BOTH cuts off, untruncated likelihood
  ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
    --prune-frac 0.001 --poi all --vtx-npz $R/gun_vtx.npz --arm cf \
    --max-chi2-ndof 0 --max-abs-vtxz 0 \
    -o $R/cards/gun_vtx_bothoff.hdf5 2>&1 | tee logs_chi2/card_gun_vtx_bothoff.log
  ;;
cards-dy)
  for A in $ARMS; do
    NM=${A%%:*}; Q=${A##*:}
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --mass-npz $R/dy_mass.npz --arm cf \
      --m-ref $MZ --m-window 30 --keep-mask $R/dy_mass_gensig.npz \
      --corr-form fluctuation \
      --max-chi2-ndof $Q --alpha --mass-corrections \
      -o $R/cards/dy_mass_$NM.hdf5 2>&1 | tee logs_chi2/card_dy_mass_$NM.log
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --vtx-npz $R/dy_vtx.npz --arm cf \
      --keep-mask $R/dy_vtx_gensig.npz --max-chi2-ndof $Q \
      -o $R/cards/dy_vtx_$NM.hdf5 2>&1 | tee logs_chi2/card_dy_vtx_$NM.log
  done
  ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
    --prune-frac 0.001 --poi all --vtx-npz $R/dy_vtx.npz --arm cf \
    --keep-mask $R/dy_vtx_gensig.npz --max-chi2-ndof 0 --max-abs-vtxz 0 \
    -o $R/cards/dy_vtx_bothoff.hdf5 2>&1 | tee logs_chi2/card_dy_vtx_bothoff.log
  ;;
fits)
  for c in $(cd $R/cards && ls *.hdf5 2>/dev/null | sed 's/\.hdf5$//'); do
    [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "skip $c (done)"; continue; }
    # THE FORM OF THE TWO CORRECTIONS.  rabbit's `auto` reads the term's
    # `kernel` attribute, and `make_vtx_card.py` passes its physics kernel as
    # a TABULATED `phik` while leaving `kernel` at the default `DeltaKernel`.
    # On the gun that is the truth -- every candidate has the same gen mass,
    # so the kernel IS a delta and the residual form is exact.  On DY it is
    # not: the gen Z mass has rms 6.9 GeV against sigma_m = 1.08 GeV, and the
    # residual form would evaluate the width at the candidate's distance from
    # `m_ref`, i.e. at the LINESHAPE and not at the resolution fluctuation.
    # The wide-kernel term is the fluctuation form, and it is forced here.
    EX=""; case $c in dy_mass_*) EX="--unbinnedDeltaKernelForm fluctuation";; esac
    echo "=== fit $c $EX"; R=$R ./run_fit.sh $c $EX 2>&1 | tee logs_chi2/fit_$c.log
  done ;;
report)
  ./run_tf.sh python3 -u chi2_report.py --fits $R/fits --groups $GRP \
    2>&1 | tee logs_chi2/report.log ;;
dytable)
  $PY -u sel_dytable.py --npz $R/dy_vtx.npz 2>&1 | tee logs_chi2/dytable.log ;;
*) echo "usage: run_chi2.sh extract|extract-dy|genmask|cards|cards-dy|fits|report|dytable"; exit 2 ;;
esac

#!/bin/bash
# The TAIL study's chain (STATE.md section 16).  ceph is only readable from a
# submit node, so every stage runs there; the figures land in ~/public_html.
#
#   ./run_all_tail.sh <stage> [args]
#     extract [tag ...]   the compact per-candidate npz of each production
#     hypA | geom | align | hypBC | hypD | scan | chi2 | chi2src | subdet
#     mixture             THE CLOSURE: the scale mixture against the tail
#     ideal               the ideal- against the aligned-geometry leg
#     plots               every figure
#     prod                the ideal-geometry re-production (see run_prod_tail.sh)
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
ROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911
T=$ROOT/tail
DY=$T/dy_bs_final.npz
GUN=$T/gun_vtxon.npz
FIG=${FIG:-$HOME/public_html/ZMass/cvh/$(date +%y%m%d)_tail}
mkdir -p "$T" "$HERE/logs_tail" "$FIG"
cd $HERE
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
case ${1:-} in
extract)
  shift
  for spec in "${@:-dy_bs_final:$ROOT/beamline/dy_bs_final}"; do
    tag=${spec%%:*}; dir=${spec#*:}
    python3 -u tail_extract.py --files "$dir/task_*/globalcor_*.root" \
      --out "$T/$tag.npz" 2>&1 | tee "logs_tail/extract_$tag.log"
  done ;;
hypA)    python3 -u tail_hypA.py --npz $DY --out $T/hypA_dy.npz 2>&1 | tee logs_tail/hypA_dy.log ;;
geom)    python3 -u tail_geom.py --a $ROOT/beamline/dy_bs_final/task_0000/globalcor_0.root \
           --b $ROOT/prod_vtxon/task_0000/globalcor_0.root --labels DYreal GUNideal \
           --out $T/geom_dy_vs_gun.npz 2>&1 | tee logs_tail/geom.log ;;
align)   python3 -u tail_align.py --npz $DY \
           --files "$ROOT/beamline/dy_bs_final/task_*/globalcor_*.root" \
           --ref $ROOT/prod_vtxon/task_0000/globalcor_0.root \
           --out $T/align_dy.npz 2>&1 | tee logs_tail/align_dy.log ;;
hypBC)   python3 -u tail_hypBC.py --npz $DY \
           --files "$ROOT/beamline/dy_bs_final/task_*/globalcor_*.root" \
           --geom $T/geom_dy_vs_gun.npz --out $T/hypBC_dy.npz 2>&1 | tee logs_tail/hypBC_dy.log ;;
hypD)    python3 -u tail_hypD.py --npz $DY --gun $GUN 2>&1 | tee logs_tail/hypD_dy.log ;;
scan)    python3 -u tail_scan.py --npz $DY --gun $GUN 2>&1 | tee logs_tail/scan_dy.log ;;
chi2)    python3 -u tail_chi2.py --npz $DY --gun $GUN 2>&1 | tee logs_tail/chi2_dy.log ;;
chi2src) python3 -u tail_chi2src.py --npz $DY --gun $GUN 2>&1 | tee logs_tail/chi2src.log ;;
subdet)  python3 -u tail_subdet.py --npz $DY \
           --files "$ROOT/beamline/dy_bs_final/task_*/globalcor_*.root" 2>&1 | tee logs_tail/subdet_dy.log ;;
mixture) python3 -u tail_mixture.py --npz $DY --gun $GUN 2>&1 | tee logs_tail/mixture_dy.log ;;
ideal)   python3 -u tail_ideal.py --real ${REAL:-$T/dy_real700.npz} \
           --ideal ${IDEAL:-$T/dy_ideal700.npz} 2>&1 | tee logs_tail/ideal.log ;;
plots)   python3 -u tail_plots.py --npz $DY --gun $GUN --align $T/align_dy.npz \
           ${IDEAL:+--ideal $IDEAL} --outdir "$FIG" 2>&1 | tee logs_tail/plots.log ;;
prod)    shift; ./run_prod_tail.sh "$@" ;;
*) sed -n '2,14p' "$0" ;;
esac

#!/bin/bash
# Detached finisher for the FULL ideal-geometry leg (STATE.md section 16.10):
# waits for the six `dy_ideal` tasks, extracts the compact npz and runs the
# same-candidate comparison against `dy_bs_final`.  Run with nohup on a submit
# node so it survives the session:
#   nohup ./finish_tail_ideal.sh > logs_tail/finish_ideal.log 2>&1 &
set -uo pipefail
H=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
T=/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/tail
while [ "$(ls $T/dy_ideal/task_*/.complete 2>/dev/null | wc -l)" != "6" ]; do sleep 120; done
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
cd $H
python3 -u tail_extract.py --files "$T/dy_ideal/task_*/globalcor_*.root" \
  --out $T/dy_ideal.npz 2>&1 | tee logs_tail/extract_dy_ideal.log
python3 -u tail_ideal.py --real /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/tail/dy_bs_final.npz \
  --ideal $T/dy_ideal.npz 2>&1 | tee logs_tail/ideal_full.log
python3 -u tail_mixture.py --npz $T/dy_ideal.npz 2>&1 | tee logs_tail/mixture_ideal.log
echo "FULL IDEAL LEG DONE"

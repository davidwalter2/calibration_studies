#!/bin/bash
# Pull whatever Engaging has finished and re-make the certified table.
#   ./collect.sh [--table]
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:/work/submit/david_w/WRemnants_dev/wums:/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath:/work/submit/david_w/ZMass/calibration_studies/resolution
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4
P=/opt/venv/bin/python3
timeout 900 rsync -aq --exclude 'native/' engaging:orcd/pool/zmass/fitresults/ results/eng/
timeout 900 rsync -aq engaging:orcd/pool/zmass/fitresults/native/ results/native/
timeout 300 rsync -aq engaging:'orcd/pool/zmass/engaging/*.out' results/englogs/
timeout 1800 $P -u native_dump.py 'results/native/rabbit_*.hdf5' -o results/nativejson \
    2>&1 | grep -viE "warning|futurew|metadata|absl|oneDNN|instructions|^I0000|^E0000"
echo
$P certtable.py "$@"

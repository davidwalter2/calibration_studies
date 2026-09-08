#!/bin/bash
# One snapshot per certified-table row, at the LOWEST-NLL point any earlier fit
# of that card reached. `rabbit_vmass.sbatch` resumes from it (FRESH != 1).
# The control (f380refS) is deliberately absent: it must find -11.0643 cold.
set -uo pipefail
cd "$(dirname "$0")"
P=/opt/venv/bin/python3
seed () {  # tag card source-json
  $P -u json_to_snapshot.py --card cards/$2.hdf5 --from $3 \
     -o results/seed/rabbit_$1.snapshot.hdf5 | sed -n "1,2p" | sed "s/^/[$1] /"
}
seed SVfull   z_V_full           results/eng/fit_V_full.json
seed SVtoy    z_V_toy            results/eng/fit_V_toy.json
seed SVs6     z_V_s6             results/eng/fit_V_s6.json
seed SVs7     z_V_s7             results/eng/fit_V_s7.json
seed SVetaB   z_V_etaB           results/eng/fit_V_etaB.json
seed SVetaT   z_V_etaT           results/eng/fit_V_etaT.json
seed SVetaE   z_V_etaE           results/eng/fit_V_etaE.json
seed SVKetaB  z_VK_etaB          results/eng/fit_VK_etaB.json
seed SVKetaT  z_VK_etaT          results/eng/fit_VK_etaT.json
seed SMetaB   z_M_etaB           results/eng/fit_M_etaB.json
seed SMetaT   z_M_etaT           results/eng/fit_M_etaT.json
seed SMetaE   z_M_etaE           results/eng/fit_M_etaE.json
seed Sdc8     z_F_dc8            results/nativejson/rabbit_Rdc8X.json
seed Stoy     z_F_toy            results/eng/fit_F_toy_cv.json
seed Stoydc   z_F_toydc          results/eng/fit_F_toydc_cv.json
seed Sw70110  z_F_w70110         results/eng/fit_F_w70110_cv.json
seed Ss6      z_full380_fl_s6    results/eng/fit_f380fl_s6_cv.json
seed Ss7      z_full380_fl_s7    results/eng/fit_f380fl_s7_cv.json

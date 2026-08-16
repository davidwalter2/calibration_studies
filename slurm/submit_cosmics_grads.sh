#!/bin/bash
# Launch the cosmics CVH grads production (Run2016G+H NoBPTX
# TkAlCosmicsInCollisions) as a Slurm array, one task per repacked file.
#
# Prereq: repack/repack_cosmics_all.sh has produced the split-0 ceph files and
# repack/cosmics_ceph_filelist.txt. Each task runs runCvhCosmics.py in the
# 15_0 dev area with:
#   - fillGrads=True                 packed per-candidate grad + Hessian
#                                    (single-track maker has no factored path;
#                                    92 params -> packed is tiny)
#   - goodRunsFile=...GH.txt         source-level 3.8T run filter (drops the
#                                    era-G ramp/TS2 runs regardless of file mix)
#   - the same ScalarPot3D 50-mode init + tier-50 material groups as the J/psi
#     globalmat production, so the two share an identical global catalog and
#     combine directly in fit_global_grads.py.
set -euo pipefail

SLURM_DIR=$(dirname "$(readlink -f "$0")")
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
GOODRUNS=/work/submit/david_w/ZMass/repack/goodruns_cosmics_2016GH.txt
FILELIST=${1:-/work/submit/david_w/ZMass/repack/cosmics_ceph_filelist.txt}
OUTDIR=${2:-/ceph/submit/data/user/d/david_w/ZMass/cvh/cosmics_calib2016_grads50_260718}
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev

[[ -s "$FILELIST" ]] || { echo "filelist missing/empty: $FILELIST" >&2; exit 1; }

exec "$SLURM_DIR/submit.sh" \
  --config "$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhCosmics.py" \
  --filelist "$FILELIST" \
  --outdir "$OUTDIR" \
  --cmssw-area "$CMSSW_AREA" \
  --name cvhcosmics \
  --cpus 8 --mem 8G --time 12:00:00 \
  --cmsrun-args "nEvents=-1 numberOfThreads=8 fillGrads=True fillJac=False trackSrc=ALCARECOTkAlCosmicsInCollisions goodRunsFile=$GOODRUNS scalarPot3DInitFile=$INIT" \
  "${@:3}"

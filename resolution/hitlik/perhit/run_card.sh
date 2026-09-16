#!/bin/bash
# Build one per-hit card.  Thin wrapper over ../make_hitlik_card.py with the
# per-hit defaults: `--comps` takes a SPEC ("hit", "ref", "ref0123", "all"),
# the npz and the output live under runs/perhit, and the groups file is the
# one the maker's parmtype-15 model uses.
#   run_card.sh <name> [extra make_hitlik_card args...]
set -euo pipefail
HL=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/perhit
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
NPZ=${NPZ:-$R/perhit.npz}
name=$1; shift
mkdir -p $R/cards $R/fits logs
cd $HL
exec ./run_tf.sh python3 -u make_hitlik_card.py --npz "$NPZ" \
  --groups $GRP --whiten --prune-frac 0.001 --poi material --hit-prior 1.0 \
  -o $R/cards/$name.hdf5 "$@"

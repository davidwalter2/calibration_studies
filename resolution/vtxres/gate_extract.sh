#!/bin/bash
# The EXTRACTION half of the bit-identity gates (`run_chi2_gates.sh`).
#
# `matres/extract_groups.py` and `hitlik/extract_res5.py` used to DEFAULT to
# `--max-chi2-ndof 0` (no cut) and now default to the standard 3.0.  Two
# things are therefore owed:
#   * that at `--max-chi2-ndof 0` the new code reproduces the old one exactly
#     -- the OLD behaviour, bit for bit;
#   * what 3.0 actually removes there, quoted rather than assumed.
# Both run on a handful of files, because the question is about the SELECTION
# and not about statistics.
set -uo pipefail
CS=/work/submit/david_w/ZMass/calibration_studies
REF=${REF:-/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/head_ref}
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/chi2cut/gates}
PY=${PY:-/work/submit/david_w/ZMass/mfs/.venv/bin/python3}
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
GRP2=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
[ -f "$GRP" ] || GRP=$GRP2
NF=${NF:-1}
NC=${NC:-400}      # candidates/tracks per file: the gate is about the
                    # SELECTION, not about statistics
mkdir -p $OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0

HITF="$CEPH/resolution_trackres_mugun_ul16_260903x_m0/task_*/globalcor_resclosure_*.root"
MATF="$CEPH/resolution_trackres_jpsigun_ul16_260905d_m0/task_*/globalcor_*.root"

hit () {   # hit <where> <chi2> <tag>
  local B=$1 Q=$2 T=$3
  [ -s $OUT/hitlik_$T.npz ] && return
  $PY -u $B/resolution/hitlik/extract_res5.py --files "$HITF" --ntasks $NF \
    --groups $GRP --max-chi2-ndof $Q --max-hess 1e8 --max-grad 1e6 \
    --decimate 4 --tmax 8 --max-cands $NC -o $OUT/hitlik_$T.npz
}
mat () {   # mat <where> <chi2> <tag>
  local B=$1 Q=$2 T=$3
  [ -s $OUT/matres_$T.npz ] && return
  $PY -u $B/resolution/matres/extract_groups.py --files "$MATF" --ntasks $NF \
    --functional mass --parmtypes 14 15 --groups $GRP -j 8 \
    --max-chi2-ndof $Q --max-hess 1e8 --max-grad 1e6 --max-cands $NC \
    --decimate 4 --tmax 8 -o $OUT/matres_$T.npz
}

case ${1:-} in
hitlik)
  hit $REF 0 ref0; hit $CS 0 new0; hit $CS 3 new3
  $PY $CS/resolution/cmp_outputs.py $OUT/hitlik_ref0.npz $OUT/hitlik_new0.npz \
      --skip-keys provenance
  $PY - "$OUT/hitlik_new0.npz" "$OUT/hitlik_new3.npz" <<'PYX'
import json, sys
import numpy as np
a, b = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])
pa = json.loads(str(a["provenance"])); pb = json.loads(str(b["provenance"]))
print(f"hitlik: chi2 off {pa['ntracks']} tracks -> chi2/ndof < 3 "
      f"{pb['ntracks']} tracks: {pa['ntracks']-pb['ntracks']} removed "
      f"({100.0*(pa['ntracks']-pb['ntracks'])/max(pa['ntracks'],1):.3f} %)")
PYX
  ;;
matres)
  mat $REF 0 ref0; mat $CS 0 new0; mat $CS 3 new3
  $PY $CS/resolution/cmp_outputs.py $OUT/matres_ref0.npz $OUT/matres_new0.npz \
      --skip-keys provenance
  $PY - "$OUT/matres_new0.npz" "$OUT/matres_new3.npz" <<'PYX'
import sys
import numpy as np
a, b = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])
na, nb = len(a["sigma"]), len(b["sigma"])
print(f"matres mass: chi2 off {na} candidates -> chi2/ndof < 3 {nb}: "
      f"{na-nb} removed ({100.0*(na-nb)/max(na,1):.3f} %)")
q = np.asarray(a["chi2ndof"], float)
print(f"  chi2/ndof: median {np.median(q):.4f}  p99 {np.quantile(q,0.99):.4f}  "
      f"max {q.max():.4g};  P(>3) = {100*np.mean(q>3):.3f} %")
PYX
  ;;
*) echo "usage: gate_extract.sh hitlik|matres"; exit 2 ;;
esac

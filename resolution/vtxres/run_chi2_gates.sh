#!/bin/bash
# THE BIT-IDENTITY GATES for moving `--max-chi2-ndof` into
# `resolution/selection.py` (STATE.md 15.8).
#
# The claim is that nothing a certified result rests on moved.  The test is
# not "compatible" but IDENTICAL: the same input and the same cut value through
# the code at HEAD and through the new code, compared array by array.
#
#   z        the full-scale Z card (`run_phase1.sh card n300k`), cut 3.0 both
#            sides -- the value it always used
#   jpsi     the full-scale J/psi leg of the joint card, ditto
#   matres   `matres/extract_groups.py` at `--max-chi2-ndof 0`, the OLD default
#   hitlik   `hitlik/extract_res5.py` at `--max-chi2-ndof 0`, ditto
#
# usage: ./run_chi2_gates.sh z|jpsi|matres|hitlik
set -uo pipefail
CS=/work/submit/david_w/ZMass/calibration_studies
REF=${REF:-/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/head_ref}
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/chi2cut/gates}
FS=$CS/fullscale
Z=$CS/zchannel
PY=${PY:-/work/submit/david_w/ZMass/mfs/.venv/bin/python3}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
LOGS=$CS/resolution/vtxres/logs_chi2
mkdir -p $OUT $LOGS
cd $CS/resolution/vtxres || exit 9

# BIT-IDENTICAL means byte for byte.  `cmp` is the whole test when it passes;
# when it does not, `cmp_outputs.py` says WHICH array moved -- and it has to
# run inside the container, because a rabbit card is written with a
# compression filter the bare venv's h5py cannot load.
cmpfiles () {
  if cmp -s "$1" "$2"; then
    echo "BYTE-IDENTICAL: $(basename $1) == $(basename $2)"
    md5sum "$1" "$2"
  else
    echo "bytes differ -- comparing array by array"
    ./run_tf.sh python3 $CS/resolution/cmp_outputs.py "$1" "$2" "${@:3}"
  fi
}

case ${1:-} in
z)
  for W in ref new; do
    [ $W = ref ] && B=$REF || B=$CS
    [ -s $OUT/z_$W.hdf5 ] && continue
    ./run_tf.sh python3 -u $B/fullscale/make_card.py \
      --pairs $FS/runs/zpairs_dyv2.npz --fsr $Z/data/kern_loose_band3.3e-4.npz \
      --acc $Z/data/acc_loose_d8.json --maxn 300000 \
      --max-chi2-ndof 3 -o $OUT/z_$W.hdf5 2>&1 | tee $LOGS/gate_z_$W.log
  done
  cmpfiles $OUT/z_ref.hdf5 $OUT/z_new.hdf5 2>&1 | tee $LOGS/gate_z_cmp.log ;;
jpsi)
  # the J/psi leg of the joint card does NOT go through `make_card.build` --
  # `make_joint_card.build_jpsi` calls `make_card.select` and assembles its own
  # term -- so the SELECTION is the whole of what could have moved there, and
  # comparing the index array is the same gate without the term assembly.
  for W in ref new; do
    [ $W = ref ] && B=$REF || B=$CS
    [ -s $OUT/jpsisel_$W.npz ] && continue
    ./run_tf.sh python3 -u $CS/resolution/vtxres/gate_select.py --base $B \
      --pairs $FS/runs/jpairs_v2_n600.npz -o $OUT/jpsisel_$W.npz -- \
      --name jpsi --channel jpsi --mref 3.0969 --window 2.9469 3.2469 \
      --max-chi2-ndof 3 --max-sigma-rel 0.10 --corr-clip 0 \
      2>&1 | tee $LOGS/gate_jpsi_$W.log
  done
  cmpfiles $OUT/jpsisel_ref.npz $OUT/jpsisel_new.npz --skip-keys make_card \
    2>&1 | tee $LOGS/gate_jpsi_cmp.log ;;
matres|hitlik)
  ./gate_extract.sh ${1} 2>&1 | tee $LOGS/gate_${1}.log ;;
*) echo "usage: run_chi2_gates.sh z|jpsi|matres|hitlik"; exit 2 ;;
esac

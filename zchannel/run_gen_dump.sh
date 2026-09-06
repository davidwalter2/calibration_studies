#!/bin/bash
# Shard dump_gen_fsr.py over the full DY MiniAOD filelist, N workers per node.
#
#   ./run_gen_dump.sh <first> <last> <node> <nproc>
#
# One npz per input file in $OUT; already-written outputs are skipped, so the
# script is resumable.  ~84 s and ~88 k events per file, I/O bound.
set -u
FIRST=${1:-0}; LAST=${2:-399}; NODE=${3:-submit50}; NPROC=${4:-16}
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
FL=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_dy_miniaod_full_260905_paths.txt
OUT=/ceph/submit/data/user/d/david_w/ZMass/zgen

ssh "$NODE" "mkdir -p $OUT; seq $FIRST $LAST | xargs -P $NPROC -I{} bash -c \
  'test -s $OUT/gen_{}.npz || $Z/fwlite.sh $Z/dump_gen_fsr.py --filelist $FL \
   --skip-files {} --nfiles 1 --progress 0 -o $OUT/gen_{}.npz > $OUT/log_{}.txt 2>&1'"

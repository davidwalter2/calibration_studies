#!/bin/bash
# Shard dump_gen_perleg.py over the DY MiniAOD filelist, N workers per node.
#   ./run_perleg_dump.sh <first> <last> <node> <nproc>
set -u
FIRST=${1:-0}; LAST=${2:-59}; NODE=${3:-submit50}; NPROC=${4:-12}
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
FL=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_dy_miniaod_full_260905_paths.txt
OUT=/ceph/submit/data/user/d/david_w/ZMass/zgen_perleg

ssh "$NODE" "mkdir -p $OUT; seq $FIRST $LAST | xargs -P $NPROC -I{} bash -c \
  'test -s $OUT/pl_{}.npz || $Z/fwlite.sh $Z/dump_gen_perleg.py --filelist $FL \
   --skip-files {} --nfiles 1 --progress 0 -o $OUT/pl_{}.npz > $OUT/log_{}.txt 2>&1'"

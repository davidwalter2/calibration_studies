#!/bin/bash
# run.sh TAG NREP NEV [photos_gen args ...]
#
# Launches 78 x NREP independent `photos_gen` processes (one per (band,
# replica); Photos++ is a static singleton and cannot be threaded), then merges
# them into data/photos/gen_<TAG>.npz.  Each band's kernel is normalised on its
# own, so the equal-per-band allocation is free: it buys uniform precision
# across the mass range instead of following the sample's population.
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
SCRATCH=${PHOTOS_SCRATCH:-/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/photos}
NPROC=${NPROC:-260}
TAG=$1; NREP=$2; NEV=$3; shift 3
D=$SCRATCH/$TAG
mkdir -p "$D"
rm -f "$D"/*.bin
cd "$Z"
for b in $(seq 0 77); do
  for r in $(seq 0 $((NREP-1))); do
    echo "$b $r"
  done
done | xargs -P "$NPROC" -n 2 bash -c '
  b=$0; r=$1
  seed=$(( (b+1)*100003 + (r+1)*7919 + '"${SEED0:-0}"' ))
  '"$Z"'/photos_standalone/photos_gen --band-first=$b --band-last=$b \
      --n='"$NEV"' --seed=$seed '"$*"' --out='"$D"'/b${b}_r${r}.bin > /dev/null 2>&1 \
    || echo "FAILED b=$b r=$r" >&2
'
n=$(ls "$D"/*.bin 2>/dev/null | wc -l)
echo "$TAG: $n / $((78*NREP)) files"
python3 "$Z"/photos_standalone/merge.py "$D/*.bin" -o "$Z/data/photos/gen_$TAG.npz" \
        --meta "$TAG: $* (nrep=$NREP nev=$NEV)"
rm -f "$D"/*.bin
rmdir "$D" 2>/dev/null

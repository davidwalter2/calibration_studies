#!/bin/bash
# run_share.sh TAG NREP NEV [photos_share args ...]
#
# Launches NREP independent `photos_share` processes on ONE m_pre band (BAND,
# default 20 = [90, 92) GeV, the peak) and merges them into
# $SCRATCH/share_<TAG>.npz.  Photos++ is a static singleton and cannot be
# threaded, so parallelism is by separate processes.
#
# Build first, inside the el7 container (the binary then runs outside it):
#   g++ -O2 -std=c++14 -o photos_share photos_share.cc \
#       -I$PH/include -L$PH/lib -lPhotospp -lPhotosppHEPEVT \
#       -Wl,-rpath,$PH/lib -Wl,-rpath,$GCC/lib64
# with $PH, $GCC as in build.sh.
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
SCRATCH=${PHOTOS_SCRATCH:-/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/photos_share}
NPROC=${NPROC:-200}
BAND=${BAND:-20}
TAG=$1; NREP=$2; NEV=$3; shift 3
D=$SCRATCH/$TAG
mkdir -p "$D"
rm -f "$D"/*.bin
cd "$Z"
for r in $(seq 0 $((NREP-1))); do echo "$r"; done | xargs -P "$NPROC" -n 1 bash -c '
  r=$0
  seed=$(( ('"$BAND"'+1)*100003 + (r+1)*7919 + '"${SEED0:-0}"' ))
  '"$Z"'/photos_standalone/photos_share --band-first='"$BAND"' --band-last='"$BAND"' \
      --n='"$NEV"' --seed=$seed '"$*"' --out='"$D"'/r${r}.bin > /dev/null 2>&1 \
    || echo "FAILED r=$r" >&2
'
n=$(ls "$D"/*.bin 2>/dev/null | wc -l)
echo "$TAG: $n / $NREP files"
python3 "$Z"/photos_standalone/share_merge.py "$D/*.bin" -o "$SCRATCH/share_$TAG.npz" \
        --meta "$TAG: band=$BAND $* (nrep=$NREP nev=$NEV)"
rm -f "$D"/*.bin
rmdir "$D" 2>/dev/null

#!/bin/bash
# Every fit benchmark of the resolution classes, in sequence (one TensorFlow
# job at a time on the node).
set -u
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
for A in "$@"; do
  set -- $A
  CUTS=$1; NG=$2; S=${3:-}
  L=data/00_fit_kclass_${CUTS}_n${NG}${S:+_single}.log
  echo "=== $(date +%H:%M:%S)  $CUTS n$NG $S -> $L"
  ./run_kclass_fit.sh $CUTS $NG $S > $L 2>&1
  echo "=== $(date +%H:%M:%S)  exit $?"
done

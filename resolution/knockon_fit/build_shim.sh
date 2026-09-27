#!/bin/bash
# Build the knock-on validation shim (cxx/knockonshim.cc) against the cvhcf
# sources of a CMSSW tree (default: the dev2 area; override with CMSSW_SRC),
# compiled unmodified, as cxx/build_cvhcf.sh does for cvhcfshim.
#   usage: build_shim.sh [OUT.so]      (default knockon_fit/libknockonshim.so)
# At run time CMSSW_SRC must name the same tree (the Moliere shape table).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CXX=$HERE/../cxx
SRC=${CMSSW_SRC:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src}
OUT=${1:-$HERE/libknockonshim.so}
ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
CLHEP_INC=$(ls -d /cvmfs/cms.cern.ch/$ARCH/external/clhep/*/include 2>/dev/null | sort | tail -1 || true)
[ -n "$CLHEP_INC" ] || { echo "FATAL: no CLHEP include dir found on cvmfs"; exit 1; }
g++ -O3 -std=c++17 -fPIC -shared \
    -I"$SRC" -I"$CXX/stub" -I"$CLHEP_INC" \
    "$SRC/TrackPropagation/Geant4e/src/CGFQoPBlock.cc" \
    "$SRC/TrackPropagation/Geant4e/src/CvhCfExponents.cc" \
    "$CXX/knockonshim.cc" \
    -o "$OUT"
echo "[build] wrote $OUT"

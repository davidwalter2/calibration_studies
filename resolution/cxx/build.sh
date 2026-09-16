#!/bin/bash
# Build the offline CF's C++ kernel as a standalone .so, loadable by ctypes
# from the analysis venv (no CMSSW runtime needed).
#
# CGFQoPBlock.cc is compiled UNMODIFIED and straight out of the CMSSW tree --
# that is the entire point: the offline model and the fit must be the same
# object code path, not two transcriptions of one formula. Only two CMSSW
# headers are stubbed (Exception, ParameterSet), and both are used solely by
# `configure`, which the shim never calls.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${CMSSW_SRC:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src}
OUT=$HERE/libcgfshim.so

# CLHEP is header-only for PhysicalConstants; take it from the CMSSW externals
# so the constants are bit-identical to the ones the fit compiled against.
# Pin the ARCHITECTURE. A bare /cvmfs/cms.cern.ch/*/external glob sorts
# ubi8_ppc64le ahead of el9_amd64 and silently compiled against the PowerPC
# tree on the first build here. PhysicalConstants.h is header-only so it did
# no numerical harm, but "it happened to be only constants" is not a build
# rule. Use the release's own SCRAM_ARCH when it is set.
ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
CLHEP_INC=$(ls -d /cvmfs/cms.cern.ch/$ARCH/external/clhep/*/include 2>/dev/null | sort | tail -1 || true)
[ -n "$CLHEP_INC" ] || { echo "FATAL: no CLHEP include dir found on cvmfs"; exit 1; }
echo "[build] CLHEP: $CLHEP_INC"

g++ -O3 -std=c++17 -fPIC -shared -march=native \
    -I"$SRC" -I"$HERE/stub" -I"$CLHEP_INC" \
    "$SRC/TrackPropagation/Geant4e/src/CGFQoPBlock.cc" \
    "$HERE/cgfshim.cc" \
    -o "$OUT"
echo "[build] wrote $OUT"
nm -D --defined-only "$OUT" | grep cvhcgf_ || { echo "FATAL: shim symbols missing"; exit 1; }

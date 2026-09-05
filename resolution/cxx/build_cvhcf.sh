#!/bin/bash
# Build `cvhcf` (the IN-MAKER resolution-CF exponents) as a standalone .so,
# loadable by ctypes from the analysis venv -- the same construction, and for
# the same reason, as build.sh does for `cvhcgf`.
#
# CvhCfExponents.cc and CGFQoPBlock.cc are compiled UNMODIFIED out of the CMSSW
# tree: the point of the validation is that the maker and the validator run the
# same object code, not two transcriptions of one formula.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${CMSSW_SRC:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src}
OUT=$HERE/libcvhcfshim.so

ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
CLHEP_INC=$(ls -d /cvmfs/cms.cern.ch/$ARCH/external/clhep/*/include 2>/dev/null | sort | tail -1 || true)
[ -n "$CLHEP_INC" ] || { echo "FATAL: no CLHEP include dir found on cvmfs"; exit 1; }
echo "[build] CLHEP: $CLHEP_INC"

g++ -O3 -std=c++17 -fPIC -shared \
    -I"$SRC" -I"$HERE/stub" -I"$CLHEP_INC" \
    "$SRC/TrackPropagation/Geant4e/src/CGFQoPBlock.cc" \
    "$SRC/TrackPropagation/Geant4e/src/CvhCfExponents.cc" \
    "$HERE/cvhcfshim.cc" \
    -o "$OUT"
echo "[build] wrote $OUT"
# `grep -q` closes the pipe early, which under `pipefail` would fail the
# build on a SUCCESSFUL check; count instead.
SYMS=$(nm -D --defined-only "$OUT" | grep -c "^.* T cvhcf_" || true)
[ "${SYMS:-0}" -ge 8 ] || { echo "FATAL: shim symbols missing ($SYMS)"; exit 1; }
echo "[build] ok"

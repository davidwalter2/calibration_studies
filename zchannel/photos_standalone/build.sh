#!/bin/bash
# Compile the standalone driver against the same Photos++ 3.61 build CMSSW_10_6_26
# links (`scram tool info photospp`).  Plain g++, outside the CMSSW source tree.
set -e
PH=${PHOTOSPP_BASE:-/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/photospp/3.61-pafccj}
GCC=${GCCDIR:-/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/gcc/7.0.0-pafccj}
cd "$(dirname "$0")"
export PATH=$GCC/bin:$PATH
export LD_LIBRARY_PATH=$GCC/lib64:$GCC/lib:$PH/lib:$LD_LIBRARY_PATH
g++ -O2 -std=c++14 -Wall -Wno-unused-variable -o ${OUTBIN:-photos_gen} photos_gen.cc \
    -I"$PH/include" -L"$PH/lib" -lPhotospp -lPhotosppHEPEVT \
    -Wl,-rpath,"$PH/lib" -Wl,-rpath,"$GCC/lib64"
echo "built $(pwd)/${OUTBIN:-photos_gen}"

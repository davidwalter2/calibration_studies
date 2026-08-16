#!/bin/bash
# Build speciesdedx_g4driver against the ALREADY-BUILT CMSSW
# area.  Nothing in the CMSSW or Geant4 source trees is written to; only the
# two binaries are produced, into the scratchpad.  (Pattern copied from
# build_urban_g4driver.sh, including the wrapper that freezes the CMSSW
# LD_LIBRARY_PATH -- RUNPATH is not transitive and the CMSSW library pulls in
# dependencies the executable's own RUNPATH cannot resolve.)
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
eval $(scramv1 runtime -sh)

G4BASE=$(scram tool tag geant4core GEANT4CORE_BASE)
CLHEPBASE=$(scram tool tag clhep CLHEP_BASE)
VGBASE=$(scram tool tag vecgeom VECGEOM_BASE)
DEST=${1:-/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad}
SRCDIR=/work/submit/david_w/ZMass/calibration_studies/resolution

# NOTE on the lock: nothing here WRITES into the CMSSW area, but the link step
# READS libTrackPropagationGeant4e.so, which a concurrent `scram b` may
# rewrite.  That is exactly what the SHARED (`run`) side of cmsswlock.sh is
# for.  The lock is taken here, after cmsenv, because the script itself has to
# set CMSSW_BASE first.
LOCK=$CMSSW_BASE/src/Analysis/HitAnalyzer/test/cmsswlock.sh

OUT="$DEST/speciesdedx_g4driver"
"$LOCK" run g++ -O2 -std=c++17 -o "$OUT" "$SRCDIR/speciesdedx_g4driver.cc" \
  -I"$CMSSW_BASE/src" -I"$CMSSW_RELEASE_BASE/src" \
  -I"$G4BASE/include/Geant4" -I"$CLHEPBASE/include" -I"$VGBASE/include" \
  -L"$CMSSW_BASE/lib/$SCRAM_ARCH" -lTrackPropagationGeant4e \
  -L"$G4BASE/lib64" -lG4processes -lG4materials -lG4particles -lG4global \
     -lG4geometry -lG4track -lG4run -lG4event -lG4digits_hits -lG4intercoms \
  -L"$CLHEPBASE/lib" -lCLHEP \
  -Wl,-rpath,"$CMSSW_BASE/lib/$SCRAM_ARCH" -Wl,-rpath,"$G4BASE/lib64" \
  -Wl,-rpath,"$CLHEPBASE/lib"

cat > "$OUT.sh" <<EOS
#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
exec "$OUT" "\$@"
EOS
chmod +x "$OUT.sh"

echo "built $OUT (+ wrapper $OUT.sh)"

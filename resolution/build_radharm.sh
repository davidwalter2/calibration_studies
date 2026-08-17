#!/bin/bash
# Build radharm_g4driver against the ALREADY-BUILT CMSSW area (pattern copied
# from build_barkas.sh).  Nothing in CMSSW or Geant4 is written to.
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
eval $(scramv1 runtime -sh)

G4BASE=$(scram tool tag geant4core GEANT4CORE_BASE)
CLHEPBASE=$(scram tool tag clhep CLHEP_BASE)
VGBASE=$(scram tool tag vecgeom VECGEOM_BASE)
DEST=${1:-/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad}
SRCDIR=/work/submit/david_w/ZMass/calibration_studies/resolution
LOCK=$CMSSW_BASE/src/Analysis/HitAnalyzer/test/cmsswlock.sh

OUT="$DEST/radharm_g4driver"
"$LOCK" run g++ -O2 -std=c++17 -o "$OUT" "$SRCDIR/radharm_g4driver.cc" \
  -I"$CMSSW_BASE/src" -I"$CMSSW_RELEASE_BASE/src" \
  -I"$G4BASE/include/Geant4" -I"$CLHEPBASE/include" -I"$VGBASE/include" \
  -L"$G4BASE/lib64" -lG4processes -lG4materials -lG4particles -lG4global \
     -lG4geometry -lG4track -lG4run -lG4event -lG4digits_hits -lG4intercoms \
  -L"$CLHEPBASE/lib" -lCLHEP \
  -Wl,-rpath,"$G4BASE/lib64" -Wl,-rpath,"$CLHEPBASE/lib"

cat > "$OUT.sh" <<EOS
#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
exec "$OUT" "\$@"
EOS
chmod +x "$OUT.sh"
echo "built $OUT (+ wrapper $OUT.sh)"

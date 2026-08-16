#!/bin/bash
# Build urban_g4driver against the ALREADY-BUILT CMSSW area. Nothing in the
# CMSSW source tree is written to; only the driver binary is produced, into
# the scratchpad.
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
eval $(scramv1 runtime -sh)

G4BASE=$(scram tool tag geant4core GEANT4CORE_BASE)
CLHEPBASE=$(scram tool tag clhep CLHEP_BASE)
VGBASE=$(scram tool tag vecgeom VECGEOM_BASE)
OUT=${1:-/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad/urban_g4driver}
SRC=/work/submit/david_w/ZMass/calibration_studies/resolution/urban_g4driver.cc

g++ -O2 -std=c++17 -o "$OUT" "$SRC" \
  -I"$CMSSW_BASE/src" -I"$CMSSW_RELEASE_BASE/src" \
  -I"$G4BASE/include/Geant4" -I"$CLHEPBASE/include" -I"$VGBASE/include" \
  -L"$CMSSW_BASE/lib/$SCRAM_ARCH" -lTrackPropagationGeant4e \
  -L"$G4BASE/lib64" -lG4processes -lG4materials -lG4particles -lG4global \
     -lG4geometry -lG4track -lG4run -lG4event -lG4digits_hits -lG4intercoms \
  -L"$CLHEPBASE/lib" -lCLHEP \
  -Wl,-rpath,"$CMSSW_BASE/lib/$SCRAM_ARCH" -Wl,-rpath,"$G4BASE/lib64" \
  -Wl,-rpath,"$CLHEPBASE/lib"
# The CMSSW library pulls in transitive dependencies (libMagneticFieldRecords
# and friends) that a RUNPATH on the executable does NOT resolve -- RUNPATH is
# not transitive. Freeze the CMSSW runtime LD_LIBRARY_PATH into a wrapper so
# the driver can be launched from the calibration_studies venv, which cannot
# have cmsenv applied to it (it would replace `python`).
cat > "$OUT.sh" <<EOS
#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
exec "$OUT" "\$@"
EOS
chmod +x "$OUT.sh"
echo "built $OUT (+ wrapper $OUT.sh)"

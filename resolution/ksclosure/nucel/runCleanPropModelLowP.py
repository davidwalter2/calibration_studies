## runCleanPropModel.py with the Geant4e momentum floor lowered from its 1 GeV
## default to 0.05 GeV (the K_S refit's value), so a sub-GeV pion reference can
## be propagated.  Everything else is the original configuration.
import FWCore.ParameterSet.Config as cms
_ORIG = ('/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/'
         'HitAnalyzer/test/runCleanPropModel.py')
exec(compile(open(_ORIG).read(), _ORIG, 'exec'))
process.Geant4ePropagator.PropagationPtotLimit = cms.double(0.05)

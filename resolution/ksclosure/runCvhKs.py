## CVH two-track refit driver for K_S -> pi+ pi- in the inclusive
## B -> J/psi + X MC ALCARECO (2016 postVFP, v3 campaign, split=1, gen+sim
## truth kept).  Displaced-vertex momentum-scale closure: the vertex
## constraint is ON and the beam-spot constraint is OFF (a K_S is displaced
## by construction), no mass constraint, pion mass hypothesis on both legs.
##
## Candidate source: the persisted ALCARECOTkAlJpsiXB0KsResonances collection.
## Each entry is a B0 -> [J/psi -> mu mu] [K_S -> pi pi] pairing whose
## daughter(1) is the K_S sub-candidate with two charged-track leaves, so
## subsystemDaughter=1 selects the K_S two-track subsystem of the generic
## VertexCompositeCandidate decomposition in
## ResidualGlobalCorrectionMakerTwoTrackG4e.cc.  No new C++ is needed.
##
## Truth matching is done OFFLINE (the maker's built-in gen matching is
## muon-specific): rows are keyed by (run, lumi, event) -- unique in this
## campaign -- and joined against the SimVertex K_S -> pi+ pi- truth dumped
## by resolution/ksclosure/ks_truth_dump.py.
##
## Configuration follows runCvhJpsiGenMC.py so the K_S closure is comparable
## to the J/psi one, with the K_S-specific changes:
##   - pion mass on both legs, CvhMaster particle list pi+/pi-
##   - propagation momentum floor 0.05 GeV (soft V0 daughters)
##   - useStartingState='perigee' (measured better than midPropagated on V0s)
##   - useIdealGeometry=True (MC closure: the simulated detector is ideal)
##   - useDefaultField=True (the SIM propagated through OAE)
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.AlCa.GlobalTag import GlobalTag

opts = VarParsing.VarParsing('analysis')
opts.register('input', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string,
              'comma-separated absolute paths or root:// URLs of ALCARECO files')
opts.register('inputFileList', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string,
              'text file with one input path per line')
opts.register('nEvents', -1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'events to process (-1 = all)')
opts.register('srcCandidates', 'ALCARECOTkAlJpsiXB0KsResonances',
              VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string,
              'VertexCompositeCandidate collection to drive the refit')
opts.register('subsystemDaughter', 1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int,
              'which daughter of each candidate is the two-track subsystem to '
              'fit: 1 = the K_S of a B0 -> J/psi K_S pairing; -1 = the '
              'candidate itself (a flat K_S collection)')
opts.register('tightG4eStepper', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'match the Geant4e field-integration tolerances in the FIT to the SIM')
opts.register('useIdealGeometry', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'ideal tracker geometry: the simulated detector IS the ideal one, '
              'while the GT alignment payload is a misalignment scenario')
opts.register('useDefaultField', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'unlabelled default CMSSW field (OAE tracker parametrization), '
              'required for a gen-matched MC closure: the SIM propagated through it')
opts.register('scalarPot3DInitFile', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string,
              'scalar-potential coefficient dump; required (registers parmtype-14 modes)')
opts.register('globalTag', '106X_mcRun2_asymptotic_v17',
              VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'conditions')
opts.register('doRes', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'export the per-candidate mass-CF resolution ingredients')
opts.register('fillGradsFactored', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'per-event gradient + low-rank factored Hessian (H = B^T B)')
opts.register('fillJac', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'per-track Jacobians')
opts.register('exportCfExponents', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'in-maker CF exponents (cfmass_*)')
opts.register('exportCfGroupExponents', False, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'per-material-group CF exponents')
opts.register('exportHitResBlocks', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'parmtype-8/9 hit-resolution dV blocks')
opts.register('exportVtxResidual', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'export the vertex-constraint residual as a CF term (the constraint '
              'is ON here, so this is the diagnostic of what it costs)')
opts.register('doVtxConstraint', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'common-vertex constraint (ON)')
opts.register('bsConstraint', False, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'luminous-region constraint: OFF -- a K_S is displaced by '
              'construction (c*tau = 2.68 cm), the constraint would drag the '
              'decay vertex onto the beam line')
opts.register('doMassConstraint', False, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'K_S mass constraint: OFF -- the mass is the observable')
opts.register('minLegHits', 8, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'minimum valid hits on the weaker leg')
opts.register('minNdof', 1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'minimum pre-fit degrees of freedom')
opts.register('minPairHits', -1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'minimum valid hits over the pair (-1 = auto)')
opts.register('propagationPtotLimit', 0.05, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float,
              'G4e propagation momentum floor [GeV]; V0 daughters are soft')
opts.register('clampMomentumFloor', -1.0, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float,
              'Gauss-Newton momentum floor; <0 = 1.25 * propagationPtotLimit')
opts.register('maxBacktracks', 4, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'failed-leg GN step halving budget')
opts.register('maxSeedInflations', 2, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'iteration-0 seed inflation budget')
opts.register('useStartingState', 'perigee', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string,
              "iteration-0 reference state; 'perigee' measured better than "
              "'midPropagated' on V0s")
opts.register('propagationDirection', 'anyDirection', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'Geant4ePropagator PropagationDirection')
opts.register('numberOfThreads', 1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'framework threads (streams follow)')
opts.register('nIters', 10, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'Gauss-Newton iteration cap per phase')
opts.register('edmConvergence', 1e-5, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float, 'reference-block EDM convergence')
_defaultGroupsFile = os.path.join(os.environ.get('CMSSW_BASE', ''),
                                  'src/Analysis/HitAnalyzer/data/materialGroups50.txt')
opts.register('materialGroupsFile', _defaultGroupsFile, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'material grouping-tier rules file')
opts.register('globalMaterialModel', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool,
              'parmtype-15 global material groups (origin-independent: what makes '
              'a displaced track fittable with the same parameters as a prompt one)')
opts.register('skipHitlessSurfaces', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'hit-to-hit propagation')
opts.register('perStepFieldModes', True, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.bool, 'per-step field-mode attribution')
opts.register('outprefix', 'globalcor_ks', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'output file prefix')
opts.register('eventsToProcess', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'comma-separated run:event list')

import TrackPropagation.Geant4e.cvhSwitches as cvhSwitches
cvhSwitches.register(opts)
opts.parseArguments()

# The two-track maker has no CGF override hooks (see runCvhJpsiGenMC.py): under
# CgfQoPMode >= 1 the fluctuation model would return the untruncated ionization
# second cumulant as the leg q/p variance without the maker substituting the
# Fisher weight.
if opts.CgfQoPMode < 0:
    opts.CgfQoPMode = 0
    print('[cvh] two-track driver: CgfQoPMode defaulting to 0 (legacy truncated-Q)')
if not opts.scalarPot3DInitFile:
    raise SystemExit('scalarPot3DInitFile=<path> is required (coefficient dump file)')

process = cms.Process("BENCH", Run2_2016)

process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.GeometrySimDB_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, opts.globalTag, "")
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
        record=cms.string("GeometryFileRcd"),
        tag=cms.string("XMLFILE_Geometry_2016_81YV1_Extended2016_mc"),
        label=cms.untracked.string("Extended"),
    ),
)
process.XMLFromDBSource.label = cms.string("Extended")

process.load("TrackPropagation.Geant4e.geantRefit_cff")
if opts.tightG4eStepper:
    _fsp = process.geopro.MagneticField.ConfGlobalMFM.OCMS.StepperParam
    _fsp.DeltaOneStepTracker = 1e-5
    _fsp.DeltaIntersectionTracker = 1e-6
    _fsp.DeltaOneStep = 1e-5
    _fsp.DeltaIntersection = 1e-6
from TrackPropagation.Geant4e.cvhMaster_cfi import CvhMasterPSet

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.nEvents))

_paths = [p.strip() for p in opts.input.split(',') if p.strip()]
if opts.inputFileList:
    with open(opts.inputFileList) as _f:
        _paths += [l.strip() for l in _f if l.strip() and not l.startswith('#')]
assert _paths, "must set input=<paths> and/or inputFileList=<file>"
_urls = [p if p.startswith(("root://", "file:")) else "file:" + p for p in _paths]
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(*_urls),
    secondaryFileNames=cms.untracked.vstring(),
    # the condor MC production has a tail of zero-length / truncated files
    skipBadFiles=cms.untracked.bool(True),
)
if opts.eventsToProcess:
    process.source.eventsToProcess = cms.untracked.VEventRange(
        *[s.strip() for s in opts.eventsToProcess.split(',') if s.strip()])

process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfStreams=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1),
)
process.RandomNumberGeneratorService.globalCor = cms.PSet(
    initialSeed=cms.untracked.uint32(123456789),
    engineName=cms.untracked.string('HepJamesRandom'),
)
process.MessageLogger.cerr.FwkReport.reportEvery = 200
process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

process.globalCor = cms.EDProducer(
    "ResidualGlobalCorrectionMakerTwoTrackG4e",
    src=cms.InputTag("ALCARECOTkAlJpsiX"),
    srcCandidates=cms.InputTag(opts.srcCandidates),
    subsystemDaughter=cms.int32(int(opts.subsystemDaughter)),
    dedxSourceTracks=cms.InputTag("ALCARECOTkAlJpsiX"),
    dedxHarmonic2=cms.InputTag("ALCARECOTkAlJpsiXDeDxHarmonic2"),
    dedxPixelHarmonic2=cms.InputTag("ALCARECOTkAlJpsiXDeDxPixelHarmonic2"),
    dedxAllHarmonic2=cms.InputTag("ALCARECOTkAlJpsiXDeDxAllHarmonic2"),
    fitFromGenParms=cms.bool(False),
    fitFromSimParms=cms.bool(False),
    fillTrackTree=cms.bool(True),
    fillGrads=cms.bool(False),
    fillGradsFactored=cms.untracked.bool(bool(opts.fillGradsFactored)),
    fillJac=cms.bool(bool(opts.fillJac)),
    fillRunTree=cms.bool(True),
    # The maker's own gen matching is muon-specific (status-1, |pdgId| == 13),
    # so it finds nothing for pions; requireGen MUST stay False or every
    # candidate is silently rejected. Truth matching is done offline.
    doGen=cms.bool(False),
    genParticles=cms.InputTag("genParticles"),
    pileupInfo=cms.InputTag("addPileupInfo"),
    doSim=cms.bool(False),
    requireGen=cms.bool(False),
    doMuons=cms.bool(False),
    doMuonAssoc=cms.bool(False),
    doTrigger=cms.bool(False),
    triggers=cms.vstring(),
    doL1Trigger=cms.bool(False),
    l1Results=cms.InputTag('gtDigis', '', 'RECO'),
    l1Triggers=cms.vstring(),
    doRes=cms.bool(bool(opts.doRes)),
    exportStepRecords=cms.bool(False),
    exportCfExponents=cms.bool(bool(opts.exportCfExponents)),
    exportCfGroupExponents=cms.bool(bool(opts.exportCfGroupExponents)),
    exportHitResBlocks=cms.bool(bool(opts.exportHitResBlocks)),
    exportVtxResidual=cms.bool(bool(opts.exportVtxResidual)),
    useIdealGeometry=cms.bool(bool(opts.useIdealGeometry)),
    bsConstraint=cms.bool(bool(opts.bsConstraint)),
    doVtxConstraint=cms.bool(bool(opts.doVtxConstraint)),
    applyHitQuality=cms.bool(True),
    minNdof=cms.int32(int(opts.minNdof)),
    minPairHits=cms.int32(int(opts.minPairHits)),
    minLegHits=cms.int32(int(opts.minLegHits)),
    doMassConstraint=cms.bool(bool(opts.doMassConstraint)),
    massConstraint=cms.double(0.497611),
    massConstraintWidth=cms.double(7.351e-15),
    daughterParticleName1=cms.string("pi"),
    daughterParticleName2=cms.string("pi"),
    corFiles=cms.vstring(),
    MagneticFieldLabel=cms.string(""),
    scalarPotentialInitFile=cms.string(opts.scalarPot3DInitFile),
    nIters=cms.uint32(int(opts.nIters)),
    edmConvergence=cms.double(float(opts.edmConvergence)),
    materialGroupsFile=cms.string(opts.materialGroupsFile),
    globalMaterialModel=cms.bool(bool(opts.globalMaterialModel)),
    perStepFieldModes=cms.bool(bool(opts.perStepFieldModes)),
    skipHitlessSurfaces=cms.bool(bool(opts.skipHitlessSurfaces) and bool(opts.globalMaterialModel)),
    useStartingState=cms.string(opts.useStartingState),
    maxBacktracks=cms.uint32(int(opts.maxBacktracks)),
    maxSeedInflations=cms.uint32(int(opts.maxSeedInflations)),
    outprefix=cms.untracked.string(opts.outprefix),
    CvhMaster=CvhMasterPSet.clone(Particles=cms.vstring("pi+", "pi-")),
)

if opts.useDefaultField:
    fieldlabel = ""
else:
    from MagneticField.ParametrizedEngine.parametrizedMagneticField_ScalarPot3D_cfi \
        import ParametrizedMagneticFieldProducer as ScalarPot3DMagneticFieldProducer
    process.ScalarPot3DMagneticFieldProducer = ScalarPot3DMagneticFieldProducer.clone()
    process.ScalarPot3DMagneticFieldProducer.parameters.InitFile = opts.scalarPot3DInitFile
    fieldlabel = "ScalarPot3DMf"
    process.ScalarPot3DMagneticFieldProducer.label = fieldlabel
process.geopro.MagneticFieldLabel = fieldlabel
process.Geant4ePropagator.MagneticFieldLabel = fieldlabel
from TrackPropagation.Geant4e.cvhMasterESProducer_cfi import cvhMasterESProducer
process.cvhMasterESProducer = cvhMasterESProducer.clone()
process.cvhMasterESProducer.MagneticFieldLabel = cms.string(fieldlabel)
process.Geant4ePropagator.ForCVH = cms.bool(True)
process.Geant4ePropagator.PropagationDirection = cms.string(opts.propagationDirection)
process.Geant4ePropagator.PropagationPtotLimit = cms.double(float(opts.propagationPtotLimit))

_clampFloor = (float(opts.clampMomentumFloor) if float(opts.clampMomentumFloor) > 0.
               else 1.25 * float(opts.propagationPtotLimit))
if _clampFloor <= float(opts.propagationPtotLimit):
    raise RuntimeError("clampMomentumFloor must be above propagationPtotLimit")
process.globalCor.clampMomentumFloor = cms.double(_clampFloor)
print("[cvh] effective: PropagationPtotLimit=%g GeV, clampMomentumFloor=%g GeV"
      % (float(opts.propagationPtotLimit), _clampFloor))
cvhSwitches.apply(process, opts)
process.globalCor.MagneticFieldLabel = cms.string(fieldlabel)

process.reconstruction_step = cms.Path(process.offlineBeamSpot * process.globalCor)
process.schedule = cms.Schedule(process.reconstruction_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

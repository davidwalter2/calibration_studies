## CVH two-track refit PROFILING driver.
##
## One config for every input we want to compare like-for-like, so the maker
## is instantiated identically and only the tracks change:
##   inputType=miniaod  -> slimmedMuons -> TrackProducerFromPatMuons
##                         (exactly runCvhDimuonMiniAOD.py)
##   inputType=tracks   -> an existing reco::TrackCollection (ALCARECO
##                         TkAlJpsiMuMu, generalTracks of the J/psi gun, ...)
## Both then go through the SAME diMuonTrackVertexCandidates + the SAME
## ResidualGlobalCorrectionMakerTwoTrackG4e, so a per-candidate cost
## comparison is not confounded by a different producer chain.
##
## Instrumentation: Timing service in per-event mode (TimeEvent> lines) +
## wantSummary (TimeReport per module).  The output tree carries run/lumi/
## event, so per-event wall time can be attributed to candidates offline.
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.AlCa.GlobalTag import GlobalTag

opts = VarParsing.VarParsing('analysis')
_S = VarParsing.VarParsing.multiplicity.singleton
_T = VarParsing.VarParsing.varType
opts.register('input', '', _S, _T.string, 'input file (comma separated ok)')
opts.register('inputType', 'miniaod', _S, _T.string, 'miniaod | tracks')
opts.register('trackSrc', '', _S, _T.string, 'reco::TrackCollection label for inputType=tracks')
opts.register('nEvents', 200, _S, _T.int, 'events')
opts.register('numberOfThreads', 1, _S, _T.int, 'threads/streams')
opts.register('scalarPot3DInitFile', '', _S, _T.string, 'parmtype-14 coefficient dump')
opts.register('massMin', 60.0, _S, _T.float, 'min mu-mu mass')
opts.register('massMax', 120.0, _S, _T.float, 'max mu-mu mass')
opts.register('outprefix', 'globalcor_profile', _S, _T.string, 'output tree prefix')
opts.register('globalTag', '106X_mcRun2_asymptotic_v17', _S, _T.string, 'conditions')
opts.register('doRes', False, _S, _T.bool, 'in-maker CF exports')
opts.register('exportStepRecords', False, _S, _T.bool, 'raw per-step export (huge)')
opts.register('exportCfExponents', True, _S, _T.bool, 'CF exponents on the tau grid')
opts.register('fillGrads', False, _S, _T.bool, 'packed Hessian')
opts.register('fillGradsFactored', False, _S, _T.bool, 'factored Hessian H=B^T B')
opts.register('fillJac', False, _S, _T.bool, 'per-track Jacobians')
opts.register('fillTrackTree', True, _S, _T.bool, 'per-candidate tree')
opts.register('produceValueMaps', True, _S, _T.bool, 'EDM ValueMaps')
opts.register('tightG4eStepper', False, _S, _T.bool, 'tighten geopro field-integration tolerances')
opts.register('perStepFieldModes', True, _S, _T.bool, 'per-G4-step field-mode attribution')
opts.register('globalMaterialModel', True, _S, _T.bool, 'parmtype-15 material groups')
opts.register('nIters', 10, _S, _T.int, 'GN iteration cap')
opts.register('edmConvergence', 1e-5, _S, _T.float, 'EDM convergence')
opts.register('useIdealGeometry', False, _S, _T.bool, 'ideal tracker geometry')
# The CVH energy-loss / CGF physics switches as command-line options, so this
# driver can PIN the estimator instead of silently inheriting the cfi default.
# CgfQoPMode in particular: the cfi default is 1 (Fisher weight, one 262144-
# point FFT per propagate call), the J/psi production drivers pin 0.
import TrackPropagation.Geant4e.cvhSwitches as cvhSwitches
cvhSwitches.register(opts)
opts.parseArguments()
assert opts.input, "set input=<file>"
assert opts.scalarPot3DInitFile, "set scalarPot3DInitFile=<coeff dump>"

process = cms.Process("CVHPROF", Run2_2016)
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometrySimDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, opts.globalTag, "")
process.GlobalTag.toGet = cms.VPSet(cms.PSet(
    record=cms.string("GeometryFileRcd"),
    tag=cms.string("XMLFILE_Geometry_2016_81YV1_Extended2016_mc"),
    label=cms.untracked.string("Extended"),
))
process.XMLFromDBSource.label = cms.string("Extended")
process.load("TrackPropagation.Geant4e.geantRefit_cff")
if opts.tightG4eStepper:
    _fsp = process.geopro.MagneticField.ConfGlobalMFM.OCMS.StepperParam
    _fsp.DeltaOneStepTracker = 1e-5
    _fsp.DeltaIntersectionTracker = 1e-6
    _fsp.DeltaOneStep = 1e-5
    _fsp.DeltaIntersection = 1e-6

process.MessageLogger.cerr.FwkReport.reportEvery = 25
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.nEvents))
_paths = [p.strip() for p in opts.input.split(',') if p.strip()]
_urls = [p if p.startswith(("root://", "file:")) else "file:" + p for p in _paths]
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(*_urls),
                            duplicateCheckMode=cms.untracked.string('noDuplicateCheck'))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfStreams=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1),
    wantSummary=cms.untracked.bool(True),
)
# Per-event wall/cpu: "TimeEvent> <ev> <run> <lumi> <cpu> <wall>"
process.Timing = cms.Service("Timing",
                             summaryOnly=cms.untracked.bool(False),
                             useJobReport=cms.untracked.bool(False))

fieldlabel = ""   # MC: default CMSSW field, consistent with the simulation
process.Geant4ePropagator.ForCVH = cms.bool(True)
process.Geant4ePropagator.PropagationDirection = cms.string("anyDirection")
cvhSwitches.apply(process, opts)
from TrackPropagation.Geant4e.cvhMasterESProducer_cfi import cvhMasterESProducer
process.cvhMasterESProducer = cvhMasterESProducer.clone()
process.cvhMasterESProducer.MagneticFieldLabel = cms.string(fieldlabel)

from Analysis.HitAnalyzer.diMuonTrackVertexCandidates_cfi import diMuonTrackVertexCandidates
from Analysis.HitAnalyzer.ResidualGlobalCorrectionMakerDiMuonG4e_cfi import ResidualGlobalCorrectionMakerDiMuonG4e

if opts.inputType == 'miniaod':
    process.tracksfrommuons = cms.EDProducer("TrackProducerFromPatMuons",
        src=cms.InputTag("slimmedMuons"), innerTrackOnly=cms.bool(False), ptMin=cms.double(-1.))
    _tracks = "tracksfrommuons"
    _pre = process.tracksfrommuons
else:
    assert opts.trackSrc, "inputType=tracks needs trackSrc="
    _tracks = opts.trackSrc
    _pre = None

process.diMuonTrackVertexCandidates = diMuonTrackVertexCandidates.clone(
    src=_tracks, massMin=opts.massMin, massMax=opts.massMax)
process.trackrefitdimuon = ResidualGlobalCorrectionMakerDiMuonG4e.clone(
    src=cms.InputTag(_tracks),
    srcCandidates=cms.InputTag("diMuonTrackVertexCandidates"),
    MagneticFieldLabel=cms.string(fieldlabel),
    scalarPotentialInitFile=cms.string(opts.scalarPot3DInitFile),
    fillTrackTree=cms.bool(bool(opts.fillTrackTree)),
    fillGrads=cms.bool(bool(opts.fillGrads)),
    fillGradsFactored=cms.untracked.bool(bool(opts.fillGradsFactored)),
    fillJac=cms.bool(bool(opts.fillJac)),
    doRes=cms.bool(bool(opts.doRes)),
    exportStepRecords=cms.bool(bool(opts.exportStepRecords)),
    exportCfExponents=cms.bool(bool(opts.exportCfExponents)),
    perStepFieldModes=cms.bool(bool(opts.perStepFieldModes)),
    globalMaterialModel=cms.bool(bool(opts.globalMaterialModel)),
    nIters=cms.uint32(int(opts.nIters)),
    edmConvergence=cms.double(float(opts.edmConvergence)),
    useIdealGeometry=cms.bool(bool(opts.useIdealGeometry)),
    produceValueMaps=cms.bool(bool(opts.produceValueMaps)),
    outprefix=cms.untracked.string(opts.outprefix),
)
process.RandomNumberGeneratorService.trackrefitdimuon = cms.PSet(
    initialSeed=cms.untracked.uint32(423456789), engineName=cms.untracked.string('HepJamesRandom'))

process.p = cms.Path((_pre * process.diMuonTrackVertexCandidates * process.trackrefitdimuon)
                     if _pre is not None else
                     (process.diMuonTrackVertexCandidates * process.trackrefitdimuon))
process.schedule = cms.Schedule(process.p)

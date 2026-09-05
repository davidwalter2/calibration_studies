## Two-track (dimuon) CVH refit driven straight off UL16 MiniAODv2.
## Mirrors runCvhDimuon.py, but the track source is the pat::Muon tracker
## tracks of `slimmedMuons` (TrackProducerFromPatMuons), i.e. exactly the
## path the WMass custom NanoAOD uses (PhysicsTools/NanoAOD muons_cff
## tracksfrommuons -> diMuonTrackVertexCandidates -> trackrefitdimuon).
## MC -> default CMSSW field (no ScalarPot3D override), per CLAUDE.md.
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.AlCa.GlobalTag import GlobalTag

opts = VarParsing.VarParsing('analysis')
opts.register('input', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'MiniAOD file')
opts.register('nEvents', 100, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'events')
opts.register('numberOfThreads', 1, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'threads/streams')
opts.register('scalarPot3DInitFile', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'scalar-potential coeff dump (parmtype-14 basis)')
opts.register('massMin', 60.0, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float, 'min mu-mu mass')
opts.register('massMax', 120.0, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float, 'max mu-mu mass')
opts.register('outprefix', 'globalcor_dimuon_miniaod', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'output tree prefix')
opts.parseArguments()
assert opts.input, "set input=<MiniAOD>"
assert opts.scalarPot3DInitFile, "set scalarPot3DInitFile=<coeff dump>"

process = cms.Process("CVHDIMUMINI", Run2_2016)
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometrySimDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# Conditions the MC was produced with (106X UL2016 MC chain).
process.GlobalTag = GlobalTag(process.GlobalTag, "106X_mcRun2_asymptotic_v17", "")
process.GlobalTag.toGet = cms.VPSet(cms.PSet(
    record=cms.string("GeometryFileRcd"),
    tag=cms.string("XMLFILE_Geometry_2016_81YV1_Extended2016_mc"),
    label=cms.untracked.string("Extended"),
))
process.XMLFromDBSource.label = cms.string("Extended")
process.load("TrackPropagation.Geant4e.geantRefit_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.nEvents))
_url = opts.input if opts.input.startswith(("root://", "file:")) else "file:" + opts.input
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(_url))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfStreams=cms.untracked.uint32(int(opts.numberOfThreads)),
    numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1),
)

# --- MC: keep the DEFAULT CMSSW field (label ""), consistent with the sim.
fieldlabel = ""
process.Geant4ePropagator.ForCVH = cms.bool(True)
process.Geant4ePropagator.PropagationDirection = cms.string("anyDirection")
from TrackPropagation.Geant4e.cvhMasterESProducer_cfi import cvhMasterESProducer
process.cvhMasterESProducer = cvhMasterESProducer.clone()
process.cvhMasterESProducer.MagneticFieldLabel = cms.string(fieldlabel)

# --- MiniAOD muon tracker tracks -> reco::TrackCollection (+ pat::Muon assoc).
process.tracksfrommuons = cms.EDProducer("TrackProducerFromPatMuons",
    src=cms.InputTag("slimmedMuons"),
    innerTrackOnly=cms.bool(False),
    ptMin=cms.double(-1.),
)

from Analysis.HitAnalyzer.diMuonTrackVertexCandidates_cfi import diMuonTrackVertexCandidates
from Analysis.HitAnalyzer.ResidualGlobalCorrectionMakerDiMuonG4e_cfi import ResidualGlobalCorrectionMakerDiMuonG4e
process.diMuonTrackVertexCandidates = diMuonTrackVertexCandidates.clone(
    src="tracksfrommuons", massMin=opts.massMin, massMax=opts.massMax)
process.trackrefitdimuon = ResidualGlobalCorrectionMakerDiMuonG4e.clone(
    src=cms.InputTag("tracksfrommuons"),
    srcCandidates=cms.InputTag("diMuonTrackVertexCandidates"),
    MagneticFieldLabel=cms.string(fieldlabel),
    scalarPotentialInitFile=cms.string(opts.scalarPot3DInitFile),
    fillTrackTree=cms.bool(True),
    produceValueMaps=cms.bool(True),
    outprefix=cms.untracked.string(opts.outprefix),
)
process.RandomNumberGeneratorService.trackrefitdimuon = cms.PSet(
    initialSeed=cms.untracked.uint32(423456789), engineName=cms.untracked.string('HepJamesRandom'))

process.p = cms.Path(process.tracksfrommuons *
                     process.diMuonTrackVertexCandidates *
                     process.trackrefitdimuon)
process.schedule = cms.Schedule(process.p)

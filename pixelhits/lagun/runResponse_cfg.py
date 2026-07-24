## TrackRefitter + PixelLAResponse tree on the muon-gun GEN-SIM-RECO
## files (Lorentz-angle response study). Usage:
##   cmsRun runResponse_cfg.py input=<file> output=<tree.root>
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.AlCa.GlobalTag import GlobalTag

opts = VarParsing.VarParsing('analysis')
opts.register('input', '', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'input GEN-SIM-RECO file')
opts.register('output', 'response.root', VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.string, 'output tree file')
opts.parseArguments()

process = cms.Process("LARESP", Run2_2016)
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, "106X_mcRun2_asymptotic_v17", "")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring("file:" + opts.input))
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

# Refit with the analysis TTRH builder (template CPE with track angle).
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = "generalTracks"
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"
process.TrackRefitter.NavigationSchool = ""

process.TFileService = cms.Service("TFileService",
    fileName=cms.string(opts.output))

process.laResponse = cms.EDAnalyzer("PixelLAResponse",
    trajTrackAsso=cms.InputTag("TrackRefitter"))

process.p = cms.Path(process.MeasurementTrackerEvent * process.TrackRefitter * process.laResponse)

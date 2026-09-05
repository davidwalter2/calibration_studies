import os
import FWCore.ParameterSet.Config as cms
inpath  = os.environ["REPACK_IN"]
outpath = os.environ["REPACK_OUT"]
nmax    = int(os.environ.get("REPACK_N", "-1"))
inurl = inpath if inpath.startswith(("root://", "file:")) else "file:" + inpath
process = cms.Process("REPACK")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(inurl))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(nmax))
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(1))
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.out = cms.OutputModule("PoolOutputModule",
    fileName=cms.untracked.string("file:" + outpath),
    splitLevel=cms.untracked.int32(0),
    overrideInputFileSplitLevels=cms.untracked.bool(True),
    outputCommands=cms.untracked.vstring("keep *"),
)
process.ep = cms.EndPath(process.out)

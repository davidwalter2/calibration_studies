# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: --python_filename step2_digireco.py --filein file:step1.root --fileout file:step2.root --mc --eventcontent RECOSIM --datatier GEN-SIM-RECO --conditions auto:run2_design --step DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --era Run2_2016 --pileup NoPileUp --geometry DB:Ideal --no_exec -n 10 --customise Analysis/HitAnalyzer/simprod_customise.customise_nocastor --customise_commands process.RECOSIMoutput.outputCommands.extend(['keep PSimHits_g4SimHits_TrackerHits*_*','keep SimTracks_g4SimHits__*','keep SimVertexs_g4SimHits__*'])
import os
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016

process = cms.Process('RECO',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('--python_filename nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
# Conditions MUST match what the CVH refit uses, because the fit
# RE-EVALUATES hit positions with its own CPEs: a mismatched
# SiPixelLorentzAngle / SiPixelTemplate payload shifts local-x per
# module and fakes a pixel hit-quality bias. Measured 2026-08-07:
# with auto:run2_design (-> 131X_mcRun2_design_v3, whose pixel
# templates are SiPixelTemplates38T_2010_2011_mc) against a fit on
# 106X_mcRun2_asymptotic_v17, 97.5% of BPix L1 modules carried a
# >3sigma mean local-x residual; matching the GTs took that to 0.6%.
# UL16 is also simply the right description of the 2016 detector.
# 150X_mcRun2_asymptotic_v1 (auto:run2_mc) is the RELEASE-NATIVE Run2 GT
# for CMSSW_15_0 and carries the IDENTICAL UL16 pixel payloads to the
# fit's 106X_mcRun2_asymptotic_v17:
#   SiPixelLorentzAngle(Sim)  _2016_ultralegacymc_v2
#   SiPixelTemplateDBObject   _38T_2016_ultralegacymc_v2
#   SiPixelGenErrorDBObject   _38T_2016_ultralegacymc_v2
# Forcing the 106X GT itself through a 15_0 simulation does NOT work --
# 15_0 digitisation wants records 106X never carried (L1TCaloParamsO2ORcd,
# EcalSimComponentShapeRcd, ...). Pass the SAME GT to the fit
# (globalTag=150X_mcRun2_asymptotic_v1) so the two sides are identical.
_GT = os.environ.get('SIMPROD_GT', '150X_mcRun2_asymptotic_v1')
process.GlobalTag = GlobalTag(process.GlobalTag, _GT, '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
# L1 emulation is KEPT: DigiToRaw packs L1 (gtStage2Raw needs
# simGmtStage2Digis), so dropping L1simulation_step breaks RAW. It was
# briefly dropped while trying to force a 106X GT through this 15_0
# simulation; with the release-native 150X GT the L1 records exist and
# the workaround is unnecessary.
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RECOSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from Analysis.HitAnalyzer.simprod_customise
from Analysis.HitAnalyzer.simprod_customise import customise_nocastor 

#call to customisation function customise_nocastor imported from Analysis.HitAnalyzer.simprod_customise
process = customise_nocastor(process)

# End of customisation functions


# Customisation from command line

process.RECOSIMoutput.outputCommands.extend(['keep PSimHits_g4SimHits_TrackerHits*_*','keep SimTracks_g4SimHits__*','keep SimVertexs_g4SimHits__*'])
#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

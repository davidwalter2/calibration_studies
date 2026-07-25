# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt10_pythia8_cfi --era Run2_2016 --conditions 106X_mcRun2_asymptotic_v17 --beamspot Realistic25ns13TeV2016Collision -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,RECO --datatier GEN-SIM-RECO --eventcontent FEVTDEBUG -n 10 --nThreads 4 --no_exec --mc --python_filename gun_base_cfg.py --fileout file:gun.root
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
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeV2016Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleMuPt10_pythia8_cfi nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:gun.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_mcRun2_asymptotic_v17', '')

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(10.01),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(9.99),
        ParticleID = cms.vint32(-13)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single mu pt 10')
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.raw2digi_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)


# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

# --- LA response study: flat true Lorentz angle, NOMINAL ---
process.mix.digitizers.pixel.LorentzAngle_DB = cms.bool(False)
process.mix.digitizers.pixel.TanLorentzAnglePerTesla_BPix = cms.double(0.106)
process.mix.digitizers.pixel.TanLorentzAnglePerTesla_FPix = cms.double(0.106)
process.FEVTDEBUGoutput.fileName = cms.untracked.string('file:gun_nominal.root')

# --- production settings ---
process.maxEvents.input = cms.untracked.int32(20000)
process.options.numberOfThreads = cms.untracked.uint32(16)
process.options.numberOfStreams = cms.untracked.uint32(16)
process.FEVTDEBUGoutput.fileName = cms.untracked.string('file:/ceph/submit/data/user/d/david_w/ZMass/cvh/lagun/gun_nominal.root')

# --- 100k edge-statistics run ---
process.maxEvents.input = cms.untracked.int32(100000)
process.options.numberOfThreads = cms.untracked.uint32(32)
process.options.numberOfStreams = cms.untracked.uint32(32)
process.FEVTDEBUGoutput.fileName = cms.untracked.string('file:/ceph/submit/data/user/d/david_w/ZMass/cvh/lagun/gun100k_nominal.root')

# 32-stream config OOM'd under node memory pressure; proven 16-stream setup.
process.options.numberOfThreads = cms.untracked.uint32(16)
process.options.numberOfStreams = cms.untracked.uint32(16)

# --- slurm chunking: GUN_CHUNK selects the seed block + output suffix.
# Chunk 0 = the original default seeds (= the existing 20k twin samples),
# so batch production runs chunks >= 1 and the analysis combines them.
import os as _os
_chunk = int(_os.environ.get("GUN_CHUNK", "-1"))
if _chunk >= 0:
    process.maxEvents.input = cms.untracked.int32(20000)
    for _n in process.RandomNumberGeneratorService.parameterNames_():
        _p = getattr(process.RandomNumberGeneratorService, _n)
        if hasattr(_p, "initialSeed"):
            _p.initialSeed = _p.initialSeed.value() + 12345 * _chunk
    _f = process.FEVTDEBUGoutput.fileName.value()
    process.FEVTDEBUGoutput.fileName = _f.replace(".root", "_c%d.root" % _chunk)

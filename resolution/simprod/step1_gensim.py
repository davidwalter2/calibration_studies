# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Analysis/HitAnalyzer/python/simprod_JpsiGun_cfi.py --python_filename step1_gensim.py --fileout file:step1.root --mc --eventcontent RAWSIM --datatier GEN-SIM --conditions auto:run2_design --beamspot Realistic25ns13TeV2016Collision --step GEN,SIM --era Run2_2016 --geometry DB:Ideal --no_exec -n 10
import os
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016

process = cms.Process('SIM',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeV2016Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("EmptySource")

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
    annotation = cms.untracked.string('Analysis/HitAnalyzer/python/simprod_JpsiGun_cfi.py nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(1),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
    fileName = cms.untracked.string('file:step1.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
if hasattr(process, "XMLFromDBSource"): process.XMLFromDBSource.label="Ideal"
if hasattr(process, "DDDetectorESProducerFromDB"): process.DDDetectorESProducerFromDB.label="Ideal"
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
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

# --- Geant4 field-integration precision in the tracker -----------------------
# Josh: "really really really important" for the CVH momentum scale. The
# official UL16 SIM sets the GLOBAL DeltaOneStep=1e-5 / DeltaIntersection=1e-6
# (CMSSW_10_6 has no region-specific variants, so the globals are what the
# tracker uses there -- which is why the B->J/psi+X MC, produced in 10_6_20,
# genuinely runs at the 100x-looser 1e-4).
#
# CMSSW_15_0 is different: CMSFieldManager::setChordFinderForTracker applies
# DeltaOneStepTracker / DeltaIntersectionTracker whenever the track has
# E > EnergyThTracker (0.2 GeV) and is inside RmaxTracker (8 m) -- always true
# for our muons. Of those, DeltaIntersectionTracker is ALREADY 1e-6 by default,
# so the surface-intersection precision was never the loose one here; only
# DeltaOneStepTracker (1e-4) sits 10x above the official target.
#
# Set the tracker pair to the official targets, and the globals too so that
# tracks below 0.2 GeV or outside the tracker region are covered as well.
_sp = process.g4SimHits.MagneticField.ConfGlobalMFM.OCMS.StepperParam
_sp.DeltaOneStepTracker = 1e-5
_sp.DeltaIntersectionTracker = 1e-6
_sp.DeltaOneStep = 1e-5
_sp.DeltaIntersection = 1e-6

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(False),
        MaxEta = cms.double(2.4),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(30.0),
        MinEta = cms.double(-2.4),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(5.0),
        ParticleID = cms.vint32(443)
    ),
    PythiaParameters = cms.PSet(
        jpsiDecay = cms.vstring(
            '443:onMode = off',
            '443:onIfMatch = 13 -13'
        ),
        parameterSets = cms.vstring('jpsiDecay'),
        pythia8CommonSettingsBlock = cms.PSet(
            parameterSets = cms.vstring()
        )
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('Jpsi mumu flat pT gun')
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.eventSetup.numberOfConcurrentIOVs = 1
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)



# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

# Shared configuration for the dymc_8p5M_260906_v2 production (Z -> mumu leg,
# HTCondor).  Sourced by submit_dymc_v2.sh, resume_dymc_v2.sh and
# status_dymc_v2.sh.
#
# It lives in one file for the same reason config_dymc8p5M.sh does: a resumed
# task that ran a DIFFERENT configuration from the original submission would
# silently produce a mixed sample -- the outputs look identical and nothing
# downstream would catch it.
#
# WHAT IS DIFFERENT FROM dymc_8p5M_260905 (the slurm production):
#   1. the CMSSW area is dev2 @ cvh-exports-260906, which carries fab515e
#      (the `ndof == 0` abort that cost 28 % of the first finished tasks);
#   2. the four re-production export switches of PRODUCTIONS.md §3;
#   3. HTCondor instead of slurm -- and, because the submit condor pool has no
#      ceph-mounted execute capacity at all (one 1-CPU local test slot; every
#      other slot is a glidein at DESY/IIHE/... with no /ceph, no /work), that
#      means a GRID job: cvmfs base release + a payload overlay, the input
#      streamed through the CMS global redirector, the output xrdcp'd back;
#   4. numberOfThreads -- see NTHREADS below.
# The chunking is IDENTICAL (the same 380-line chunk list), so task_NNNN means
# the same event range in both productions and the two can be compared task by
# task.

TAG=dymc_8p5M_260906_v2
OUTBASE=${OUTBASE:-/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
CHUNKLIST=${CHUNKLIST:-/work/submit/david_w/ZMass/calibration_studies/production/chunks_dymc_8p5M_260905.txt}
NAME=dymc_v2

# --- sizing, from the thread scan (STATE_dy_v2.md, "Thread scan") ----------
# 4 threads, on the grid as well as on slurm.
#   * on a 22 120-event chunk the scan projects 3.94x (the 3.43x it measures on
#     2000 events is diluted by a 40 s serial head that a real chunk amortises);
#   * it turns a 2.25 h task into a 34 min one, which matters far more on a
#     preemptible glidein than on a slurm node -- an evicted job loses whatever
#     it had done;
#   * memory per core drops from 3.6 GB to 1.25 GB.
# The BtoJpsiX production runs 1 thread, which is NOT a matchability result:
# its own note says the shipped cfg ran SINGLE-THREADED so 3 of the 4 cores sat
# idle, and that what actually blocked matching was the missing +DESIRED_Sites.
# Measured here: a 4-core request matched at Caltech about a minute after
# submission.
NTHREADS=${NTHREADS:-4}
# 3.56 GB (the worst 1-thread task of the 260905 productions) + 3 x 51 MB per
# extra stream, x1.3. The scan's own 2.00 GB peak is NOT the sizing number: a
# 2000-event arm does not reach the tail that a 22 120-event chunk does.
REQMEM=${REQMEM:-5000}
REQDISK=${REQDISK:-4000000}     # KB: 64 MB payload + 228 MB unpacked area + output

# --- grid routing ----------------------------------------------------------
OUTHOST=${OUTHOST:-root://submit50.mit.edu/}
# The xrootd-side twin of OUTBASE: submit50's namespace root IS /ceph/submit.
OUTROOT=${OUTROOT:-${OUTBASE#/ceph/submit}}
REDIR=${REDIR:-root://cms-xrd-global.cern.ch/}
RELEASE=${RELEASE:-CMSSW_15_0_19_patch2}
SCRAMARCH=${SCRAMARCH:-el9_amd64_gcc12}
CFGREL=${CFGREL:-src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
# Sites that run an el9 container. mit_tier3 is deliberately absent: those
# pilots are native el7 with no singularity and cannot run this release.
DESIRED_SITES=${DESIRED_SITES:-"T2_BE_IIHE,T2_BE_UCL,T2_BR_UERJ,T2_CH_CERN,T2_CN_Beijing,T2_DE_DESY,T2_DE_RWTH,T2_ES_CIEMAT,T2_FR_CCIN2P3,T2_FR_GRIF_IRFU,T2_FR_GRIF_LLR,T2_FR_IPHC,T2_HU_Budapest,T2_IN_TIFR,T2_IT_Legnaro,T2_IT_Pisa,T2_IT_Rome,T2_KR_KISTI,T2_PL_Swierk,T2_PT_NCG_Lisbon,T2_RU_JINR,T2_TW_NCHC,T2_UK_SGrid_RALPP,T2_US_Caltech,T2_US_Florida,T2_US_MIT,T2_US_Nebraska,T2_US_Purdue,T2_US_UCSD,T2_US_Wisconsin,T3_US_HEPCloud,T3_US_NERSC,T3_US_OSG,T3_CH_CERN_CAF"}
# Node fences carried over from BtoJpsiX_MCprod/.../submits/base_v3.sub. NOTE
# the regexp form: `=!=` is the ClassAd IDENTITY operator and is CASE SENSITIVE,
# while MIT T2 advertises Machine in UPPERCASE, so a lowercase `=!=` fence
# silently never excludes anything.
# physik.rwth-aachen.de and jinr.ru: fenced from THIS production's own first
# half hour. 16 of 380 first attempts died with SIGILL (rc 132) --
# "illegal instruction" inside `XrdCl::PostMaster::Start()` in the CVMFS
# release's own libXrdCl, on the `stagein` Prepare call while PoolSource is
# being constructed -- and ALL SIXTEEN were at those two sites, 8 each. It is
# not our payload (the release loads its scram_x86-64-v2 variants) and it is
# not the chunk; it is those sites' CPUs against that external. Retries carried
# them, but a job can only retry onto a bad site so many times.
# Black-hole nodes fenced from the J/psi v2 submission's own first half
# hour: 34 of 36 SIGILLs were at five ultralight.org machines
# (compute-6-34 alone ate 16) plus t2bat0310, this time crashing in
# edm::StreamSchedule::fillWorkers rather than in XrdCl -- i.e. the node
# cannot run the release's binaries at all. The rest of Caltech and MIT T2
# ran hundreds of jobs fine, so fence the NODES, not the sites.
REQUIREMENTS=${REQUIREMENTS:-'(regexp("swan.hcc.unl.edu", Machine, "i") =!= true) && (regexp("cmsplt02", Machine, "i") =!= true) && (regexp("node-0011.hepgrid.uerj.br", Machine, "i") =!= true) && (regexp("cism.ucl.ac.be", Machine, "i") =!= true) && (regexp("physik.rwth-aachen.de", Machine, "i") =!= true) && (regexp("jinr.ru", Machine, "i") =!= true) && (regexp("compute-6-34\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-6-6\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-22-12\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-12n-23\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-21-22\.ultralight\.org", Machine, "i") =!= true) && (regexp("t2bat0310\.cmsaf\.mit\.edu", Machine, "i") =!= true)'}

# The 260905 configuration verbatim, plus the four PRODUCTIONS.md §3
# switches.  numberOfThreads is NOT here: the wrapper takes it from $NTHREADS
# so that a resume cannot silently change the stream count (and with it the
# number of output files a task is checked for).
EXTRA="doRes=True exportCfExponents=True exportStepRecords=False \
 exportCfGroupExponents=True \
 exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False doSimHits=False \
 doGen=True requireGen=False \
 genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
 doTrigger=False applyHltFilter=False \
 massMin=60 massMax=120 \
 useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
 doMassConstraint=False \
 CgfQoPMode=0 tightG4eStepper=True \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT"

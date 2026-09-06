# Shared configuration for the jpsimc_20M_260906_v2 production (J/psi leg,
# HTCondor).  Sourced by submit_jpsimc_v2.sh, resume_jpsimc_v2.sh and
# status_jpsimc_v2.sh.
#
# It lives in one file for the same reason config_jpsimc20M.sh does: a resumed
# task that ran a DIFFERENT configuration from the original submission would
# silently produce a mixed sample -- the outputs look identical and nothing
# downstream would catch it.
#
# WHAT IS DIFFERENT FROM jpsimc_20M_260905 (the slurm production):
#   1. the CMSSW area is dev2 @ cvh-exports-260906;
#   2. the four re-production export switches of PRODUCTION_NEXT.md §2;
#   3. HTCondor -> CMS global pool, i.e. a GRID job (no condor slot reachable
#      from submit mounts /ceph; see condor_dymc_v2/STATE_dy_v2.md §2);
#   4. numberOfThreads=4.
# The chunking is IDENTICAL (the same 1642-line chunk list, including the 12
# lines repointed at the re-staged copies), so task_NNNN means the same event
# range in both productions and the two can be compared task by task.

TAG=jpsimc_20M_260906_v2
OUTBASE=${OUTBASE:-/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py}
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
# The 260905 chunk list with a FOURTH field: the file's size in bytes, taken on
# the ceph copy at submit time. The wrapper compares what a door serves against
# it and refuses a mismatch -- the guard against ever reading the central
# un-repacked file that shares the same LFN.
CHUNKLIST=${CHUNKLIST:-/work/submit/david_w/ZMass/calibration_studies/production/condor_jpsimc_v2/chunks_jpsimc_20M_260906_v2.txt}
NAME=jpsimc_v2

# --- sizing, from the 2026-09-06 thread scan (condor_dymc_v2/STATE_dy_v2.md) -
# The J/psi arm measured 3.81x at 4 threads on 2000 events and fits
# `wall(N) = 36.0 + 2003.9/N`; on a real 12 185-event chunk that projects to
# 3.97x (3.40 h -> 0.86 h). Total CPU is FLAT (2028 s at 1 thread, 2024 s at 8).
NTHREADS=${NTHREADS:-4}
# 3.56 GB (the worst 1-thread task of the 1616 in jpsimc_20M_260905) + 3 x
# 76 MB per extra stream, x1.3 = 4.9 GB. LESS memory per task than today's 6 G
# and four times the cores.
REQMEM=${REQMEM:-5000}
# payload 64 MB + unpacked area 228 MB + ~856 MB of output (4 x 13.09 MB
# runtree + 12 148 x 66.2 kB).
REQDISK=${REQDISK:-6000000}

# --- grid routing ----------------------------------------------------------
OUTHOST=${OUTHOST:-root://submit50.mit.edu/}
OUTROOT=${OUTROOT:-${OUTBASE#/ceph/submit}}
RELEASE=${RELEASE:-CMSSW_15_0_19_patch2}
SCRAMARCH=${SCRAMARCH:-el9_amd64_gcc12}
CFGREL=${CFGREL:-src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py}

# ===========================================================================
# THE INPUT DOOR -- READ THIS BEFORE CHANGING ANYTHING ABOUT THE INPUT PATH
# ===========================================================================
# The J/psi inputs are OUR OWN repacked split-1 copies, written into the group
# store under the CMS /store namespace. That path ALSO EXISTS CENTRALLY, with
# DIFFERENT CONTENT -- the un-repacked split-99 original. Measured 2026-09-06
# on the first file of the list:
#
#     root://submit50.mit.edu/         -> 1 783 846 343 B   (ours, split-1)
#     root://cms-xrd-global.cern.ch/   -> 1 840 069 145 B   (the ORIGINAL)
#
# So the DY leg's trick of mapping the ceph path to an LFN and letting the
# global redirector find a replica IS NOT SAFE HERE: it would silently feed the
# production files whose split level makes the CVH refit fail on ~99 % of
# candidates (project_jpsi_mc_repack_split1), and every output would still look
# valid. The MIT T2 door does not serve this tree either ("Too many attempts to
# gain dfs read access"), because it is the T3/submit CephFS, not the T2 store.
#
# The ONLY doors that serve these bytes are submit50-55, whose xrootd namespace
# root IS /ceph/submit. The mapping is therefore a pure prefix strip,
# `/ceph/submit/X -> root://submit5N.mit.edu//X`, which is exact, has no
# redirector in the path, and works uniformly for BOTH the group-store files
# and the 12 chunk lines repointed at the re-staged copies in the user store.
#
# The door is chosen per task index so that 1642 jobs do not all read through
# one login node: ~274 jobs each. All six were verified to serve the file.
INDOORS=${INDOORS:-"submit50.mit.edu submit51.mit.edu submit52.mit.edu submit53.mit.edu submit54.mit.edu submit55.mit.edu"}

# Sites that run an el9 container, minus the two fenced below.
DESIRED_SITES=${DESIRED_SITES:-"T2_BE_IIHE,T2_BE_UCL,T2_BR_UERJ,T2_CH_CERN,T2_CN_Beijing,T2_DE_DESY,T2_DE_RWTH,T2_ES_CIEMAT,T2_FR_CCIN2P3,T2_FR_GRIF_IRFU,T2_FR_GRIF_LLR,T2_FR_IPHC,T2_HU_Budapest,T2_IN_TIFR,T2_IT_Legnaro,T2_IT_Pisa,T2_IT_Rome,T2_KR_KISTI,T2_PL_Swierk,T2_PT_NCG_Lisbon,T2_RU_JINR,T2_TW_NCHC,T2_UK_SGrid_RALPP,T2_US_Caltech,T2_US_Florida,T2_US_MIT,T2_US_Nebraska,T2_US_Purdue,T2_US_UCSD,T2_US_Wisconsin,T3_US_HEPCloud,T3_US_NERSC,T3_US_OSG,T3_CH_CERN_CAF"}
# physik.rwth-aachen.de and jinr.ru: SIGILL (rc 132) inside the CVMFS release's
# own libXrdCl on the stagein Prepare -- 16 of 16 such failures in the DY v2
# submission were at those two sites, 8 each. See condor_dymc_v2/STATE_dy_v2.md.
# `regexp(..., "i")`, NOT `=!=` on the name: `=!=` is the ClassAd IDENTITY
# operator and is case sensitive, and MIT T2 advertises Machine in uppercase.
# Black-hole nodes fenced 2026-09-06 from the J/psi v2 submission's own
# first half hour: 34 of 36 SIGILLs were at five ultralight.org machines
# (compute-6-34 alone ate 16) plus t2bat0310, this time crashing in
# edm::StreamSchedule::fillWorkers rather than in XrdCl -- i.e. the node
# cannot run the release's binaries at all. The rest of Caltech and MIT T2
# ran hundreds of jobs fine, so fence the NODES, not the sites.
REQUIREMENTS=${REQUIREMENTS:-'(regexp("swan.hcc.unl.edu", Machine, "i") =!= true) && (regexp("cmsplt02", Machine, "i") =!= true) && (regexp("node-0011.hepgrid.uerj.br", Machine, "i") =!= true) && (regexp("cism.ucl.ac.be", Machine, "i") =!= true) && (regexp("physik.rwth-aachen.de", Machine, "i") =!= true) && (regexp("jinr.ru", Machine, "i") =!= true) && (regexp("compute-6-34\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-6-6\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-22-12\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-12n-23\.ultralight\.org", Machine, "i") =!= true) && (regexp("compute-21-22\.ultralight\.org", Machine, "i") =!= true) && (regexp("t2bat0310\.cmsaf\.mit\.edu", Machine, "i") =!= true)'}

# config_jpsimc20M.sh verbatim, plus the four PRODUCTION_NEXT.md §2 switches.
# numberOfThreads is NOT here: the wrapper takes it from $NTHREADS so that a
# resume cannot silently change the stream count (and with it the number of
# output files a task is checked for).
EXTRA="doRes=True exportCfExponents=True exportStepRecords=False \
 exportCfGroupExponents=True \
 exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False \
 trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True \
 doTrigger=True applyHltFilter=False doSimHits=False \
 useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
 doVtxConstraint=False doMassConstraint=False \
 CgfQoPMode=0 \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT"

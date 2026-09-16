# Shared configuration for the dymc_8p5M_260905 production (Z -> mumu leg).
# Sourced by BOTH submit_dymc8p5M.sh and resume_dy.sh.
#
# It lives in one file because a resumed task that ran a DIFFERENT
# configuration from the original submission would silently produce a mixed
# sample -- the outputs look identical and nothing downstream would catch it.
#
# See PRODUCTIONS.md §5 for the option-by-option rationale of both legs; this
# one mirrors the J/psi leg.

TAG=dymc_8p5M_260905
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
MAXARRAY=1000          # MaxArraySize here is 1001 -> indices 0..1000
MAXRUNNING=200         # concurrent tasks PER ARRAY
NAME=dymc8p5M

# useDefaultField=True: this is a closure/feasibility test, so the fit's field
# must be the field the SIM propagated through (the OAE-parametrised tracker
# field on the 160812 map). Same reasoning as the J/psi leg.
#
# CgfQoPMode=0 is passed explicitly even though the cfi default came back to 0
# at 1cd4453: mode >= 1 costs 8.5 s/candidate against 0.63 s for a CGF block
# the two-track maker has no hook to substitute (profiling/NOTES.md).
#
# doRes + fillGradsFactored TOGETHER are what turn the in-maker CF export on
# (the gate is doRes && (fillGrads || fillGradsFactored)); each alone is
# nearly free and the pair is the 0.63 s/candidate.
#
# requireGen=False, unlike the J/psi leg: an opposite-sign muon pair in a
# 60-120 GeV window on DY is the Z, so the gen cut would only couple the yield
# to the MiniAOD gen-pruning thresholds. The Mu{plus,minus}gen_* branches are
# written either way.
EXTRA="numberOfThreads=1 \
 doRes=True exportCfExponents=True exportStepRecords=False \
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

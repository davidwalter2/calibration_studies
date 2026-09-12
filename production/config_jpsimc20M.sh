# Shared configuration for the jpsimc_20M_260905 production.
# Sourced by BOTH submit_jpsimc20M.sh and resume.sh.
#
# It lives in one file because a resumed task that ran a DIFFERENT
# configuration from the original submission would silently produce a mixed
# sample -- the outputs look identical and nothing downstream would catch it.
#
# See PRODUCTIONS.md §5 for the option-by-option rationale.

TAG=jpsimc_20M_260905
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev}
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
MAXARRAY=1000          # MaxArraySize here is 1001 -> indices 0..1000
MAXRUNNING=200         # concurrent tasks PER ARRAY
NAME=jpsimc20M

# useDefaultField=True: this is a closure test, so the fit's field must be the
# field the SIM propagated through (the OAE-parametrised tracker field on the
# 160812 map). The 3D TOSCA grid leaves an eta/phi-coherent ~8e-4 dp/p pattern
# in the pull width, which on a closure test is indistinguishable from the
# thing being measured. The Opera3D variant is reserved as a later
# injected-field test on a subset.
EXTRA="numberOfThreads=1 \
 doRes=True exportCfExponents=True exportStepRecords=False \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False \
 trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True \
 doTrigger=True applyHltFilter=False doSimHits=False \
 useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
 doVtxConstraint=False doMassConstraint=False \
 CgfQoPMode=0 \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT"

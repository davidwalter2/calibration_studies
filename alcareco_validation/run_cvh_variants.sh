#!/bin/bash
# Run the CVH step-2 ntuplizer for KS, Lambda, J/psi under three constraint
# configurations so we can compare the fitted ditrack invariant mass
# (Jpsikin_mass) against the pre-refit track-pair mass (Jpsitrk_mass).
#
#   (b) nv_np : doVtxConstraint=False  doPointingConstraint=False  (current default for V0s)
#   (c)  v_np : doVtxConstraint=True   doPointingConstraint=False
#   (d)  v_p  : doVtxConstraint=True   doPointingConstraint=True   (sigma=1 mrad)
#
# Output layout:
#   /ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants/<channel>_<variant>/globalcor_<channel>_0.root
#
# Run inside the CMSSW SLC7 container, e.g.:
#   APPTAINER_BIND="/tmp,/home/submit,/work/submit,/scratch/submit,/ceph/submit,/cvmfs,/etc/grid-security,/run" \
#     cmssw-el7 --command-to-run \
#     /work/submit/david_w/ZMass/calibration_studies/alcareco_validation/run_cvh_variants.sh

set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src
eval $(scramv1 runtime -sh)

OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants
mkdir -p $OUTBASE

inject_variant() {
    # $1 = source runner full path, $2 = process attribute name, $3 = output cfg path,
    # $4 = doVtxConstraint (True/False), $5 = doPointingConstraint (True/False)
    cp $1 $3
    python -c "
s = open('$3').read()
inject = '''process.$2.doVtxConstraint        = cms.bool($4)
process.$2.doPointingConstraint   = cms.bool($5)
process.$2.pointingSigma          = cms.double(1.e-3)

'''
s = s.replace('process.p = cms.Path(', inject + 'process.p = cms.Path(')
open('$3','w').write(s)
"
}

prepare_one() {
    # Prepare the cfg + outdir for one variant, but don't launch.
    # Echoes a single-line "label cfg outdir outfile" record on stdout for
    # the caller to use when scheduling the parallel cmsRun calls.
    label=$1; src=$2; attr=$3; outfile=$4; variant=$5; vtx=$6; pt=$7
    outdir=$OUTBASE/${label}_${variant}
    mkdir -p $outdir
    rm -f $outdir/*.root
    cfg=/tmp/run_${label}_${variant}.py
    inject_variant $src $attr $cfg $vtx $pt
    echo "${label}_${variant} $cfg $outdir $outfile"
}

run_one_bg() {
    # Launch one cmsRun in the background with stdout/stderr -> outdir/run.log.
    tag=$1; cfg=$2; outdir=$3; outfile=$4
    cd $outdir
    cmsRun $cfg > run.log 2>&1 &
    pid=$!
    echo "  launched $tag pid=$pid"
}

KS_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runGlobalCorRecKsFromAlca.py
LM_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runGlobalCorRecLambdaFromAlca.py
JP_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runCvhJpsiCandidateDriven.py

# Prepare all 9 cfgs (sequential -- cheap; just template substitution).
declare -a JOBS
for spec in \
    "ks     $KS_RUNNER globalCorKs     globalcor_ks" \
    "lambda $LM_RUNNER globalCorLambda globalcor_lambda" \
    "jpsi   $JP_RUNNER globalCorJpsi   globalcor_jpsi"
do
    set -- $spec
    label=$1; runner=$2; attr=$3; outfile=$4
    JOBS+=("$(prepare_one $label $runner $attr $outfile nv_np False False)")
    JOBS+=("$(prepare_one $label $runner $attr $outfile v_np  True  False)")
    JOBS+=("$(prepare_one $label $runner $attr $outfile v_p   True  True)")
done

# Launch all 9 cmsRun jobs in parallel.
echo ">>> Launching ${#JOBS[@]} parallel cmsRun jobs"
PIDS=()
for j in "${JOBS[@]}"; do
    set -- $j  # tag cfg outdir outfile
    cd $3
    cmsRun $2 > run.log 2>&1 &
    PIDS+=($!)
    echo "  launched $1 pid=$!"
done

# Wait for all to finish.
echo ">>> Waiting for ${#PIDS[@]} jobs"
fail=0
for p in "${PIDS[@]}"; do
    if ! wait $p; then
        fail=$((fail + 1))
    fi
done

echo
echo "Done.  failures=${fail}/${#PIDS[@]}"
echo "Outputs:"
find $OUTBASE -maxdepth 2 -name '*.root' -printf '%p  %s\n'

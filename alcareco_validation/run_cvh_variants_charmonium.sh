#!/bin/bash
# Companion to run_cvh_variants.sh: same 9-variant scan (KS, Lambda, J/psi
# x nv_np / v_np / v_p) but with all three channels sourced from the same
# Charmonium-based ALCAReco at /ceph/.../charmonium_full/. The original
# script reads Ks/Lambda from a SingleMuon-derived multifile compilation;
# this one keeps everything on one HLT slice for cross-channel comparison.
#
# Output layout:
#   /ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants_charmonium/
#     <channel>_<variant>/globalcor_<channel>_0.root
#
# Run inside the CMSSW SLC7 container:
#   APPTAINER_BIND="/tmp,/home/submit,/work/submit,/scratch/submit,/ceph/submit,/cvmfs,/etc/grid-security,/run" \
#     cmssw-el7 --command-to-run \
#     /work/submit/david_w/ZMass/calibration_studies/alcareco_validation/run_cvh_variants_charmonium.sh

set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src
eval $(scramv1 runtime -sh)

OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants_charmonium
mkdir -p $OUTBASE

inject_variant() {
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
    label=$1; src=$2; attr=$3; outfile=$4; variant=$5; vtx=$6; pt=$7
    outdir=$OUTBASE/${label}_${variant}
    mkdir -p $outdir
    rm -f $outdir/*.root
    cfg=/tmp/run_${label}_${variant}_charmonium.py
    inject_variant $src $attr $cfg $vtx $pt
    echo "${label}_${variant} $cfg $outdir $outfile"
}

KS_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runGlobalCorRecKsFromCharmoniumAlca.py
LM_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runGlobalCorRecLambdaFromCharmoniumAlca.py
JP_RUNNER=/work/submit/david_w/ZMass/CMSSW_10_6_26/src/Analysis/HitAnalyzer/test/runCvhJpsiCandidateDriven.py

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

echo ">>> Launching ${#JOBS[@]} parallel cmsRun jobs"
PIDS=()
for j in "${JOBS[@]}"; do
    set -- $j
    cd $3
    cmsRun $2 > run.log 2>&1 &
    PIDS+=($!)
    echo "  launched $1 pid=$!"
done

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

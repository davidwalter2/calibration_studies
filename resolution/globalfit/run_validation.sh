#!/bin/bash
# Validation ladder for the unified calibration card (quadratic hit-chi2 term
# + unbinned mass term sharing the global calibration parameters by name).
#
# The production is simulation, so the physics truth is "no correction": every
# floated global parameter should come out compatible with zero, and the
# injection tests must return exactly what was put in.
#
#   0  extract      one pass over the production -> npz with G, K, the
#                   per-candidate mass inputs and the D rows
#   1  reference    offline solve of the quadratic problem (= fit_global_grads)
#   2  quad         rabbit, quadratic term only        -> must equal (1)
#   3  mass         rabbit, mass term only, with alpha -> must equal step 2 of
#                   the mass-likelihood chain (cf_masslik_fit / the rabbit
#                   unbinned-mass card)
#   4  joint        rabbit, both terms
#   5  inject       both terms shifted by a known dtheta -> pull table
#   5b injectmass   only m_i^0 shifted   (mass term alone must see it)
#   5c injectquad   only G shifted       (quadratic term alone must see it)
#
# Usage:  ./run_validation.sh [STEP ...]      (default: all)
set -u
cd "$(dirname "$0")"

PROD=${PROD:-/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_btojpsix_v3_260904f_m0}
FILES="${PROD}/task_*/globalcor_*.root"
NTASKS=${NTASKS:-48}
JOBS=${JOBS:-16}
TAG=${TAG:-btojpsix_v3_260904f}
PARMTYPES=${PARMTYPES:-"14 15"}
GROUPS=${GROUPS:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
# one unit of bfield_mode0 = dB/B 8.2e-4, so dB/B = 1e-4 is 0.12195
INJ=${INJ:-bfield_mode0:0.12195}

RUNS=runs/globalfit
CARDS=$RUNS/cards
OUT=$RUNS/fits
LOGS=$RUNS/logs
mkdir -p $CARDS $OUT $LOGS

VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate
RUNTF=${RUNTF:-/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh}
EXTRACT=$RUNS/extract_${TAG}.npz
REF=$RUNS/reference_${TAG}.npz

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(0 1 2 3 4 5 5b 5c)
has () { for s in "${steps[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

py () { ( source $VENV && python3 -u "$@" ); }
card () {  # card <name> <extra args...>
    nm=$1; shift
    [ -s "$CARDS/${nm}.hdf5" ] && { echo "=== skip card $nm (exists)"; return; }
    echo "=== card $nm"
    $RUNTF python3 -u make_global_term.py -i $EXTRACT --parmtypes $PARMTYPES \
        --groups $GROUPS -o $CARDS/${nm}.hdf5 "$@" > $LOGS/card_${nm}.log 2>&1
    echo "    rc=$? -> $LOGS/card_${nm}.log"
}
fit () {  # fit <name> <paramModel args...>
    nm=$1; shift
    [ -s "$OUT/${nm}/fitresults.hdf5" ] && { echo "=== skip fit $nm (exists)"; return; }
    echo "=== fit $nm"
    $RUNTF rabbit_fit.py $CARDS/${nm}.hdf5 -o $OUT/${nm} -t 0 --unblind \
        "$@" > $LOGS/fit_${nm}.log 2>&1
    echo "    rc=$? -> $LOGS/fit_${nm}.log"
}

# --- 0: extraction ---------------------------------------------------------
if has 0 && [ ! -s "$EXTRACT" ]; then
    echo "=== extract $TAG"
    py extract.py --files "$FILES" --ntasks $NTASKS --parmtypes $PARMTYPES \
        -j $JOBS -o $EXTRACT 2>&1 | tee $LOGS/extract.log
fi

# --- 1: offline reference --------------------------------------------------
if has 1; then
    echo "=== reference solve"
    py solve_reference.py -i $EXTRACT --parmtypes $PARMTYPES --groups $GROUPS \
        -o $REF 2>&1 | tee $LOGS/reference.log
fi

# --- 2: quadratic term only ------------------------------------------------
if has 2; then
    card quad --no-mass
    fit  quad --paramModel ExternalParams bundle:global_params
    $RUNTF python3 -u compare_fit.py --fit $OUT/quad/fitresults.hdf5 \
        --ref $REF --label "rabbit quadratic" 2>&1 | tee $LOGS/cmp_quad.log
fi

# --- 3: mass term only -----------------------------------------------------
if has 3; then
    card mass --no-quadratic --no-jac --with-alpha
    fit  mass --paramModel UnbinnedParams
    $RUNTF python3 -u compare_fit.py --fit $OUT/mass/fitresults.hdf5 \
        --card $CARDS/mass.hdf5 --breakdown 2>&1 | tee $LOGS/cmp_mass.log
fi

# --- 4: joint --------------------------------------------------------------
if has 4; then
    card joint
    fit  joint --paramModel UnbinnedParams
    $RUNTF python3 -u compare_fit.py --fit $OUT/joint/fitresults.hdf5 \
        --fit2 $OUT/quad/fitresults.hdf5 --label joint --label2 quadratic \
        --card $CARDS/joint.hdf5 --breakdown --corr bfield_mode0 \
        2>&1 | tee $LOGS/cmp_joint.log
    # the mass-implied scale with and without the quadratic term
    card massjac --no-quadratic
    fit  massjac --paramModel UnbinnedParams
    $RUNTF python3 -u compare_fit.py --fit $OUT/massjac/fitresults.hdf5 \
        --fit2 $OUT/joint/fitresults.hdf5 --label "mass only" --label2 joint \
        2>&1 | tee $LOGS/cmp_massjac.log
fi

# --- 5: injections ---------------------------------------------------------
if has 5; then
    card inject --inject $INJ
    fit  inject --paramModel UnbinnedParams
    $RUNTF python3 -u compare_fit.py --fit $OUT/inject/fitresults.hdf5 \
        --fit2 $OUT/joint/fitresults.hdf5 --label injected --label2 baseline \
        --injected $INJ 2>&1 | tee $LOGS/cmp_inject.log
fi
if has 5b; then
    card injectmass --no-quadratic --inject $INJ --inject-mass-only
    fit  injectmass --paramModel UnbinnedParams
    $RUNTF python3 -u compare_fit.py --fit $OUT/injectmass/fitresults.hdf5 \
        --fit2 $OUT/massjac/fitresults.hdf5 --label injected --label2 baseline \
        --injected $INJ 2>&1 | tee $LOGS/cmp_injectmass.log
fi
if has 5c; then
    card injectquad --no-mass --inject $INJ --inject-quad-only
    fit  injectquad --paramModel ExternalParams bundle:global_params
    py solve_reference.py -i $EXTRACT --parmtypes $PARMTYPES --groups $GROUPS \
        --inject $INJ -o $RUNS/reference_inject.npz 2>&1 | tee $LOGS/reference_inject.log
    $RUNTF python3 -u compare_fit.py --fit $OUT/injectquad/fitresults.hdf5 \
        --ref $RUNS/reference_inject.npz --label "rabbit quadratic injected" \
        2>&1 | tee $LOGS/cmp_injectquad.log
fi

echo "ALL DONE"

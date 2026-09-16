#!/bin/bash
#SBATCH --job-name=zprecond
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=11:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# Re-fit the cards that could not be certified, WITH the trust-region
# preconditioner (rabbit PRs #156/#157, on `vmass-conditioning` since the
# 2026-09-13 merge of origin/main).  Structure and env follow
# `engaging/rabbit_vmass_batch.sbatch`: several rows in ONE allocation, each a
# separate `rabbit_fit.py` process, so a row that fails does not take the
# others with it.  Partition, account and GPU come from the `sbatch` line, not
# from the header.
#
# WHY.  Both rows fail for the same reason and the diagnosis is in
# `RESOLUTION.md` sec. 6.4 / sec. 10 items 1-3:
#   * `SVs9`  the v-form K(m)=9 rung runs ~59 Hessians with no density NaN and
#             then dies on an INDEFINITE Hessian whose diagonal spans 0.447 to
#             6.24e13 -- fourteen orders of magnitude.  The v-form ladder ends
#             at 7 because of this, and K(m) truncation is the one open item on
#             the full-scale closure.
#   * `P2X`   the phase-2 joint card descends (EDM 1.4e5 -> ~150 over 19
#             Hessians in 6 h 15) and then sits on a shelf at condition number
#             ~1e15.  STATE's named remedy is preconditioning or 2-GPU
#             candidate sharding, NOT a resubmit.
# `--preconditionTransform spectral` is the one to use, not the `ridge`
# default: ridge needs a single scalar larger than |lambda_min|, which swamps
# every direction softer than itself -- precisely the 14-orders-of-magnitude
# case -- while spectral whitens by |H| = Q|Lambda|Q^T, giving each
# eigendirection its own scale AND KEEPING THE SIGN of the negative curvature
# that trust-krylov/trust-exact use to escape a saddle.  With spectral a larger
# block is always better or equal, so `--preconditionBlocks none` (one
# factorisation over the whole scope) is right and
# `--preconditionBlockThreshold` would work against it.
#
# THE CONTROL ROW IS NOT OPTIONAL.  `SVs7P` is the already-certified K=7 rung
# re-fitted with the same preconditioner options; the acceptance for the
# preconditioner itself (RESOLUTION.md sec. 10 item 2) is that an already
# converged cell reproduces to 0.01 MeV.  Run it in the same allocation.
#
# STAGING.  `engaging/stage_native.sh code` rsyncs `engaging/` wholesale but
# only `*.py`/`*.sh` out of `fullscale/`, so THIS FILE IS NOT STAGED BY IT.
# After `./stage_native.sh code` (which needs David's Duo login for
# `eng-master`), also:
#     rsync -a /work/submit/david_w/ZMass/calibration_studies/fullscale/precond_refit.sbatch \
#           engaging:orcd/pool/zmass/engaging/
#
# SUBMIT (from ~/orcd/pool/zmass/engaging on Engaging):
#   # the K(m) ladder: control + the rung that failed, one H200, ~6 h
#   sbatch -A mit_general -p mit_preemptable -G h200:1 \
#          --export=ALL,ROWS="SVs7P SVs9P" precond_refit.sbatch
#   # phase 2, its own allocation -- 6 h+ and a 141.4 GB Hessian on a 143.8 GB H200
#   sbatch -A mit_general -p mit_preemptable -G h200:1 --time=1-12:00:00 \
#          --export=ALL,ROWS="P2XP" precond_refit.sbatch
#   # the J/psi FSR kernel: phase 2 with the kernel (its own allocation), and
#   # the two J/psi-only rows together (3 M candidates, ~63 GB Hessian)
#   sbatch -A mit_general -p mit_preemptable -G h200:1 --time=1-12:00:00 \
#          --export=ALL,ROWS="P2K" precond_refit.sbatch
#   sbatch -A mit_general -p mit_preemptable -G h200:1 --time=1-00:00:00 \
#          --export=ALL,ROWS="J0 JK" precond_refit.sbatch
#
# WHAT TO LOOK FOR, in order:
#   1. `INFO:preconditioner.py: Preconditioned N of 1 block(s), M parameters;
#      condition number median X, worst X -> median Y, worst Y ...`  The
#      before/after pair is the TRUE condition number (not the correlation
#      one).  If it says "(WORSE than unpreconditioned)" or a block was
#      dropped, the transform is not doing its job and the row is not a test of
#      the preconditioner.
#   2. `SVs7P` must reproduce the certified `SVs7`:
#        NLL (nllvalreduced) -9827109.7690, EDM 1.34e-09,
#        m_Z -3.874045 +- 2.2772, Gamma_Z +12.8587 +- 4.3546,
#        shape5 +0.063675 +- 0.006349   (errors raw; x1.109 for N_eff/N)
#      m_Z to 0.01 MeV or the preconditioner has changed an answer.
#   3. `SVs9P` certifies only with EDM < 1e-3 AND a finite postfit Hessian.
#      Its m_Z is then the number that closes RESOLUTION.md sec. 6.4: the
#      v-form truncation systematic is currently quoted as 2.4 MeV over
#      K = 5..7, and K=9 either confirms it or replaces it.  Compare the m-form
#      K=9 rung `Ss9`, which DID converge: NLL 11074904.2322, EDM 1.31e-11,
#      m_Z +9.388832 +- 2.3114, shape9 +0.003683 +- 0.000154.
#   4. `P2XP` certifies only with EDM < 1e-3.  There is no certified reference;
#      the nearest is `P2smoke` on a DIFFERENT (500k+500k) card -- EDM 6.2e-19,
#      m_Z +31.86 +- 5.78 MeV -- so compare the shape of the answer, never the
#      NLL.
#   A postfit `ValueError: Cholesky decomposition failed, Hessian is not
#   positive-definite` means the minimiser LEFT the near-stationary point along
#   a genuinely negative-curvature direction.  That is information, not a bug:
#   it says the direction is a real feature of the likelihood and the cure is a
#   prior or a freeze, not a better minimiser (this is exactly what the
#   preconditioner did to the vtxres `hitres_str_N5_hi` direction).
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
# SOURCE THE ENVIRONMENT BEFORE `set -f`, not after.  `setup_env_engaging.sh`
# puts the CUDA wheel lib dirs on LD_LIBRARY_PATH with a glob, and under noglob
# that loop matched nothing: TF then skipped registering the GPU (at a log
# level TF_CPP_MIN_LOG_LEVEL=2 hides) and jobs 22679406/22679407 spent 4 h each
# on an H200 they never touched, at ~1/100 the speed.  The env script is
# noglob-safe as of 2026-09-13 as well, so this ordering is the second lock on
# the same door.
source $ZMASS/engaging/setup_env_engaging.sh
# noglob: PRECOND is expanded UNQUOTED (it has to word-split into flags) and
# carries the regex `.*`, which bash would otherwise expand against the cwd.
set -f
RABBIT=${RABBIT:-$ZMASS/rabbit-vmass}
PYBASE=$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}

# Per-row card, param model and freeze list. The `P` suffix on every tag keeps
# these results and their snapshots away from the unpreconditioned ones.
declare -A CARDS=(
  [SVs7P]=z_V_s7                 # the certified control
  [SVs9P]=z_V_s9                 # the v-form rung that failed
  [SVs12P]=z_V_s12               # the next rung, if 9 certifies
  [Ss9P]=z_full380_fl_s9         # the m-form K=9, converged -- a second control
  [P2XP]=joint_ok_full           # phase 2, full
  # --- the J/psi FSR kernel (`build_jpsi_fsr.sh`) ---------------------------
  # P2K is P2XP with ONE change: the J/psi term's delta at MJPSI becomes
  # delta (x) K with the sample's own kernel.  J0/JK are the same J/psi term
  # and the same quadratic term with NO Z leg, so what moves between them is
  # attributable to the J/psi FSR treatment alone.
  [P2K]=joint_fsrmc              # phase 2, full, J/psi kernel = mc
  [J0]=jpsi_nok                  # J/psi + the 92, delta kernel
  [JK]=jpsi_fsrmc                # J/psi + the 92, mc kernel
)
declare -A MODELS=(
  [SVs7P]="UnbinnedParams"  [SVs9P]="UnbinnedParams"  [SVs12P]="UnbinnedParams"
  [Ss9P]="UnbinnedParams"
  [P2XP]="ExternalParams bundle:global_params"
  [P2K]="ExternalParams bundle:global_params"
  [J0]="ExternalParams bundle:global_params"
  [JK]="ExternalParams bundle:global_params"
)
# The phase-2 freeze list is the one phase 2 was run with: the four resolution
# knobs plus the four material groups the hit-chi2 term carries NO information
# on (beampipe, thermal screen, support tube, pp1 cables). FREEZE3=1 keeps
# material_beampipe floating, as in the cross-check.
_P2FREEZE="k_hit k_ms k_ioni k_rad material_thermal_screen material_support_tube material_pp1_cables"
[ "${FREEZE3:-0}" = 1 ] || _P2FREEZE="$_P2FREEZE material_beampipe"
declare -A FREEZES=(
  [SVs7P]="k_hit k_ms k_ioni k_rad"  [SVs9P]="k_hit k_ms k_ioni k_rad"
  [SVs12P]="k_hit k_ms k_ioni k_rad" [Ss9P]="k_hit k_ms k_ioni k_rad"
  [P2XP]="$_P2FREEZE"
  [P2K]="$_P2FREEZE"  [J0]="$_P2FREEZE"  [JK]="$_P2FREEZE"
)

METHOD=${METHOD:-trust-exact}
# `--preconditionParams '.*'` rather than the default: the default scope is
# every UNCONSTRAINED parameter, which is all 15 on the K(m) cards (they carry
# no Gaussian priors) but leaves out the 42 material groups on `joint_ok_full`,
# and those are part of the conditioning problem there. Frozen parameters are
# excluded from the scope automatically -- naming one explicitly would raise
# ("preconditioning a FROZEN parameter is not supported").
# `--stallRelTol 1e-4`: SVs9 cycled its EDM between 9e3 and 2.4e4 without ever
# triggering the default stall test (which fires only on LITERALLY no
# improvement), so it never got its transform rebuilt at the point it had
# reached. 1e-4 is two orders below the 0.0198 at which a healthy fit
# converging at 0.1 % per iteration over a 20-iteration window would be
# flagged. With --maxRestarts (default -1, restart while the loss keeps
# falling) each restart rebuilds the preconditioner where the fit now is.
PRECOND=${PRECOND:---precondition --preconditionParams .* --preconditionBlocks none --preconditionTransform spectral --preconditionFrom hessian --stallRelTol 1e-4}

OUT=$ZMASS/fitresults/native
mkdir -p $OUT
cd $ZMASS/fullscale
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for SPEC in ${ROWS:?set ROWS="tag tag ..."}; do
    TAG=${SPEC%%@*}
    if [ "$SPEC" = "$TAG" ]; then RB=$RABBIT; else RB=$ZMASS/${SPEC#*@}; fi
    export PYTHONPATH=$RB:$PYBASE
    CARD=$ZMASS/cards/${CARDS[$TAG]:?unknown row $TAG}.hdf5
    SNAP=$OUT/rabbit_${TAG}.snapshot.hdf5
    RESUME=""
    # FRESH defaults to 1 here, unlike rabbit_vmass_batch.sbatch: the control
    # row's whole point is to arrive at the certified minimum on its own.
    if [ -f "$SNAP" ] && [ "${FRESH:-1}" != "1" ]; then RESUME="--externalPostfit $SNAP"; fi
    echo "############ $TAG  $(date +%H:%M:%S)  card $CARD  method $METHOD ${RESUME:+(warm)}"
    echo "############   RABBIT=$RB  ($(git -C $RB log --oneline -1 2>/dev/null))"
    echo "############   MODEL=${MODELS[$TAG]}  FREEZE=${FREEZES[$TAG]}"
    echo "############   PRECOND=$PRECOND"
    python3 -u $RB/bin/rabbit_fit.py "$CARD" $RESUME \
        --paramModel ${MODELS[$TAG]} --minimizerMethod "$METHOD" \
        --freezeParameters ${FREEZES[$TAG]} \
        -t 0 --unblind --diagnostics \
        $PRECOND \
        --snapshotFile $SNAP --snapshotInterval 0.25 \
        --outpath $OUT --outname rabbit_${TAG}.hdf5 ${EXTRA:-}
    rc=$?
    echo "############ $TAG done $(date +%H:%M:%S) rc=$rc"
done

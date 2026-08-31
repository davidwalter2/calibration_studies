#!/bin/bash
# The two hit-resolution ablation samples, A1 and A2.
#
# WHY. The CF closure cannot separate the hit term from MS: the hits are 11 %
# of Var(q/p) and MS 82 %, so once the MS scale is free the two hit treatments
# are degenerate (NOTES_HITRES s15). Removing the trajectory noise takes the
# hit share from 0.17 to ~0.94 and gives the closure full power on the hit
# model. High pT does NOT do this: the measured hit share grows only as
# pT^0.47 -- because the radiative term grows with momentum -- so even at
# pT = 1 TeV it reaches only ~0.50 (NOTES_HITRES s16).
#
#   A1  msc + CoulombScat off for mu+/-        no Coulomb scattering
#   A2  A1 + muBrems + muPairProd off          also no radiative straggling
#
# CoulombScat is in A1 deliberately. FTFP_BERT_EMM uses the COMBINED model:
# `msc` below a polar-angle limit and `CoulombScat` (single scattering) above
# it. Deactivating msc alone leaves 275 primary CoulombScat steps per 60
# events -- measured -- and those are the large-angle RUTHERFORD TAIL, i.e.
# exactly the part that would keep the trajectory noise heavy-tailed while
# looking ablated.
#
# muIoni is NEVER deactivated: the tracker digitiser builds the cluster charge
# from PSimHit::energyLoss, so without it there are no hits at all. It is also
# unnecessary -- the ionization block is < 0.2 % of Var(q/p) (s11).
#
# The proof is the [procact] step census in step1.log, not the flag: the
# deactivated processes must be ABSENT from the primary's row while the
# electrons' TransportationWithMsc stays non-zero (delta-ray transport, hence
# cluster shapes, untouched) and muIoni is unchanged.
#
# usage: ./run_simprod_ablation.sh [ntasks] [nparallel] [nevents]
set -uo pipefail
NTASK=${1:-80}
NPAR=${2:-80}
NEVT=${3:-1000}
PTMIN=${PTMIN:-2}
PTMAX=${PTMAX:-20}
SIMPROD=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOGD=${LOGD:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/260829_hitres}
mkdir -p "$LOGD"

launch() {
  local tag=$1 inact=$2
  # to STDERR: this function's stdout is captured by $(launch ...) and must
  # contain ONLY the pid
  echo "[ablation] $tag : inactivate '$inact' for mu-,mu+" >&2
  SIMPROD_INACTIVATE="$inact" SIMPROD_INACT_PARTICLES="mu-,mu+" \
  OUTROOT="$CEPH/resolution_simprod_${tag}" \
    "$SIMPROD/run_simprod_mugun.sh" "$NPAR" 0 $((NTASK - 1)) "$NEVT" "$PTMIN" "$PTMAX" \
      > "$LOGD/simprod_${tag}.log" 2>&1 &
  echo $!
}

p1=$(launch mugun_lowpt_noms     "msc,CoulombScat")
p2=$(launch mugun_lowpt_nomsrad  "msc,CoulombScat,muBrems,muPairProd")
fail=0
for p in $p1 $p2; do wait "$p" || fail=1; done
for tag in mugun_lowpt_noms mugun_lowpt_nomsrad; do
  d="$CEPH/resolution_simprod_${tag}"
  echo "[ablation] $tag: $(ls $d/task_*/step2.root 2>/dev/null | wc -l)/$NTASK step2.root"
  ls "$d"/task_*/step2.root 2>/dev/null | sort > "$SIMPROD/filelist_${tag}.txt"
done
exit $fail

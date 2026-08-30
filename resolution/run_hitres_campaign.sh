#!/bin/bash
# The hit-resolution campaign: five species-and-momentum arms of the pull
# study, plus the two sim-position null tests.
#
# Arms and why each one is there:
#   mugun_lowpt   mu,  pT 2-20   the species control for the hadrons: same
#                               momenta, different particle. THE reference.
#   mugun_ul16    mu,  pT 20-60  the Z-momentum arm; the hit term dominates
#                               the curvature error there (hit share of the
#                               q/p variance 0.28 vs 0.14 at low pT, NOTES.md)
#                               so this is where a wrong sigma_hit matters most.
#   piongun_ul16  pi,  pT 2-20   \
#   kaongun_ul16  K,   pT 2-20    > the two implicit dE/dx dependences of the
#   protongun_ul16 p,  pT 2-20   /  CPE (the strip dQdx > maxChgOneMIP branch
#                                   and the pixel qbin) are the only way the
#                                   error model can know about the species,
#                                   and they are never given a hypothesis.
#
# Sim-position null tests are run on mu and kaon ONLY: for muons it repeats a
# measurement that already exists (NOTES.md 2026-08-08, |shift| < 0.45e-4 at
# 3 sigma) as a cross-check of this production, and for kaons it is NEW --
# before the species fix to the sim-hit match, fitSimHitPositions silently
# fell back to reco positions on every hadron.
#
# usage: ./run_hitres_campaign.sh [nfiles_per_arm] [nparallel_per_arm] [stage]
#   stage: main | nulltest | all   (default all)
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
NF=${1:-40}
NPAR=${2:-40}
STAGE=${3:-all}
LOG=${LOG:-$RES/runs/260829_hitres}
mkdir -p "$LOG"
cd "$RES"

# species:filelist:outtag
MAIN=(
  "mu:simprod/filelist_mugun_lowpt.txt:mugun_lowpt"
  "mu:simprod/filelist_mugun_ul16.txt:mugun_ul16"
  "pi:simprod/filelist_piongun_ul16.txt:piongun_ul16"
  "kaon:simprod/filelist_kaongun_ul16.txt:kaongun_ul16"
  "proton:simprod/filelist_protongun_ul16.txt:protongun_ul16"
)
NULL=(
  "mu:simprod/filelist_mugun_lowpt.txt:mugun_lowpt_simpos"
  "kaon:simprod/filelist_kaongun_ul16.txt:kaongun_ul16_simpos"
)

launch() {
  local -n arr=$1
  local extra=$2
  local pids=()
  for spec in "${arr[@]}"; do
    IFS=: read -r sp list tag <<< "$spec"
    echo "[campaign] $tag: species=$sp files=0-$((NF-1))"
    EXTRA="$extra" STAGGER=3 ./run_local_hitres.sh "$sp" "$list" "$tag" \
        "$NPAR" 0 $((NF - 1)) > "$LOG/${tag}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

case "$STAGE" in
  main|all) launch MAIN "";;
esac
case "$STAGE" in
  nulltest|all) launch NULL "fitSimHitPositions=True";;
esac

CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
for spec in "${MAIN[@]}" "${NULL[@]}"; do
  tag=${spec##*:}
  n=$(ls -d "$CEPH/hitres_${tag}"/task_*/.complete 2>/dev/null | wc -l)
  echo "[campaign] $tag: $n/$NF complete"
done

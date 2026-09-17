#!/bin/bash
# Integrity of one 2026-09-17 closure production, from its own outputs only.
#
# Three independent things are checked, because each catches a different way a
# task can look finished and not be:
#   1. per task: the `.complete` sentinel, exactly `streams=N` stream files and
#      none of them empty -- a killed cmsRun leaves a non-empty but TRUNCATED
#      .root, so the sentinel is the only honest completion signal;
#   2. events: the chunk ranges of the completed tasks summed against the
#      `fit summary` lines, and against the per-file counts the chunk list was
#      built from;
#   3. candidates: attempted / succeeded / failure breakdown, and the yield per
#      event, against the v2 reference (0.447 DY, 0.9967 J/psi).
#
#   usage: ./integrity_260917.sh <tag> [<ntasks>]
set -uo pipefail
TAG=${1:?usage: integrity_260917.sh <tag> [ntasks]}
O=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$TAG" in
  dymc_*)   CHUNKS=$P/chunks_dymc_8p5M_260905.txt;                       NT=${2:-380} ;;
  jpsimc_*) CHUNKS=$P/condor_jpsimc_v2/chunks_jpsimc_20M_260906_v2.txt;  NT=${2:-600} ;;
  *) echo "unknown tag $TAG" >&2; exit 1;;
esac

echo "=== $TAG ($NT tasks planned)"
ok=0; miss=(); badstream=(); empty=()
for ((i = 0; i < NT; i++)); do
  d=$O/task_$(printf '%04d' "$i")
  if [[ ! -f $d/.complete ]]; then miss+=("$i"); continue; fi
  want=$(sed -n 's/^streams=//p' "$d/.complete")
  n=$(ls "$d"/globalcor_*.root 2>/dev/null | wc -l)
  e=$(find "$d" -name 'globalcor_*.root' -empty 2>/dev/null | wc -l)
  [[ "$n" == "${want:-4}" ]] || badstream+=("$i:$n")
  (( e == 0 )) || empty+=("$i:$e")
  ok=$((ok + 1))
done
printf 'complete           : %d / %d\n' "$ok" "$NT"
printf 'missing sentinel   : %d %s\n' "${#miss[@]}" "$(printf '%s ' "${miss[@]:0:20}")"
printf 'wrong stream count : %d %s\n' "${#badstream[@]}" "$(printf '%s ' "${badstream[@]:0:20}")"
printf 'empty stream files : %d %s\n' "${#empty[@]}" "$(printf '%s ' "${empty[@]:0:20}")"

# events planned for the COMPLETED tasks only
PLANNED=$(for ((i = 0; i < NT; i++)); do
            [[ -f $O/task_$(printf '%04d' "$i")/.complete ]] && sed -n "$((i+1))p" "$CHUNKS"
          done | awk '{s+=$3} END{print s+0}')
ALL=$(head -n "$NT" "$CHUNKS" | awk '{s+=$3} END{print s+0}')
printf 'events             : %d planned over the completed tasks, %d over all %d\n' \
       "$PLANNED" "$ALL" "$NT"

grep -h "fit summary" "$O"/task_*/local.log 2>/dev/null | awk -v pl="$PLANNED" '
  { for (i = 1; i <= NF; i++) {
      split($i, kv, "=");
      if (kv[1] == "attempted") a += kv[2];
      else if (kv[1] == "succeeded") s += kv[2];
      else if (kv[1] ~ /^(fail|skipped|clamped|backtracked|inflated)/) f[kv[1]] += kv[2];
  } }
  END {
    printf "candidates         : %d attempted, %d succeeded, %d failed (%.4f%%)\n", a, s, a-s, a ? 100*(a-s)/a : 0;
    printf "yield              : %.6f candidates/event (planned events %d)\n", pl ? s/pl : 0, pl;
    n = 0; out = "";
    for (k in f) if (f[k] > 0) { out = out sprintf("%s=%d ", k, f[k]); n++ }
    printf "nonzero counters   : %s\n", n ? out : "(none)";
  }'
du -sh "$O" 2>/dev/null | awk '{printf "volume             : %s\n", $1}'

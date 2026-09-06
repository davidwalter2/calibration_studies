#!/bin/bash
# Parallel wrapper for cf_track_resolution.py --extract.
#
# The extraction is CPU-bound and embarrassingly parallel over input files
# (every track is independent), but the script itself is a serial loop --
# measured ~9 min/file, i.e. ~6 h for a 40-file production on one core, on a
# 192-core box. This shards by file and merges the per-shard caches.
#
# usage: ./masspairs_parallel.sh <production-dir> <out.npz> [nshards]
set -euo pipefail
INDIR=$1
OUT=$2
NSHARD=${3:-20}

SELF=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=prodfiles.sh
source "$SELF/prodfiles.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/masspairs_shards.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# EVERY STREAM OF EVERY USABLE TASK. `ls task_*/globalcor_0.root` took a
# quarter of the candidates from a numberOfThreads=4 production, and picked up
# the leftover streams of any task whose sentinel is missing.
mapfile -t FILES < <(pf_files "$INDIR")
N=${#FILES[@]}
echo "[masspairs_parallel] $N files ($(pf_task_dirs "$INDIR" | wc -l) tasks) -> $NSHARD shards"

run_shard() {
  local i=$1
  local list="$TMP/shard_$i.txt"
  : > "$list"
  local j=$i
  while [ "$j" -lt "$N" ]; do echo "${FILES[$j]}" >> "$list"; j=$((j + NSHARD)); done
  [ -s "$list" ] || return 0
  # One shard = one explicit file list, handed to --files directly. This used
  # to be a directory of symlinks named task_NNNN/globalcor_0.root, which a
  # multi-stream production breaks twice over: the fake tasks carry no
  # `.complete`, and a task's four streams would have had to be renamed apart.
  python3 "$SELF/cf_mass_likelihood.py" --pairs-tt \
      --files "$list" \
      --ntasks 0 --pairs-cache "$TMP/out_$i.npz" > "$TMP/log_$i.txt" 2>&1 \
    || { echo "[FAIL] shard $i"; tail -3 "$TMP/log_$i.txt"; return 1; }
}
export -f run_shard
export TMP N NSHARD SELF
# FILES is an array; re-export via a serialized form the subshells can read
printf '%s\n' "${FILES[@]}" > "$TMP/all_files.txt"
run_shard_wrap() { mapfile -t FILES < "$TMP/all_files.txt"; export FILES; run_shard "$1"; }
export -f run_shard_wrap

seq 0 $((NSHARD - 1)) | xargs -P "$NSHARD" -I{} bash -c 'mapfile -t FILES < "$TMP/all_files.txt"; run_shard {}'

echo "[masspairs_parallel] merging"
# Write to a temp name and rename only on success. np.savez_compressed on a
# multi-GB cache takes minutes, during which the final path already exists
# and is a TRUNCATED zip -- any reader that waits on `[ -f ... ]` gets
# BadZipFile. Rename is atomic within a filesystem, so the final path never
# exists in a partial state. (Same trap as the step2/simprod outputs.)
python3 - "$TMP" "$OUT.tmp.npz" <<'PY'
import glob, sys
import numpy as np
tmp, out = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{tmp}/out_*.npz"))
if not fs:
    raise SystemExit("no shard outputs -- see logs in " + tmp)
parts = [np.load(f) for f in fs]
keys = list(parts[0].files)
merged = {}
for k in keys:
    a = [p[k] for p in parts]
    # tgrid is the shared tau grid, identical in every shard: keep one copy
    if k == "tgrid":
        for x in a[1:]:
            assert np.allclose(x, a[0]), "tgrid differs between shards"
        merged[k] = a[0]
    else:
        merged[k] = np.concatenate(a, axis=0)
np.savez_compressed(out, **merged)
n = len(merged["z"])
print(f"[masspairs_parallel] {len(fs)} shards -> {out}  ({n} tracks)")
PY
mv -f "$OUT.tmp.npz" "$OUT"
echo "[masspairs_parallel] -> $OUT"

#!/bin/bash
# Parallel wrapper for cf_track_resolution.py --extract.
#
# The extraction is CPU-bound and embarrassingly parallel over input files
# (every track is independent), but the script itself is a serial loop --
# measured ~9 min/file, i.e. ~6 h for a 40-file production on one core, on a
# 192-core box. This shards by file and merges the per-shard caches.
#
# usage: ./extract_parallel.sh <indir-glob-dir> <out.npz> [nshards]
set -euo pipefail
INDIR=$1
OUT=$2
NSHARD=${3:-20}

SELF=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/extract_shards.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

mapfile -t FILES < <(ls "$INDIR"/task_*/globalcor_resclosure_0.root | sort)
N=${#FILES[@]}
echo "[extract_parallel] $N files -> $NSHARD shards"

run_shard() {
  local i=$1
  local list="$TMP/shard_$i.txt"
  : > "$list"
  local j=$i
  while [ "$j" -lt "$N" ]; do echo "${FILES[$j]}" >> "$list"; j=$((j + NSHARD)); done
  [ -s "$list" ] || return 0
  # one shard = one explicit file list; --files takes a glob, so stage the
  # shard's files into a private directory of symlinks and glob that.
  local sd="$TMP/d_$i"; mkdir -p "$sd"
  local k=0
  while read -r f; do
    mkdir -p "$sd/task_$(printf '%04d' $k)"
    ln -sf "$f" "$sd/task_$(printf '%04d' $k)/globalcor_resclosure_0.root"
    k=$((k + 1))
  done < "$list"
  python3 "$SELF/cf_track_resolution.py" --extract \
      --files "$sd/task_*/globalcor_resclosure_0.root" \
      --ntasks 10000 --cache "$TMP/out_$i.npz" > "$TMP/log_$i.txt" 2>&1 \
    || { echo "[FAIL] shard $i"; tail -3 "$TMP/log_$i.txt"; return 1; }
}
export -f run_shard
export TMP N NSHARD SELF
# FILES is an array; re-export via a serialized form the subshells can read
printf '%s\n' "${FILES[@]}" > "$TMP/all_files.txt"
run_shard_wrap() { mapfile -t FILES < "$TMP/all_files.txt"; export FILES; run_shard "$1"; }
export -f run_shard_wrap

seq 0 $((NSHARD - 1)) | xargs -P "$NSHARD" -I{} bash -c 'mapfile -t FILES < "$TMP/all_files.txt"; run_shard {}'

echo "[extract_parallel] merging"
python3 - "$TMP" "$OUT" <<'PY'
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
print(f"[extract_parallel] {len(fs)} shards -> {out}  ({n} tracks)")
PY

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
EXTRA_ARGS=${EXTRA_ARGS:-}

# ONE BLAS/OpenMP THREAD PER SHARD. numpy's backends default to one thread
# per core, so NSHARD=160 on a 192-core box asks for ~21 600 threads from ONE
# python3 process group -- and two samples extracting at once put this user at
# 32 373 threads against a `ulimit -u` of 32 768. At that point NOTHING can
# create a thread any more: every cmsRun launched afterwards died with an
# immediate segmentation violation before its first log line, including a
# trivial EmptySource job, while the machine still had 1.2 TB of free memory.
# (Measured 2026-08-30: it killed 40 CGF refit tasks and 69 more in the next
# sample, and looked exactly like a physics crash.)
#
# The shard work is per-track and serial; the threads buy nothing here.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

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
  # EXTRA_ARGS is how a run caps the work per shard. Wall time here is set by
  # ONE file -- each shard gets one -- and the per-track cost is dominated by
  # the exact-delta ionization exponent (_kokoulin_exponent), measured at over
  # an hour per file on the doRes productions. So `--max-tracks N` is the only
  # knob that shortens the run; reducing the FILE count does not.
  # shellcheck disable=SC2086
  python3 "$SELF/cf_track_resolution.py" --extract \
      --files "$sd/task_*/globalcor_resclosure_0.root" \
      --ntasks 10000 --cache "$TMP/out_$i.npz" ${EXTRA_ARGS:-} > "$TMP/log_$i.txt" 2>&1 \
    || { echo "[FAIL] shard $i"; tail -3 "$TMP/log_$i.txt"; return 1; }
}
export -f run_shard
export TMP N NSHARD SELF EXTRA_ARGS
# FILES is an array; re-export via a serialized form the subshells can read
printf '%s\n' "${FILES[@]}" > "$TMP/all_files.txt"
run_shard_wrap() { mapfile -t FILES < "$TMP/all_files.txt"; export FILES; run_shard "$1"; }
export -f run_shard_wrap

seq 0 $((NSHARD - 1)) | xargs -P "$NSHARD" -I{} bash -c 'mapfile -t FILES < "$TMP/all_files.txt"; run_shard {}'

echo "[extract_parallel] merging"
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
# A shard set with different key sets can only come from mixing code versions
# in one run (the `ioni_charge_signed` provenance flag is the case that
# introduced this), and silently merging it would produce a cache whose sign
# convention differs between its halves. Fail instead.
for f, p in zip(fs[1:], parts[1:]):
    assert set(p.files) == set(keys), (
        f"{f} has key set {sorted(set(p.files) ^ set(keys))} different from "
        f"{fs[0]} -- shards were written by different code versions")
merged = {}
for k in keys:
    a = [p[k] for p in parts]
    # tgrid is the shared tau grid, hitclsnames the canonical class list and
    # ioni_charge_signed the scalar provenance flag saying the shard's Sio_im
    # already carries the charge of the ionization q/p map; all three are
    # identical in every shard, so keep one copy instead of concatenating
    # (concatenating hitclsnames would also make its length depend on the
    # shard count, and concatenating the flag would turn a 0-d marker into an
    # array whose length is the shard count).  `rad_model` joins them: the 0/1
    # provenance value saying whether the radiative block was built, identical
    # across shards by construction (extract() refuses an input set that mixes
    # productions with and without the `radstepv` export).
    if k in ("tgrid", "hitclsnames", "ioni_charge_signed", "rad_model"):
        for x in a[1:]:
            assert np.array_equal(x, a[0]), f"{k} differs between shards"
        merged[k] = a[0]
    else:
        merged[k] = np.concatenate(a, axis=0)
np.savez_compressed(out, **merged)
n = len(merged["z"])
print(f"[extract_parallel] {len(fs)} shards -> {out}  ({n} tracks)")
PY
mv -f "$OUT.tmp.npz" "$OUT"
echo "[extract_parallel] -> $OUT"

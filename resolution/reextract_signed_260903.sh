#!/bin/bash
# Re-extract the six current track-resolution caches on the CHARGE-SIGNED
# ionization convention (2026-09-03, Documents/Resolution/NOTES.md s3).
#
# `ioniurbanv[:,10] = cs = E/p^3` is positive for every track and the physical
# map is d(q/p) = q cs dE, so the ionization step weight carries the charge.
# `cf_track_resolution.extract` now passes `chg * wstd` and stamps the cache
# with `ioni_charge_signed`.  Only Im S changes, and only on the mu- half:
#
#     Re S(-w) = Re S(+w),   Im S(-w) = -Im S(+w)   EXACTLY,
#
# so every other array in the cache must come back BIT-IDENTICAL.  That is
# what this script checks before it replaces anything, on the full sample:
# the new cache is written to <name>_k0s.npz, verified against the existing
# <name>_k0.npz, and only then rotated into place (the old cache is KEPT as
# <name>_k0_unsigned.npz -- it is the input to the 2026-09-02/03 tables).
#
# ONE SAMPLE AT A TIME, and the SHARD COUNT MUST MATCH THE ORIGINAL RUN: the
# merge concatenates shards in lexical order of the shard index, so a
# different NSHARD is a different track ORDER in the merged arrays and the
# element-wise comparison below would be meaningless (the numbers would all
# be there, permuted). 160 files -> 160 shards, 80 files -> 80 shards, i.e.
# one file per shard, exactly as in finish_muon_closure_260903.sh and
# chain_noms_260903.sh.
#
# usage: ./reextract_signed_260903.sh [sample-key ...]     (default: all six)
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CVH_IONI_KOKOULIN=0   # project convention since 2026-09-03
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/signfix260903; mkdir -p "$LOG"

# key | ceph sample | cache basename | nshard | extract args
SAMPLES=(
  "lowpt_m0|mugun_lowpt_260830_m0|cf_trackres_mugun_lowpt_260830_m0_k0|160|"
  "ul16_m0|mugun_ul16_260830_m0|cf_trackres_mugun_ul16_260830_m0_k0|160|"
  "lowpt_raw|mugun_lowpt_260830|cf_trackres_mugun_lowpt_260830_raw_k0|160|--ioni-norm raw"
  "ul16_raw|mugun_ul16_260830|cf_trackres_mugun_ul16_260830_raw_k0|160|--ioni-norm raw"
  "noms|mugun_lowpt_noms_cf|cf_trackres_mugun_lowpt_noms_k0|80|"
  "nomsrad|mugun_lowpt_nomsrad_cf|cf_trackres_mugun_lowpt_nomsrad_k0|80|"
)

do_one() {
  local key=$1 samp=$2 base=$3 nsh=$4 extra=$5
  local old="runs/$base.npz" new="runs/${base}s.npz" keep="runs/${base}_unsigned.npz"
  echo "=== $key  ($samp -> $base, $nsh shards, args '${extra}')  $(date +%H:%M:%S)"
  if [ ! -s "$old" ]; then echo "    [skip] no $old to compare against"; return 0; fi
  if grep -q "^ioni_charge_signed" <(python3 -c "
import numpy as np,sys
print('\n'.join(np.load('$old').files))" 2>/dev/null); then
    echo "    [skip] $old is ALREADY charge-signed"; return 0; fi
  if [ ! -s "$new" ]; then
    EXTRA_ARGS="$extra" ./extract_parallel.sh "$CEPH/resolution_trackres_$samp" \
        "$new" "$nsh" > "$LOG/ext_$key.log" 2>&1 \
      || { echo "    [FAIL] extract $key (see $LOG/ext_$key.log)"; return 1; }
  else
    echo "    reusing existing $new"
  fi
  python3 - "$old" "$new" > "$LOG/verify_$key.log" 2>&1 <<'PY'
import sys
import numpy as np
o, n = (np.load(f) for f in sys.argv[1:3])
ko, kn = set(o.files), set(n.files)
assert kn - ko == {"ioni_charge_signed"}, f"unexpected new keys {sorted(kn-ko)}"
assert not ko - kn, f"keys LOST: {sorted(ko-kn)}"
q = o["charge"]


def eq(a, b):
    """array_equal, but NaN == NaN.  `eta` carries one genuine NaN per ~3e5
    tracks (a fit with lambda just past pi/2 makes -log(tan((pi/2-lam)/2))
    undefined); it is present and identical in BOTH caches, and a plain
    array_equal would report the whole column as a mismatch because
    NaN != NaN.  equal_nan is only defined for inexact dtypes."""
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


assert eq(q, n["charge"]), "charge column differs -> different track set/order"
bad = []
for k in sorted(ko):
    if k == "Sio_im":
        ok = eq(n[k], (q[:, None] * o[k]).astype(n[k].dtype))
        print(f"  Sio_im == charge * Sio_im_old : {ok}")
    else:
        ok = eq(o[k], n[k])
        nnan = int(np.isnan(o[k]).sum()) if o[k].dtype.kind == "f" else 0
        print(f"  {k:14s} bit-identical         : {ok}"
              + (f"   ({nnan} NaN, identical in both)" if nnan else ""))
    if not ok:
        bad.append(k)
nq = int((q > 0).sum())
print(f"  tracks {len(q)}  ({nq} mu+, {len(q)-nq} mu-)")
print(f"  <Im S(tau=1)> old {o['Sio_im'][:, 32].mean():+.6e}  "
      f"new {n['Sio_im'][:, 32].mean():+.6e}")
raise SystemExit(f"MISMATCH in {bad}" if bad else 0)
PY
  if [ $? -ne 0 ]; then echo "    [FAIL] verify $key:"; tail -20 "$LOG/verify_$key.log"; return 1; fi
  grep -E "Sio_im|tracks|Im S" "$LOG/verify_$key.log" | sed 's/^/      /'
  mv -f "$old" "$keep" && mv -f "$new" "$old" \
    && echo "    [ok] $base rotated (old kept as $(basename "$keep"))  $(date +%H:%M:%S)"
}

WANT=("$@")
for s in "${SAMPLES[@]}"; do
  IFS='|' read -r key samp base nsh extra <<< "$s"
  if [ ${#WANT[@]} -gt 0 ]; then
    printf '%s\n' "${WANT[@]}" | grep -qx "$key" || continue
  fi
  do_one "$key" "$samp" "$base" "$nsh" "$extra"
done
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="

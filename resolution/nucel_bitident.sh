#!/bin/bash
# Bit-identity of the nuclear-elastic channel's OFF state, measured against a
# GENUINE REVERT rather than argued from the code.
#
# The project bar (HANDOFF_NUCLEAR_ELASTIC.md s4) is bit-identity when the
# switch is off, demonstrated by actually reverting and re-running -- "the code
# path is guarded" has been wrong before.
#
# WHY A COPY AND NOT `git stash`.  The first version of this script stashed the
# four CF files in place.  That mutates the LIVE tree for the duration of the
# run, and when the reference leg died the EXIT trap did not restore them --
# the working copy was left holding the reverted files with the real work
# sitting in a stash entry.  Anything else running out of that directory at the
# time silently switched physics underneath itself.  Reverting into a COPY
# leaves the live tree untouched no matter how this script exits.
#
# SCOPE.  Only the four CF files are reverted.  hadron_probe.py's changes in
# this branch (the `elonly`/`inelonly` arms, the `_s1??` seed pin, the
# outermost row, the cmd_live restore assertion) are intentional behaviour
# changes, not the channel -- the pin in particular restores the muon `off`
# sample from 11 files to the published 10, so a whole-tree revert would
# "fail" bit-identity for a reason with nothing to do with this channel.
#
# Usage:  ./nucel_bitident.sh
set -euo pipefail
REPO=/work/submit/david_w/ZMass/calibration_studies
SCRATCH=/tmp/claude-125124/-work-submit-david-w-ZMass/cf54f95f-40aa-4db0-8dcc-8a9a64f5439a/scratchpad
CF_FILES="cf_propagation_test.py cgf_channels.py fisher_norm.py barkas_probe.py"
REF_DIR="$SCRATCH/nucel_ref_tree"

PROBE='
import sys
import numpy as np
import hadron_probe as hp
out = {}
for pdg in (13, -211, -321, -2212, 2212):
    for func in ("qop", "locx"):
        r = hp._rows(pdg, "off", corr="on", func=func)
        out[f"{pdg}_{func}_rows"] = r["rows"]
        out[f"{pdg}_{func}_errs"] = r["errs"]
        out[f"{pdg}_{func}_sF"]   = r["sc"]["sF"]
np.savez(sys.argv[1], **out)
print("  wrote", sys.argv[1], len(out), "arrays", flush=True)
'

cd "$REPO"

echo "=== [1/4] building a reverted COPY of resolution/ (live tree untouched) ==="
rm -rf "$REF_DIR"; mkdir -p "$REF_DIR"
cp -a resolution/. "$REF_DIR"/
for f in $CF_FILES; do
  git show "HEAD:resolution/$f" > "$REF_DIR/$f"
done
# the channel module must not exist in the reference either
rm -f "$REF_DIR/cf_nucel_exact.py"
for f in $CF_FILES; do
  if grep -q cf_nucel_exact "$REF_DIR/$f"; then
    echo "*** $f still references the channel after revert" >&2; exit 2
  fi
done
echo "  reverted: $CF_FILES"

echo "=== [2/4] REFERENCE run (pre-change code) ==="
( source setup_env.sh >/dev/null 2>&1; cd "$REF_DIR"
  RES_NO_PHI_CACHE=1 RES_CACHE=0 python -u -c "$PROBE" "$SCRATCH/nucel_bitident_ref.npz" )

echo "=== [3/4] CURRENT run, NUCEL_CHANNEL=0 ==="
( source setup_env.sh >/dev/null 2>&1; cd resolution
  NUCEL_CHANNEL=0 RES_NO_PHI_CACHE=1 RES_CACHE=0 python -u -c "$PROBE" "$SCRATCH/nucel_bitident_new.npz" )

echo "=== [4/4] comparing, BITWISE ==="
source setup_env.sh >/dev/null 2>&1
python - "$SCRATCH/nucel_bitident_ref.npz" "$SCRATCH/nucel_bitident_new.npz" <<'PY'
import sys
import numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
ka, kb = sorted(a.files), sorted(b.files)
if ka != kb:
    raise SystemExit(f"*** key mismatch: only-ref={set(ka)-set(kb)} only-new={set(kb)-set(ka)}")
bad = []
for k in ka:
    x, y = a[k], b[k]
    if x.shape != y.shape or not np.array_equal(x, y):
        d = np.max(np.abs(x - y)) if x.shape == y.shape else float("nan")
        bad.append((k, d))
print(f"  {len(ka)} arrays compared")
for k, d in bad:
    print(f"  *** DIFFERS {k}   max|d| = {d:.3e}")
if bad:
    raise SystemExit(f"*** NOT bit-identical: {len(bad)}/{len(ka)} arrays differ")
print(f"  BIT-IDENTICAL: {len(ka)}/{len(ka)} arrays, every element")
PY

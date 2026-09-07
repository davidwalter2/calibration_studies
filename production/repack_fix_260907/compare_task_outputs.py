"""Candidate-by-candidate comparison of two CVH task outputs.

Used to decide whether the NUL runs found in two inputs' ParameterSets blobs
were inert.  The two runs cannot be compared file by file: with 4 EDM streams
the assignment of a candidate to a stream file depends on event scheduling and
is not reproducible.  So both sides are read as ONE set, sorted on
(run, lumi, event) with a kinematic tiebreaker, and every branch is compared on
that common order.

Flat branches are compared bitwise (`np.array_equal` on the raw values, NaNs
matched as equal).  Jagged branches are compared on their per-candidate counts
and their flattened values.  `runtree` is compared separately and in full.

  python3 compare_task_outputs.py <dirA> <dirB> [--branches N]
exit 0 = identical, 1 = differences (listed).
"""
import sys, glob
import numpy as np
import uproot, awkward as ak

KEY = ["run", "lumi", "event"]
TIE = ["Muplus_pt", "Muminus_pt"]


def files(d):
    fs = sorted(glob.glob(d.rstrip("/") + "/globalcor_*.root"))
    if not fs:
        sys.exit(f"no globalcor_*.root in {d}")
    return [uproot.open(f) for f in fs]


def order(fh):
    """Concatenated sort permutation over the streams of one run."""
    cols = {k: np.concatenate([f["tree"][k].array(library="np") for f in fh])
            for k in KEY + TIE}
    # lexsort: last key is primary
    perm = np.lexsort(tuple(cols[k] for k in (TIE + KEY)[::-1]))
    ids = np.stack([cols[k][perm].astype(np.float64) for k in KEY])
    return perm, ids, len(perm)


def branch(fh, name, perm):
    parts = [f["tree"][name].array() for f in fh]
    a = ak.concatenate(parts) if len(parts) > 1 else parts[0]
    return a[perm]


def same(a, b):
    """Bitwise comparison. Flat branches go through to_numpy; jagged ones fall
    back to (counts, flattened values), which to_numpy raises on."""
    try:
        na, nb = ak.to_numpy(a), ak.to_numpy(b)
        if na.shape != nb.shape:
            return False, f"shape {na.shape} vs {nb.shape}"
        if na.dtype.kind == "f":
            ok = np.array_equal(na, nb, equal_nan=True)
        else:
            ok = np.array_equal(na, nb)
        return ok, "" if ok else f"{int((na != nb).sum())} differing values"
    except Exception:
        ca, cb = ak.num(a, axis=1), ak.num(b, axis=1)
        if not np.array_equal(ak.to_numpy(ca), ak.to_numpy(cb)):
            return False, "differing per-candidate counts"
        fa = ak.to_numpy(ak.flatten(a)); fb = ak.to_numpy(ak.flatten(b))
        if fa.shape != fb.shape:
            return False, f"flat shape {fa.shape} vs {fb.shape}"
        ok = np.array_equal(fa, fb, equal_nan=True) if fa.dtype.kind == "f" \
             else np.array_equal(fa, fb)
        return ok, "" if ok else f"{int((fa != fb).sum())} differing flat values"


def main():
    dA, dB = sys.argv[1], sys.argv[2]
    fA, fB = files(dA), files(dB)
    pA, idA, nA = order(fA)
    pB, idB, nB = order(fB)
    print(f"A {dA}\n  streams={len(fA)} candidates={nA}")
    print(f"B {dB}\n  streams={len(fB)} candidates={nB}")
    if nA != nB:
        print(f"DIFFER: candidate count {nA} vs {nB}")
        return 1
    if not np.array_equal(idA, idB):
        nd = int((idA != idB).any(axis=0).sum())
        print(f"DIFFER: {nd} candidates have a different (run,lumi,event) after sorting")
        return 1
    print(f"  (run,lumi,event) order matches for all {nA} candidates")

    names = [k for k in fA[0]["tree"].keys()]
    bad = []
    for i, n in enumerate(names):
        ok, why = same(branch(fA, n, pA), branch(fB, n, pB))
        if not ok:
            bad.append((n, why)); print(f"  DIFF {n}: {why}")
        if (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(names)} branches checked", flush=True)
    # runtree: the global parameter map, must be identical outright
    rt = []
    for k in fA[0]["runtree"].keys():
        a = fA[0]["runtree"][k].array(); b = fB[0]["runtree"][k].array()
        ok, why = same(a, b)
        if not ok:
            rt.append((k, why)); print(f"  DIFF runtree/{k}: {why}")
    print(f"\nchecked {len(names)} tree branches + {len(fA[0]['runtree'].keys())} runtree branches")
    if bad or rt:
        print(f"RESULT: DIFFERENT ({len(bad)} tree, {len(rt)} runtree branches)")
        return 1
    print(f"RESULT: BIT-IDENTICAL over {nA} candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

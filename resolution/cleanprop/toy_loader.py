#!/usr/bin/env python3
"""Read the toy clean-propagation sim and express it in the model's plane frames.

The toy watcher (Analysis/HitAnalyzer/plugins/ToyStateNtuplizer.cc) writes the
true state in GLOBAL coordinates on purpose: the local frame is a property of the
SURFACE, fixed by the reference trajectory, and the watcher does not know the
reference. An earlier version tried to pick a frame there and produced locx
identically zero. The frame is applied here instead, using the SAME plane
definitions that were handed to G4ePropagationExport, so the two sides are
guaranteed to agree by construction rather than by coincidence.

Frame convention, matching the barrel modules and makePlaneTarget():
    local z = normal  = radial
    local x = u       = r-phi   (the BENDING direction)
    local y = uz x ux = global z (the NON-bending direction)

Returns the same dict layout as cf_propagation_test.load_sim, so the existing
comparison machinery reads it without change.
"""
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plane_frames(origin, normal, uaxis):
    """(N,3) origins and (N,3,3) rotation matrices whose ROWS are the local axes.

    u is orthogonalized against the normal exactly as makePlaneTarget does, so a
    caller-supplied pair that is not perfectly orthogonal gives the same frame on
    both sides.
    """
    o = np.asarray(origin, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    u = np.asarray(uaxis, dtype=np.float64).reshape(-1, 3)
    uz = n / np.linalg.norm(n, axis=1, keepdims=True)
    u = u - uz * np.sum(uz * u, axis=1, keepdims=True)
    ux = u / np.linalg.norm(u, axis=1, keepdims=True)
    uy = np.cross(uz, ux)
    R = np.stack([ux, uy, uz], axis=1)      # (N,3,3), rows = local axes
    return o, R


def load_toy_sim(path, origin, normal, uaxis):
    """Toy sim states in the plane frames, laid out like load_sim's output.

    `path` may be a glob, which is how a sim split across seeded jobs is read
    back (runToyGeomCheck.py `seed=`): the initial state is fixed and only the
    Geant4 seed varies between events, so the chunks concatenate into one
    sample with no bookkeeping. Files are taken in sorted order so the event
    ordering -- and therefore every per-event statistic -- is reproducible.
    """
    br = ["detid", "qop", "globx", "globy", "globz",
          "globpx", "globpy", "globpz", "pabs", "eloss", "globr"]
    files = (sorted(glob.glob(path)) if any(c in path for c in "*?[")
             else [path])
    if not files:
        raise FileNotFoundError(f"no toy sim files match {path}")
    if len(files) == 1:
        a = uproot.open(files[0])["simstates"].arrays(br, library="np")
    else:
        parts = [uproot.open(f)["simstates"].arrays(br, library="np")
                 for f in files]
        a = {k: np.concatenate([p[k] for p in parts]) for k in br}

    o, R = plane_frames(origin, normal, uaxis)
    nlayer = len(o)
    nev = len(a["detid"])

    out = {k: np.full((nev, nlayer), np.nan) for k in
           ("qop", "dxdz", "dydz", "locx", "locy", "locz", "pabs", "eloss", "globr")}
    valid = np.zeros((nev, nlayer), dtype=bool)

    # FLATTEN ONCE rather than looping over events (2026-08-15). The per-event
    # loop was 5.7 s of the 13.5 s `toy_closure` run at 100k events, all of it
    # Python overhead: the arithmetic per event is nine 14-element rotations.
    # Concatenating the jagged branches and scattering the result back with a
    # (event, plane) index pair does exactly the same arithmetic -- the SAME
    # einsum call with the SAME subscripts, so every output element is still
    # the same three-term sum in the same order -- on one long axis instead of
    # nev short ones. Verified bit-identical to the loop on hsK1 (100k events).
    nper = np.fromiter((len(d) for d in a["detid"]), dtype=np.int64, count=nev)
    ev = np.repeat(np.arange(nev, dtype=np.int64), nper)
    if len(ev):
        idx = np.concatenate([np.asarray(d, dtype=int) for d in a["detid"]])

        def _cat(name):
            return np.concatenate([np.asarray(v, dtype=np.float64)
                                   for v in a[name]])

        pos = np.stack([_cat(c) for c in ("globx", "globy", "globz")], axis=1)
        mom = np.stack([_cat(c) for c in ("globpx", "globpy", "globpz")],
                       axis=1)
        # rotate into each plane's own frame
        d = pos - o[idx]
        lp = np.einsum("kij,kj->ki", R[idx], d)
        lm = np.einsum("kij,kj->ki", R[idx], mom)
        ok = lm[:, 2] != 0.
        out["locx"][ev, idx] = lp[:, 0]
        out["locy"][ev, idx] = lp[:, 1]
        out["locz"][ev, idx] = lp[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["dxdz"][ev, idx] = np.where(ok, lm[:, 0] / lm[:, 2], np.nan)
            out["dydz"][ev, idx] = np.where(ok, lm[:, 1] / lm[:, 2], np.nan)
        for name in ("qop", "pabs", "eloss", "globr"):
            out[name][ev, idx] = _cat(name)
        valid[ev, idx] = True

    out["valid"] = valid
    out["ntot"] = nev
    out["nkept"] = int(valid.all(axis=1).sum())
    out["detid"] = np.arange(nlayer, dtype=np.uint32)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", required=True)
    p.add_argument("--planes", default=None,
                   help="toyPlanes_*.py; default = the pt3 one in the CMSSW test dir")
    args = p.parse_args()

    pl = args.planes or ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
                         "Analysis/HitAnalyzer/test/toyPlanes_pt3.py")
    ns = {}
    exec(open(pl).read(), ns)
    sim = load_toy_sim(args.sim, ns["origin"], ns["normal"], ns["uaxis"])

    v = sim["valid"]
    print(f"events {sim['ntot']}, complete {sim['nkept']} "
          f"({100.*sim['nkept']/sim['ntot']:.2f} %)")
    print(f"{'k':>3} {'r[cm]':>7} {'locx mean':>11} {'locx rms':>10} "
          f"{'locy rms':>10} {'locz rms':>10}")
    print("-" * 58)
    for k in range(v.shape[1]):
        s = v[:, k]
        if not s.any():
            continue
        print(f"{k:3d} {np.nanmedian(sim['globr'][s, k]):7.1f} "
              f"{1e4*np.nanmean(sim['locx'][s, k]):11.1f} "
              f"{1e4*np.nanstd(sim['locx'][s, k]):10.1f} "
              f"{1e4*np.nanstd(sim['locy'][s, k]):10.1f} "
              f"{1e4*np.nanstd(sim['locz'][s, k]):10.1f}")
    print("\nall in um. locz is the NORMAL offset: it measures how far the "
          "cylinder\ncrossing sits off the tangent plane, i.e. the one "
          "approximation in this setup.")

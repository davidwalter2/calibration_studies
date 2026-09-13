#!/usr/bin/env python3
"""TAIL study, hypothesis B: WHAT GEOMETRY does each production refit with?

The maker writes a `runtree` with one row per global parameter, carrying the
module's `rawdetid`, `subdet`, `layer` and its GLOBAL POSITION `x, y, z` as
the fit sees it.  Comparing two productions row by row on `rawdetid` therefore
MEASURES the difference between the two tracker geometries directly -- no
condition-database lookup and no assumption about what a global tag contains.

usage: python3 tail_geom.py --a <fileA.root> --b <fileB.root> [--labels A B]
"""
import argparse
import numpy as np
import uproot

SUBDET = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}


def load(fn):
    rt = uproot.open(fn + ":runtree")
    a = rt.arrays(["iidx", "parmtype", "rawdetid", "subdet", "layer",
                   "x", "y", "z", "rho", "eta", "phi", "dx", "dy", "dz",
                   "dtheta"], library="np")
    # one row per MODULE: parmtype 0 is the first alignment dof of each module
    m = a["parmtype"] == 0
    out = {k: v[m] for k, v in a.items()}
    _, u = np.unique(out["rawdetid"], return_index=True)
    return {k: v[u] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    print(f"# {a.labels[0]}: {len(A['rawdetid'])} modules   "
          f"{a.labels[1]}: {len(B['rawdetid'])} modules")
    common, ia, ib = np.intersect1d(A["rawdetid"], B["rawdetid"],
                                    return_indices=True)
    print(f"# common rawdetid: {len(common)}")
    dx = A["x"][ia] - B["x"][ib]
    dy = A["y"][ia] - B["y"][ib]
    dz = A["z"][ia] - B["z"][ib]
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    sub = A["subdet"][ia]
    lay = A["layer"][ia]
    rho = A["rho"][ia]
    phi = A["phi"][ia]

    print(f"\n=== |r_{a.labels[0]} - r_{a.labels[1]}| per module, um")
    hdr = (f"  {'subdet':<8s}{'layer':>6s}{'N':>7s}{'median':>10s}{'mean':>10s}"
           f"{'rms':>10s}{'p99':>10s}{'max':>12s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in sorted(set(sub.tolist())):
        for l in sorted(set(lay[sub == s].tolist())):
            m = (sub == s) & (lay == l)
            v = dr[m] * 1e4
            print(f"  {SUBDET.get(int(s), str(s)):<8s}{int(l):>6d}{int(m.sum()):>7d}"
                  f"{np.median(v):>10.2f}{v.mean():>10.2f}{v.std():>10.2f}"
                  f"{np.percentile(v,99):>10.2f}{v.max():>12.2f}")
    v = dr * 1e4
    print(f"  {'ALL':<8s}{'':>6s}{len(v):>7d}{np.median(v):>10.2f}{v.mean():>10.2f}"
          f"{v.std():>10.2f}{np.percentile(v,99):>10.2f}{v.max():>12.2f}")

    print(f"\n=== the components, um (all modules)")
    for nm, q in (("dx", dx), ("dy", dy), ("dz", dz)):
        u = q * 1e4
        print(f"  {nm}: mean {u.mean():+8.3f}  rms {u.std():8.3f}  "
              f"p1 {np.percentile(u,1):+8.2f}  p99 {np.percentile(u,99):+8.2f}"
              f"  max|.| {np.abs(u).max():9.2f}")
    # the R-PHI component, which is what a curvature / d0 cares about
    drphi = (-np.sin(phi) * dx + np.cos(phi) * dy) * 1e4
    drad = (np.cos(phi) * dx + np.sin(phi) * dy) * 1e4
    print(f"  r-phi: mean {drphi.mean():+8.3f}  rms {drphi.std():8.3f}  "
          f"p1 {np.percentile(drphi,1):+8.2f}  p99 {np.percentile(drphi,99):+8.2f}")
    print(f"  radial: mean {drad.mean():+8.3f}  rms {drad.std():8.3f}")
    print(f"  MODULES MOVED by > 1 um: "
          f"{int((dr*1e4 > 1).sum())} of {len(dr)} "
          f"({100*(dr*1e4>1).mean():.2f} %)")
    print(f"  MODULES MOVED by > 10 um: {int((dr*1e4 > 10).sum())}")
    print(f"  MODULES MOVED by > 100 um: {int((dr*1e4 > 100).sum())}")

    # a COHERENT mode would show as a non-zero mean of dr-phi against phi
    print(f"\n=== the r-phi shift against module phi (a COHERENT mode is what "
          "a vertex / d0 residual sees)")
    edges = np.linspace(-np.pi, np.pi, 13)
    for i in range(12):
        m = (phi >= edges[i]) & (phi < edges[i + 1])
        if m.sum() < 5:
            continue
        print(f"  phi {edges[i]:+6.2f}..{edges[i+1]:+6.2f}  N {int(m.sum()):5d}"
              f"  <r-phi> {drphi[m].mean():+8.3f} +- "
              f"{drphi[m].std()/np.sqrt(m.sum()):.3f} um"
              f"  <dz> {dz[m].mean()*1e4:+8.3f} um")

    if a.out:
        np.savez_compressed(a.out, rawdetid=common, subdet=sub, layer=lay,
                            rho=rho, phi=phi, dx=dx, dy=dy, dz=dz,
                            drphi=drphi / 1e4)
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()

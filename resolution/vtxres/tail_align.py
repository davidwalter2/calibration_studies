#!/usr/bin/env python3
"""TAIL study, hypothesis B quantified: THE MISALIGNMENT AS A HIT-NOISE FAMILY.

The maker itself exports the misalignment: `runtree` carries, per module,
`dx/dy/dz = r_ideal - r_aligned` in the global frame and `dtheta`, the angle
between the local x axis of the two surfaces (filled against
`globalGeometryIdeal`).  On the J/psi gun, which refits with
`useIdealGeometry=True`, all four are identically zero; on DY they are not.

What matters to a track fit is not |dr| but its projection on the MEASUREMENT
direction -- local x for every module, local y for the 2-D ones -- because
that is the direction in which the hit is displaced from the trajectory the
particle actually flew.  This script projects it, weights it by the modules
the candidates actually use, and tests whether the residual tail follows the
worst-misaligned hit a candidate carries.

usage:
  python3 tail_align.py --npz <tail/dy_bs_final.npz> \
      --files '<prod>/task_*/globalcor_*.root' [--ref <gun file>]
"""
import argparse, glob, os, sys
import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402

SUBDET = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}
# nominal single-hit resolutions, um, local x -- for scale only
HITRES = {0: 12.0, 1: 12.0, 2: 23.0, 3: 35.0, 4: 30.0, 5: 40.0}


def modules(fn):
    a = uproot.open(fn + ":runtree").arrays(
        ["iidx", "parmtype", "rawdetid", "subdet", "layer", "rho", "phi", "z",
         "dx", "dy", "dz", "dtheta", "lxx", "lxy", "lxz",
         "lyx", "lyy", "lyz"], library="np")
    m = a["parmtype"] == 0
    d = {k: v[m] for k, v in a.items()}
    dr = np.stack([d["dx"], d["dy"], d["dz"]], axis=1)
    lx = np.stack([d["lxx"], d["lxy"], d["lxz"]], axis=1)
    ly = np.stack([d["lyx"], d["lyy"], d["lyz"]], axis=1)
    d["dlocx"] = (dr * lx).sum(1)
    d["dlocy"] = (dr * ly).sum(1)
    d["dnorm"] = np.sqrt((dr ** 2).sum(1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(a.files))
    M = modules(files[0])
    print(f"# {len(M['iidx'])} modules in the runtree")
    if a.ref:
        R = modules(a.ref)
        print(f"# REFERENCE production: max |dlocx| = "
              f"{np.abs(R['dlocx']).max()*1e4:.3e} um, max |dtheta| = "
              f"{np.abs(R['dtheta']).max():.3e} rad  "
              f"(zero == the ideal geometry)")

    print("\n=== the misalignment in the MEASUREMENT direction, um "
          "(local x; `local y` only where it is measured)")
    hdr = (f"  {'subdet':<8s}{'N':>7s}{'rms locx':>11s}{'p99 locx':>11s}"
           f"{'max locx':>11s}{'rms locy':>11s}{'kurt locx':>11s}"
           f"{'sigma_hit':>11s}{'ratio':>8s}")
    print(hdr)
    for s in sorted(SUBDET):
        m = M["subdet"] == s
        if not m.any():
            continue
        vx = M["dlocx"][m] * 1e4
        vy = M["dlocy"][m] * 1e4
        ku = ((vx - vx.mean()) ** 4).mean() / vx.var() ** 2
        print(f"  {SUBDET[s]:<8s}{int(m.sum()):>7d}{vx.std():>11.3f}"
              f"{np.percentile(np.abs(vx),99):>11.3f}{np.abs(vx).max():>11.2f}"
              f"{vy.std():>11.3f}{ku:>11.2f}{HITRES[s]:>11.1f}"
              f"{vx.std()/HITRES[s]:>8.3f}")
    vx = M["dlocx"] * 1e4
    print(f"  {'ALL':<8s}{len(vx):>7d}{vx.std():>11.3f}"
          f"{np.percentile(np.abs(vx),99):>11.3f}{np.abs(vx).max():>11.2f}")
    print(f"  dtheta: rms {M['dtheta'].std()*1e3:.4f} mrad, p99 "
          f"{np.percentile(M['dtheta'],99)*1e3:.4f} mrad, max "
          f"{M['dtheta'].max()*1e3:.3f} mrad "
          f"(x 2 cm half-length -> {M['dtheta'].std()*2e4:.2f} um rms at the edge)")

    # ------- per candidate: the modules the fit actually put a hit on -------
    a0 = uproot.open(files[0] + ":runtree").arrays(["iidx", "parmtype"],
                                                   library="np")
    nmax = int(a0["iidx"].max()) + 1
    ptype = np.full(nmax, -1, np.int8)
    sub = np.full(nmax, -1, np.int8)
    dlx = np.full(nmax, np.nan)
    dly = np.full(nmax, np.nan)
    rho = np.full(nmax, np.nan)
    ptype[M["iidx"]] = 0
    sub[M["iidx"]] = M["subdet"]
    dlx[M["iidx"]] = M["dlocx"]
    dly[M["iidx"]] = M["dlocy"]
    rho[M["iidx"]] = M["rho"]
    isal0 = np.zeros(nmax, bool)
    isal0[a0["iidx"][a0["parmtype"] == 0]] = True

    rows = []
    for fn in files:
        d = uproot.open(fn + ":tree").arrays(
            ["event", "globalidxv"], library="np")
        for i in range(len(d["event"])):
            gi = np.asarray(d["globalidxv"][i], np.int64)
            gi = gi[isal0[gi]]                      # one entry per HIT module
            v = np.abs(dlx[gi]) * 1e4
            pix = sub[gi] <= 1
            inner = rho[gi] < 20.0
            rows.append((len(gi), np.nanmax(v), np.sqrt(np.nanmean(v ** 2)),
                         np.nanmax(v[pix]) if pix.any() else np.nan,
                         np.nanmax(v[inner]) if inner.any() else np.nan,
                         np.nanmin(rho[gi])))
    A = np.array(rows)
    print(f"\n# per candidate, over the modules it has a hit on: "
          f"{A.shape[0]} candidates, median {np.median(A[:,0]):.0f} hit modules")

    d = dict(np.load(a.npz, allow_pickle=False))
    assert len(d["run"]) == A.shape[0]
    base, chi2n, nl = TC.baseline(d, verbose=False)
    basenoz, _, _ = TC.baseline(d, max_abs_vtxz=0.0, verbose=False)
    cls, names, _ = GB.classify(d)
    isig = names.index("signal")
    z1, z2 = d["bsz"][:, 0], d["bsz"][:, 1]
    zv = np.asarray(d["vtxz"], float)
    sig = base & (cls == isig)
    signoz = basenoz & (cls == isig)
    dmaxall, drms, dmaxpix, dmaxin = A[:, 1], A[:, 2], A[:, 3], A[:, 4]

    print("\n=== does the tail follow the WORST-misaligned hit the candidate "
          "uses?   (|dlocx| in um; gen SIGNAL)")
    for nm, q, sel in (("max |dlocx| over all hits", dmaxall, sig),
                       ("max |dlocx| over PIXEL hits", dmaxpix, sig),
                       ("max |dlocx| at rho < 20 cm", dmaxin, sig),
                       ("rms |dlocx| over all hits", drms, sig)):
        e = np.nanpercentile(q[sel], [0, 25, 50, 75, 90, 100])
        print(f"\n  -- {nm}")
        hdr = (f"  {'quartile':<22s}{'N':>7s}{'range um':>18s}"
               f"{'P(|z_1|>3)':>22s}{'P(|z_1|>5)':>22s}{'Var z_1':>10s}")
        print(hdr)
        for i in range(len(e) - 1):
            b = sel & (q >= e[i]) & (q < e[i + 1] if i < len(e) - 2 else q <= e[i + 1])
            if b.sum() < 20:
                continue
            print(f"  {'':<22s}{int(b.sum()):>7d}{e[i]:>8.2f}..{e[i+1]:<8.2f}"
                  f"{TC.pm(int((np.abs(z1[b])>3).sum()), int(b.sum())):>22s}"
                  f"{TC.pm(int((np.abs(z1[b])>5).sum()), int(b.sum())):>22s}"
                  f"{np.var(z1[b]):>10.4f}")

    print("\n=== the z_v tail WITHOUT the |z_v| < 5 cut (gen SIGNAL, "
          f"{int(signoz.sum())} candidates)")
    TC.moments(zv[signoz], "z_v, no |z_v| cut")
    for nm, q in (("max |dlocx| all", dmaxall), ("max |dlocx| pixel", dmaxpix)):
        e = np.nanpercentile(q[signoz], [0, 50, 90, 100])
        for i in range(len(e) - 1):
            b = signoz & (q >= e[i]) & (q <= e[i + 1])
            if b.sum() < 20:
                continue
            print(f"  {nm:<20s} {e[i]:7.2f}..{e[i+1]:<7.2f} um  N {int(b.sum()):6d}"
                  f"   P(|z_v|>5) {TC.pm(int((np.abs(zv[b])>5).sum()), int(b.sum()))}"
                  f"   Var {np.var(zv[b]):.4f}")

    if a.out:
        np.savez_compressed(a.out, nmod=A[:, 0], dmaxall=dmaxall, drms=drms,
                            dmaxpix=dmaxpix, dmaxin=dmaxin, rhomin=A[:, 5],
                            base=base, basenoz=basenoz, cls=cls,
                            dlocx_mod=M["dlocx"], dlocy_mod=M["dlocy"],
                            sub_mod=M["subdet"], lay_mod=M["layer"],
                            rho_mod=M["rho"], dtheta_mod=M["dtheta"])
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()

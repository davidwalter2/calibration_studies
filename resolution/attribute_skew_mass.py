#!/usr/bin/env python3
"""The class-share attribution on the ACTUAL Z legs (the DY groups cache).

The gun said the charge-even odd moment runs -4.9 / -3.6 / +0.1 across the
three |eta| bands and that tracks whose curvature is carried by single-strip
(N1) clusters are more negatively skewed in all three -- at ~1 sigma each. This
runs the same split on `gzpairs_dyv2_n50.npz`, 487 742 Z candidates with their
per-hit class labels and per-hit influence weights, which is the sample the
mass fit actually uses.

Two differences from the track-level version, both forced by the data:
  * it is a two-track MASS cache, so there is no per-candidate charge and the
    charge split stays a track-level measurement;
  * the pull-normalisation artefact at mass level is `a` from sec. 0f.33,
    MEASURED at `a/(sigma/m) = 1.211` inclusively rather than the spec's
    `1 + vgf`. `x = z/(1 - a z)` uses the measured coefficient.

    python3 attribute_skew_mass.py
"""
import argparse

import numpy as np

PROBES = (0.05, 0.2)


def odd(z, u, w=None):
    return np.average(z * np.exp(-u * z * z), weights=w)


def model_odd_jensen(srel, fang, u):
    """The MODEL's own odd moment, leading (and dominant) term.

    At TRACK level the model's odd moment is ~0 (measured: 5e-5), so a raw data
    odd moment IS `data - model` there. At MASS level it is NOT: the model
    carries the exact Jensen mean shift `d_i = 1.5 s_i^2 m_i` with
    `s_i^2 = (sigma_i/m_i)^2 (1 + f_ang,i)`, i.e. a displacement of
    `d_i/sigma_i = 1.5 (sigma_i/m_i)(1 + f_ang,i)` in the standardized
    variable. For a unit-width density displaced by `mu`,
    `<z e^{-u z^2}> = mu/(1+2u)^{3/2}` to first order in `mu`, so

        model_odd_i = 1.5 (sigma_i/m_i)(1 + f_ang,i) / (1 + 2u)^{3/2}

    which at the endcap's `sigma/m = 0.018` and `u = 0.05` is **+23.4e-3** --
    the same size as everything measured here. It grows with `sigma/m`, and
    `sigma/m` is what the pT bins and the pixel-share split are largely
    selecting on. NOT included: the ionisation and radiative skews of the CF
    itself, which are negative and largest at low pT; the residual below is
    therefore `data - Jensen`, not the full `data - model`.
    """
    return 1.5 * srel * (1.0 + fang) / (1.0 + 2.0 * u) ** 1.5


def odd_err(z, u, nboot, rng, w=None):
    n = len(z)
    if n < 50:
        return np.nan
    b = np.empty(nboot)
    for i in range(nboot):
        k = rng.integers(0, n, n)
        b[i] = odd(z[k], u, None if w is None else w[k])
    return b.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/work/submit/david_w/ZMass/"
                    "calibration_studies/fullscale/runs/gzpairs_dyv2_n50.npz")
    ap.add_argument("--nboot", type=int, default=150)
    ap.add_argument("--zmax", type=float, default=8.0)
    ap.add_argument("--acoef", type=float, default=1.211,
                    help="a/(sigma/m), MEASURED in sec. 0f.33 (the spec's "
                         "1+vgf is 1.2625 and is high)")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)

    d = np.load(a.cache, allow_pickle=True)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mgen = d["mgen"].astype(np.float64) if "mgen" in d.files else None
    m = z * sig + mgen
    etal = np.maximum(np.abs(d["etap"].astype(np.float64)),
                      np.abs(d["etam"].astype(np.float64)))
    w = d["w"].astype(np.float64) if "w" in d.files else np.ones(len(z))
    cls = d["hit_cls"].astype(np.int64)
    hv = d["hit_v"].astype(np.float64)
    ptr = d["hit_ptr"].astype(np.int64)
    names = [str(s) for s in d["hit_classes"]]

    a_i = a.acoef * sig / m
    den = 1.0 - a_i * z
    x = z if a.raw else np.where(np.abs(den) > 1e-3, z / den, np.nan)
    ok = np.isfinite(x) & (np.abs(x) < a.zmax) & np.isfinite(w) & (sig / m < 0.10)

    ntr = len(z)
    cnt = np.diff(ptr)
    trk = np.repeat(np.arange(ntr), cnt)
    tot = np.bincount(trk, weights=hv, minlength=ntr)
    ncl = len(names)
    share = np.zeros((ntr, ncl))
    for c in range(ncl):
        mm = cls == c
        share[:, c] = np.bincount(trk[mm], weights=hv[mm], minlength=ntr)
    share /= np.maximum(tot, 1e-300)[:, None]
    idx = {n: i for i, n in enumerate(names)}
    nhit = cnt.astype(np.float64)
    ptlead = np.maximum(d["ptp"].astype(np.float64), d["ptm"].astype(np.float64))
    def grp(pref):
        cs = [i for n, i in idx.items() if n.startswith(pref)]
        return share[:, cs].sum(1) if cs else np.zeros(ntr)
    n1 = grp("str_N1"); pix = grp("pix"); strp = grp("str")
    pixx = grp("pix_x"); pixy = grp("pix_y")

    print(f"{a.cache.split('/')[-1]}: {ntr} candidates, {len(cls)} hits, "
          f"{int(ok.sum())} used")
    print(f"classes: {names}")
    print(f"influence-weighted shares: pixel {np.median(pix):.3f}, strip "
          f"{np.median(strp):.3f}, single-strip {np.median(n1):.3f}")
    print(f"a = {a.acoef} sigma/m: median {np.median(a_i):.5f}; using "
          f"{'RAW z' if a.raw else 'x = z/(1-a z)'}\n")

    fang = d["fang"].astype(np.float64) if "fang" in d.files else np.zeros(ntr)
    srel_all = sig / m

    def table(title, groups):
        print(title)
        hdr = (f"  {'cut':34s} {'n':>8s}" +
               "".join(f"{'u='+str(u)+': data / model / d-m':>34s}"
                       for u in PROBES))
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for lab, mk in groups:
            mk = mk & ok
            if mk.sum() < 500:
                print(f"  {lab:34s} {int(mk.sum()):8d}   (too few)"); continue
            row = f"  {lab:34s} {int(mk.sum()):8d}"
            for u in PROBES:
                dv = odd(x[mk], u, w[mk])
                de = odd_err(x[mk], u, a.nboot, rng, w[mk])
                mv = np.average(model_odd_jensen(srel_all[mk], fang[mk], u),
                                weights=w[mk])
                row += (f"  {1e3*dv:+8.2f}+-{1e3*de:4.2f} {1e3*mv:+8.2f} "
                        f"{1e3*(dv-mv):+8.2f}")
            print(row)
        print()

    BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"),
             (1.6, 3.0, "1.6-3.0")]
    table("A. per |eta| band", [(l, (etal >= lo) & (etal < hi))
                               for lo, hi, l in BANDS]
          + [("inclusive", np.ones(ntr, bool))])
    for lo, hi, lab in BANDS:
        b = (etal >= lo) & (etal < hi)
        if (b & ok).sum() < 5000:
            continue
        q = np.percentile(n1[b & ok], [50, 90])
        qp = np.percentile(pix[b & ok], [33, 67])
        table(f"B. hit composition, {lab}",
              [(f"{lab}, single-strip < {q[0]:.3f}", b & (n1 < q[0])),
               (f"{lab}, single-strip > {q[1]:.3f}", b & (n1 >= q[1])),
               (f"{lab}, pixel share < {qp[0]:.3f}", b & (pix < qp[0])),
               (f"{lab}, pixel share > {qp[1]:.3f}", b & (pix >= qp[1]))])

    # ---- CONTROLS: is the pixel-share split a proxy for pT or hit count? ----
    b = (etal >= 1.6) & (etal < 3.0)          # where the 4.7 sigma sits
    qp = np.percentile(pix[b & ok], [33, 67])
    qt = np.percentile(ptlead[b & ok], [33, 67])
    qn = np.percentile(nhit[b & ok], [33, 67])
    print("C. CONTROLS in 1.6-3.0: the pixel-share split AT FIXED pT and AT "
          "FIXED hit count.\n   If the 4.7 sigma is a proxy for either, it "
          "must collapse inside those slices.\n")
    rows = []
    for i, (tlo, thi) in enumerate([(0, qt[0]), (qt[0], qt[1]), (qt[1], 1e9)]):
        t = b & (ptlead >= tlo) & (ptlead < thi)
        rows += [(f"pT bin {i+1} [{tlo:.0f},{min(thi,999):.0f}], pix LOW",
                  t & (pix < qp[0])),
                 (f"pT bin {i+1} [{tlo:.0f},{min(thi,999):.0f}], pix HIGH",
                  t & (pix >= qp[1]))]
    table("C1. at fixed leading-muon pT", rows)
    rows = []
    for i, (nlo, nhi) in enumerate([(0, qn[0]), (qn[0], qn[1]), (qn[1], 1e9)]):
        t = b & (nhit >= nlo) & (nhit < nhi)
        rows += [(f"nhit bin {i+1} [{nlo:.0f},{min(nhi,999):.0f}], pix LOW",
                  t & (pix < qp[0])),
                 (f"nhit bin {i+1} [{nlo:.0f},{min(nhi,999):.0f}], pix HIGH",
                  t & (pix >= qp[1]))]
    table("C2. at fixed hit count", rows)

    # ---- the local coordinate: a forward-disk incidence/drift effect must
    # sit in ONE local coordinate, and on the tilted Phase-0 turbine blades
    # that is the drift direction, not both ----
    for lo, hi, lab in BANDS:
        b = (etal >= lo) & (etal < hi)
        if (b & ok).sum() < 5000:
            continue
        qx = np.percentile(pixx[b & ok], [33, 67])
        qy = np.percentile(pixy[b & ok], [33, 67])
        table(f"D. LOCAL COORDINATE, {lab}   (pix_x vs pix_y share)",
              [(f"{lab}, pix_X share < {qx[0]:.3f}", b & (pixx < qx[0])),
               (f"{lab}, pix_X share > {qx[1]:.3f}", b & (pixx >= qx[1])),
               (f"{lab}, pix_Y share < {qy[0]:.3f}", b & (pixy < qy[0])),
               (f"{lab}, pix_Y share > {qy[1]:.3f}", b & (pixy >= qy[1]))])


if __name__ == "__main__":
    main()

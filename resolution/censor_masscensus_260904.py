#!/usr/bin/env python3
"""TASK 2 (mass side) of the 2026-09-04 censoring test.

Uses `censoring_aux.py`'s output, which is the candidate pairs cache with the
selection columns rebuilt from the SAME trees in the SAME order and the
alignment PROVED (z bit-identical), plus the tree rows the cache DROPPED.

Sections
  1. counts        tree -> cache, and what the dropped rows look like
  2. the window    is the ALCARECO 2.95-3.25 GeV cut visible in Jpsitrk_mass
                   (the PRE-refit mass, which IS the cut variable)?
  3. the smearing  delta = m_refit - m_prefit, i.e. by how much the refit
                   moves a candidate across a sharp pre-refit window edge.
                   A(m) = Pr(delta in [m-hi, m-lo]) is the EFFECTIVE
                   post-refit acceptance -- the object a first-principles
                   truncation term needs, and the reason a sharp post-refit
                   window over-corrects.
  4. edge test     the odd moment <z e^{-u z^2}> in bins of the min daughter
                   momentum / pT.  A candidate at pT_min = 10 GeV is nowhere
                   near any reconstruction threshold; if the asymmetry is the
                   same there as at pT_min = 2 GeV it is not an acceptance
                   effect.
  5. quality       the same odd moment vs normchi2, niter cap and the
                   momentum-floor-clamp fingerprint `frozen`.
"""
import argparse, os
import numpy as np

MJ = 3.0969
PROBES = (0.05, 0.2, 1.0)
TRIMS = (1., 2., 5.)


def od(z, probes=PROBES):
    return [float(np.mean(z * np.exp(-u * z ** 2))) for u in probes]


def qline(v, name, qs=(0., .001, .01, .5, .99, .999, 1.)):
    v = v[np.isfinite(v)]
    if not len(v):
        return f"   {name:>16s}  (empty)"
    return (f"   {name:>16s}  " +
            "  ".join(f"q{q*100:g}%={np.quantile(v,q):.5g}" for q in qs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aux", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--win", type=float, nargs=2, default=None,
                   help="the production's PRE-refit mass window, if any")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    L = []

    def log(s=""):
        print(s, flush=True)
        L.append(s)

    d = np.load(a.aux)
    c = np.load(a.cache)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mg = c["eta"].astype(np.float64)
    m = z * sig + mg
    mtrk, mkin = d["mtrk"].astype(np.float64), d["mkin"].astype(np.float64)
    n = len(z)
    ntree = n + len(d["drop_z"])

    log(f"{'='*104}")
    log(f"# {a.label}")
    log(f"#   aux   {a.aux}")
    log(f"#   cache {a.cache}")
    log("")
    log("## 1. COUNTS")
    log(f"   tree rows {ntree}   cache candidates {n}   "
        f"DROPPED {ntree-n} = {100.*(ntree-n)/ntree:.4f} %")
    dz, ds = d["drop_z"].astype(np.float64), d["drop_sigma"].astype(np.float64)
    nbad = int((~np.isfinite(ds)).sum() + (ds <= 0).sum())
    log(f"   of the dropped: sigma_m NaN or <=0 {nbad} "
        f"({100.*nbad/max(ntree-n,1):.1f} % of the drops)")
    fin = np.isfinite(dz)
    if fin.any():
        log(f"   dropped rows WITH a finite pull: {int(fin.sum())};  "
            f"z: med {np.median(dz[fin]):+.4f}  "
            f"mean {dz[fin].mean():+.4f}  min {dz[fin].min():+.4g}  "
            f"max {dz[fin].max():+.4g}")
    dmk = d["drop_mkin"].astype(np.float64)
    dmt = d["drop_mtrk"].astype(np.float64)
    ok = np.isfinite(dmt) & (dmt > 0)
    if ok.any():
        log(f"   dropped rows' PRE-refit mass m_trk: med {np.median(dmt[ok]):.4f}"
            f"  q1 {np.quantile(dmt[ok],.01):.4f}  q99 {np.quantile(dmt[ok],.99):.4f}")
    log("   the analysis-level drop conditions are (cf_mass_likelihood.build_pairs_tt):")
    log("     sigma_m not finite or <= 0 ;  |m_gen - M| > 0.35 GeV ;  empty")
    log("     reseigidx ;  a per-block failure -- NONE of them is a cut on the")
    log("     OBSERVED mass, so they truncate the observed distribution only")
    log("     through their correlation with a pathological fit.")
    log("")

    log("## 2. THE PRODUCTION WINDOW, seen in the PRE-refit mass")
    log(qline(mtrk, "m_trk (pre-fit)"))
    log(qline(mkin, "m_kin (vtx seed)"))
    log(qline(m, "m (refit)"))
    if a.win:
        lo, hi = a.win
        out = (mtrk < lo) | (mtrk > hi)
        outp = (m < lo) | (m > hi)
        log(f"   window [{lo}, {hi}] GeV on the PRE-refit mass:")
        log(f"     m_trk outside: {int(out.sum())} = {100.*out.mean():.4f} %   "
            f"(0 % is the signature of the cut being ON m_trk)")
        log(f"     m_refit outside: {int(outp.sum())} = {100.*outp.mean():.4f} %  "
            f"-- the refit moves this fraction back across the edge")
        log(f"     window half-width / median sigma_m = "
            f"{0.5*(hi-lo)/np.median(sig[np.isfinite(sig)]):.2f}")
    log("")

    log("## 3. THE REFIT SMEARING delta = m_refit - m_prefit, and the")
    log("##    EFFECTIVE post-refit acceptance")
    dl = m - mtrk
    good = np.isfinite(dl) & (np.abs(dl) < 1.)
    log(f"   delta [MeV]: n {int(good.sum())}  mean {1e3*dl[good].mean():+.3f}  "
        f"med {1e3*np.median(dl[good]):+.3f}  rms {1e3*dl[good].std():.3f}  "
        f"q1 {1e3*np.quantile(dl[good],.01):+.3f}  "
        f"q99 {1e3*np.quantile(dl[good],.99):+.3f}")
    log(f"   |delta| > 10 MeV on {100.*np.mean(np.abs(dl[good])>0.010):.3f} %,"
        f"  > 30 MeV on {100.*np.mean(np.abs(dl[good])>0.030):.4f} %")
    if a.win:
        lo, hi = a.win
        dd = np.sort(dl[good])
        F = lambda x: np.searchsorted(dd, x) / len(dd)
        log("   A(m) = Pr(delta in [m-hi, m-lo]) from the empirical delta:")
        log(f"   {'m [GeV]':>10s}{'A(m)':>10s}   {'m [GeV]':>10s}{'A(m)':>10s}")
        grid = [lo - 0.06, lo - 0.03, lo - 0.01, lo, lo + 0.01, lo + 0.03,
                MJ, hi - 0.03, hi - 0.01, hi, hi + 0.01, hi + 0.03, hi + 0.06]
        A = [F(x - lo) - F(x - hi) for x in grid]
        for i in range(0, len(grid), 2):
            s = f"   {grid[i]:>10.4f}{A[i]:>10.4f}"
            if i + 1 < len(grid):
                s += f"   {grid[i+1]:>10.4f}{A[i+1]:>10.4f}"
            log(s)
        log("   A SHARP post-refit window is A = 1 inside / 0 outside; the "
            "measured A is")
        log("   smooth over ~+-30 MeV, i.e. ~1 sigma_m, so treating the cut as "
            "sharp in the")
        log("   refit mass OVER-corrects the tail it removes.")
    log("")

    log("## 4. ACCEPTANCE-EDGE TEST: the odd moment vs the min daughter momentum")
    ptmin, pmin = d["ptmin"].astype(np.float64), d["pmin"].astype(np.float64)
    for nm, v, edges in (("min mu pT [GeV]", ptmin, (0., 2., 3., 4., 5., 7., 10., 1e9)),
                         ("min mu |p| [GeV]", pmin, (0., 5., 8., 12., 20., 40., 1e9))):
        log(f"   {nm:<18s}{'n':>9}{'<sigma_m>':>11}"
            + "".join(f"{'<ze-'+format(u,'g')+'z2>':>14}" for u in PROBES)
            + "".join(f"{'<z>|z|<'+format(T,'g'):>13}" for T in TRIMS))
        for l_, h_ in zip(edges[:-1], edges[1:]):
            s_ = (v >= l_) & (v < h_) & np.isfinite(z) & (sig < 0.15)
            if s_.sum() < 200:
                continue
            zz = z[s_]
            log(f"   [{l_:>6.1f},{h_ if h_<1e8 else 999:>6.1f})"
                f"{int(s_.sum()):>9}{np.median(sig[s_]):>11.4f}"
                + "".join(f"{v_:>+14.5f}" for v_ in od(zz))
                + "".join(f"{zz[np.abs(zz)<T].mean():>+13.5f}" for T in TRIMS))
        log("")

    log("## 5. FIT-QUALITY / CLAMP DEPENDENCE (sigma_m < 0.15 GeV throughout)")
    base = np.isfinite(z) & (sig < 0.15)
    c2 = d["normchi2"].astype(np.float64)
    it = d["niter"].astype(np.float64)
    fz = d["frozen"].astype(np.float64) > 0.5
    dr = d["drmax"].astype(np.float64)
    log(f"   normchi2: med {np.median(c2[base]):.3f} q99 {np.quantile(c2[base],.99):.3f}"
        f"  frac>3 {np.mean(c2[base]>3):.4f}")
    log(f"   niter>=10 {np.mean(it[base]>=10):.4f}   frozen "
        f"{np.mean(fz[base]):.5f}   drmax q99 {np.quantile(dr[base],.99):.4f}")
    log(f"   {'cut':<24}{'n':>9}{'frac':>9}"
        + "".join(f"{'<ze-'+format(u,'g')+'z2>':>14}" for u in PROBES)
        + "".join(f"{'<z>|z|<'+format(T,'g'):>13}" for T in TRIMS))
    rows = [("(no cut, sane)", base)]
    for cc in (5., 3., 2., 1.5):
        rows.append((f"normchi2 < {cc:g}", base & (c2 < cc)))
    rows.append(("niter < 10", base & (it < 10)))
    rows.append(("not frozen", base & ~fz))
    rows.append(("drmax < 0.02", base & (dr < 0.02)))
    for nm, s_ in rows:
        if s_.sum() < 200:
            continue
        zz = z[s_]
        log(f"   {nm:<24}{int(s_.sum()):>9}{s_.sum()/base.sum():>9.5f}"
            + "".join(f"{v_:>+14.5f}" for v_ in od(zz))
            + "".join(f"{zz[np.abs(zz)<T].mean():>+13.5f}" for T in TRIMS))
    log("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

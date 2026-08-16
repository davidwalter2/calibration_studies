#!/usr/bin/env python3
"""Decision-gate checks for the CGF/resolution workstream (CRITIQUE_260813 pt 1).

Three items, three subcommands:

  median   The mode-vs-median CAVEAT (NOTES 2026-08-12 III).  The MEDIAN of the
           per-block loss-deviation density, per plane and per momentum, by
           EXACT FFT inversion of the block CF -- NOT from the saddlepoint,
           whose density is 2-8x wrong below the mode and up to 140x wrong in
           the power-law tail (NOTES_XXII_msrad section 4).  The
           Lugannani-Rice median is computed too, purely as a cross-check, and
           its error is quoted.  The same quantities are measured directly in
           Geant4 (cleanprop sim) so the prediction never has to be corrected
           by a model/G4 mode ratio taken at one momentum and assumed flat.

  dedx     The ds_req comparison of NOTES 2026-08-12 (III), redone against BOTH
           references (mode and median), with a bootstrap that carries the
           error on the response R as well as on the bias, and with the
           reference measured at four momenta instead of interpolated between
           two.

  bin61    The p ~ 6.1 GeV bin: binning, pairing, eta composition, statistics.

usage:
  python gate_checks.py median [--planes 0,9,18] [--models pt3,pt10,pt40,pt100]
  python gate_checks.py dedx
  python gate_checks.py bin61
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop"

# (label, model file, sim glob).  The sim/model pairing is ASSERTED on the
# detid sequence in every case -- regen_models.sh mis-targeted a model once
# already (cgf_phase0_validate docstring).
SAMPLES = {
    "pt3":   (f"{BASE}/model/model_mu_pt3_eta0.30.root",
              f"{BASE}/sim_260808tight_pt3_eta0.30_phi0.70/simstates_*.root"),
    "pt10":  (f"{BASE}/model/model_pt10_eta0.30_phi0.20.root",
              f"{BASE}/sim_260804_pt10_eta0.30_phi0.20/simstates_*.root"),
    "pt40":  (f"{BASE}/model/model_mu_pt40_eta0.30_phi0.70.root",
              f"{BASE}/sim_260808match_pt40_eta0.30_phi0.70/simstates_*.root"),
    "pt40b": (f"{BASE}/model/model_mu_pt40_eta0.30.root",
              f"{BASE}/sim_260806_pt40_eta0.30_phi0.50/simstates_*.root"),
    "pt100": (f"{BASE}/model/model_mu_pt100_eta0.30.root",
              f"{BASE}/sim_260806_pt100_eta0.30_phi0.10/simstates_*.root"),
    "pt10e10": (f"{BASE}/model/model_mu_pt10_eta1.00.root",
                f"{BASE}/sim_260806_pt10_eta1.00_phi0.10/simstates_*.root"),
    "pt10e16": (f"{BASE}/model/model_mu_pt10_eta1.60.root",
                f"{BASE}/sim_260806_pt10_eta1.60_phi0.10/simstates_*.root"),
}


# ====================================================================== part 1
def cdf_median(z, p, ret_diag=False):
    """Median of a density given on a (sorted, uniform) z grid.

    The block density has a 1/z^2 power-law tail, so a finite grid ALWAYS
    misses some mass.  Rather than pretend otherwise, the missing mass is
    measured and the median is bracketed two ways:
      * normalizing on the grid (implicitly spreading the missing mass in
        proportion to what is there), and
      * charging ALL the missing mass to the tail side (the truth, up to the
        mass beyond the grid on the light side, which is astronomically small).
    The two differ by << the grid spacing here; the spread is returned so the
    claim can be checked rather than asserted.

    The CDF is monotone by construction, so interpolating it IS safe -- unlike
    psi, which is not monotone and must never be inverted this way.
    """
    dz = z[1] - z[0]
    pc = np.maximum(p, 0.0)                       # inversion noise -> tiny <0
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pc[1:] + pc[:-1]) * dz)])
    tot = c[-1]
    neg = float(np.sum(np.minimum(p, 0.0)) * dz)  # size of the noise, for QA

    def _cross(cc, target):
        i = int(np.searchsorted(cc, target))
        if i <= 0 or i >= len(cc):
            return np.nan
        f = (target - cc[i - 1]) / max(cc[i] - cc[i - 1], 1e-300)
        return float(z[i - 1] + f * dz)

    m_norm = _cross(c, 0.5 * tot)
    miss = 1.0 - tot
    # heavy tail on the low side => the missing mass sits below the grid
    m_low = _cross(c, 0.5 - miss) if miss < 0.5 else np.nan
    m_high = _cross(c, 0.5)
    if not ret_diag:
        return m_norm
    return m_norm, dict(mass=tot, miss=miss, negmass=neg,
                        med_lo=min(m_low, m_high), med_hi=max(m_low, m_high))


def lr_cdf_curve(blk, thetas, block_cgf_derivs):
    """(x, F_LR) along the saddlepoint curve.

    Lugannani-Rice:  F(x) = Phi(w) + phi(w) (1/w - 1/u),
        w = sgn(th) sqrt(2(th K'(th) - K(th))),   u = th sqrt(K''(th)),
        x = K'(th).
    Parameterizing by theta (rather than solving K'(th) = x) means no root
    finding, and x(th) is monotone because K'' > 0, so the returned curve is
    ordered.  th = 0 is excluded: the formula is 0/0 there and the limit
    -lambda3/6 is itself the object under test.
    """
    K, K1, K2 = block_cgf_derivs(blk, thetas, order=2)[:3]
    ok = np.isfinite(K) & np.isfinite(K1) & np.isfinite(K2) & (K2 > 0)
    arg = 2.0 * (thetas * K1 - K)
    ok &= np.isfinite(arg) & (arg > 0) & (thetas != 0.0)
    w = np.where(ok, np.sign(thetas) * np.sqrt(np.abs(arg)), np.nan)
    u = np.where(ok, thetas * np.sqrt(np.abs(K2)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / w - 1.0 / u
        F = ndtr(w) + np.exp(-0.5 * w ** 2) / np.sqrt(2 * np.pi) * d
    # ROUND-OFF MASK.  1/w and 1/u both diverge like 1/(theta sqrt(kappa2)) as
    # theta -> 0 while their difference tends to lambda3/6, so the term is a
    # textbook catastrophic cancellation; and BELOW it there is a second, worse
    # problem -- K(theta) itself has an absolute round-off floor (~4e-13 here,
    # from summing ~200 step contributions), so 2(theta K' - K) stops being
    # kappa2 theta^2 at all.  MEASURED, not assumed: rel is the departure of K
    # from its own quadratic limit; it must be small at small theta and grow
    # with theta.  Wherever it instead grows as theta -> 0, K is round-off.
    kap2 = block_cgf_derivs_at0(blk)[0]
    # MEASURE the absolute round-off floor of K: at the smallest |theta| on the
    # grid the true K = kappa2 theta^2 / 2 is astronomically below it, so what
    # is left there IS the noise.
    a = np.abs(thetas)
    tiny = a <= np.quantile(a[a > 0], 0.05)
    knoise = float(np.median(np.abs(K[tiny]))) if tiny.any() else 0.0
    # error on d = 1/w - 1/u is ~ 1.5 knoise / arg^{3/2}; require it < 1e-3
    argmin = (1500.0 * max(knoise, 1e-300)) ** (2.0 / 3.0)
    x = np.where(ok, K1, np.nan)
    good = ok & np.isfinite(F) & np.isfinite(x) & (arg > argmin)
    return x[good], F[good], thetas[good], np.full(int(good.sum()), argmin)


def block_cgf_derivs_at0(blk):
    from cgf_channels import block_cgf_derivs as _d
    K, K1, K2, K3 = _d(blk, np.array([0.0]), order=3)
    return float(K2[0]), float(K3[0])


def lr_median(blk, block_cgf_derivs, block_theta_grid, n=20001, reltol=0.05):
    """Median from the Lugannani-Rice CDF, and diagnostics.

    Returns (median, info).  F is NOT assumed monotone -- it is not, and near
    theta = 0 it is not even bounded before the round-off mask is applied.
    Every 0.5 crossing on the numerically clean part of the curve is found and
    reported; the median is quoted only when exactly one survives.
    """
    th = block_theta_grid(blk, n=n)
    x, F, thg, rel = lr_cdf_curve(blk, th, block_cgf_derivs)
    keep = np.ones(len(thg), dtype=bool)
    o = np.argsort(x[keep])
    xk, Fk = x[keep][o], F[keep][o]
    s = np.sign(Fk - 0.5)
    cross = np.where(np.diff(s) != 0)[0]
    roots = []
    for i in cross:
        f0, f1 = Fk[i] - 0.5, Fk[i + 1] - 0.5
        if f1 == f0:
            continue
        roots.append(float(xk[i] - f0 * (xk[i + 1] - xk[i]) / (f1 - f0)))
    # F at the MEAN (x = 0) is the LR value the asymptotics are supposed to
    # deliver; |F| > 1 there is the diagnostic that the expansion has failed
    j = int(np.argmin(np.abs(xk)))
    info = dict(nroot=len(roots), roots=roots, F_at_mean=float(Fk[j]),
                Fmin=float(np.min(Fk)), Fmax=float(np.max(Fk)),
                nkept=int(keep.sum()), ntot=int(len(x)),
                monotone=bool(np.all(np.diff(Fk) >= -1e-12)))
    if len(roots) == 1:
        return roots[0], info
    if len(roots) > 1:
        return roots[-1], info
    return np.nan, info


def run_median(args):
    from cf_propagation_test import (load_model, load_sim, model_variance,
                                     FUNCTIONALS, SIM_BRANCH, REF_BRANCH)
    from cgf_channels import (collect_block, block_cgf_derivs, block_theta_grid,
                              block_mode, exact_density, mode_from_density,
                              auto_tau, invert_cf, block_cf_exponent)

    avec = FUNCTIONALS["qop"]
    labels = args.models.split(",")
    refs = []
    print("EXACT median vs mode of the per-block q/p residual density.")
    print("Route: FFT inversion of the full block CF (ioni+ms+rad) on a "
          "tau grid matched per block")
    print("       (auto_tau, lncut = -60).  The saddlepoint density is NOT "
          "used for the median.\n")

    for lab in labels:
        mpath, spath = SAMPLES[lab]
        legs = load_model(mpath)
        nl = len(legs)
        planes = ([int(x) for x in args.planes.split(",")] if args.planes
                  else [0, nl // 2, nl - 1])
        planes = [min(k, nl - 1) for k in planes]
        p0 = legs[0]["refp"]
        dE = 1e3 * (legs[0]["refp"] - legs[-1]["refp"])
        print("=" * 108)
        print(f"{lab}: {os.path.basename(mpath)}  {nl} planes, p = {p0:.3f} GeV, "
              f"reference loss over the ladder = {dE:.3f} MeV")
        print("=" * 108)

        sim = None
        if not args.nosim:
            try:
                sim = load_sim(spath, acceptance="perplane")
                sd, ld = sim["detid"], np.array([l["detid"] for l in legs])
                if len(sd) != len(ld) or not np.all(sd == ld):
                    print(f"  [sim/model detid sequence MISMATCH "
                          f"({len(sd)} vs {len(ld)}) -- G4 columns suppressed]")
                    sim = None
            except Exception as e:                       # noqa: BLE001
                print(f"  [no sim: {e}]")
                sim = None

        hdr = (f"{'pl':>3} {'r[cm]':>6} {'sigma':>10} | {'mode':>8} {'median':>8} "
               f"{'med/mode':>8} | {'LRmed':>8} {'LR-ex':>7} {'LR/ex':>6} | "
               f"{'mass':>9} {'dmed':>8}")
        if sim is not None:
            hdr += f" | {'G4mode':>7} {'G4med':>7} {'n':>7}"
        print(hdr)
        print("-" * len(hdr))
        rows = []
        for k in planes:
            var, _, _ = model_variance(legs, k, avec)
            sig = float(np.sqrt(var))
            tau = auto_tau(legs, k, avec, sig)
            S = block_cf_exponent(legs, k, avec, sig, tau)
            z, p = invert_cf(S, tau, npad=args.npad, nt=args.nt, deriv=False)
            md = mode_from_density(z, p)
            me, di = cdf_median(z, p, ret_diag=True)
            blk = collect_block(legs, k, avec, sig)
            lr, lri = lr_median(blk, block_cgf_derivs, block_theta_grid)
            g4mode = g4med = np.nan
            g4mi = {}
            ng4 = 0
            if sim is not None:
                good = (sim["valid"][:, k]
                        & np.isfinite(sim[SIM_BRANCH["qop"]][:, k]))
                d = sim[SIM_BRANCH["qop"]][:, k][good] - legs[k][REF_BRANCH["qop"]]
                zz = d / sig
                ng4 = int(good.sum())
                g4med = float(np.median(zz))
                g4mode, g4mi = sample_mode(zz, nboot=args.nboot, ret_scan=True)
            rr = float(np.nanmedian(sim["globr"][:, k])) if sim is not None else np.nan
            line = (f"{k:3d} {rr:6.1f} {sig:10.3e} | {md:8.4f} {me:8.4f} "
                    f"{me/md:8.4f} | {lr:8.4f} {lr-me:+7.4f} "
                    f"{lr/me:6.3f} | {di['mass']:9.6f} "
                    f"{di['med_hi']-di['med_lo']:8.1e}")
            if sim is not None:
                line += f" | {g4mode:7.3f} {g4med:7.3f} {ng4:7d}"
            print(line)
            rows.append(dict(plane=k, sigma=sig, mode=md, median=me, lr=lr,
                             lrinfo=lri, g4mode=g4mode, g4med=g4med,
                             g4mi=g4mi, mass=di["mass"], n=ng4))

        # conversion to the quantity the ds_req comparison needs
        print()
        print(f"{'pl':>3} | {'mode':>9} {'median':>9}  (dp/p, e-4) | "
              f"{'mode':>8} {'median':>8}  (MeV) | {'mode/dE':>8} {'med/dE':>8}"
              f" | {'G4mode/dE':>10} {'G4med/dE':>9}")
        print("-" * 108)
        for r in rows:
            k = r["plane"]
            q = abs(legs[k]["refqop"])
            pk = legs[k]["refp"]
            f = r["sigma"] / q                      # z -> dp/p
            g = f * pk * 1e3                        # z -> MeV
            print(f"{k:3d} | {r['mode']*f*1e4:9.2f} {r['median']*f*1e4:9.2f}"
                  f"               | {r['mode']*g:8.3f} {r['median']*g:8.3f}"
                  f"          | {r['mode']*g/dE:8.4f} {r['median']*g/dE:8.4f}"
                  f" | {r['g4mode']*g/dE:10.4f} {r['g4med']*g/dE:9.4f}")
        print()
        for r in rows:
            i = r["lrinfo"]
            print(f"  LR diag plane {r['plane']:2d}: {i['nroot']} clean "
                  f"crossing(s) of F = 0.5 ({i['nkept']}/{i['ntot']} theta "
                  f"points survive the cancellation mask), F(mean) = "
                  f"{i['F_at_mean']:+.3f}, F in [{i['Fmin']:+.3f}, "
                  f"{i['Fmax']:+.3f}]")
        for r in rows:
            g = r.get("g4mi") or {}
            if g:
                sc = g["scan"]
                print(f"  G4 mode plane {r['plane']:2d}: "
                      + " ".join(f"{k}={v:.3f}" for k, v in sc.items())
                      + f"  spread {g['spread']:.3f}  boot {g['boot']:.3f}")
        r = rows[-1]                                    # outermost plane
        k = r["plane"]
        g = r["sigma"] * legs[k]["refp"] * 1e3 / abs(legs[k]["refqop"])
        gi = r.get("g4mi") or {}
        refs.append(dict(label=lab, p=p0, dE=dE, plane=k, zmev=g,
                         mode=r["mode"], median=r["median"], lr=r["lr"],
                         g4mode=r["g4mode"], g4med=r["g4med"],
                         g4mode_spread=gi.get("spread", np.nan),
                         g4mode_boot=gi.get("boot", np.nan),
                         f_mode=r["mode"] * g / dE, f_med=r["median"] * g / dE,
                         f_g4mode=r["g4mode"] * g / dE,
                         f_g4med=r["g4med"] * g / dE))
        print()
    if args.refs:
        _write_refs(args.refs, refs)


def _write_refs(path, rows):
    import json
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n[reference table written to {path}]")


def sample_mode(z, smooth=0.15, nboot=0, rng=None, ret_scan=False):
    """Mode of a SAMPLE, by smoothing a fine histogram.

    The peak of the G4 residual density at the outer planes is very flat (it
    varies by ~3 % over 2 z units at pT = 3, plane 18), so an unsmoothed
    histogram mode is a Poisson coin-flip between bins: the same 200k sample
    returns 3.23 with 0.05-wide bins and 3.33 with 0.2-wide bins.  A fixed
    fine binning (0.05 z) plus an explicit Gaussian smoothing scale makes the
    estimator reproducible, and the scan over that scale plus the bootstrap
    say how well the mode is determined AT ALL -- which is the number that
    actually matters here, since the mode is one of the two candidate
    references the gate turns on.
    """
    z = np.asarray(z)
    z = z[np.isfinite(z)]
    if len(z) < 2000:
        return (np.nan, {}) if ret_scan else np.nan
    from scipy.ndimage import gaussian_filter1d
    lo, hi = np.quantile(z, [0.02, 0.995])
    pad = 0.3 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    bw = (hi - lo) / 2000.0

    def _m(zz, s):
        h, e = np.histogram(zz, bins=2000, range=(lo, hi))
        hs = gaussian_filter1d(h.astype(float), s / bw)
        c = 0.5 * (e[1:] + e[:-1])
        i = int(np.argmax(hs))
        if 0 < i < len(hs) - 1:
            y0, y1, y2 = hs[i - 1], hs[i], hs[i + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                return float(c[i] - 0.5 * bw * (y2 - y0) / den)
        return float(c[i])

    # smoothing is quoted in z units, i.e. in units of the propagator's own
    # sigma, so the same number means the same thing at every plane and every
    # momentum -- that is the point of standardizing by sigma in the first place
    m = _m(z, smooth)
    if not ret_scan:
        return m
    scan = {f"s{f:g}": _m(z, f) for f in (0.05, 0.1, 0.15, 0.3, 0.6)}
    err = np.nan
    if nboot:
        rng = rng or np.random.default_rng(3)
        bs = [_m(z[rng.integers(0, len(z), len(z))], smooth)
              for _ in range(nboot)]
        err = float(np.std(bs))
    return m, dict(scan=scan, boot=err,
                   spread=float(max(scan.values()) - min(scan.values())))


def kde_mode(z):
    return sample_mode(z)


# ====================================================================== part 2
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
CONFIGS = [("jpsigun_ul16", 1.000), ("jpsigun_dedx096", 0.960),
           ("jpsigun_dedx093", 0.930), ("jpsigun_dedx887", 0.887)]
BRANCHES = ["run", "lumi", "event",
            "Muplus_pt", "Muplus_eta", "Muplus_phi",
            "Muminus_pt", "Muminus_eta", "Muminus_phi",
            "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
            "Muplus_nvalid", "Muminus_nvalid",
            "Muplus_nvalidpixel", "Muminus_nvalidpixel",
            "Jpsi_mass", "Jpsigen_mass"]


def load_scan(nfiles=0, cache=""):
    """Key-matched candidates common to all four dE/dx configurations."""
    import glob
    import uproot
    if cache and os.path.exists(cache):
        z = np.load(cache)
        scales = sorted({float(k.split("|")[0]) for k in z.files}, reverse=True)
        out = {s: {k.split("|")[1]: z[k] for k in z.files
                   if float(k.split("|")[0]) == s} for s in scales}
        print(f"  [scan loaded from cache {cache}: "
              f"{len(out[scales[0]]['run'])} matched candidates]")
        return out, scales
    data = {}
    for tag, s in CONFIGS:
        fs = sorted(glob.glob(f"{CEPH}/resolution_trackres_{tag}/task_*/globalcor_0.root"))
        if nfiles:
            fs = fs[:nfiles]
        cols = {k: [] for k in BRANCHES}
        for fn in fs:
            try:
                a = uproot.open(fn)["tree"].arrays(BRANCHES, library="np")
            except Exception:                            # noqa: BLE001
                continue
            for k in BRANCHES:
                cols[k].append(np.asarray(a[k], dtype=np.float64))
        d = {k: np.concatenate(v) for k, v in cols.items()}
        d["key"] = np.array([f"{int(r)}:{int(l)}:{int(e)}:{gp:.5f}"
                             for r, l, e, gp in zip(d["run"], d["lumi"],
                                                    d["event"],
                                                    d["Muplusgen_pt"])])
        data[s] = d
        print(f"  loaded {tag:20s} s={s:.3f}  n={len(d['key'])}  ({len(fs)} files)")
    scales = sorted(data, reverse=True)
    common = set(data[scales[0]]["key"])
    for s in scales[1:]:
        common &= set(data[s]["key"])
    common = np.array(sorted(common))
    idx = {}
    for s in scales:
        pos = {k: i for i, k in enumerate(data[s]["key"])}
        idx[s] = np.array([pos[k] for k in common])
    out = {s: {k: v[idx[s]] for k, v in data[s].items() if k != "key"}
           for s in scales}
    print(f"  key-matched candidates common to all {len(scales)}: {len(common)}")
    if cache:
        np.savez_compressed(cache, **{f"{s}|{k}": v for s in scales
                                      for k, v in out[s].items()})
    return out, scales


def scan_observable(m, mode="pair"):
    """(value, momentum, eta, extra) per ENTRY, where an entry is a candidate
    (mode='pair') or a single leg (mode='leg').

    The pair observable is the one the earlier scan used; the leg observable
    exists because the pair one bins on the HARMONIC momentum of two legs,
    which is not the momentum either leg has, and which mixes (pT, eta) in a
    way that can manufacture structure in a p bin.
    """
    pp = m["Muplus_pt"] * np.cosh(m["Muplus_eta"])
    pm = m["Muminus_pt"] * np.cosh(m["Muminus_eta"])
    gp = m["Muplusgen_pt"] * np.cosh(m["Muplusgen_eta"])
    gm = m["Muminusgen_pt"] * np.cosh(m["Muminusgen_eta"])
    if mode == "pair":
        v = 0.5 * ((pp - gp) / gp + (pm - gm) / gm)
        p = 2.0 / (1.0 / gp + 1.0 / gm)
        eta = 0.5 * (np.abs(m["Muplusgen_eta"]) + np.abs(m["Muminusgen_eta"]))
        pt = 0.5 * (m["Muplusgen_pt"] + m["Muminusgen_pt"])
    else:
        v = np.concatenate([(pp - gp) / gp, (pm - gm) / gm])
        p = np.concatenate([gp, gm])
        eta = np.abs(np.concatenate([m["Muplusgen_eta"], m["Muminusgen_eta"]]))
        pt = np.concatenate([m["Muplusgen_pt"], m["Muminusgen_pt"]])
    grp = (np.arange(len(m["run"])) if mode == "pair"
           else np.tile(np.arange(len(m["run"])), 2))
    return v, p, eta, pt, grp


def ds_req_bins(vals, scales, mom, sel, edges, nboot=400, seed=11, group=None):
    """(pmed, n, bias1, R, ds_req, err) per bin, with a PAIRED bootstrap.

    The earlier version propagated only the error on bias(s=1) and treated R
    as exact.  R is measured on the SAME candidates, so the two are correlated;
    resampling the candidate index once per replica and recomputing both is
    the only way to get the error on their ratio right.

    `group` (one id per entry) makes the bootstrap resample GROUPS rather than
    entries.  It must be used for the single-leg observable: the two legs of a
    J/psi candidate are correlated through the common vertex and mass
    constraint of the two-track CVH fit, so resampling legs independently
    understates the error.
    """
    rng = np.random.default_rng(seed)
    sarr = np.array(scales)
    out = []
    for i in range(len(edges) - 1):
        s = sel & (mom >= edges[i]) & (mom < edges[i + 1])
        n = int(s.sum())
        if n < 200:
            continue
        idx = np.where(s)[0]
        V = np.stack([vals[sc][idx] for sc in scales])          # (nscale, n)

        def _one(cols):
            ms = np.median(V[:, cols], axis=1)
            R = np.polyfit(sarr, ms, 1)[0]
            b1 = ms[np.argmin(np.abs(sarr - 1.0))]
            return b1, R, (b1 / R if R != 0 else np.nan)

        b1, R, dsr = _one(np.arange(n))
        if group is None:
            draws = (rng.integers(0, n, n) for _ in range(nboot))
        else:
            g = group[idx]
            uq, inv = np.unique(g, return_inverse=True)
            members = [np.where(inv == j)[0] for j in range(len(uq))]
            ng = len(uq)
            draws = (np.concatenate([members[j] for j in
                                     rng.integers(0, ng, ng)])
                     for _ in range(nboot))
        bs = np.array([_one(c) for c in draws])
        out.append(dict(p=float(np.median(mom[idx])), n=n, bias=b1, R=R,
                        ds=dsr, ds_err=float(np.std(bs[:, 2])),
                        bias_err=float(np.std(bs[:, 0])),
                        R_err=float(np.std(bs[:, 1]))))
    return out


def _ref_interp(refs, key, p):
    """Log-p interpolation of a reference fraction, CLAMPED at the anchors.

    Extrapolating a log-p straight line below 3 GeV or above 100 GeV is not
    supported by anything, so it is not done: outside the anchor range the
    nearest anchor is returned and the fact is visible in the table.
    """
    pa = np.array([r["p"] for r in refs])
    fa = np.array([r[key] for r in refs])
    o = np.argsort(pa)
    return float(np.interp(np.log(p), np.log(pa[o]), fa[o]))


def run_dedx(args):
    import json
    refs = json.load(open(args.refs))
    refs = [r for r in refs if r["label"] in args.use.split(",")]
    print("REFERENCE TABLE (cleanprop, outermost plane, q/p functional):")
    print(f"{'label':>8} {'p':>8} {'dE MeV':>8} | {'mode':>8} {'median':>8} "
          f"{'G4mode':>8} {'G4med':>8} | {'f_mode':>7} {'f_med':>7} "
          f"{'f_G4mode':>8} {'f_G4med':>8}")
    for r in sorted(refs, key=lambda x: x["p"]):
        print(f"{r['label']:>8} {r['p']:8.3f} {r['dE']:8.3f} | {r['mode']:8.3f} "
              f"{r['median']:8.3f} {r['g4mode']:8.3f} {r['g4med']:8.3f} | "
              f"{r['f_mode']:7.4f} {r['f_med']:7.4f} {r['f_g4mode']:8.4f} "
              f"{r['f_g4med']:8.4f}")
    print("\n  f_X = (shift in MeV) / (reference energy loss over the ladder); "
          "this is what a\n  dE/dx SCALE change of that size would remove, so "
          "it is directly comparable to ds_req.\n")

    data, scales = load_scan(args.nfiles, cache=args.cache)
    for obs in args.obs.split(","):
        vals, mom, eta, pt, gid = {}, None, None, None, None
        for s in scales:
            v, p, e, t, grp = scan_observable(data[s], obs)
            vals[s] = v
            if mom is None:
                mom, eta, pt, gid = p, e, t, grp
        ok = np.ones(len(mom), dtype=bool)
        for s in scales:
            ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
        ok &= np.isfinite(mom) & (mom > 0)
        edges = np.quantile(mom[ok], np.linspace(0, 1, args.nbin + 1))
        rows = ds_req_bins(vals, scales, mom, ok, edges, nboot=args.nboot, group=gid)
        print("=" * 112)
        print(f"observable = {obs};  {args.nbin} quantile bins;  "
              f"{int(ok.sum())} entries")
        print("=" * 112)
        print(f"{'<p>':>7} {'n':>7} {'bias e-4':>14} {'R e-4':>9} "
              f"{'ds_req':>15} | {'mode ref':>9} {'pull':>6} | "
              f"{'med ref':>9} {'pull':>6}")
        print("-" * 112)
        chi2 = {"mode": 0.0, "med": 0.0}
        for r in rows:
            fm = _ref_interp(refs, "f_g4mode", r["p"])
            fd = _ref_interp(refs, "f_g4med", r["p"])
            pm = (r["ds"] - fm) / r["ds_err"]
            pd = (r["ds"] - fd) / r["ds_err"]
            chi2["mode"] += pm ** 2
            chi2["med"] += pd ** 2
            print(f"{r['p']:7.2f} {r['n']:7d} {r['bias']*1e4:+8.2f}"
                  f"+-{r['bias_err']*1e4:4.2f} {r['R']*1e4:9.1f} "
                  f"{r['ds']:8.3f}+-{r['ds_err']:5.3f} | {fm:9.4f} "
                  f"{pm:+6.1f} | {fd:9.4f} {pd:+6.1f}")
        nb = len(rows)
        print("-" * 112)
        print(f"  chi2 vs G4 MODE reference   = {chi2['mode']:8.1f} / {nb}")
        print(f"  chi2 vs G4 MEDIAN reference = {chi2['med']:8.1f} / {nb}")
        pref = "MODE" if chi2["mode"] < chi2["med"] else "MEDIAN"
        print(f"  -> the data prefer the {pref} reference "
              f"(ratio {max(chi2.values())/max(min(chi2.values()),1e-9):.1f}x)")
        print()


# ====================================================================== part 3
def run_bin61(args):
    """What is the p ~ 6.1 GeV bin?

    Candidate explanations, each with a test that can fail:
      A. binning artefact          -> vary nbin and the binning variable
      B. pairing artefact          -> per-LEG observable, binned in the leg's p
      C. eta composition           -> the p_harm bins mix (pT, eta) differently
      D. statistics                -> bootstrap + a split-half check
      E. selection/acceptance      -> nvalid / pixel-hit composition per bin
    """
    data, scales = load_scan(args.nfiles, cache=args.cache)
    import json
    refs = json.load(open(args.refs))

    for obs in ("pair", "leg"):
        vals, mom, eta, pt, gid = {}, None, None, None, None
        for s in scales:
            v, p, e, t, grp = scan_observable(data[s], obs)
            vals[s] = v
            if mom is None:
                mom, eta, pt, gid = p, e, t, grp
        ok = np.ones(len(mom), dtype=bool)
        for s in scales:
            ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
        ok &= np.isfinite(mom) & (mom > 0)

        print("=" * 104)
        print(f"A/B.  binning and pairing:  observable = {obs}")
        print("=" * 104)
        for nb in (3, 4, 5, 6, 8, 10):
            edges = np.quantile(mom[ok], np.linspace(0, 1, nb + 1))
            rows = ds_req_bins(vals, scales, mom, ok, edges, nboot=args.nboot, group=gid)
            print(f"  nbin={nb:2d}: " + "  ".join(
                f"p={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
                for r in rows))
        # fixed (not quantile) edges, so the bin CONTENT is not tied to nbin
        for eds in ([2, 4, 5, 7, 9, 12, 20, 60], [2, 5, 8, 12, 20, 60]):
            rows = ds_req_bins(vals, scales, mom, ok, np.array(eds, float),
                               nboot=args.nboot, group=gid)
            print(f"  fixed {eds}: " + "  ".join(
                f"p={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
                for r in rows))
        print()

    # ---- C: what IS in the 6.1 bin -------------------------------------
    vals, mom, eta, pt, gid = {}, None, None, None, None
    for s in scales:
        v, p, e, t, grp = scan_observable(data[s], "pair")
        vals[s] = v
        if mom is None:
            mom, eta, pt, gid = p, e, t, grp
    ok = np.ones(len(mom), dtype=bool)
    for s in scales:
        ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
    ok &= np.isfinite(mom) & (mom > 0)
    edges = np.quantile(mom[ok], np.linspace(0, 1, 6))
    print("=" * 104)
    print("C.  composition of the five quantile bins (pair observable)")
    print("=" * 104)
    print(f"{'<p>':>7} {'n':>7} {'<pt>':>7} {'pt 16-84%':>16} {'<|eta|>':>8} "
          f"{'|eta| 16-84%':>16} {'nvalid':>7} {'npix':>6}")
    m = data[1.000]
    nval = 0.5 * (m["Muplus_nvalid"] + m["Muminus_nvalid"])
    npix = 0.5 * (m["Muplus_nvalidpixel"] + m["Muminus_nvalidpixel"])
    for i in range(len(edges) - 1):
        s = ok & (mom >= edges[i]) & (mom < edges[i + 1])
        q = np.quantile(pt[s], [0.16, 0.84])
        qe = np.quantile(eta[s], [0.16, 0.84])
        print(f"{np.median(mom[s]):7.2f} {int(s.sum()):7d} "
              f"{np.mean(pt[s]):7.2f} [{q[0]:6.2f},{q[1]:6.2f}] "
              f"{np.mean(eta[s]):8.3f} [{qe[0]:6.2f},{qe[1]:6.2f}] "
              f"{np.mean(nval[s]):7.2f} {np.mean(npix[s]):6.2f}")

    # ---- C2: ds_req in (p, |eta|) cells --------------------------------
    print()
    print("=" * 104)
    print("C2.  ds_req split by |eta| -- the cleanprop reference is an "
          "eta = 0.30 ray, and the")
    print("     tracker material per unit path is strongly eta dependent.")
    print("=" * 104)
    ecuts = [(0.0, 0.8), (0.8, 1.4), (1.4, 3.0)]
    for lo, hi in ecuts:
        sel = ok & (eta >= lo) & (eta < hi)
        if sel.sum() < 2000:
            print(f"  |eta| in [{lo}, {hi}): only {int(sel.sum())} -- skipped")
            continue
        eds = np.quantile(mom[sel], np.linspace(0, 1, 5))
        rows = ds_req_bins(vals, scales, mom, sel, eds, nboot=args.nboot, group=gid)
        print(f"  |eta| in [{lo},{hi}) n={int(sel.sum()):6d}: " + "  ".join(
            f"p={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
            for r in rows))

    # ---- D: split-half stability ---------------------------------------
    print()
    print("=" * 104)
    print("D.  split-half stability of the 5-bin quantile result "
          "(even/odd candidate index)")
    print("=" * 104)
    for name, half in (("even", np.arange(len(mom)) % 2 == 0),
                       ("odd", np.arange(len(mom)) % 2 == 1)):
        rows = ds_req_bins(vals, scales, mom, ok & half, edges,
                           nboot=args.nboot, group=gid)
        print(f"  {name:4s}: " + "  ".join(
            f"p={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
            for r in rows))

    # ---- E: is it the RESPONSE or the BIAS that is odd? -----------------
    print()
    print("=" * 104)
    print("E.  ds_req = bias/R is a RATIO.  Which factor carries the "
          "non-monotonicity?")
    print("=" * 104)
    rows = ds_req_bins(vals, scales, mom, ok, edges, nboot=args.nboot, group=gid)
    print(f"{'<p>':>7} {'bias e-4':>16} {'R e-4':>16} {'ds_req':>16} "
          f"{'R*p (MeV)':>10}")
    for r in rows:
        print(f"{r['p']:7.2f} {r['bias']*1e4:+8.2f}+-{r['bias_err']*1e4:5.2f} "
              f"{r['R']*1e4:10.1f}+-{r['R_err']*1e4:4.1f} "
              f"{r['ds']:9.3f}+-{r['ds_err']:5.3f} {r['R']*r['p']*1e3:10.2f}")
    print("  R*p is the energy the full dE/dx table is worth in that bin; if "
          "the model is sane\n  it must be ~the reference loss (25-40 MeV) and "
          "vary smoothly with p.")


def run_probe(args):
    """Follow-ups on the p ~ 6 GeV dip that the first battery did not settle.

      F. is the response LINEAR in s?  ds_req = bias/R takes a linear fit over
         four scales; if the response bends, R is not what the ratio assumes.
      G. charge:  a charge-ODD residual is alignment/field-like, a charge-EVEN
         one is material-like.  The mechanism under test is charge-even.
      H. pT rather than p:  the gun's pT range is bounded, so a p bin near the
         edge of it has a different (pT, eta) mix from its neighbours.
      I. does the dip move with |eta| or with pT at FIXED p?
    """
    data, scales = load_scan(args.nfiles, cache=args.cache)
    sarr = np.array(scales)
    m = data[1.000]

    for obs in ("pair", "leg"):
        vals, mom, eta, pt, gid = {}, None, None, None, None
        for s in scales:
            v, p, e, t, grp = scan_observable(data[s], obs)
            vals[s] = v
            if mom is None:
                mom, eta, pt, gid = p, e, t, grp
        ok = np.ones(len(mom), dtype=bool)
        for s in scales:
            ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
        ok &= np.isfinite(mom) & (mom > 0)
        edges = np.quantile(mom[ok], np.linspace(0, 1, 6))
        print("=" * 100)
        print(f"F.  linearity of the response in s  ({obs} observable)")
        print("=" * 100)
        print(f"{'<p>':>7} | " + " ".join(f"{f's={s:.3f}':>10}" for s in scales)
              + f" | {'lin fit resid (e-4)':>22}")
        for i in range(len(edges) - 1):
            sel = ok & (mom >= edges[i]) & (mom < edges[i + 1])
            ms = np.array([np.median(vals[s][sel]) for s in scales])
            c = np.polyfit(sarr, ms, 1)
            res = (ms - np.polyval(c, sarr)) * 1e4
            print(f"{np.median(mom[sel]):7.2f} | "
                  + " ".join(f"{v*1e4:+10.2f}" for v in ms)
                  + " | " + " ".join(f"{r:+5.2f}" for r in res))
        print()

    # ---- G: charge --------------------------------------------------------
    print("=" * 100)
    print("G.  by charge (single-leg observable): charge-even = material-like")
    print("=" * 100)
    gp = m["Muplusgen_pt"] * np.cosh(m["Muplusgen_eta"])
    gm = m["Muminusgen_pt"] * np.cosh(m["Muminusgen_eta"])
    for lab, gen, rec in (("mu+", "Muplusgen", "Muplus"),
                          ("mu-", "Muminusgen", "Muminus")):
        v = {}
        for s in scales:
            d = data[s]
            g = d[f"{gen}_pt"] * np.cosh(d[f"{gen}_eta"])
            r = d[f"{rec}_pt"] * np.cosh(d[f"{rec}_eta"])
            v[s] = (r - g) / g
        mo = gp if lab == "mu+" else gm
        ok = np.ones(len(mo), dtype=bool)
        for s in scales:
            ok &= np.isfinite(v[s]) & (np.abs(v[s]) < 0.2)
        eds = np.quantile(mo[ok], np.linspace(0, 1, 6))
        rows = ds_req_bins(v, scales, mo, ok, eds, nboot=args.nboot)
        print(f"  {lab}: " + "  ".join(
            f"p={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
            for r in rows))

    # ---- H/I: pT, and (p, eta) / (p, pt) cells ----------------------------
    vals, mom, eta, pt, gid = {}, None, None, None, None
    for s in scales:
        vv, p, e, t, grp = scan_observable(data[s], "leg")
        vals[s] = vv
        if mom is None:
            mom, eta, pt, gid = p, e, t, grp
    ok = np.ones(len(mom), dtype=bool)
    for s in scales:
        ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
    print()
    print("=" * 100)
    print("H.  binned in gen pT instead of gen p (leg observable)")
    print("=" * 100)
    eds = np.quantile(pt[ok], np.linspace(0, 1, 6))
    rows = ds_req_bins(vals, scales, pt, ok, eds, nboot=args.nboot, group=gid)
    print("  " + "  ".join(f"pt={r['p']:5.2f} ds={r['ds']:+6.3f}+-{r['ds_err']:.3f}"
                           for r in rows))
    print()
    print("=" * 100)
    print("I.  the 4-8 GeV leg sample, split by |eta| and by pT")
    print("=" * 100)
    win = ok & (mom > 4.0) & (mom < 8.0)
    print(f"  window 4 < p < 8 GeV: n = {int(win.sum())}")
    for name, var, cuts in (("|eta|", eta, [0.0, 0.5, 1.0, 3.0]),
                            ("pT", pt, [0.0, 4.0, 6.0, 30.0])):
        rows = ds_req_bins(vals, scales, var, win, np.array(cuts, float),
                           nboot=args.nboot, group=gid)
        print(f"  by {name:5s}: " + "  ".join(
            f"<{name}>={r['p']:5.2f} n={r['n']:5d} ds={r['ds']:+6.3f}"
            f"+-{r['ds_err']:.3f}" for r in rows))
    print()
    print("  and the same window with a FINE p scan:")
    rows = ds_req_bins(vals, scales, mom, ok,
                       np.array([2., 3., 4., 5., 6., 7., 8., 10., 13., 18., 30., 100.]),
                       nboot=args.nboot, group=gid)
    for r in rows:
        print(f"    p={r['p']:6.2f} n={r['n']:6d} bias={r['bias']*1e4:+7.2f}"
              f"+-{r['bias_err']*1e4:4.2f} R={r['R']*1e4:7.1f}"
              f"+-{r['R_err']*1e4:4.1f} ds={r['ds']:+7.3f}+-{r['ds_err']:.3f}"
              f"  R*p={r['R']*r['p']*1e3:6.1f} MeV")


def run_synth(args):
    """The gate arithmetic, done in BIAS space rather than in ds_req space.

    ds_req = bias/R is a ratio of two measured things; its momentum structure
    is partly the structure of R.  The model predicts a BIAS,
        bias_pred(p) = c * f_ref(p) * R(p),
    with c = 1 if the fit inherits the whole reference shift.  Fitting c and
    quoting the chi2 separates "is the SHAPE right" from "is the SIZE right",
    which the ds_req table conflates.
    """
    import json
    refs = json.load(open(args.refs))
    refs = [r for r in refs if np.isfinite(r["f_g4mode"])]
    data, scales = load_scan(args.nfiles, cache=args.cache)

    for obs in ("leg", "pair"):
        vals, mom, eta, pt, gid = {}, None, None, None, None
        for s in scales:
            v, p, e, t, grp = scan_observable(data[s], obs)
            vals[s] = v
            if mom is None:
                mom, eta, pt, gid = p, e, t, grp
        ok = np.ones(len(mom), dtype=bool)
        for s in scales:
            ok &= np.isfinite(vals[s]) & (np.abs(vals[s]) < 0.2)
        ok &= np.isfinite(mom) & (mom > 0)
        eds = np.array([0.8, 1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0,
                        20.0, 30.0, 60.0, 200.0])
        rows = ds_req_bins(vals, scales, mom, ok, eds, nboot=args.nboot, group=gid)
        print("=" * 106)
        print(f"FINE MOMENTUM SCAN, {obs} observable "
              f"(fixed edges, so the bin CONTENT does not move with nbin)")
        print("=" * 106)
        print(f"{'<p>':>7} {'n':>6} {'bias e-4':>15} {'R e-4':>13} "
              f"{'ds_req':>15} {'f_mode':>7} {'f_med':>7}")
        for r in rows:
            r["fm"] = _ref_interp(refs, "f_g4mode", r["p"])
            r["fd"] = _ref_interp(refs, "f_g4med", r["p"])
            print(f"{r['p']:7.2f} {r['n']:6d} {r['bias']*1e4:+8.2f}"
                  f"+-{r['bias_err']*1e4:4.2f} {r['R']*1e4:7.1f}"
                  f"+-{r['R_err']*1e4:4.1f} {r['ds']:+8.3f}+-{r['ds_err']:5.3f}"
                  f" {r['fm']:7.4f} {r['fd']:7.4f}")

        for pcut in (0.0, 2.5, 4.0, 8.0):
            sub = [r for r in rows if r["p"] > pcut]
            b = np.array([r["bias"] for r in sub]) * 1e4
            e = np.array([r["bias_err"] for r in sub]) * 1e4
            w = 1.0 / e ** 2
            print(f"\n  --- bins with p > {pcut} GeV ({len(sub)} of {len(rows)})")
            c0 = float(np.sum(w * b) / np.sum(w))
            chi0 = float(np.sum(w * (b - c0) ** 2))
            print(f"  CONSTANT bias hypothesis: b = {c0:+.2f} +- "
                  f"{1/np.sqrt(np.sum(w)):.2f} e-4, chi2 = {chi0:.1f} / "
                  f"{len(sub)-1}")
            for key, name in (("fm", "G4 MODE"), ("fd", "G4 MEDIAN")):
                pr = np.array([r[key] * r["R"] for r in sub]) * 1e4
                cc = float(np.sum(w * b * pr) / np.sum(w * pr ** 2))
                ce = float(1.0 / np.sqrt(np.sum(w * pr ** 2)))
                chi1 = float(np.sum(w * (b - pr) ** 2))
                chic = float(np.sum(w * (b - cc * pr) ** 2))
                print(f"  {name:10s} reference: c = {cc:+.3f} +- {ce:.3f}   "
                      f"chi2(c=1) = {chi1:6.1f} / {len(sub)}   "
                      f"chi2(c free) = {chic:5.1f} / {len(sub)-1}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("median")
    m.add_argument("--models", default="pt3,pt10,pt40,pt100")
    m.add_argument("--planes", default="")
    m.add_argument("--nt", type=int, default=1 << 17)
    m.add_argument("--npad", type=int, default=32)
    m.add_argument("--nosim", action="store_true")
    m.add_argument("--nboot", type=int, default=60)
    m.add_argument("--refs", default="", help="write the reference table here")
    m.set_defaults(func=run_median)

    d = sub.add_parser("dedx")
    d.add_argument("--refs", default="gate_refs.json")
    d.add_argument("--use", default="pt3,pt10,pt40,pt100")
    d.add_argument("--obs", default="pair,leg")
    d.add_argument("--nbin", type=int, default=5)
    d.add_argument("--nfiles", type=int, default=0)
    d.add_argument("--nboot", type=int, default=400)
    d.add_argument("--cache", default="")
    d.set_defaults(func=run_dedx)

    b = sub.add_parser("bin61")
    b.add_argument("--refs", default="gate_refs.json")
    b.add_argument("--cache", default="gate_scan_cache.npz")
    b.add_argument("--nfiles", type=int, default=0)
    b.add_argument("--nboot", type=int, default=300)
    b.set_defaults(func=run_bin61)

    q = sub.add_parser("probe")
    q.add_argument("--cache", default="gate_scan_cache.npz")
    q.add_argument("--nfiles", type=int, default=0)
    q.add_argument("--nboot", type=int, default=300)
    q.set_defaults(func=run_probe)

    y = sub.add_parser("synth")
    y.add_argument("--refs", default="gate_refs.json")
    y.add_argument("--cache", default="gate_scan_cache.npz")
    y.add_argument("--nfiles", type=int, default=0)
    y.add_argument("--nboot", type=int, default=400)
    y.set_defaults(func=run_synth)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""CF-0: offline characteristic-function-space estimator prototype for the
CVH resolution scales, running entirely on existing gen-closure output
(no C++ changes, no re-production).

Physics/statistics basis (see Documents/Resolution/NOTES.md):
- With gen-frozen reference states the CVH covariance V is block diagonal
  (per-hit measurement blocks + per-layer process-noise blocks), R = V^-1,
  and for each resolution parameter i the stored gradient decomposes as
    gradv_i     = nu_i - q_i      (full gradient, log-det + chi2 term)
    gradchisqv_i =      -q_i      (chi2 term alone)
  with nu_i = tr(dV_i R) > 0 and q_i = (Rr)^T dV_i (Rr) >= 0.
- z2 = q/nu is the squared standardized residual of that block: for rank-1
  dV (strip hit-x) exactly z2 ~ chi2_1 under the Gaussian model; for
  higher-rank blocks (MS) the correct reference is the EXACT eigenvalue
  spectrum of the block, not a single effective rank.
- ECF/Laplace estimator for the tied variance scale s of a family: solve
    mean_i exp(-u * z2_i)  =  mean_i prod_j (1 + 2 u lam~_ij s)^(-1/2)
  for s at probe u, with lam~_ij = lam_ij / sum_j lam_ij the normalized
  block spectrum. The statistic has bounded influence (a catastrophic
  track moves it by <= 1/N), and u sets the effective truncation scale
  z2 ~ 1/u smoothly -- no data-dependent selection, no censoring
  arithmetic. u -> 0 recovers the (tail-dominated) moment fit.

Three model conventions for the reference, in increasing fidelity:
  r = 1                 rank-1 (exact for hit_x/hit_y/ioni)
  r = r_eff = nu^2/H_ii two-moment approximation from the Fisher diagonal
                        (--extract-full; the ONLY option before the
                        reseigv export existed)
  exact spectrum        lam_j from reseigidx/reseigv (--extract-exact);
                        prod_j (1+2u lam~_j s)^(-1/2) reduces to the
                        chi2_r form iff all lam_j are equal. Real MS
                        spectra are strongly asymmetric (0.60/0.31/0.07/..),
                        so this is where the two-moment version breaks.

Stages (cached to npz between stages):
  --extract        read (q, nu, chi2/ndof) per entry per family from task files
  --extract-full   ... plus per-entry r_eff from hesspackedv (cache _full)
  --extract-exact  ... plus the exact block spectra, nu from gradllv
                   (cache _exact; needs the reseigv/gradllv export)
  --reff           calibrate c and r_eff per family from hesspackedv (subset)
  --fit            spectroscopy plots + ECF probe scan + result table
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2dist

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

FAMILIES = {8: "hit_x", 9: "hit_y", 10: "ms", 11: "ioni"}
NU_MIN = 1e-4  # float32 cancellation guard on nu = gradv - gradchisqv
# gradllv stores the log-det trace nu directly, so the cancellation guard is
# not needed there: ionization entries (h ~ 1e-6) become usable instead of
# being cut down to the ~5% thick-crossing survivors that NU_MIN leaves.
NU_MIN_LL = 1e-12
NEIG = 5      # reseigv is padded to 5 eigenvalues per row
NLAM = 2 * NEIG   # a parameter can carry more than one leg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_eig_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root",
                   help="glob of gen-closure output files (needs gradchisqv; "
                        "gradllv + reseigidx/reseigv for --extract-exact). The "
                        "_eig production is the same sample and statistics as "
                        "the older _censor one, with the later exports added.")
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf0_cache.npz"))
    p.add_argument("--reff-cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "runs/cf0_reff.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--extract-full", action="store_true",
                   help="like --extract but also reads hesspackedv on every "
                        "file and stores per-entry r_eff = nu^2/H_ii aligned "
                        "with the z2 arrays (cache suffix _full)")
    p.add_argument("--extract-exact", action="store_true",
                   help="like --extract but takes nu from gradllv and stores "
                        "the EXACT per-entry eigenvalue spectrum from "
                        "reseigidx/reseigv (cache suffix _exact)")
    p.add_argument("--reff", action="store_true")
    p.add_argument("--reff-files", type=int, default=5,
                   help="number of files for the hesspackedv r_eff calibration")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("-v", "--verbose", type=int, default=3,
                   help="logging verbosity (wums levels; 3 = info)")
    p.add_argument("--compare-models", action="store_true",
                   help="with an _exact cache: also solve with the two-moment "
                        "chi2_r reference derived FROM the same spectra "
                        "(r_eff = 1/sum(lam~^2)), so the exact-vs-r_eff "
                        "difference is isolated from any selection change")
    return p.parse_args()


def default_outpath():
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/calibration_studies/{today}_cf0_ecf/")


def extract(files, cachepath, with_hess=False, exact=False):
    """Per-family arrays of (z2, nu, track chi2/ndof[, r_eff][, spectrum]).

    with_hess: also read hesspackedv and store r_eff = nu^2/H_ii per entry
    (c = 1 exactly, validated on rank-1 hit blocks). Entries with a
    non-positive or missing H_ii fall back to r_eff = 1 so the arrays stay
    aligned with the z2 selection.

    exact: take nu from gradllv (no float32 cancellation) and store the
    exact block eigenvalue spectrum, normalized to sum 1, padded to NLAM.
    sum(lam) reproduces nu to 1e-7 by construction of the export, which is
    checked here and reported.
    """
    keys = (["z2", "nu", "c2n"] + (["reff"] if with_hess else [])
            + (["lam"] if exact else []))
    acc = {f: {k: [] for k in keys} for f in FAMILIES.values()}
    ndrop = {f: 0 for f in FAMILIES.values()}
    ntr = 0
    pt = None
    numin = NU_MIN_LL if exact else NU_MIN
    branches = ["globalidxv", "gradchisqv", "chisqval", "ndof"]
    branches += ["gradllv"] if exact else ["gradv"]
    if exact:
        branches += ["reseigidx", "reseigv"]
    if with_hess:
        branches.append("hesspackedv")
    sumlam_dev = []
    for ifn, fn in enumerate(files):
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            if pt is None:
                pt = f["runtree"]["parmtype"].array(library="np")
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        if t.num_entries == 0:
            continue
        a = t.arrays(branches, library="np")
        for ic in range(len(a["chisqval"])):
            gi = np.asarray(a["globalidxv"][ic])
            gc = np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            if exact:
                nu_all = np.asarray(a["gradllv"][ic], dtype=np.float64)
                eidx = np.asarray(a["reseigidx"][ic])
                ev = np.asarray(a["reseigv"][ic], dtype=np.float64).reshape(-1, NEIG)
                # sort once per candidate so each parameter's legs are a slice
                eorder = np.argsort(eidx, kind="stable")
                eidx_s, ev_s = eidx[eorder], ev[eorder]
            else:
                nu_all = (np.asarray(a["gradv"][ic], dtype=np.float64) - gc)
            tp = pt[gi]
            c2n = a["chisqval"][ic] / max(a["ndof"][ic], 1)
            ntr += 1
            if with_hess:
                h = np.asarray(a["hesspackedv"][ic], dtype=np.float64)
                hdiag = h[packed_diag_indices(len(gi))]
            for code, fam in FAMILIES.items():
                m = tp == code
                if not m.any():
                    continue
                nu = nu_all[m]
                q = -gc[m]
                ok = nu > numin
                if exact:
                    # a block with no exported spectrum cannot be modelled
                    gq = gi[m]
                    lo = np.searchsorted(eidx_s, gq, "left")
                    hi_ = np.searchsorted(eidx_s, gq, "right")
                    ok &= (hi_ > lo)
                ndrop[fam] += int((~ok).sum())
                if not ok.any():
                    continue
                acc[fam]["z2"].append(np.maximum(q[ok], 0.) / nu[ok])
                acc[fam]["nu"].append(nu[ok])
                acc[fam]["c2n"].append(np.full(ok.sum(), c2n, dtype=np.float32))
                if with_hess:
                    hi = hdiag[m][ok]
                    nui = nu[ok]
                    r = np.where(hi > 0., nui * nui / np.maximum(hi, 1e-300), 1.)
                    acc[fam]["reff"].append(np.clip(r, 1.0, None))
                if exact:
                    sel = np.where(ok)[0]
                    lam = np.zeros((len(sel), NLAM), dtype=np.float32)
                    for kk, jj in enumerate(sel):
                        v = ev_s[lo[jj]:hi_[jj]].ravel()
                        v = v[v > 0.][:NLAM]
                        if not len(v):
                            continue
                        tot = v.sum()
                        lam[kk, :len(v)] = v / tot
                        sumlam_dev.append(abs(tot / nu[jj] - 1.))
                    acc[fam]["lam"].append(lam)
        if (ifn + 1) % 20 == 0:
            logger.info(f"extract: {ifn+1}/{len(files)} files, {ntr} tracks")
    out = {}
    for fam, d in acc.items():
        for k in d:
            if not d[k]:
                out[f"{fam}_{k}"] = np.zeros(0, dtype=np.float64)
            elif k == "lam":
                lam = np.concatenate(d[k], axis=0)
                # drop trailing all-zero columns (rank never reached them)
                nz = np.flatnonzero(lam.any(axis=0))
                out[f"{fam}_lam"] = lam[:, :(nz[-1] + 1)] if len(nz) else lam[:, :1]
            else:
                out[f"{fam}_{k}"] = np.concatenate(d[k])
        msg = (f"{fam}: {len(out[f'{fam}_z2'])} entries "
               f"({ndrop[fam]} dropped by nu>{numin:g}"
               f"{'/no spectrum' if exact else ''})")
        if exact and f"{fam}_lam" in out and out[f"{fam}_lam"].size:
            msg += f", spectrum width {out[f'{fam}_lam'].shape[1]}"
        logger.info(msg)
    if sumlam_dev:
        d = np.asarray(sumlam_dev)
        logger.info(f"sum(lambda) vs nu: median |dev| = {np.median(d):.2e}, "
                    f"q99 = {np.percentile(d, 99):.2e} (export self-check)")
    out["ntracks"] = np.array(ntr)
    os.makedirs(os.path.dirname(cachepath), exist_ok=True)
    np.savez_compressed(cachepath, **out)
    logger.info(f"wrote {cachepath}")


def packed_diag_indices(n):
    """Indices of diagonal elements in upper-triangular row-major packing."""
    r = np.arange(n)
    return (r * n - r * (r - 1) // 2).astype(np.int64)


def calibrate_reff(files, nfiles, cachepath):
    """c from rank-1 hit-x entries (H_ii = c nu^2), then r_eff = c nu^2/H_ii."""
    ratios = {f: [] for f in FAMILIES.values()}  # H_ii / nu^2
    pt = None
    for fn in files[:nfiles]:
        f = uproot.open(fn)
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradv", "gradchisqv", "hesspackedv"],
                     library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            n = len(gi)
            g = np.asarray(a["gradv"][ic], dtype=np.float64)
            gc = np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            h = np.asarray(a["hesspackedv"][ic], dtype=np.float64)
            hdiag = h[packed_diag_indices(n)]
            nu = g - gc
            tp = pt[gi]
            for code, fam in FAMILIES.items():
                m = (tp == code) & (nu > NU_MIN) & (hdiag > 0)
                if m.any():
                    ratios[fam].append(hdiag[m] / nu[m] ** 2)
    ratios = {f: (np.concatenate(v) if v else np.zeros(0)) for f, v in ratios.items()}
    # rank-1 reference: strip-dominated hit_x -> c = mode of H/nu^2
    c = np.median(ratios["hit_x"])
    out = {"c": np.array(c)}
    logger.info(f"Fisher-convention factor c = {c:.4f} "
                f"(hit_x H/nu^2 median; q16/q84 = "
                f"{np.percentile(ratios['hit_x'], 16):.3f}/"
                f"{np.percentile(ratios['hit_x'], 84):.3f})")
    for fam, v in ratios.items():
        if len(v) == 0:
            continue
        reff = c / v  # r_eff = c nu^2 / H_ii
        out[f"{fam}_reff_med"] = np.array(np.median(reff))
        out[f"{fam}_reff_q16"] = np.array(np.percentile(reff, 16))
        out[f"{fam}_reff_q84"] = np.array(np.percentile(reff, 84))
        logger.info(f"{fam}: r_eff median = {np.median(reff):.2f} "
                    f"[{np.percentile(reff, 16):.2f}, {np.percentile(reff, 84):.2f}]")
    np.savez(cachepath, **out)
    logger.info(f"wrote {cachepath}")


def ecf_solve(z2, r, u, lam=None):
    """Solve mean exp(-u z2) = model(s) for the tied log variance scale s.

    lam given: EXACT spectrum model, mean_i prod_j (1 + 2 u lam~_ij s)^(-1/2),
    with lam a (n, m) array of per-entry normalized eigenvalues (zero padded;
    a zero column contributes a factor 1, so padding is inert).

    lam None: chi2_r fallback, mean_i (1 + 2 u s / r_i)^(-r_i/2), where r may
    be a scalar or a per-entry array. This is the special case of the above
    with all r eigenvalues equal to 1/r, i.e. a two-moment approximation of
    the true spectrum.

    Delta-method error from the sample spread of the bounded statistic.
    """
    e = np.exp(-u * z2)
    F = np.mean(e)
    sigF = np.std(e) / np.sqrt(len(z2))

    if lam is not None:
        def model(logs):
            return np.mean(np.prod(
                1. / np.sqrt(1. + 2. * u * np.exp(logs) * lam), axis=1)) - F
    else:
        def model(logs):
            return np.mean((1. + 2. * u * np.exp(logs) / r) ** (-r / 2.)) - F

    try:
        logs = brentq(model, -8., 8., xtol=1e-10)
    except ValueError:
        return np.nan, np.nan
    eps = 1e-4
    dm = (model(logs + eps) - model(logs - eps)) / (2 * eps)  # d model/d log s
    sig_logs = sigF / abs(dm)
    return logs, sig_logs


def spectroscopy_plot(cache, outdir, args):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"hit_x": "#1f77b4", "hit_y": "#d62728", "ms": "#2ca02c", "ioni": "#9467bd"}
    edges = np.logspace(-3, 6, 200)
    for fam in FAMILIES.values():
        z2 = cache[f"{fam}_z2"]
        if len(z2) == 0:
            continue
        sf = 1.0 - np.searchsorted(np.sort(z2), edges) / len(z2)
        ax.plot(edges, sf, label=f"{fam} (N={len(z2):,})", color=colors[fam], lw=2)
    ax.plot(edges, chi2dist.sf(edges, 1), "k--", lw=1.5, label=r"$\chi^2_1$ (Gaussian)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z^2 = q/\nu$ (standardized block residual$^2$)")
    ax.set_ylabel("survival fraction")
    ax.set_ylim(1e-7, 2)
    ax.legend(fontsize=18)
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["cf0_z2_survival", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))


def ecf_scan_plot(results, outdir, args):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"hit_x": "#1f77b4", "hit_y": "#d62728", "ms": "#2ca02c"}
    markers = {"all": "o", "presel": "s"}
    for (fam, sel), (us, ths, errs) in results.items():
        if fam == "ioni" or fam not in colors:
            continue
        off = 1.0 if sel == "all" else 1.05
        ax.errorbar(np.array(us) * off, ths, yerr=errs,
                    marker=markers[sel], ls="-" if sel == "all" else "--",
                    color=colors[fam], markersize=7,
                    label=f"{fam} ({'no preselection' if sel == 'all' else 'chi2/ndof<10'})")
    ax.axhline(0., color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"ECF probe $u$  (effective truncation $z^2 \sim 1/u$)")
    ax.set_ylabel(r"fitted $\ln$ variance scale $\hat{\theta}$")
    ax.legend(fontsize=15, ncol=1)
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["cf0_ecf_scan", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))


def pick_cache(args):
    """Prefer the highest-fidelity cache present: exact > full > plain."""
    for suffix in ("_exact", "_full", ""):
        p = args.cache.replace(".npz", f"{suffix}.npz") if suffix else args.cache
        if os.path.exists(p):
            return p
    return args.cache


def fit(args, outdir):
    cachepath = pick_cache(args)
    logger.info(f"fit stage using {cachepath}")
    cache = np.load(cachepath)
    reff = np.load(args.reff_cache) if os.path.exists(args.reff_cache) else None
    spectroscopy_plot(cache, outdir, args)

    results = {}
    print(f"\n{'family':8s} {'model':7s} {'sel':10s} {'u':>6s} {'theta':>10s} "
          f"{'err':>8s} {'<z2> (moment)':>14s}")
    for fam in FAMILIES.values():
        z2 = cache[f"{fam}_z2"]
        c2n = cache[f"{fam}_c2n"]
        if len(z2) == 0:
            continue
        lam, r, tag = None, 1.0, "rank1"
        if f"{fam}_lam" in cache.files and cache[f"{fam}_lam"].size:
            lam = cache[f"{fam}_lam"].astype(np.float64)
            tag = "exact"
            nnz = (lam > 0).sum(axis=1)
            logger.info(f"{fam}: exact spectra, median rank "
                        f"{np.median(nnz):.0f}, max {nnz.max()}, "
                        f"width {lam.shape[1]}")
        elif f"{fam}_reff" in cache.files:
            r = cache[f"{fam}_reff"]  # per-entry rank (--extract-full)
            tag = "r_eff"
            logger.info(f"{fam}: per-entry r_eff, median {np.median(r):.2f}")
        elif reff is not None and f"{fam}_reff_med" in reff:
            r = max(float(reff[f"{fam}_reff_med"]), 1.0)
            tag = "r_eff"
        # two-moment rank implied by the SAME spectra: r_eff = 1/sum(lam~^2).
        # Fitting with this instead of the full product isolates exactly what
        # the r_eff approximation was costing, on identical entries.
        r_from_lam = None
        if lam is not None and args.compare_models:
            r_from_lam = 1. / np.maximum((lam ** 2).sum(axis=1), 1e-300)
            logger.info(f"{fam}: r_eff from spectra, median "
                        f"{np.median(r_from_lam):.2f} "
                        f"[{np.percentile(r_from_lam, 16):.2f}, "
                        f"{np.percentile(r_from_lam, 84):.2f}]")

        for sel, mask in (("all", np.ones(len(z2), bool)), ("presel", c2n < 10)):
            us, ths, errs = [], [], []
            rmask = r[mask] if isinstance(r, np.ndarray) else r
            lmask = lam[mask] if lam is not None else None
            for u in args.probes:
                th, sig = ecf_solve(z2[mask], rmask, u, lam=lmask)
                us.append(u)
                ths.append(th)
                errs.append(sig)
                line = (f"{fam:8s} {tag:7s} {sel:10s} {u:6.2f} {th:10.4f} "
                        f"{sig:8.4f} {np.mean(z2[mask]):14.3f}")
                if r_from_lam is not None:
                    th2, _ = ecf_solve(z2[mask], r_from_lam[mask], u)
                    line += f"   r_eff:{th2:8.4f}  delta:{th - th2:+8.4f}"
                print(line)
            results[(fam, sel)] = (us, ths, errs)
    ecf_scan_plot(results, outdir, args)
    np.savez(os.path.join(outdir, "cf0_results.npz"),
             **{f"{fam}_{sel}": np.array(v) for (fam, sel), v in results.items()})


def main():
    args = parse_args()
    logging.setup_logger(__file__, args.verbose)
    files = sorted(glob.glob(args.input))
    if not files and (args.extract or args.extract_full
                      or args.extract_exact or args.reff):
        sys.exit(f"no files match {args.input}")
    if args.extract:
        extract(files, args.cache)
    if args.extract_full:
        full = args.cache.replace(".npz", "_full.npz")
        extract(files, full, with_hess=True)
    if args.extract_exact:
        exact = args.cache.replace(".npz", "_exact.npz")
        extract(files, exact, exact=True)
    if args.reff:
        calibrate_reff(files, args.reff_files, args.reff_cache)
    if args.fit:
        outdir = output_tools.make_plot_dir(args.outpath or default_outpath())
        logger.info(f"writing to {outdir}")
        fit(args, outdir)


if __name__ == "__main__":
    main()

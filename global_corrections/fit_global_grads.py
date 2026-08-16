#!/usr/bin/env python3
"""Combined least-squares fit of the CVH global correction blocks from the
per-candidate gradients/Hessians stored by the residual-makers
(fillGrads=True): scalar-potential field modes (parmtype 14) and global
material groups (parmtype 15).

The stored gradv / hesspackedv are already marginalized over the
per-track parameters (Millepede factorization), so the global fit is
    delta = - (sum_i H_i + P)^{-1} (sum_i g_i)
restricted to the selected parameter types; all other global parameters
(alignment, per-module blocks) are held at their current values, i.e.
their rows/columns are dropped. P is a diagonal prior matrix
(1/sigma^2), with sigma from the materialGroups file for parmtype 15 and
--field-prior for parmtype 14.

The result is a delta on the current corparms, i.e. directly usable as a
corFiles iteration (option --write-parmtree) or for closure tests.

Example (V3 closure: fit must recover minus the injected k):
  python fit_global_grads.py -i 'run_dir/globalcor_*.root' \
      --groups .../materialGroups50.txt --parmtypes 15
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot
from scipy.stats import chi2 as chi2dist


def load_runtree_meta(fname):
    rt = uproot.open(fname)["runtree"]
    parmtype = rt["parmtype"].array(library="np")
    iidx = rt["iidx"].array(library="np")
    return parmtype, iidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="glob pattern of globalcor output files")
    ap.add_argument("--groups", default=None,
                    help="materialGroups tier file (group names + priors for parmtype 15)")
    ap.add_argument("--parmtypes", type=int, nargs="+", default=[14, 15],
                    help="global parameter types to float (default: 14 15)")
    ap.add_argument("--field-prior", type=float, default=0.0,
                    help="Gaussian prior sigma for parmtype-14 modes (0 = none)")
    ap.add_argument("--material-prior-scale", type=float, default=1.0,
                    help="scale factor on the per-group prior sigmas from the groups file "
                         "(0 = no material priors)")
    ap.add_argument("--write-parmtree", default=None,
                    help="write the fitted deltas as a corFiles-compatible parmtree ROOT file")
    ap.add_argument("--nmr", default=None,
                    help="NMR constraint npz (mfs/make_nmr_constraints.py output: g, resid, "
                         "sigma on the parmtype-14 mode basis) to anchor the field block")
    ap.add_argument("--save-info", default=None,
                    help="save the accumulated per-channel information (grad, hess on the "
                         "floated params, before priors/NMR) + metadata to an npz, so channels "
                         "can be combined and compared offline without re-aggregating")
    ap.add_argument("--max-chi2-per-hit", type=float, default=0.,
                    help="drop candidates with chisqval/(nValidHits) above this "
                         "(0 = no cut). Trimmed-closure diagnostic for the "
                         "resolution parameters: the raw Gaussian variance "
                         "estimator is dominated by catastrophic outliers "
                         "(decays in flight, hard delta rays), so the fitted "
                         "scales are reported as a function of this trimming")
    ap.add_argument("--censor-cut", type=float, default=0.,
                    help="censored-likelihood trimming: drop candidates with "
                         "chisqval/ndof above this AND apply the model-side "
                         "truncation correction to the resolution-parameter "
                         "gradients. Conditioning a correct Gaussian model on "
                         "chi2_tot < T is spherically symmetric in whitened "
                         "coordinates, so E[q_param | pass] = lambda * nu with "
                         "lambda = F_{k+2}(T)/F_k(T) (chi2 CDFs, k = ndof) "
                         "exactly; the consistent trimmed gradient is "
                         "g' = g - (1-lambda)*nu, with nu = gradv - gradchisqv "
                         "(the per-parameter log-det trace term). Requires the "
                         "gradchisqv branch (produced with fillGrads since "
                         "2026-07-24). Mutually exclusive with "
                         "--max-chi2-per-hit.")
    ap.add_argument("--censor-raw", action="store_true",
                    help="with --censor-cut: apply the chisqval/ndof trimming "
                         "but SKIP the model-side truncation correction "
                         "(cut-matched comparison baseline)")
    ap.add_argument("--tie", action="store_true",
                    help="tie all parameters of each floated parmtype to a single common "
                         "scale (one fitted number per type). For log-variance parameters "
                         "(resolution parmtypes 8-11) the common scale is exact, not an "
                         "approximation: exp(c) applied to every module is one global "
                         "rescaling. Used by the resolution gen-closure diagnostic.")
    args = ap.parse_args()

    # -i is a glob, or a .txt filelist (one path per line) for chunked runs
    if args.input.endswith(".txt") and os.path.isfile(args.input):
        files = [ln.strip() for ln in open(args.input) if ln.strip()]
    else:
        files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")

    parmtype, _ = load_runtree_meta(files[0])
    nglobal = len(parmtype)
    sel = np.isin(parmtype, args.parmtypes)
    fitidx = np.where(sel)[0]
    if args.tie:
        # one fitted parameter per floated parmtype: every global parameter
        # of that type maps to the same fit index
        types = sorted(set(parmtype[fitidx].tolist()))
        nfit = len(types)
        g2f = -np.ones(nglobal, dtype=np.int64)
        for k, ptv in enumerate(types):
            g2f[parmtype == ptv] = k
        print(f"{nglobal} global parameters; floating {nfit} TIED type scales "
              f"(types {types}), rest frozen")
    else:
        nfit = len(fitidx)
        # map global index -> fit index (-1 = frozen)
        g2f = -np.ones(nglobal, dtype=np.int64)
        g2f[fitidx] = np.arange(nfit)
        print(f"{nglobal} global parameters; floating {nfit} "
              f"(types {sorted(set(parmtype[fitidx].tolist()))}), rest frozen")

    # group names/priors for reporting
    _TYPENAMES = {8: "res_hit_x", 9: "res_hit_y", 10: "res_ms", 11: "res_ioni",
                  14: "bfield", 15: "material"}
    if args.tie:
        names = [f"tied_{_TYPENAMES.get(ptv, f'type{ptv}')}" for ptv in types]
        priors = np.zeros(nfit)
    else:
        names = [f"idx{gi}" for gi in fitidx]
        priors = np.zeros(nfit)
    if args.field_prior > 0. and not args.tie:
        priors[parmtype[fitidx] == 14] = 1. / args.field_prior**2
    gnames = {0: "other"}
    gprior = {0: 0.2}
    if args.groups:
        for line in open(args.groups):
            if line.startswith("RULE"):
                f = line.split("\t")
                gnames[int(f[1])] = f[2]
                gprior[int(f[1])] = float(f[10])
    if not args.tie:
        # parmtype-15 sentinel DetId(groupIdx): recover groupIdx from runtree
        # ordering -- entries of a given parmtype are ordered by DetId, i.e. by
        # groupIdx, so enumerate them in order.
        m15 = np.where(parmtype[fitidx] == 15)[0]
        for order, fi in enumerate(m15):
            names[fi] = gnames.get(order, f"group{order}")
            sig = gprior.get(order, 0.2) * args.material_prior_scale
            if args.material_prior_scale > 0.:
                priors[fi] = 1. / sig**2
        m14 = np.where(parmtype[fitidx] == 14)[0]
        for order, fi in enumerate(m14):
            names[fi] = f"bfield_mode{order}"

    grad = np.zeros(nfit)
    hess = np.zeros((nfit, nfit))
    ncand = 0
    fmtcount = {"packed": 0, "factored": 0}
    maxdrop = 0.
    nskip = 0
    for fname in files:
        # skip empty / treeless files (e.g. a grads task that crashed
        # mid-write leaves 1.3 GB of orphaned baskets with no finalized
        # TTree) instead of aborting the whole aggregation
        try:
            f = uproot.open(fname)
            if "tree" not in f:
                nskip += 1
                print(f"WARNING: no 'tree' in {fname} -- skipping", file=sys.stderr)
                continue
            t = f["tree"]
        except Exception as e:
            nskip += 1
            print(f"WARNING: cannot open {fname} ({type(e).__name__}) -- skipping", file=sys.stderr)
            continue
        if t.num_entries == 0:
            continue
        keys = set(t.keys())
        # Two storage conventions produce the SAME per-candidate global
        # Hessian and are combined transparently:
        #   packed  ("hesspackedv") -- upper-triangular H, single-track maker
        #           (ResidualGlobalCorrectionMakerG4e; e.g. cosmics grads)
        #   factored("hessfactorv") -- B (nRank x nParms, row-major, rows
        #           sqrt(lambda)*v^T); H = B^T B, factor 2 already inside.
        #           Two-track maker (fillGradsFactored); e.g. J/psi grads.
        if "hessfactorv" in keys:
            fmt = "factored"
            facs = t["hessfactorv"].array(library="np")
            nranks = t["nRank"].array(library="np")
            hdrops = t["hessdroppedmass"].array(library="np")
        elif "hesspackedv" in keys:
            fmt = "packed"
            hesss = t["hesspackedv"].array(library="np")
        else:
            sys.exit(f"{fname}: neither hesspackedv nor hessfactorv present")
        fmtcount[fmt] += 1
        gidxs = t["globalidxv"].array(library="np")
        grads = t["gradv"].array(library="np")
        chi2ok = None
        if args.max_chi2_per_hit > 0.:
            chi2 = t["chisqval"].array(library="np")
            nvh = t["nValidHits"].array(library="np")
            chi2ok = chi2 / np.maximum(nvh, 1) < args.max_chi2_per_hit
        lam = None
        if args.censor_cut > 0.:
            if args.max_chi2_per_hit > 0.:
                sys.exit("--censor-cut and --max-chi2-per-hit are mutually exclusive")
            if "gradchisqv" not in keys:
                sys.exit(f"{fname}: no gradchisqv branch -- --censor-cut needs "
                         "output produced with the 2026-07-24 fillGrads gating")
            chi2 = t["chisqval"].array(library="np")
            kdof = t["ndof"].array(library="np").astype(np.float64)
            gradchis = t["gradchisqv"].array(library="np")
            chi2ok = chi2 / np.maximum(kdof, 1) < args.censor_cut
            if not args.censor_raw:
                T = args.censor_cut * kdof
                lam = chi2dist.cdf(T, kdof + 2) / np.maximum(chi2dist.cdf(T, kdof), 1e-300)
        for ic in range(len(gidxs)):
            if chi2ok is not None and not chi2ok[ic]:
                continue
            gi = np.asarray(gidxs[ic])
            fi = g2f[gi]
            keep = fi >= 0
            if not keep.any():
                continue
            ncand += 1
            g = np.asarray(grads[ic], dtype=np.float64)
            if lam is not None:
                # model-side truncation: g' = g - (1-lambda)*nu, where
                # nu = gradv - gradchisqv is the log-det trace term (zero for
                # non-resolution parameters by construction)
                nu = g - np.asarray(gradchis[ic], dtype=np.float64)
                g = g - (1. - lam[ic]) * nu
            # np.add.at: with --tie a candidate hits the same fit index
            # repeatedly (fancy += silently drops duplicates)
            np.add.at(grad, fi[keep], g[keep])
            kk = np.where(keep)[0]
            n = len(gi)
            if fmt == "packed":
                # unpack upper-triangular packed hessian rows, keep fit block
                h = np.asarray(hesss[ic], dtype=np.float64)
                H = np.zeros((n, n))
                pos = 0
                for r in range(n):
                    m = n - r
                    H[r, r:] = h[pos:pos + m]
                    pos += m
                H = H + np.triu(H, 1).T
                Hkk = H[np.ix_(kk, kk)]
            else:
                # H = B^T B; restrict B to kept columns before the product
                # (nkeep ~ 90 << nParms up to ~350) -> syrk on the fit block.
                nrank = int(nranks[ic])
                B = np.asarray(facs[ic], dtype=np.float64).reshape(nrank, n)
                Bk = B[:, kk]
                Hkk = Bk.T @ Bk
                maxdrop = max(maxdrop, float(hdrops[ic]))
            np.add.at(hess, (fi[kk][:, None], fi[kk][None, :]), Hkk)
    print(f"aggregated {ncand} candidates from {len(files) - nskip}/{len(files)} files "
          f"(packed files: {fmtcount['packed']}, factored files: {fmtcount['factored']}, "
          f"skipped empty/bad: {nskip})")
    if fmtcount["factored"]:
        print(f"max dropped mass-constraint eigenfraction (factored): {maxdrop:.2e}")

    # Save the raw per-channel information (grad, hess) on the floated params,
    # BEFORE priors/NMR, for offline combination and correlation comparison.
    if args.save_info:
        np.savez(args.save_info,
                 grad=grad, hess=hess,
                 fitidx=(np.array(types) if args.tie else fitidx),
                 parmtype=(np.array(types) if args.tie else parmtype[fitidx]),
                 names=np.array(names), priors=priors,
                 ncand=ncand, nfiles=len(files))
        print(f"saved per-channel info ({nfit} params) to {args.save_info}")

    # NMR absolute-|B| constraints on the field block (parmtype 14). Track
    # data alone leaves the field undetermined at the probe radius (harmonic
    # continuation from the tracker interior blows up); the four probes pin
    # those boundary directions. chi2_p = (resid_p - g_p . dc)^2 / sigma_p^2,
    # linearized about the base field. Same H = d2chi2/dp2 convention as the
    # prior above (factor 2 on the outer product).
    if args.tie and args.nmr:
        sys.exit("--nmr is not supported together with --tie")
    m14 = np.where(parmtype[fitidx] == 14)[0]
    if args.nmr:
        if len(m14) == 0:
            sys.exit("--nmr given but no parmtype-14 (field) parameters are floated")
        nmr = np.load(args.nmr)
        gm, res, sig = nmr["g"], nmr["resid"], nmr["sigma"]
        if gm.shape[1] != len(m14):
            sys.exit(f"--nmr basis mismatch: {gm.shape[1]} modes vs {len(m14)} floated")
        print(f"adding {len(res)} NMR rows (resid [mT] {np.round(res*1e3,2)}, "
              f"sigma [mT] {np.round(sig*1e3,3)})")
        for p in range(len(res)):
            w = 1. / sig[p]**2
            grad[m14] += -2. * res[p] * w * gm[p]
            hess[np.ix_(m14, m14)] += 2. * w * np.outer(gm[p], gm[p])

    hess_prior = hess + np.diag(2. * priors)  # chi2 convention: H = d2chi2/dp2
    delta = -np.linalg.solve(hess_prior, grad)
    cov = 2. * np.linalg.inv(hess_prior)
    err = np.sqrt(np.diag(cov))

    print(f"\n{'parameter':28s} {'delta':>12s} {'error':>10s} {'pull':>8s}")
    order = np.argsort(-np.abs(delta / np.maximum(err, 1e-30)))
    for i in order[:30]:
        print(f"{names[i]:28s} {delta[i]:12.5f} {err[i]:10.5f} "
              f"{delta[i]/max(err[i],1e-30):8.2f}")

    if args.write_parmtree:
        out = np.zeros(nglobal, dtype=np.float32)
        if args.tie:
            # broadcast each tied type scale to all parameters of that type
            # (also usable as an injection corfile for closure tests)
            for k, ptv in enumerate(types):
                out[parmtype == ptv] = delta[k]
        else:
            out[fitidx] = delta
        with uproot.recreate(args.write_parmtree) as f:
            f["parmtree"] = {"x": out}
        print(f"\nwrote {args.write_parmtree}")


if __name__ == "__main__":
    main()

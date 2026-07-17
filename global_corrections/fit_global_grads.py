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
import sys

import numpy as np
import uproot


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
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")

    parmtype, _ = load_runtree_meta(files[0])
    nglobal = len(parmtype)
    sel = np.isin(parmtype, args.parmtypes)
    fitidx = np.where(sel)[0]
    nfit = len(fitidx)
    # map global index -> fit index (-1 = frozen)
    g2f = -np.ones(nglobal, dtype=np.int64)
    g2f[fitidx] = np.arange(nfit)
    print(f"{nglobal} global parameters; floating {nfit} "
          f"(types {sorted(set(parmtype[fitidx].tolist()))}), rest frozen")

    # group names/priors for reporting
    names = [f"idx{gi}" for gi in fitidx]
    priors = np.zeros(nfit)
    if args.field_prior > 0.:
        priors[parmtype[fitidx] == 14] = 1. / args.field_prior**2
    gnames = {0: "other"}
    gprior = {0: 0.2}
    if args.groups:
        for line in open(args.groups):
            if line.startswith("RULE"):
                f = line.split("\t")
                gnames[int(f[1])] = f[2]
                gprior[int(f[1])] = float(f[10])
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
    for fname in files:
        t = uproot.open(fname)["tree"]
        if t.num_entries == 0:
            continue
        gidxs = t["globalidxv"].array(library="np")
        grads = t["gradv"].array(library="np")
        hesss = t["hesspackedv"].array(library="np")
        for ic in range(len(gidxs)):
            gi = np.asarray(gidxs[ic])
            fi = g2f[gi]
            keep = fi >= 0
            if not keep.any():
                continue
            ncand += 1
            g = np.asarray(grads[ic], dtype=np.float64)
            grad[fi[keep]] += g[keep]
            # unpack upper-triangular packed hessian rows
            n = len(gi)
            h = np.asarray(hesss[ic], dtype=np.float64)
            H = np.zeros((n, n))
            pos = 0
            for r in range(n):
                m = n - r
                H[r, r:] = h[pos:pos + m]
                pos += m
            H = H + np.triu(H, 1).T
            kk = np.where(keep)[0]
            hess[np.ix_(fi[kk], fi[kk])] += H[np.ix_(kk, kk)]
    print(f"aggregated {ncand} candidates from {len(files)} files")

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
        out[fitidx] = delta
        with uproot.recreate(args.write_parmtree) as f:
            f["parmtree"] = {"x": out}
        print(f"\nwrote {args.write_parmtree}")


if __name__ == "__main__":
    main()

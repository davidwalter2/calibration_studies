#!/usr/bin/env python3
"""Offline reference solution of the quadratic (hit-chi2) problem alone.

Same algebra and the same conventions as
``global_corrections/fit_global_grads.py``:

    chi2(theta) = chi2_0 + G^T theta + 0.5 theta^T K theta   (chi2 units)
    theta_hat   = -(K + 2 P)^-1 G
    cov         =  2 (K + 2 P)^-1

with ``P = diag(1/sigma_prior^2)``. This is what the rabbit external term
must reproduce: written in NLL units (``g = G/2``, ``H = K/2``) with the
priors declared through the ParamModel prior mechanism, rabbit minimizes
``g^T x + 0.5 x^T H x + 0.5 sum (x/sigma)^2``, whose minimum is
``-(H + P)^-1 g = -(K + 2P)^-1 G`` and whose covariance is
``(H + P)^-1 = 2 (K + 2P)^-1`` -- identical.

It can also compare against the arrays saved by ``fit_global_grads.py
--save-info``, which is the check that ``extract.py`` accumulates the same
``G`` and ``K`` as the established reader.

Usage::

    python solve_reference.py -i runs/globalfit_btojpsix.npz --parmtypes 14
    python solve_reference.py -i runs/globalfit_btojpsix.npz \\
        --compare-info runs/fitglobalgrads_info.npz
"""

import argparse
import sys

import numpy as np

from make_global_term import name_params, param_scales


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="extract.py npz")
    p.add_argument("--parmtypes", type=int, nargs="+", default=None)
    p.add_argument("--groups", default=None)
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument(
        "--inject",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="apply G -> G - K dtheta before solving (mirrors "
        "make_global_term.py --inject)",
    )
    p.add_argument(
        "--compare-info",
        default=None,
        help="npz from fit_global_grads.py --save-info to compare G and K "
        "against (element-wise, on the common parameter names)",
    )
    p.add_argument(
        "--whiten",
        action="store_true",
        help="solve in the physical basis (see make_global_term.py --whiten); "
        "must match how the card was written for the comparison to be exact",
    )
    p.add_argument("--coeffs", default=None)
    p.add_argument("--top", type=int, default=25, help="rows to print")
    p.add_argument("-o", "--output", default=None, help="save the solution npz")
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.input)
    parmtype, subidx = d["parmtype"], d["subidx"]
    sel = (
        np.isin(parmtype, args.parmtypes)
        if args.parmtypes
        else np.ones(len(parmtype), dtype=bool)
    )
    isel = np.where(sel)[0]
    names, prior_sigmas = name_params(
        parmtype[isel],
        subidx[isel],
        args.groups,
        args.field_prior,
        args.material_prior_scale,
    )
    G = d["grad"][isel].astype(np.float64)
    K = d["hess"][np.ix_(isel, isel)].astype(np.float64)
    n = len(names)
    if args.whiten:
        pscale, _ = param_scales(parmtype[isel], subidx[isel], args.coeffs, args.groups)
        inv = 1.0 / np.maximum(pscale, 1e-300)
        G = G * inv
        K = K * np.outer(inv, inv)
        prior_sigmas = prior_sigmas * pscale
        print(
            "whitened: 1 unit = 1 T of RMS |dB| (parmtype 14) / 1/gprior of "
            "d ln(dE/dx) (parmtype 15); scale range "
            f"{pscale.min():.3e} .. {pscale.max():.3e}"
        )

    if args.inject:
        idx = {nm: i for i, nm in enumerate(names)}
        inj = np.zeros(n)
        for spec in args.inject:
            nm, val = spec.rsplit(":", 1)
            inj[idx[nm]] = float(val)
        G = G - K @ inj
        print(f"injected dtheta on {(inj != 0).sum()} parameter(s)")

    if args.compare_info:
        ref = np.load(args.compare_info, allow_pickle=True)
        rnames = [str(s) for s in ref["names"]]
        common = [nm for nm in names if nm in set(rnames)]
        if not common:
            print(
                f"WARNING: no common parameter names with {args.compare_info}\n"
                f"  here:  {names[:4]} ...\n  there: {rnames[:4]} ..."
            )
        else:
            i_here = [names.index(nm) for nm in common]
            i_there = [rnames.index(nm) for nm in common]
            dg = G[i_here] - ref["grad"][i_there]
            dk = K[np.ix_(i_here, i_here)] - ref["hess"][np.ix_(i_there, i_there)]
            sg = np.abs(ref["grad"][i_there]).max()
            sk = np.abs(ref["hess"][np.ix_(i_there, i_there)]).max()
            print(
                f"vs fit_global_grads.py on {len(common)} common parameters "
                f"(candidates here {int(d['ncand_quadratic'])}, there "
                f"{int(ref['ncand'])}):\n"
                f"  max |dG| = {np.abs(dg).max():.6e}  (scale {sg:.6e}, rel "
                f"{np.abs(dg).max()/max(sg,1e-300):.3e})\n"
                f"  max |dK| = {np.abs(dk).max():.6e}  (scale {sk:.6e}, rel "
                f"{np.abs(dk).max()/max(sk,1e-300):.3e})"
            )

    priors = np.where(np.isfinite(prior_sigmas), 1.0 / prior_sigmas**2, 0.0)
    Kp = K + np.diag(2.0 * priors)
    theta = -np.linalg.solve(Kp, G)
    cov = 2.0 * np.linalg.inv(Kp)
    err = np.sqrt(np.diag(cov))

    print(f"\n{'parameter':28s} {'theta':>14s} {'error':>12s} {'pull':>8s}")
    order = np.argsort(-np.abs(theta / np.maximum(err, 1e-30)))
    for i in order[: args.top]:
        print(
            f"{names[i]:28s} {theta[i]:14.8f} {err[i]:12.8f} "
            f"{theta[i]/max(err[i],1e-30):8.2f}"
        )

    if args.output:
        np.savez(
            args.output,
            names=np.array(names),
            theta=theta,
            err=err,
            cov=cov,
            grad=G,
            hess=K,
            prior_sigmas=prior_sigmas,
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

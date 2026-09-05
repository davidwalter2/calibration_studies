#!/usr/bin/env python3
"""Compare rabbit fit results against each other or against the offline
reference solution, and report the NLL breakdown of a joint card.

Typical uses::

    # (a) the quadratic term through rabbit vs solve_reference.py
    python compare_fit.py --fit out_quad/fitresults.hdf5 \\
        --ref runs/reference_quad.npz

    # (c) the joint fit vs the quadratic-only and mass-only fits
    python compare_fit.py --fit out_joint/fitresults.hdf5 \\
        --fit2 out_quad/fitresults.hdf5 --label2 quadratic-only

    # NLL breakdown + correlations of the joint card at its minimum
    python compare_fit.py --fit out_joint/fitresults.hdf5 \\
        --card cards/joint.hdf5 --breakdown --corr bfield_mode0

    # (d) injection recovery
    python compare_fit.py --fit out_inj/fitresults.hdf5 \\
        --fit2 out_base/fitresults.hdf5 --injected bfield_mode0:0.122

Run it inside the rabbit TF environment (it imports rabbit).
"""

import argparse
import json
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", required=True, help="rabbit fitresults.hdf5")
    p.add_argument("--fit2", default=None, help="second fitresults.hdf5 to compare to")
    p.add_argument("--label", default="fit")
    p.add_argument("--label2", default="fit2")
    p.add_argument(
        "--ref", default=None, help="solve_reference.py npz (names, theta, err)"
    )
    p.add_argument(
        "--card", default=None, help="the datacard, for the NLL breakdown / index map"
    )
    p.add_argument("--breakdown", action="store_true", help="print the NLL breakdown")
    p.add_argument(
        "--corr",
        default=None,
        help="comma separated parameters whose correlations with the others "
        "are printed",
    )
    p.add_argument(
        "--injected",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="expected shift of --fit relative to --fit2; prints a pull table",
    )
    p.add_argument("--select", default=None, help="regex on parameter names")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-6, help="pass/fail tolerance")
    return p.parse_args()


def read_fit(path):
    from rabbit import io_tools

    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    val = np.asarray(h.values(), dtype=np.float64)
    err = np.sqrt(np.asarray(h.variances(), dtype=np.float64))
    cov = None
    if "cov" in fr:
        cov = np.asarray(fr["cov"].get().values(), dtype=np.float64)
    nll = {}
    for k in ("nllvalfull", "nllvalreduced", "edmval", "ndfsat", "satchi2"):
        if k in fr:
            try:
                nll[k] = float(np.asarray(fr[k].get()))
            except Exception:  # noqa: BLE001
                pass
    return names, val, err, cov, nll


def main():
    args = parse_args()
    names, val, err, cov, nll = read_fit(args.fit)
    sel = list(range(len(names)))
    if args.select:
        import re

        rx = re.compile(args.select)
        sel = [i for i, nm in enumerate(names) if rx.search(nm)]

    print(f"=== {args.label}: {args.fit}")
    if nll:
        print("    " + "  ".join(f"{k}={v:.6f}" for k, v in nll.items()))

    ok = True
    if args.ref:
        r = np.load(args.ref, allow_pickle=True)
        rn = [str(s) for s in r["names"]]
        common = [nm for nm in names if nm in set(rn)]
        print(f"\n--- vs reference {args.ref} ({len(common)} common parameters)")
        print(f"{'parameter':28s} {'fit':>15s} {'ref':>15s} {'d/err':>10s} "
              f"{'err fit':>12s} {'err ref':>12s} {'derr/err':>10s}")
        dmax = emax = 0.0
        for nm in common:
            i, j = names.index(nm), rn.index(nm)
            d = val[i] - r["theta"][j]
            de = err[i] - r["err"][j]
            rel = abs(d) / max(r["err"][j], 1e-300)
            rele = abs(de) / max(r["err"][j], 1e-300)
            dmax = max(dmax, rel)
            emax = max(emax, rele)
        for nm in sorted(
            common,
            key=lambda nm: -abs(val[names.index(nm)] - r["theta"][rn.index(nm)]),
        )[: args.top]:
            i, j = names.index(nm), rn.index(nm)
            print(
                f"{nm:28s} {val[i]:15.8f} {r['theta'][j]:15.8f} "
                f"{(val[i]-r['theta'][j])/max(r['err'][j],1e-300):10.2e} "
                f"{err[i]:12.8f} {r['err'][j]:12.8f} "
                f"{(err[i]-r['err'][j])/max(r['err'][j],1e-300):10.2e}"
            )
        print(
            f"  max |value difference| / error = {dmax:.3e}\n"
            f"  max |error difference| / error = {emax:.3e}"
        )
        ok &= dmax < args.tol and emax < args.tol
        print("  PASS" if ok else f"  FAIL (tolerance {args.tol:g})")

    if args.fit2:
        n2, v2, e2, _, nll2 = read_fit(args.fit2)
        print(f"\n--- {args.label} vs {args.label2} ({args.fit2})")
        if nll2:
            print("    " + "  ".join(f"{k}={v:.6f}" for k, v in nll2.items()))
        common = [nm for nm in names if nm in set(n2)]
        inj = {}
        for spec in args.injected:
            nm, v = spec.rsplit(":", 1)
            inj[nm] = float(v)
        head = f"{'parameter':28s} {args.label:>15s} {args.label2:>15s} {'diff':>14s}"
        if inj:
            head += f" {'injected':>14s} {'pull':>8s}"
        head += f" {'err':>12s}"
        print(head)
        pulls = []
        for nm in common:
            i, j = names.index(nm), n2.index(nm)
            d = val[i] - v2[j]
            line = f"{nm:28s} {val[i]:15.8f} {v2[j]:15.8f} {d:14.6e}"
            if inj:
                want = inj.get(nm, 0.0)
                pull = (d - want) / max(err[i], 1e-300)
                pulls.append(pull)
                line += f" {want:14.6e} {pull:8.2f}"
            line += f" {err[i]:12.8f}"
            if nm in inj or abs(d) > 0.0:
                print(line)
        if pulls:
            pulls = np.asarray(pulls)
            print(
                f"  pulls: mean {pulls.mean():+.3f}, rms {pulls.std():.3f}, "
                f"max |pull| {np.abs(pulls).max():.3f}"
            )

    if args.card and (args.breakdown or args.corr):
        import tensorflow as tf

        from rabbit import fitter, inputdata
        from rabbit.param_models.helpers import load_models

        indata = inputdata.FitInputData(args.card)
        aux = getattr(indata, "auxiliary", {}) or {}
        if "global_index_map" in aux and "provenance" in aux["global_index_map"]:
            print("\n--- card provenance")
            print("    " + json.dumps(json.loads(aux["global_index_map"]["provenance"][0]), indent=6)[6:])
        models = []
        if getattr(indata, "unbinned_terms", []):
            models.append(["UnbinnedParams"])
        if "global_params" in aux:
            models.append(["ExternalParams", "bundle:global_params"])
        if not models:
            sys.exit("cannot infer the param model of the card")

        class _Opt:
            earlyStopping = -1
            noBinByBinStat = True
            binByBinStatMode = "lite"
            binByBinStatType = "automatic"
            covarianceFit = False
            chisqFit = False
            diagnostics = False
            minimizerMethod = "trust-krylov"
            prefitUnconstrainedNuisanceUncertainty = 0.0
            freezeParameters = []
            setConstraintMinimum = []
            unblind = []
            blindingGroup = []

        f = fitter.Fitter(indata, load_models(models, indata), _Opt())
        f.set_nobs(f.indata.data_obs)
        fparms = list(f.parms.astype(str))
        x = f.x.numpy()
        for i, nm in enumerate(fparms):
            if nm in names:
                x[i] = val[names.index(nm)]
        f.x.assign(tf.constant(x, dtype=f.x.dtype))
        if args.breakdown:
            print("\n--- NLL breakdown at the fitted point")
            tot = float(f.full_nll().numpy())
            ext = f._compute_external_nll(full_nll=True)
            unb = f._compute_unbinned_nll(full_nll=True)
            lc = float(f._compute_lc(full_nll=True).numpy())
            print(f"    total (full)          {tot:18.6f}")
            if ext is not None:
                print(f"    external (hit chi2)   {float(ext.numpy()):18.6f}")
            if unb is not None:
                print(f"    unbinned (mass)       {float(unb.numpy()):18.6f}")
            print(f"    constraints/priors    {lc:18.6f}")

    if args.corr and cov is not None:
        targets = [s for s in args.corr.split(",") if s]
        print("\n--- correlations")
        for tgt in targets:
            if tgt not in names:
                print(f"  {tgt}: not in the fit")
                continue
            i = names.index(tgt)
            rho = cov[i] / np.sqrt(np.maximum(cov[i, i] * np.diag(cov), 1e-300))
            order = np.argsort(-np.abs(rho))
            print(f"  {tgt} (err {err[i]:.6g}):")
            for j in order[: args.top]:
                if j == i:
                    continue
                print(f"      {names[j]:28s} rho = {rho[j]:+.4f}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

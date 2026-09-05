#!/usr/bin/env python3
"""Systematics of the Z-channel prototype: what each modelling choice costs.

Builds several variants of the term from the same candidates and, for each,
reports (a) the expected errors at ``--project`` candidates from the
reference-point (Asimov) information, and (b) the fitted ``m_Z`` / ``Gamma_Z``
with the resolution scales held fixed.  The *shift* of the fitted ``m_Z``
between a variant and the baseline is the modelling bias that choice induces.

Variants:

``baseline``      additive FSR kernel from gen events inside the reconstruction
                  acceptance (pT > 5 GeV, abs(eta) < 2.4), Born
                  support 50-130 GeV, truncated likelihood on 60-120 GeV
``no-window-norm``the same without the truncation normalisation -- the
                  likelihood the term had before ``norm_window`` existed
``norm-exact``    one resolution class per candidate instead of 32
``kernel-pt26``   the FSR kernel built with a pT > 26 GeV gen acceptance
``kernel-noacc``  the FSR kernel with no gen acceptance at all
``kernel-mult``   the FSR kernel built from ``dm/m_pre * m_Z``, i.e. the best
                  single additive kernel if FSR is a pure rescaling
``no-fsr``        no FSR kernel at all (shows the size of the effect)
``born-200``      Born support 50-200 GeV instead of 50-130
``born-60-120``   Born support equal to the selection window (wrong on purpose:
                  it forbids the Born masses that FSR moves into the window)
``sigma-max-5``   drop the tail of very poorly measured pairs

Usage::

    python z_variants.py --pairs data/zpairs_smoke.npz --kdir data \\
        --project 3.9e6
"""

import argparse
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

import fit_z
import make_z_card

DTYPE = tf.float64


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--kdir", default="data", help="directory of the FSR kernels")
    p.add_argument("--project", type=float, default=3.9e6)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--k-prior", type=float, nargs="+", default=[0.0, 0.01, 0.001],
                   help="external Gaussian constraint on the resolution scales")
    return p.parse_args()


def variants(args):
    K = args.kdir
    base = ["--pairs", args.pairs, "--kernel", f"{K}/zfsr_kernel_reco.npz",
            "--norm-classes", "32"]

    def alt(*extra):
        return base + list(extra)

    return {
        "baseline": base,
        "no-window-norm": alt("--no-window-norm"),
        "norm-exact": alt("--norm-classes", "0"),
        "kernel-pt26": alt("--kernel", f"{K}/zfsr_kernel_acc.npz"),
        "kernel-noacc": alt("--kernel", f"{K}/zfsr_kernel_noacc.npz"),
        "kernel-mult": alt("--kernel", f"{K}/zfsr_kernel_mult.npz"),
        "no-fsr": alt("--kernel", f"{K}/zfsr_kernel_none.npz"),
        "born-200": alt("--born-window", "50", "200"),
        "born-60-120": alt("--born-window", "60", "120"),
        "sigma-max-5": alt("--sigma-max", "5"),
    }


def asimov(term, names, k_prior):
    """Errors and rho(m_Z, Gamma_Z) from the reference-point information.

    ``k_prior`` > 0 puts a Gaussian of that width on every resolution scale,
    standing in for the constraint the J/psi channel supplies in the unified
    likelihood; 0 leaves them free.  Returns ``(None, nan)`` if the resulting
    information is not positive definite -- at a few hundred candidates the Z
    alone does not determine the resolution, and the reference point is then
    not even a local minimum in those directions.
    """
    sig = np.asarray(term.param_prior_sigmas, dtype=np.float64).copy()
    if k_prior > 0:
        for i, nm in enumerate(names):
            if nm.startswith("k_"):
                sig[i] = min(sig[i], k_prior) if np.isfinite(sig[i]) else k_prior
    keep = term.param_prior_sigmas
    term.param_prior_sigmas = sig
    try:
        obj = fit_z.Objective(term)
        H = obj.hess(obj.x0)
    finally:
        term.param_prior_sigmas = keep
    if not np.all(np.isfinite(H)) or np.min(np.linalg.eigvalsh(H)) <= 0:
        return None, np.nan
    C = np.linalg.inv(H)
    e = np.sqrt(np.diag(C))
    rho = C[names.index("m_Z"), names.index("Gamma_Z")] / (
        e[names.index("m_Z")] * e[names.index("Gamma_Z")])
    return dict(zip(names, e)), rho


def main():
    args = parse_args()
    vs = variants(args)
    if args.only:
        vs = {k: v for k, v in vs.items() if k in args.only}

    rows = []
    for name, argv in vs.items():
        print(f"\n{'='*70}\n=== {name}\n{'='*70}")
        try:
            term, datasets, decl, info = make_z_card.build(
                make_z_card.parse_args(argv), log=lambda *a: None)
        except SystemExit as e:
            print(f"  SKIP: {e}")
            continue
        names = list(term.param_names)
        t0 = time.time()
        f = np.sqrt(term.n / args.project)
        # (a) the resolution scales profiled under an external constraint of
        #     width --k-prior. Free (0) is included and is normally indefinite
        #     at 449 candidates: the Z alone does not determine them.
        prof = {}
        for kp in args.k_prior:
            e, rho = asimov(term, names, kp)
            prof[kp] = (e, rho)
            tag = "free" if kp == 0 else f"+-{kp:g}"
            if e is None:
                print(f"    k prior {tag:>8s}: information indefinite")
            else:
                print(f"    k prior {tag:>8s}: s(m_Z) = {e['m_Z']*f:7.3f} MeV, "
                      f"s(Gamma_Z) = {e['Gamma_Z']*f:7.3f} MeV, "
                      f"rho(m_Z,Gamma_Z) = {rho:+.3f}")
        # (b) resolution fixed: Asimov information and the actual fit
        free = [i for i, nm in enumerate(names) if nm in ("m_Z", "Gamma_Z")]
        obj2 = fit_z.Objective(term, free)
        Hf = obj2.hess(obj2.x0[free])
        e_fix = dict(zip(obj2.freenames, np.sqrt(np.diag(np.linalg.inv(Hf)))))
        res = minimize(obj2.value_grad, obj2.x0[free], jac=True, hess=obj2.hess,
                       method="trust-exact")
        Hm = obj2.hess(res.x)
        emz, egz = np.sqrt(np.diag(np.linalg.inv(Hm)))
        rows.append(dict(
            name=name, n=term.n, f=f, prof=prof,
            mz_fix=e_fix["m_Z"] * f, gz_fix=e_fix["Gamma_Z"] * f,
            mz_fit=res.x[0], gz_fit=res.x[1], emz=emz, egz=egz,
            t=time.time() - t0))
        print(f"    resolution fixed: s(m_Z) = {rows[-1]['mz_fix']:7.3f} MeV, "
              f"s(Gamma_Z) = {rows[-1]['gz_fix']:7.3f} MeV")
        print(f"    fit: m_Z = {res.x[0]:+.1f} +- {emz:.1f} MeV, "
              f"Gamma_Z = {res.x[1]:+.1f} +- {egz:.1f} MeV  "
              f"[{rows[-1]['t']:.0f} s]")

    if not rows:
        return
    ref = next((r for r in rows if r["name"] == "baseline"), rows[0])
    kps = args.k_prior
    print(f"\n\n{'='*118}")
    print(f"Expected errors at N = {args.project:.3g} candidates, MeV "
          f"(reference-point information scaled by N/n = {args.project/ref['n']:.0f})")
    print(f"{'='*118}")
    hdr = "".join(f" {('free' if k==0 else f'k+-{k:g}'):>16s}" for k in kps)
    print(f"{'variant':>15s} {'n':>5s} |{hdr} {'k fixed':>16s} |"
          f" {'fit m_Z':>9s} {'+-':>7s} {'fit G_Z':>9s} {'d(m_Z)':>8s}")
    print(f"{'':>15s} {'':>5s} |" + "".join(f" {'m_Z / Gamma_Z':>16s}"
                                            for _ in kps)
          + f" {'m_Z / Gamma_Z':>16s} |")
    for r in rows:
        cells = ""
        for k in kps:
            e, _ = r["prof"][k]
            cells += (f" {'  indefinite':>16s}" if e is None else
                      f" {e['m_Z']*r['f']:7.3f}/{e['Gamma_Z']*r['f']:8.3f}")
        d = r["mz_fit"] - ref["mz_fit"]
        print(f"{r['name']:>15s} {r['n']:5d} |{cells} "
              f"{r['mz_fix']:7.3f}/{r['gz_fix']:8.3f} |"
              f" {r['mz_fit']:+9.1f} {r['emz']:7.1f} {r['gz_fit']:+9.1f} "
              f"{d:+8.1f}")
    print("\n  'k +- x' constrains every resolution scale with a Gaussian of "
          "width x -- which is what\n  the J/psi channel of the unified "
          "likelihood supplies. 'free' is what the Z alone can do\n  (normally "
          "nothing: the information is indefinite at 449 candidates).")
    print("  The last three columns are the actual fit with the resolution "
          "held at 1; d(m_Z) is the\n  shift with respect to the baseline, "
          "i.e. the modelling bias of that choice.")
    for r in rows:
        e, rho = r["prof"][kps[-1]]
        if e is not None:
            print(f"\n  {r['name']}: correlations at k +- {kps[-1]:g} -> "
                  f"rho(m_Z, Gamma_Z) = {rho:+.4f}")
            break


if __name__ == "__main__":
    main()

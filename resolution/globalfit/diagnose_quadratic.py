#!/usr/bin/env python3
"""Diagnose the hit-chi2-only (quadratic) problem over the global parameters.

Solving ``theta = -K^-1 G`` on the 50 scalar-potential field modes + 42
material groups from MC gives coefficients of order 1e2-1e3 with 300-1000
sigma pulls, which looks alarming until one notices that **the raw
coefficients are not comparable to each other**: a parmtype-14 coefficient is
in T cm and multiplies a basis function ``(R/r_scale)^(l-1)/r_scale``, so a
high-l mode needs a huge coefficient to move the field at all inside the
tracker (at R = 50 cm, ``(50/320)^6 ~ 1e-5``). This script therefore works in
a **whitened** basis where every parameter is scaled by its own physical unit:

* parmtype 14: ``s_j`` = RMS over the tracker volume of ``|dB_j|`` per unit
  coefficient (Tesla), evaluated with ``mfs/harmonic_basis.py`` on the same
  (l, m, cos/sin) assignment as the coefficient dump the production used. So
  ``theta_j * s_j`` is the RMS field change in Tesla that mode contributes.
* parmtype 15: ``s_j`` = the per-group prior sigma ``gprior`` from the
  materialGroups tier file (column 11), so the whitened value is
  ``gprior * k`` for a log energy-loss scale ``k``.

In that metric it reports

1. the eigenspectrum of ``K_tilde = S K S`` (condition number; the near-null
   directions and which parameters they mix),
2. the projection of the gradient on the eigenbasis, ``theta_i = -g_i/lambda_i``
   and its pull ``-g_i/sqrt(2 lambda_i)`` -- which eigen-directions carry the
   large corrections and the large significances,
3. the solve with and without the calibration's priors, and
4. the solve restricted to the field modes alone and to the material groups
   alone (the 14 x 15 cross-block is where most of the degeneracy lives).

Usage::

    python diagnose_quadratic.py -i runs/globalfit/extract_quadonly_*.npz \\
        --groups .../materialGroups50.txt --coeffs .../custom50.txt
"""

import argparse
import os
import sys

import numpy as np

from make_global_term import (  # noqa: F401
    B_NOMINAL,
    DEFAULT_COEFFS,
    MFS,
    R_SCALE_CM,
    field_scales,
    name_params,
    read_groups,
    read_modes,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="extract.py npz")
    p.add_argument("--groups", default=None)
    p.add_argument("--coeffs", default=DEFAULT_COEFFS, help="50-mode dump")
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument("--rmax", type=float, default=110.0, help="tracker r [cm]")
    p.add_argument("--zmax", type=float, default=280.0, help="tracker |z| [cm]")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("-o", "--output", default=None, help="save the scales npz")
    return p.parse_args()


# ---------------------------------------------------------------------------
def show_spectrum(label, Kt, names, top):
    lam, V = np.linalg.eigh(Kt)
    lam = np.maximum(lam, 0.0)
    pos = lam[lam > 0]
    cond = pos.max() / pos.min() if len(pos) else np.inf
    print(f"\n--- {label}: {len(lam)} parameters")
    print(
        f"    lambda: max {lam.max():.4e}  min {lam.min():.4e}  "
        f"cond {cond:.3e}  (rank>1e-12*max: "
        f"{int((lam > 1e-12*lam.max()).sum())})"
    )
    print("    softest directions (parameter content > 0.15):")
    for k in range(min(top, len(lam))):
        v = V[:, k]
        big = np.argsort(-np.abs(v))
        content = ", ".join(
            f"{names[i]}({v[i]:+.2f})" for i in big[:5] if abs(v[i]) > 0.15
        )
        print(f"      lam[{k}] = {lam[k]:.4e}  {content}")
    return lam, V


def show_solve(label, K, G, names, scale, priors=None, top=10, field=None):
    """Solve (K + 2P) theta = -G in the WHITENED basis.

    Each entry of ``theta`` is the raw coefficient times its scale ``s_j``:
    Tesla of RMS field change for a parmtype-14 mode, ``gprior * k`` for a
    parmtype-15 group.
    """
    P = np.zeros(len(names)) if priors is None else priors
    Kp = K + np.diag(2.0 * P)
    lam0 = np.linalg.eigvalsh(Kp)
    dead = np.where(np.abs(np.diag(Kp)) < 1e-12 * np.abs(np.diag(Kp)).max())[0]
    if lam0.min() <= 1e-12 * lam0.max():
        # a parameter no track ever touches (or an exact degeneracy): report it
        # and fall back to the pseudo-inverse so the rest of the block is still
        # solvable. This is a real feature of the problem, not a numerical
        # accident -- e.g. material_pp1_cables sits outside every J/psi track.
        print(
            f"\n--- {label}: SINGULAR (min/max eigenvalue "
            f"{lam0.min()/max(lam0.max(),1e-300):.2e}); using the pseudo-inverse."
            + (f" Zero-information parameters: {[names[i] for i in dead][:6]}"
               if len(dead) else "")
        )
        Kinv = np.linalg.pinv(Kp, rcond=1e-10)
    else:
        print(f"\n--- {label}")
        Kinv = np.linalg.inv(Kp)
    theta = -Kinv @ G
    cov = 2.0 * Kinv
    err = np.sqrt(np.abs(np.diag(cov)))
    pull = theta / np.maximum(err, 1e-300)
    print(
        f"    max|pull| {np.abs(pull).max():8.1f}   "
        f"rms pull {np.sqrt((pull**2).mean()):8.1f}   "
        f"chi2 improvement {float(-G @ theta):.4e}"
    )
    if field is not None and field.any():
        tot = np.sqrt(np.sum(theta[field] ** 2))
        print(
            f"    quadrature sum of the field modes: |dB|_rms = {tot*1e3:.4f} mT "
            f"= {tot/B_NOMINAL*1e3:.4f} e-3 relative"
        )
        print(
            f"    raw coefficient scale: max |theta_raw| = "
            f"{np.abs(theta/np.maximum(scale,1e-300)).max():.4g} (T cm)"
        )
    print(
        f"    {'parameter':26s} {'value':>13s} {'err':>11s} {'pull':>8s}  unit"
    )
    for i in np.argsort(-np.abs(pull))[:top]:
        unit = "T (|dB|rms)" if (field is not None and field[i]) else "k * gprior"
        print(
            f"    {names[i]:26s} {theta[i]:13.5g} {err[i]:11.5g} "
            f"{pull[i]:8.1f}  {unit}"
        )
    return theta, err, pull


def main():
    args = parse_args()
    d = np.load(args.input)
    parmtype, subidx = d["parmtype"], d["subidx"]
    names, prior_sigmas = name_params(
        parmtype, subidx, args.groups, args.field_prior, args.material_prior_scale
    )
    G = d["grad"].astype(np.float64)
    K = d["hess"].astype(np.float64)
    n = len(names)
    print(
        f"{args.input}\n  {int(d['ncand_quadratic'])} candidates, {n} parameters "
        f"{ {int(p): int((parmtype==p).sum()) for p in sorted(set(parmtype.tolist()))} }"
    )

    # ---- physical scales -------------------------------------------------
    modes, r_scale, cmssw_norm = read_modes(args.coeffs)
    print(
        f"  coefficient dump: {len(modes)} modes, r_scale {r_scale:.4f} cm, "
        f"cmssw_norm {cmssw_norm}"
    )
    fs = field_scales(modes, r_scale, cmssw_norm, args.rmax, args.zmax)
    _, gprior = read_groups(args.groups)
    scale = np.ones(n)
    is14 = parmtype == 14
    is15 = parmtype == 15
    for i in np.where(is14)[0]:
        j = int(subidx[i])
        scale[i] = fs[j] if j < len(fs) else 1.0
    for i in np.where(is15)[0]:
        scale[i] = gprior.get(int(subidx[i]), 0.02)
    print("\n  mode |dB| RMS in the tracker per unit coefficient [T]:")
    for j in (0, 1, 3, 10, 20, 30, 40, 49):
        if j < len(modes):
            l, m, cs = modes[j]
            print(
                f"    mode{j:<3d} (l={l:2d}, m={m:2d}, {cs})  "
                f"{fs[j]:.4e} T   -> 1 unit = {fs[j]/B_NOMINAL*1e3:.3e} e-3 relative"
            )

    # whiten: theta_tilde_j = theta_j * s_j is the physical quantity, so
    # theta = S^-1 theta_tilde and K_tilde = S^-1 K S^-1, G_tilde = G / s.
    inv = 1.0 / np.maximum(scale, 1e-300)
    Kt = K * np.outer(inv, inv)
    Gt = G * inv

    # ---- 1. eigenspectrum -------------------------------------------------
    show_spectrum("whitened K (all)", Kt, names, args.top)
    if is14.any():
        show_spectrum(
            "whitened K (field modes only)",
            Kt[np.ix_(is14, is14)],
            [names[i] for i in np.where(is14)[0]],
            5,
        )
    if is15.any():
        show_spectrum(
            "whitened K (material groups only)",
            Kt[np.ix_(is15, is15)],
            [names[i] for i in np.where(is15)[0]],
            5,
        )

    # ---- 2. gradient in the eigenbasis ------------------------------------
    lam, V = np.linalg.eigh(Kt)
    gi = V.T @ Gt
    ti = -gi / np.maximum(lam, 1e-300)
    pi = -gi / np.sqrt(np.maximum(2.0 * lam, 1e-300))
    print("\n--- gradient in the eigenbasis (largest |pull| first)")
    print(f"    {'i':>4s} {'lambda':>12s} {'g_i':>12s} {'theta_i':>12s} {'pull_i':>10s}  content")
    for k in np.argsort(-np.abs(pi))[: args.top]:
        v = V[:, k]
        big = np.argsort(-np.abs(v))
        content = ", ".join(
            f"{names[i]}({v[i]:+.2f})" for i in big[:4] if abs(v[i]) > 0.2
        )
        print(
            f"    {k:4d} {lam[k]:12.4e} {gi[k]:12.4e} {ti[k]:12.4e} "
            f"{pi[k]:10.1f}  {content}"
        )

    # ---- 3/4. solves -------------------------------------------------------
    fld = is14
    show_solve("no priors, all 92", Kt, Gt, names, scale, None, args.top, fld)

    # material priors: one whitened unit per group
    pri_mat = np.where(is15, 1.0, 0.0)
    show_solve(
        "material priors only (their own prior sigmas)",
        Kt, Gt, names, scale, pri_mat, args.top, fld,
    )

    for fp in (1e-3, 1e-4):
        sig = fp * B_NOMINAL  # whitened sigma in Tesla
        pri = np.where(is14, 1.0 / sig**2, pri_mat)
        show_solve(
            f"field prior |dB|rms/B < {fp:g} per mode ({sig*1e3:.3f} mT) "
            "+ material priors",
            Kt, Gt, names, scale, pri, args.top, fld,
        )

    if is14.any():
        i14 = np.where(is14)[0]
        show_solve(
            "field modes only, no priors",
            Kt[np.ix_(i14, i14)], Gt[i14], [names[i] for i in i14],
            scale[i14], None, 6, np.ones(len(i14), dtype=bool),
        )
    if is15.any():
        i15 = np.where(is15)[0]
        show_solve(
            "material groups only, no priors",
            Kt[np.ix_(i15, i15)], Gt[i15], [names[i] for i in i15],
            scale[i15], None, 6, np.zeros(len(i15), dtype=bool),
        )

    if args.output:
        np.savez(
            args.output,
            names=np.array(names),
            scale=scale,
            field_scale=fs,
            parmtype=parmtype,
            subidx=subidx,
            lam=lam,
            V=V,
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

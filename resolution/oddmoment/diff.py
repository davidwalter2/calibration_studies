#!/usr/bin/env python3
"""Differential dependence of the charge-decomposed q/p odd moment.

For every binning variable it reports, per bin,
    A = odd-in-q part  = <q z>            ("momentum scale" channel)
    S = even-in-q part = <z>              ("sagitta bias" channel)
in two estimators: the bounded <z e^{-uz^2}> at u = 0.05 (data and CF model,
per track) and the trimmed mean <z>_{|z|<5} (data; model from the pooled CF
of the bin, inverted to a density).

Writes a markdown table and an .npz for the figure step.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)
import decomp as D                                            # noqa: E402

TG = D.TG


def trim_mean(z, T=5.0):
    s = z[np.abs(z) < T]
    return float(s.mean()) if len(s) else np.nan


def model_trim(phibar, T=5.0, zmax=12., nz=4801):
    zg = np.linspace(-zmax, zmax, nz)
    p = D.density(phibar, zg)
    s = np.abs(zg) < T
    return float(np.trapezoid(zg[s] * p[s], zg[s]) / np.trapezoid(p[s], zg[s]))


def binnings(d):
    eta = d["eta"].astype(np.float64)
    phi = d["phi"].astype(np.float64)
    out = []
    out.append(("eta", eta, np.linspace(-2.4, 2.4, 13)))
    out.append(("abseta", np.abs(eta), np.array([0., .4, .8, 1.2, 1.6, 2.0, 2.4])))
    out.append(("phi", phi, np.linspace(-np.pi, np.pi, 13)))
    out.append(("genpt", d["genpt"], np.quantile(d["genpt"], np.linspace(0, 1, 7))))
    nv = d["nvalidhits"]
    out.append(("nvalidhits", nv, np.array([0, 10.5, 12.5, 14.5, 16.5, 40])))
    out.append(("normchi2", d["normchi2"],
                np.quantile(d["normchi2"], np.linspace(0, 1, 6))))
    out.append(("sigma", d["sigma"], np.quantile(d["sigma"], np.linspace(0, 1, 6))))
    out.append(("vgf", d["vgf"], np.quantile(d["vgf"], np.linspace(0, 1, 6))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--trim", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    args = D.Args()
    args.krad = a.krad
    d = D.load(a.cache)
    z = d["z"]
    q = np.sign(d["charge"]).astype(np.float64)
    zh = q * z                              # folded: <zh> = A, <z> = S
    n = len(z)

    probes = (a.u,)
    L = []
    P = L.append
    lab = a.label or os.path.basename(a.cache)
    P(f"# {lab}   n = {n}   krad = {a.krad}   u = {a.u}   trim = |z| < {a.trim}")
    store = {}

    for name, v, edges in binnings(d):
        nb = len(edges) - 1
        bid = np.clip(np.digitize(v, edges[1:-1]), 0, nb - 1)
        # per-track model odd moment, pooled phi per bin (and per charge)
        pool = bid * 2 + (q < 0)
        Om, Em, phis, cnt = D.model_per_track(d, args, probes,
                                              pool_id=pool, npool=2 * nb)
        P("")
        P(f"## {name}")
        P("| bin | n | A_dat(u) | A_mod(u) | dA(u) | A_dat(trim) | A_mod(trim) "
          "| dA(trim) | S_dat(trim) |")
        P("|---|---|---|---|---|---|---|---|---|")
        rows = []
        for k in range(nb):
            msk = bid == k
            nk = int(msk.sum())
            if nk < 200:
                continue
            zp, zm = z[msk & (q > 0)], z[msk & (q < 0)]
            Adu = 0.5 * ((zp * np.exp(-a.u * zp ** 2)).mean()
                         - (zm * np.exp(-a.u * zm ** 2)).mean())
            Amu = 0.5 * (Om[msk & (q > 0), 0].mean() - Om[msk & (q < 0), 0].mean())
            Adt = 0.5 * (trim_mean(zp, a.trim) - trim_mean(zm, a.trim))
            Sdt = 0.5 * (trim_mean(zp, a.trim) + trim_mean(zm, a.trim))
            Amt = 0.5 * (model_trim(phis[2 * k], a.trim)
                         - model_trim(phis[2 * k + 1], a.trim))
            # bootstrap-free analytic error on the trimmed A
            sp = zp[np.abs(zp) < a.trim]; sm = zm[np.abs(zm) < a.trim]
            eA = 0.5 * np.hypot(sp.std(ddof=1) / np.sqrt(len(sp)),
                                sm.std(ddof=1) / np.sqrt(len(sm)))
            P(f"| {edges[k]:.3g}..{edges[k+1]:.3g} | {nk} | {Adu:+.5f} | {Amu:+.5f} "
              f"| {Adu-Amu:+.5f} | {Adt:+.5f} | {Amt:+.5f} | {Adt-Amt:+.5f} +- {eA:.5f} "
              f"| {Sdt:+.5f} |")
            rows.append((0.5 * (edges[k] + edges[k + 1]), nk, Adu, Amu,
                         Adt, Amt, eA, Sdt))
        store[name] = np.array(rows)
        store[name + "_edges"] = edges

    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    np.savez_compressed(a.out.replace(".txt", ".npz"), **store)


if __name__ == "__main__":
    main()

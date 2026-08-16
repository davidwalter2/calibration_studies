#!/usr/bin/env python3
"""Where in z does the layered-toy `qop` clean-propagation non-closure live, and
which Geant4 process puts it there?

CONTEXT.  NOTES_TOY_PT40 measured the residual; NOTES_RADOFF excluded radiation
(brems + pair) as the cause; NOTES_URBANSAMPLING excluded the ionization
straggling LAW and, more strongly, any change of the ionization SECOND CUMULANT
of any size.  The residual peaks at u = 0.03, i.e. |z| ~ 6 s_F, so the search
moves to rare, hard, one-sided q/p deviations.

THIS MODULE DOES TWO THINGS, and the first needs no new simulation:

    zloc     Decompose the closure number itself into |z| bins.
             closure(u) = <e^{-u z^2}>_data - <e^{-u z^2}>_model is an integral
             over z of (p_data - p_model) e^{-u z^2}.  Binning both halves in
             |z| says WHERE the two densities disagree, which is a far sharper
             statement than the u-curve.  The model half is the exact FFT
             inversion of the SAME CF `weier_scalar` integrates, and the sum
             over bins is checked against `weier_scalar` itself.

    tailobs  For the events in each |z| bin, print the observables the existing
             ntuple already carries -- cumulative momentum loss, per-shell
             loss -- so the tail is characterised before any new watcher runs.

Nothing here writes to the CMSSW source tree and nothing re-runs a simulation;
the inputs are the archived 400k NOTES_TOY_PT40 samples.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                               # noqa: E402
import cgf_channels as cc                                       # noqa: E402
import fisher_norm as fn                                        # noqa: E402
import toy_radoff as tr                                         # noqa: E402
from cf_propagation_test import FUNCTIONALS                     # noqa: E402

UCURVE = fn.UCURVE

# |z| bin edges in units of s_F.  The probe grid runs u = 1e-3 ... 10, i.e.
# characteristic |z| = 1/sqrt(u) from 32 down to 0.32, so the binning has to
# resolve that whole range.
ZEDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0,
                   40.0, np.inf])

# SIGNED edges.  The residual is strongly one-sided -- for mu- (q/p = -1/p) a
# larger-than-reference energy loss moves q/p DOWN, so the hard-collision tail
# is entirely at z < 0, while the Landau mode sits at z > 0 because the
# reference subtracts the MEAN loss.  Folding |z| hides that completely.
SEDGES = np.array([-np.inf, -40, -20, -12, -8, -6, -4, -3, -2, -1.5, -1, -0.5,
                   0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12,
                   np.inf])


def _model_density(legs, k, avec, sF, channels):
    """(z, p) of the model's standardized residual, z in units of s_F.

    `sigma=sF` is the same standardization `_closure_model_one` uses when it
    calls `model_phi(legs, k, avec, s_F, tau)`, so this density and the closure
    number are the same object; `zloc` checks that explicitly by comparing
    int p e^{-u z^2} dz against `weier_scalar`.
    """
    return cc.exact_density(legs, k, avec, sF, channels=channels,
                            deriv=False)[:2]


def _cum(z, p, u):
    """Cumulative int_{-inf}^{z} p e^{-u z'^2} dz' on the FFT grid.

    Bin values are DIFFERENCES of this, interpolated at the edges, rather than
    a trapezoid over the points inside each bin.  The naive form drops one grid
    spacing at every bin boundary -- with 28 boundaries and dz = 0.0025 that
    lost 0.5 % of the model's mass, i.e. as much as the closure number being
    decomposed.  Differencing a cumulative integral sums to the total exactly
    by construction.
    """
    w = p * np.exp(-u * z ** 2)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z))])
    return c


def _cbin(z, c, edges):
    """Bin integrals from a cumulative integral, edges clipped to the grid."""
    e = np.clip(np.asarray(edges, dtype=float), z[0], z[-1])
    v = np.interp(e, z, c)
    return np.diff(v)


def _binint(z, p, edges, u):
    """int_bin p(z) e^{-u z^2} dz over |z| bins, on the FFT grid."""
    c = _cum(z, p, u)
    out = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        out[i] = (_cbin(z, c, [lo, hi])[0]        # positive half
                  + _cbin(z, c, [-hi, -lo])[0])   # negative half
    return out


def _sbinint(z, p, edges, u):
    """int_bin p(z) e^{-u z^2} dz over SIGNED bins, on the FFT grid."""
    return _cbin(z, _cum(z, p, u), edges)


def _zdata(tag, func="qop"):
    """(z, valid) per event per plane, plus the per-plane scale and legs."""
    cfg = tr.CONFIGS[tag]
    tr.set_rad(cfg["rad"])
    legs, sim = tr.legs_of(tag), tr.sim_of(tag)
    sc = fn.plane_scales(legs, func, tag=tag, channels=tr.chans(cfg["rad"]))
    nl = min(sim["valid"].shape[1], len(legs))
    z = np.full(sim["valid"].shape[:2], np.nan)
    for k in range(nl):
        good = sim["valid"][:, k] & np.isfinite(sim[cpt.SIM_BRANCH[func]][:, k])
        z[good, k] = ((sim[cpt.SIM_BRANCH[func]][good, k]
                       - legs[k][cpt.REF_BRANCH[func]]) / float(sc["sF"][k]))
    tr.set_rad(True)
    return z, sim, legs, sc, nl


# ==========================================================================
# subcommand: zloc
# ==========================================================================

def cmd_zloc(args):
    print("=" * 100)
    print("CLOSURE DECOMPOSED IN |z|:  closure(u) = sum_bins [ <e^{-u z^2}>_data"
          " - int_bin p_model e^{-u z^2} ]")
    print("=" * 100)
    for tag in args.configs:
        cfg = tr.CONFIGS[tag]
        chans = tr.chans(cfg["rad"])
        z, sim, legs, sc, nl = _zdata(tag, args.func)
        tr.set_rad(cfg["rad"])
        tau = fn.closure_tau(float(np.max(UCURVE)))
        avec = FUNCTIONALS[args.func]
        print(f"\n### {tag}  ({args.func}, {sim['valid'].shape[0]} events, "
              f"{nl} planes, channels={chans})")

        # ---- occupancy first: p_data vs p_model per |z| bin, u = 0
        occ_d = np.zeros(len(ZEDGES) - 1)
        occ_m = np.zeros(len(ZEDGES) - 1)
        ktot = 0
        dec_d = np.zeros((len(UCURVE), len(ZEDGES) - 1))
        dec_m = np.zeros((len(UCURVE), len(ZEDGES) - 1))
        wsum = np.zeros(len(UCURVE))
        for k in range(nl):
            good = np.isfinite(z[:, k])
            if good.sum() < 100:
                continue
            zk = z[good, k]
            azk = np.abs(zk)
            zm, pm = _model_density(legs, k, avec, float(sc["sF"][k]), chans)
            phi = cpt.model_phi(legs, k, avec, float(sc["sF"][k]), tau)
            for iu, u in enumerate(UCURVE):
                e = np.exp(-u * zk ** 2)
                for i in range(len(ZEDGES) - 1):
                    m = (azk >= ZEDGES[i]) & (azk < ZEDGES[i + 1])
                    dec_d[iu, i] += e[m].sum() / good.sum()
                dec_m[iu] += _binint(zm, pm, ZEDGES, u)
                wsum[iu] += cpt.weier_scalar(phi, u, tau)
            for i in range(len(ZEDGES) - 1):
                occ_d[i] += ((azk >= ZEDGES[i]) & (azk < ZEDGES[i + 1])
                             ).sum() / good.sum()
            occ_m += _binint(zm, pm, ZEDGES, 0.0)
            ktot += 1
        dec_d /= ktot
        dec_m /= ktot
        wsum /= ktot
        occ_d /= ktot
        occ_m /= ktot
        tr.set_rad(True)

        # sanity: the binned model must reproduce weier_scalar
        print("  closure-integral consistency  (binned model vs weier_scalar):")
        print("    " + "".join(f"{u:>11.3g}" for u in UCURVE))
        print("    " + "".join(f"{v:11.6f}" for v in dec_m.sum(axis=1))
              + "   binned")
        print("    " + "".join(f"{v:11.6f}" for v in wsum) + "   weier")
        print("    " + "".join(f"{v:11.2e}" for v in dec_m.sum(axis=1) - wsum)
              + "   diff")

        print(f"\n  OCCUPANCY per |z| bin (mean over {ktot} planes)")
        print(f"    {'|z| bin':>14}  {'data':>12} {'model':>12} "
              f"{'data-model':>12} {'d/m':>9}")
        for i in range(len(ZEDGES) - 1):
            lab = f"{ZEDGES[i]:g}-{ZEDGES[i+1]:g}"
            r = occ_d[i] / occ_m[i] if occ_m[i] > 1e-12 else np.nan
            print(f"    {lab:>14}  {occ_d[i]:12.6f} {occ_m[i]:12.6f} "
                  f"{occ_d[i]-occ_m[i]:+12.6f} {r:9.3f}")
        print(f"    {'TOTAL':>14}  {occ_d.sum():12.6f} {occ_m.sum():12.6f} "
              f"{occ_d.sum()-occ_m.sum():+12.6f}")

        print(f"\n  CLOSURE CONTRIBUTION per |z| bin (data - model)")
        print(f"    {'|z| bin':>14}" + "".join(f"{u:>11.3g}" for u in UCURVE))
        for i in range(len(ZEDGES) - 1):
            lab = f"{ZEDGES[i]:g}-{ZEDGES[i+1]:g}"
            print(f"    {lab:>14}"
                  + "".join(f"{v:+11.5f}" for v in dec_d[:, i] - dec_m[:, i]))
        print(f"    {'SUM':>14}"
              + "".join(f"{v:+11.5f}" for v in (dec_d - dec_m).sum(axis=1)))

        # ---- SIGNED decomposition.  This is the table that matters: the
        # residual is one-sided and folding |z| hides which side it is on.
        tr.set_rad(cfg["rad"])
        sd = np.zeros(len(SEDGES) - 1)
        sm = np.zeros(len(SEDGES) - 1)
        sdu = np.zeros((len(UCURVE), len(SEDGES) - 1))
        smu = np.zeros((len(UCURVE), len(SEDGES) - 1))
        zlo = zhi = mass = 0.0
        for k in range(nl):
            good = np.isfinite(z[:, k])
            if good.sum() < 100:
                continue
            zk = z[good, k]
            zm, pm = _model_density(legs, k, avec, float(sc["sF"][k]), chans)
            zlo += zm.min()
            zhi += zm.max()
            mass += np.trapezoid(pm, zm)
            idx = np.digitize(zk, SEDGES) - 1
            sd += np.bincount(idx, minlength=len(SEDGES) - 1) / good.sum()
            sm += _sbinint(zm, pm, SEDGES, 0.0)
            for iu, u in enumerate(UCURVE):
                e = np.exp(-u * zk ** 2)
                sdu[iu] += np.bincount(idx, weights=e,
                                       minlength=len(SEDGES) - 1) / good.sum()
                smu[iu] += _sbinint(zm, pm, SEDGES, u)
        sd /= ktot
        sm /= ktot
        sdu /= ktot
        smu /= ktot
        tr.set_rad(True)
        print(f"\n  model FFT grid: z in [{zlo/ktot:.1f}, {zhi/ktot:.1f}], "
              f"int p dz = {mass/ktot:.6f}   (mass outside the grid is the "
              f"binned-vs-weier defect above)")
        print(f"\n  SIGNED occupancy and closure contribution "
              f"(mean over {ktot} planes)")
        print(f"    {'z bin':>14}  {'data':>10} {'model':>10} {'d-m':>10} "
              f"{'d/m':>7}  " + "".join(f"{'c(u='+str(u)+')':>12}"
                                        for u in (0.01, 0.03, 0.1)))
        iu3 = [int(np.argmin(np.abs(UCURVE - u))) for u in (0.01, 0.03, 0.1)]
        for i in range(len(SEDGES) - 1):
            lab = f"{SEDGES[i]:g}:{SEDGES[i+1]:g}"
            r = sd[i] / sm[i] if sm[i] > 1e-9 else np.nan
            print(f"    {lab:>14}  {sd[i]:10.6f} {sm[i]:10.6f} "
                  f"{sd[i]-sm[i]:+10.6f} {r:7.3f}  "
                  + "".join(f"{sdu[j, i]-smu[j, i]:+12.5f}" for j in iu3))
        print(f"    {'TOTAL':>14}  {sd.sum():10.6f} {sm.sum():10.6f} "
              f"{sd.sum()-sm.sum():+10.6f} {'':7s}  "
              + "".join(f"{(sdu[j]-smu[j]).sum():+12.5f}" for j in iu3))


# ==========================================================================
# subcommand: tailobs -- what the existing ntuple says about the tail events
# ==========================================================================

def cmd_tailobs(args):
    print("=" * 100)
    print("TAIL EVENT OBSERVABLES from the archived ntuple (no new sim)")
    print("=" * 100)
    for tag in args.configs:
        z, sim, legs, sc, nl = _zdata(tag, args.func)
        k = nl - 1
        good = np.isfinite(z[:, k])
        zk = z[good, k]
        # cumulative momentum loss to this plane, in MeV
        dE = 1e3 * (np.asarray(sim["pabs"][good, 0])
                    - np.asarray(sim["pabs"][good, k]))
        print(f"\n### {tag}  plane {k}  (r = {sim['globr'][good, k].mean():.1f} cm)"
              f"  n = {good.sum()}")
        print(f"    s_F = {sc['sF'][k]:.4e}  sigma = {sc['sigma'][k]:.4e}  "
              f"1/I = {sc['invI'][k]:.4f}")
        print(f"    {'|z| bin':>12} {'n(z<0)':>9} {'n(z>0)':>9} "
              f"{'<dE> z<0':>11} {'<dE> z>0':>11} {'max dE':>11}")
        for i in range(len(ZEDGES) - 1):
            lo, hi = ZEDGES[i], ZEDGES[i + 1]
            mn = (zk <= -lo) & (zk > -hi)
            mp = (zk >= lo) & (zk < hi)
            lab = f"{lo:g}-{hi:g}"
            print(f"    {lab:>12} {mn.sum():9d} {mp.sum():9d} "
                  f"{(dE[mn].mean() if mn.sum() else np.nan):11.3f} "
                  f"{(dE[mp].mean() if mp.sum() else np.nan):11.3f} "
                  f"{(dE[mn].max() if mn.sum() else np.nan):11.3f}")
        print(f"    all events: <dE> = {dE.mean():.3f} MeV, median "
              f"{np.median(dE):.3f}, p99 {np.percentile(dE,99):.3f}, "
              f"max {dE.max():.3f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    for name, f in (("zloc", cmd_zloc), ("tailobs", cmd_tailobs)):
        q = s.add_parser(name)
        q.add_argument("--configs", nargs="+",
                       default=["pt3_radon", "pt40_radon"])
        q.add_argument("--func", default="qop")
        q.set_defaults(fn=f)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

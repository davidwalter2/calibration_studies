#!/usr/bin/env python3
"""CF / lineshape figures for the ADDITIONAL directions of the 5D track state.

The published pair `result_qop.*` / `result_ms.*` (make_slide_figs.py) and the
toy set `result_{qop,ms}_{homo,lay1,lay4,lay16}_pt3_fisher*` cover two axis
directions -- `qop` and `locx`.  This script draws the SAME two-panel figure
(characteristic function Re/Im model vs Geant4, beside the lineshape) for the
directions added by `dir_closure.py`: the non-bending position `locy`, the two
angles `dxdz` / `dydz`, and the MIXED directions that are the only probe of the
correlations.

NAMING.  `result_*` is taken (including `_k03` and
`_{homo,lay1,lay4,lay16,real}_pt3_fisher` variants), so nothing here is called
`result_`.  The scheme is

    dircl_<direction>_<geom>_pt3_fisher.{png,pdf}

with <direction> in locy, dxdz, dydz, locxPdxdz, locxMdxdz, locyMdydz, eig4,
eig5 ... and <geom> in lay1, real.  P/M are the + and - of a mixed direction
(a literal '+' in a filename is a nuisance).

WHAT IS DIFFERENT FROM `make_slide_figs._panel`.  That function takes a
FUNCTIONAL NAME and looks the a-vector up in `cf_propagation_test.FUNCTIONALS`;
it cannot draw an arbitrary direction, and it applies the a-vector in the
CURVILINEAR basis, which is only correct for `qop` and `locx` (see
`curv2local.py`).  The panel is therefore reimplemented here with the same
colours, line widths, axes, figure size and dpi -- and `--cross-check` redraws
`locx` and `qop` through this code and compares the CF extrema and the closure
against the published route, so a reimplementation artefact would show up.

usage (from calibration_studies/resolution, after `source ../setup_env.sh`):
    python cleanprop/make_dir_figs.py --cross-check
    python cleanprop/make_dir_figs.py
"""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mplhep as hep                                             # noqa: E402
from wums import logging, output_tools, plot_tools               # noqa: E402,F401

import dir_closure as dcl                                        # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import make_slide_figs as msf                                    # noqa: E402
from cf_propagation_test import TAU, ecf, model_phi              # noqa: E402

hep.style.use(hep.style.ROOT)

RED, BLUE, GREY = msf.RED, msf.BLUE, msf.GREY
OUT = msf.OUT                              # cleanprop/slides/assets
DATED = os.path.expanduser(
    f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_directions/")

IU1 = fn.IHEAD[-1]

# direction -> (filename token, panel label, one-line physics tag)
DIRS = {
    "locy":      ("locy", r"local $y$",
                  "multiple scattering, NON-bending plane"),
    "locx":      ("locx", r"local $x$", "multiple scattering, bending plane"),
    "qop":       ("qop", r"$q/p$", "ionization straggling"),
    "dxdz":      ("dxdz", r"$dx/dz$", "MS angle, bending plane"),
    "dydz":      ("dydz", r"$dy/dz$", "MS angle, non-bending plane"),
    "locx+dxdz": ("locxPdxdz", r"$(x + dx/dz)/\sqrt{2}$, oriented",
                  "mixed, reinforcing -- WELL conditioned"),
    "locx-dxdz": ("locxMdxdz", r"$(x - dx/dz)/\sqrt{2}$, oriented",
                  "mixed, near-cancelling -- ILL conditioned"),
    "locy-dydz": ("locyMdydz", r"$(y - dy/dz)/\sqrt{2}$, oriented",
                  "mixed, near-cancelling -- ILL conditioned"),
    "eig4":      ("eig4", "4th eigen-direction",
                  "mixed, near-cancelling"),
    "eig5":      ("eig5", "5th eigen-direction",
                  "mixed, most near-cancelling -- ILL conditioned"),
}

GEOMLAB = {
    "lay1": "layered toy — 1 mm shells, 14 planes",
    "real": "real tracker — 19 barrel modules",
}


def project(g, name, k, ms):
    """(z, phi) for one direction on one plane, on the drawing grid TAU."""
    legs, s = dcl.model(g), dcl.sim(g)
    r = ms[(k, name)]
    resid = np.stack([s[b][:, k] - legs[k][rf]
                      for b, rf in zip(dcl.LOCAL, dcl.REF)], axis=-1)
    good = s["valid"][:, k] & np.isfinite(resid).all(axis=-1)
    z = (resid[good] @ r["a_local"]) / r["sF"]
    phi = model_phi(legs, k, r["a_curv"], r["sF"], TAU)
    return z, phi


def make(g, name, ms, rows, err, ks, k=None):
    legs, s = dcl.model(g), dcl.sim(g)
    if k is None:
        k = len(legs) - 1
    kk = int(np.where(ks == k)[0][0])
    z, phi = project(g, name, k, ms)
    r = float(np.nanmedian(s["globr"][:, k]))
    cl, e = rows[name][:, IU1], float(err[name][IU1])
    cond = float(ms[(k, name)]["cond"])

    trunc = float(np.max(np.abs(phi[TAU > 0.9 * TAU[-1]])))
    nd = int(np.clip(1 - np.floor(np.log10(max(e, 1e-12))), 3, 6))
    note = (r"closure $\langle e^{-uz^2}\rangle_{\rm data}-"
            r"\langle e^{-uz^2}\rangle_{\rm model}$ at $u=1$" "\n"
            f"{cl[kk]:+.{nd}f} here  ·  {len(cl)}-plane mean "
            f"{cl.mean():+.{nd}f} ± {e:.{nd}f}")
    lab, tag = DIRS[name][1], DIRS[name][2]
    title = (f"{lab} at $r={r:.0f}$ cm — {tag}\n"
             f"{GEOMLAB[g]} · $\\mu$, $p_T=3$ GeV, $\\eta=0.30$ · "
             f"predicted variance of this direction {cond:.3f}")

    # Four axes: CF + CF difference, lineshape + lineshape ratio.  The lower
    # panels are the whole point -- the closure statistic is an integral over
    # the CF difference, so plotting it is plotting the thing being measured.
    #
    # NO tight_layout here: it is incompatible with a height-ratio gridspec and
    # the failure it produces (ratio y-labels outside the canvas, x-labels
    # colliding with the panel above) is exactly what make_radsp_figs records.
    # Explicit margins instead, identical geometry to that set so the two sit
    # together in a deck.
    fig, axs = plt.subplots(2, 2, figsize=(14.4, 7.8),
                            gridspec_kw=dict(height_ratios=[3, 1]))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.095,
                        hspace=0.08, wspace=0.21)
    a1, a2 = axs[0, 0], axs[0, 1]
    msf.panels(a1, axs[1, 0], a2, axs[1, 1], z, phi, title, note)
    fs = msf._fit_title(fig, a1, a2)
    msf.save(fig, f"dircl_{DIRS[name][0]}_{g}_pt3_fisher", dated=DATED)
    return dict(plane=cl[kk], mean=float(cl.mean()), err=e, cond=cond,
                trunc=trunc, fs=fs, nev=int(len(z)))


def cmd_cross_check(args):
    """Redraw `locx`/`qop` through THIS code and compare with the published
    route, so a reimplementation artefact would be visible rather than assumed
    away."""
    print("=" * 92)
    print("CROSS-CHECK: this panel code vs make_slide_figs._panel, real "
          "geometry, outermost plane")
    print("=" * 92)
    g = "real"
    legs, s = dcl.model(g), dcl.sim(g)
    k = len(legs) - 1
    ms = dcl.model_side(g, ["locx", "qop"], useH=False, nproc=args.nproc)
    print(f"  {'func':<6} {'route':<14} {'model Re min':>13} "
          f"{'model Im max':>13} {'G4 Re min':>11} {'G4 Im max':>11} "
          f"{'sF':>12}")
    for name in ("qop", "locx"):
        z, phi = project(g, name, k, ms)
        e = ecf(z, TAU)
        print(f"  {name:<6} {'dir_closure':<14} {phi.real.min():13.5f} "
              f"{phi.imag.max():13.5f} {e.real.min():11.5f} "
              f"{e.imag.max():11.5f} {ms[(k, name)]['sF']:12.5e}")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
        zz, pp, sg, _, iI = msf._panel(name, k, s, legs, a1, a2, "", "fisher")
        plt.close(fig)
        ee = ecf(zz, TAU)
        print(f"  {'':<6} {'make_slide':<14} {pp.real.min():13.5f} "
              f"{pp.imag.max():13.5f} {ee.real.min():11.5f} "
              f"{ee.imag.max():11.5f} {sg:12.5e}")
    print("\n  (NOTES_SLIDEFIGS section 4 records the published per-plane "
          "acceptance values")
    print("   for q/p as model Re min -0.4374, Im max +0.6427, Geant4 -0.1774 "
          "/ +0.4082.)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geoms", nargs="+", default=["real", "lay1"])
    ap.add_argument("--dirs", nargs="+",
                    default=["locy", "dxdz", "dydz", "locx+dxdz",
                             "locx-dxdz", "locy-dydz", "eig5"])
    ap.add_argument("--nproc", type=int, default=64)
    ap.add_argument("--cross-check", action="store_true")
    args = ap.parse_args()

    if args.cross_check:
        cmd_cross_check(args)
        return

    print(f"assets -> {OUT}")
    print(f"dated  -> {DATED}\n")
    summary = []
    for g in args.geoms:
        print(f"--- {g}: {GEOMLAB[g]}")
        ms = dcl.model_side(g, args.dirs, useH=True, nproc=args.nproc)
        rows, cev, ks, _ = dcl.closure(g, args.dirs, ms)
        err, _ = dcl.errs_from_cev(cev, args.dirs)
        for name in args.dirs:
            d = make(g, name, ms, rows, err, ks)
            flag = "" if d["trunc"] < 1e-3 else "   <-- CF NOT DEAD AT t_max"
            print(f"    {name:<10} plane {d['plane']:+.5f}  mean "
                  f"{d['mean']:+.5f} +- {d['err']:.5f}  cond {d['cond']:.3f}  "
                  f"|phi|(t_max) {d['trunc']:.1e}  title {d['fs']} pt{flag}")
            summary.append((g, name, d))
        print()

    print("=== SUMMARY")
    print(f"  {'geom':<6} {'direction':<11} {'outermost':>11} "
          f"{'ladder mean':>13} {'+-':>9} {'cond':>7}")
    for g, name, d in summary:
        print(f"  {g:<6} {name:<11} {d['plane']:+11.5f} {d['mean']:+13.5f} "
              f"{d['err']:9.5f} {d['cond']:7.3f}")


if __name__ == "__main__":
    main()

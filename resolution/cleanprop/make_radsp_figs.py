#!/usr/bin/env python3
"""CF and lineshape figures WITH A RATIO PANEL, for the eight-species
radiation-on/off closure of NOTES_CLOSURE_FINAL.

WHAT THIS ADDS TO THE PUBLISHED SET
-----------------------------------
`make_slide_figs.fig_results` and `make_toy_figs.fig_for` draw two panels --
the characteristic function (Re/Im, model vs Geant4) beside the lineshape --
with the measured closure printed on the second.  Both are ABSOLUTE
comparisons: at the eye's resolution the model and Geant4 curves lie on top of
each other, which is the point of the figure but also its limit.  Everything
this study is now chasing is at the 1e-3 level of a curve that spans 1, so it
is invisible there.

Each panel therefore gains a LOWER panel comparing simulation to model.

    result_*   (published)         radsp_*   (this script)
    +-----------+-----------+      +-----------+-----------+
    |    CF     | lineshape |      |    CF     | lineshape |
    +-----------+-----------+      +-----------+-----------+
                                   | G4 - model|  G4/model |
                                   +-----------+-----------+

WHY THE CF PANEL SHOWS A DIFFERENCE AND THE LINESHAPE PANEL A RATIO
-------------------------------------------------------------------
This is not a stylistic choice and it is the one place the figure departs from
"sim / model everywhere".

The lineshape is a DENSITY: strictly positive, so `G4 / model` is bounded,
dimensionless and reads directly as "how much too wide is the model here".
That panel is a ratio.

The characteristic function is COMPLEX and both its real and imaginary parts
cross zero repeatedly -- Re(phi) has a zero near t ~ 3 for every species here,
and Im(phi) has one at t = 0 by construction.  A pointwise ratio therefore
diverges at every crossing and the panel would be dominated by poles that carry
no information.  Worse, it would LOOK like structure.  So the CF panel shows
`Geant4 - model`, which is bounded, has the same units as the CF itself, and is
the quantity the closure statistic is built from: the closure
`<e^{-u z^2}>_data - <e^{-u z^2}>_model` is exactly the Weierstrass transform
of this difference.  The lower-left panel is thus the closure's own integrand,
plotted before it is integrated.

NAMING
------
`result_{qop,ms}_*` (with its `_k03` and `_{homo,lay1,lay4,lay16,real}`
variants) and `dircl_*` are both taken.  This set is

    radsp_<func>_<species>_<rad>_pt3_fisher.{png,pdf}

    <func>     qop | ms                (`ms` is the file token for `locx`,
                                        the same convention `make_toy_figs`
                                        uses)
    <species>  mum mup pim pip Km Kp pbar p      (charge is in the m/p suffix,
                                        `hadron_probe.tag_of`'s convention)
    <rad>      radon | radoff

so species, CHARGE, radiation state and functional are all in the name and
nothing collides with the existing sets.

usage (from calibration_studies/resolution, after `source ../setup_env.sh`):
    python cleanprop/make_radsp_figs.py --check
    python cleanprop/make_radsp_figs.py
    python cleanprop/make_radsp_figs.py --pdg 13 2212 --funcs qop locx
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

import cf_propagation_test as cpt                                # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import make_slide_figs as msf                                    # noqa: E402
import radoff_species as rs                                      # noqa: E402
from cf_propagation_test import (FUNCTIONALS, REF_BRANCH, SIM_BRANCH, TAU,
                                 ecf, model_phi)                 # noqa: E402

hep.style.use(hep.style.ROOT)

RED, BLUE, GREY = msf.RED, msf.BLUE, msf.GREY
OUT = msf.OUT                              # cleanprop/slides/assets
DATED = os.path.expanduser(
    f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_radoff_species/")

IU1 = fn.IHEAD[-1]                         # index of u = 1 inside fn.UCURVE

TAGS = {"qop": "qop", "locx": "ms"}
LABELS = {"qop": "$q/p$", "locx": "local $x$"}
WHAT = {"qop": "ionization straggling", "locx": "multiple scattering"}
RADLAB = {True: "radiation ON", False: "radiation OFF"}
RADTOK = {True: "radon", False: "radoff"}


def make(pdg, rad, func, r, args):
    """One figure.  `r` is a `radoff_species._rows` result for the same cell,
    so the number ON the figure is the number IN the table -- not a second
    evaluation that could disagree."""
    path = rs.mp(pdg, rad)
    legs = cpt.load_model(path)
    sim = hp.sim_of(pdg, rs.SIM_ARM[rad])

    # The scale, with the channel set tracking the radiation state exactly as
    # the closure does (`radoff_species.chans`): 1/I is the inversion of the
    # same CF the closure compares against.
    cpt.RAD_CHANNEL = bool(rad)
    cpt._PHI_CACHE.clear()
    fn._SCALE_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        sc = fn.plane_scales(legs, func, tag=path, channels=rs.chans(rad))
        k = len(legs) - 1
        sF = float(sc["sF"][k])
        d = sim[SIM_BRANCH[func]][:, k] - legs[k][REF_BRANCH[func]]
        good = sim["valid"][:, k] & np.isfinite(d)
        z = d[good] / sF
        phi = model_phi(legs, k, FUNCTIONALS[func], sF, TAU)
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        cpt._PHI_CACHE.clear()
        fn._SCALE_CACHE.clear()

    # THE CF TRUNCATION.  The red lineshape curve is a trapezoid inversion over
    # TAU (ceiling 40); that is legitimate only if the CF has died by then.
    # Measured per panel rather than inherited, because the eight species have
    # very different widths -- the proton's residual is 8x narrower than the
    # muon's, so its CF is 8x wider in t.
    trunc = float(np.max(np.abs(phi[TAU > 0.9 * TAU[-1]])))

    m, err = r["m"], r["err"]
    nd = int(np.clip(1 - np.floor(np.log10(max(float(err[IU1]), 1e-12))), 3, 6))
    ipk = int(np.nanargmax(np.abs(m)))
    rms = float(np.sqrt((m ** 2).mean()))
    rr = float(np.nanmedian(np.abs(sim["globr"][:, k])))

    note = (r"closure $\langle e^{-uz^2}\rangle_{\rm data}-"
            r"\langle e^{-uz^2}\rangle_{\rm model}$" "\n"
            f"$u=1$: {m[IU1]:+.{nd}f} ± {err[IU1]:.{nd}f}"
            f"   ({m[IU1]/err[IU1]:+.1f}$\\sigma$)" "\n"
            f"peak $u={fn.UCURVE[ipk]:g}$: {m[ipk]:+.{nd}f} ± "
            f"{err[ipk]:.{nd}f}   rms {rms:.{nd}f}")
    sp = hp.SPECIES[pdg]
    title = (f"{LABELS[func]} at $r={rr:.0f}$ cm — {WHAT[func]}\n"
             f"{sp['label']}, $p_T={hp.PT:g}$ GeV, $\\eta={hp.ETA}$, "
             f"$\\phi={hp.PHI}$ · layered toy, {len(legs)} planes · "
             f"{RADLAB[rad]}")

    # NO `tight_layout` HERE.  It is incompatible with a height-ratio gridspec
    # (matplotlib says so, and the first attempt at this set produced exactly
    # the failure it warns about: the two ratio panels' y-labels rendered
    # OUTSIDE the canvas and the x-labels collided with the panel above).
    # Explicit margins instead, so the geometry is fixed rather than
    # negotiated -- which is also what keeps every figure in the set the same
    # size, as the deck needs.
    fig, axs = plt.subplots(2, 2, figsize=(14.4, 7.8),
                            gridspec_kw=dict(height_ratios=[3, 1]))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.095,
                        hspace=0.08, wspace=0.21)
    info = msf.panels(axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1],
                  z, phi, title, note)
    fs = msf._fit_title(fig, axs[0, 0], axs[0, 1])
    msf.save(fig, f"radsp_{TAGS[func]}_{rs_tag(pdg)}_{RADTOK[rad]}_pt3_fisher",
             dated=DATED)
    return dict(rms=rms, u1=float(m[IU1]), err1=float(err[IU1]),
                peak_u=float(fn.UCURVE[ipk]), peak=float(m[ipk]),
                trunc=trunc, fs=fs, nev=int(good.sum()), r=rr, sF=sF, **info)


def rs_tag(pdg):
    return hp.SPECIES[pdg]["label"].replace("-", "m").replace("+", "p")


# ==========================================================================

def cmd_check(args):
    """Premise checks, before any figure.

    Everything here is something that has silently produced a plausible-looking
    figure at some point in this study: a model paired with the wrong sim (the
    detid sequences differ and nothing complains), a CF that has not decayed by
    the inversion ceiling (the red curve rings and the ringing looks like
    physics), and a closure number on the figure that is not the closure number
    in the table."""
    print("=" * 96)
    print("PREMISE CHECKS")
    print("=" * 96)

    print("\n### 1. model/sim pairing, by detid (a mis-pair is SILENT)")
    print(f"  {'species':<8}{'rad':<8}{'legs':>6}{'sim planes':>12}"
          f"{'detid seq':>12}{'events':>10}")
    bad = 0
    for pdg in args.pdg:
        for rad in args.rad:
            if not os.path.exists(rs.mp(pdg, rad)):
                print(f"  {hp.SPECIES[pdg]['label']:<8}"
                      f"{RADTOK[rad]:<8}(model missing)")
                bad += 1
                continue
            legs = cpt.load_model(rs.mp(pdg, rad))
            sim = hp.sim_of(pdg, rs.SIM_ARM[rad])
            ld = np.array([l["detid"] for l in legs])
            sd = np.asarray(sim["detid"])
            ok = len(sd) == len(ld) and bool(np.all(sd == ld))
            bad += (not ok)
            print(f"  {hp.SPECIES[pdg]['label']:<8}{RADTOK[rad]:<8}"
                  f"{len(legs):>6}{sim['valid'].shape[1]:>12}"
                  f"{'OK' if ok else '*** DIFFER ***':>12}"
                  f"{sim['valid'].shape[0]:>10}")
    print(f"  mis-paired cells: {bad}")
    if bad:
        raise SystemExit("refusing to draw: see above")

    print("\n### 2. CF truncation at the inversion ceiling "
          f"(t_max = {TAU[-1]:g}); > 1e-3 makes the red curve ring")
    print(f"  {'species':<8}{'rad':<8}{'qop':>12}{'locx':>12}")
    for pdg in args.pdg:
        for rad in args.rad:
            row = []
            legs = cpt.load_model(rs.mp(pdg, rad))
            k = len(legs) - 1
            for func in ("qop", "locx"):
                cpt.RAD_CHANNEL = bool(rad)
                cpt._PHI_CACHE.clear()
                fn._SCALE_CACHE.clear()
                try:
                    sc = fn.plane_scales(legs, func, tag=rs.mp(pdg, rad),
                                         channels=rs.chans(rad))
                    phi = model_phi(legs, k, FUNCTIONALS[func],
                                    float(sc["sF"][k]), TAU)
                    row.append(float(np.max(np.abs(phi[TAU > 0.9 * TAU[-1]]))))
                finally:
                    cpt.RAD_CHANNEL = True
                    cpt._PHI_CACHE.clear()
                    fn._SCALE_CACHE.clear()
            flag = "  <-- RINGS" if max(row) > 1e-3 else ""
            print(f"  {hp.SPECIES[pdg]['label']:<8}{RADTOK[rad]:<8}"
                  + "".join(f"{x:12.3e}" for x in row) + flag)


def main():
    os.environ.setdefault("RES_NO_PHI_CACHE", "1")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pdg", type=int, nargs="+", default=rs.ORDER8)
    ap.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    a = ap.parse_args()
    a.rad = [bool(x) for x in a.rad]

    if a.check:
        return cmd_check(a)

    print(f"assets -> {OUT}")
    print(f"dated  -> {DATED}")
    cmd_check(a)

    print()
    print("=" * 118)
    print("FIGURES")
    print("=" * 118)
    rows, dump = [], {}
    for func in a.funcs:
        for pdg in a.pdg:
            for rad in a.rad:
                r = rs._rows(pdg, rad, func)
                info = make(pdg, rad, func, r, a)
                rows.append((func, hp.SPECIES[pdg]["label"], RADTOK[rad], info))
                key = f"{func}|{hp.SPECIES[pdg]['label']}|{RADTOK[rad]}"
                dump[key + "|m"] = r["m"]
                dump[key + "|err"] = r["err"]

    print()
    print("=" * 118)
    print("SUMMARY")
    print("=" * 118)
    print(f"  {'func':<6}{'species':<8}{'rad':<8}{'r [cm]':>8}{'rms':>10}"
          f"{'u=1':>11}{'+-':>10}{'peak u':>8}{'CF trunc':>11}"
          f"{'max|G4-mod|':>13}{'title pt':>9}")
    for func, lab, rad, i in rows:
        print(f"  {func:<6}{lab:<8}{rad:<8}{i['r']:8.1f}{i['rms']:10.5f}"
              f"{i['u1']:+11.5f}{i['err1']:10.5f}{i['peak_u']:8g}"
              f"{i['trunc']:11.3e}{i['cfmax']:13.4f}{i['fs']:9d}")

    if dump:
        os.makedirs(DATED, exist_ok=True)
        p = os.path.join(DATED, "radsp_closure.npz")
        np.savez(p, u=fn.UCURVE, **dump)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

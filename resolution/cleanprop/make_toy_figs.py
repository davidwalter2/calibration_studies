#!/usr/bin/env python3
"""Clean-propagation result figures for the FOUR TOY geometries, in the Fisher
normalization -- the toy counterparts of the real-geometry `result_qop` /
`result_ms` pair that `make_slide_figs.py` produces.

WHAT THIS IS
------------
`make_slide_figs.fig_results` draws, for ONE functional at the outermost plane,
the characteristic function (Re/Im, model vs Geant4) beside the lineshape, with
the measured closure printed on it.  It does that for the REAL tracker geometry
(19 planes) and writes `result_qop.*` / `result_ms.*`, which a slide deck
references.  This script draws the SAME two figures for the four toy
configurations of NOTES_GEOMCLOSURE (homogeneous, layered NSUB = 1 / 4 / 16),
all 14 planes, pT = 3, 100k events.

    result_qop_<geom>_pt3_fisher.{png,pdf}
    result_ms_<geom>_pt3_fisher.{png,pdf}      <geom> in homo, lay1, lay4, lay16

`--real-copies` additionally copies the EXISTING real-geometry figures to
`result_{qop,ms}_real_pt3_fisher.*` so the whole set is uniformly named.  The
originals `result_qop.*` / `result_ms.*` are never written, renamed or removed
by this script -- the deck points at them.

NOTHING IS RE-DERIVED HERE
--------------------------
The panels come from `make_slide_figs._panel` verbatim and the closure numbers
from `geom_closure.closure_rows`, i.e. from the two pieces of code that produced
the published real-geometry figures and the NOTES_GEOMCLOSURE table
respectively.  This script only selects geometries, labels them and lays them
out, so a disagreement with either reference would be a real disagreement and
not a reimplementation artefact.  `--check` prints the comparison.

THE ERROR BAR ON THE PLANE MEAN is the correlated per-EVENT one of
NOTES_GEOMCLOSURE section 1 (`closure_rows`'s `err_mean`): every plane is
scored on the SAME events, so `err_plane/sqrt(nplanes)` under-states it by 2.8x.

THE PLANE THE FIGURE SHOWS IS NOT THE PLANE THE HEADLINE NUMBER DESCRIBES.
`+0.04991` (homogeneous, locx) and `+0.00002` (layered NSUB=1) are means over
the 14 planes.  These figures show the OUTERMOST plane, because that is what the
real-geometry figures show and the point of the exercise is comparability -- and
the outermost plane is exactly where the homogeneous locx residual is SMALLEST
(+0.016 against its +0.085 peak at plane 3), while the layered one has drifted
negative (-0.004).  So the outermost-plane figures show a 4x contrast, not the
2500x the plane means suggest.  Both numbers are printed on every figure, and
`--extra-plane 3` emits the same `locx` figures at the peak plane, where the
contrast the plane mean is made of is actually visible.

usage (from calibration_studies/resolution, after `source ../setup_env.sh`):
    python cleanprop/make_toy_figs.py --check     # premise checks, no figures
    python cleanprop/make_toy_figs.py             # the eight figures + copies
    python cleanprop/make_toy_figs.py --extra-plane 3 --funcs locx
"""
import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mplhep as hep                                             # noqa: E402
from wums import logging, output_tools, plot_tools               # noqa: E402,F401

import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import make_slide_figs as msf                                    # noqa: E402
from cf_propagation_test import FUNCTIONALS, TAU, model_phi      # noqa: E402

hep.style.use(hep.style.ROOT)

# `make_slide_figs` already resolves both destinations: the deck's assets
# directory and the date-derived mirror under ~/public_html/cvh/.  Reused rather
# than recomputed so the toy figures land beside the real ones.
OUT, DATED = msf.OUT, msf.DATED

# --------------------------------------------------------------------------
# The four toy configurations.
#
# Keys are `geom_closure.GEOMS` entries, so the model/sim file pairing is the
# one NOTES_GEOMCLOSURE validated and not a fresh guess.  The `H_` prefix is the
# 100k-event regeneration (scratchpad/hs_regen.sh); the 2000-event originals
# leave a +-0.006 correlated error at u = 1, larger than the whole layered-toy
# signal.
#
# hsT0.50 / hsT2.00 are DELIBERATELY ABSENT.  The generator fixes rho = 9.0, so
# varying T scales the TOTAL material: those two lose 118.6 and 473.0 MeV
# against this family's 23.75, i.e. they are not a thickness scan at fixed
# material and the comparison they were built for does not exist.
# --------------------------------------------------------------------------
#
# The geometry strings are kept SHORT on purpose: they share the title band with
# the right panel's closure annotation, and `_fit_title` shrinks the title until
# the two stop overlapping, so a verbose label just costs font size.
CONFIGS = [
    ("homo",  "H_homo",
     r"homogeneous toy — continuous, $\rho=0.107$"),
    ("lay1",  "H_layK1",
     r"layered toy — 1 mm shells, $\rho=9$, NSUB = 1"),
    ("lay4",  "H_layK4",
     r"layered toy — 1 mm shells, $\rho=9$, NSUB = 4"),
    ("lay16", "H_layK16",
     r"layered toy — 1 mm shells, $\rho=9$, NSUB = 16"),
]

# NOTES_GEOMCLOSURE section 2/3: Fisher normalization, u = 1, mean over planes.
REFERENCE = {
    "homo":  dict(locx=(+0.04991, 0.00075), qop=(+0.01136, 0.00036)),
    "lay1":  dict(locx=(+0.00002, 0.00083), qop=(+0.00081, 0.00034)),
    "lay4":  dict(locx=(+0.00055, 0.00083), qop=(+0.00131, 0.00034)),
    "lay16": dict(locx=(+0.00111, 0.00083), qop=(+0.00157, 0.00034)),
    "real":  dict(locx=(+0.00138, 0.00058), qop=(+0.01148, 0.00027)),
}

IU1 = fn.IHEAD[-1]        # index of u = 1 inside fn.UCURVE
TAGS = {"qop": "qop", "locx": "ms"}
LABELS = {"qop": "$q/p$", "locx": "local $x$"}
WHAT = {"qop": "pure ionization", "locx": "multiple scattering"}


# ==========================================================================
# premise checks -- run before any figure is interpreted
# ==========================================================================

def verify_pair(key):
    """Model/sim correspondence for one toy, by detid rather than by trust.

    A mis-targeted pair is SILENT: it returns a plausible closure number that
    means nothing (NOTES_TCUT section 0 -- `model_pt10_eta0.30_phi0.20` shares
    not one module with its supposed sim).  Four tests, the same ones
    `geom_closure.py pairs` runs:

      (1) same number of scoring surfaces on both sides;
      (2) identical detid sequence, element by element;
      (3) the model's reference sits at the sim's median radius on every plane;
      (4) the reference's energy loss is compatible with the sim's -- the model
          quotes a MEAN and the sim a MEDIAN, so the two differ by the Landau
          mean-mode gap, and what would flag a mis-pair is a gap outside the
          2.4-3.9 MeV band the eight validated pairs share.
    """
    d = gc.GEOMS[key]
    mp = d["model"] if os.path.sep in d["model"] else os.path.join(
        gc.SCRATCH, d["model"])
    m = gc._model_meta(mp)
    s = gc._toy_sim_meta(d["sim"])
    dE_m = 1e3 * float(m["refp"][0] - m["refp"][-1])
    nok = (len(m["detid"]) == s["npl"])
    seqok = nok and bool(np.array_equal(np.asarray(m["detid"]), s["seq"]))
    dr = float(np.abs(np.asarray(m["refglobr"]) - s["r"]).max()) if nok else np.nan
    gap = dE_m - s["dE"]
    ok = nok and seqok and dr < 1e-2 and 2.0 < gap < 4.5
    return dict(model=os.path.basename(mp), sim=os.path.basename(d["sim"]),
                nlegs=len(m["detid"]), nplanes=s["npl"], seqok=seqok, dr=dr,
                dE_model=dE_m, dE_sim=s["dE"], gap=gap, nev=s["nev"],
                nfull=s["nfull"], ok=ok)


def cf_truncation(legs, k, func, scale):
    """|phi(t_max)| on the grid the figure's lineshape overlay is inverted on.

    `_panel` inverts the model CF by trapezoid over TAU (ceiling 40) to draw the
    density.  That is legitimate only if the CF has actually died by then --
    otherwise the truncation rings and the red curve is an artefact.  It is
    MEASURED here for every panel rather than inherited from the real geometry,
    because the toys have their own widths and their own tails.
    """
    phi = model_phi(legs, k, FUNCTIONALS[func], scale, TAU)
    return float(np.max(np.abs(phi[TAU > 0.9 * TAU[-1]])))


# ==========================================================================
# closure numbers
# ==========================================================================

def measure(key, func):
    """Per-plane closure curve + the correlated error, from the reference code.

    `geom_closure.closure_rows` is the function that produced the
    NOTES_GEOMCLOSURE table; calling it (rather than re-deriving the statistic)
    is what makes a disagreement with that table meaningful.
    """
    legs = gc.geom_model(key)
    sim = gc.geom_sim(key)
    sc = fn.plane_scales(legs, func, tag=key)
    rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"],
                                              probes=fn.UCURVE)
    return dict(legs=legs, sim=sim, sc=sc, rows=rows, errs=errs, ks=ks, ns=ns,
                err=err)


def cmd_check(args):
    print("=" * 78)
    print("PREMISE CHECKS")
    print("=" * 78)

    print("\n### 1. model/sim pairs, by detid (a mis-pair is silent)")
    print(f"  {'geom':<7} {'model':<14} {'sim':<12} {'legs/planes':>12} "
          f"{'detid seq':>10} {'max|dr|':>9} {'dE mod/sim':>15} {'gap':>7} "
          f"{'events':>8}")
    allok = True
    for tag, key, _ in CONFIGS:
        v = verify_pair(key)
        allok &= v["ok"]
        print(f"  {tag:<7} {v['model']:<14} {v['sim']:<12} "
              f"{v['nlegs']:5d} /{v['nplanes']:5d} {str(v['seqok']):>10} "
              f"{v['dr']:9.4f} {v['dE_model']:7.3f}/{v['dE_sim']:7.3f} "
              f"{v['gap']:+7.3f} {v['nev']:8d}")
    print(f"  -> {'ALL PAIRS OK' if allok else 'A PAIR FAILED -- STOP'}")

    print("\n### 2. closure at u = 1, Fisher, mean over planes, vs "
          "NOTES_GEOMCLOSURE")
    print(f"  {'geom':<7} {'func':<5} {'measured':>20} {'reference':>20} "
          f"{'diff':>10} {'pull':>7}")
    worst = 0.0
    for tag, key, _ in CONFIGS:
        for func in ("locx", "qop"):
            r = measure(key, func)
            m = float(r["rows"][:, IU1].mean())
            e = float(r["err"][IU1])
            rm, re = REFERENCE[tag][func]
            pull = (m - rm) / max(e, 1e-12)
            worst = max(worst, abs(pull))
            print(f"  {tag:<7} {func:<5} {m:+12.5f} +- {e:.5f} "
                  f"{rm:+12.5f} +- {re:.5f} {m - rm:+10.5f} {pull:+7.2f}")
    print(f"  -> worst |pull| vs the reference table = {worst:.2f}")

    print("\n### 3. CF truncation at the top of the inversion grid")
    print("  The lineshape overlay is a trapezoid inversion of the model CF")
    print(f"  truncated at t = {TAU[-1]:.0f}. |phi| must be dead there or the")
    print("  red curve rings.  Measured per panel:")
    print(f"  {'geom':<7} {'func':<5} {'plane':>6} {'max|phi| over top 10%':>24}")
    for tag, key, _ in CONFIGS:
        for func in ("qop", "locx"):
            r = measure(key, func)
            k = len(r["legs"]) - 1
            v = cf_truncation(r["legs"], k, func, r["sc"]["sF"][k])
            flag = "" if v < 1e-3 else "   <-- NOT NEGLIGIBLE"
            print(f"  {tag:<7} {func:<5} {k:6d} {v:24.3e}{flag}")


# ==========================================================================
# the figures
# ==========================================================================

def fig_for(tag, key, geomlabel, func, res, k=None, suffix=""):
    legs, sim = res["legs"], res["sim"]
    nplane = len(legs)
    if k is None:
        k = nplane - 1                   # outermost plane, as the real figures
    r = float(np.nanmedian(sim["globr"][:, k]))
    cl = res["rows"][:, IU1]
    kk = int(np.where(res["ks"] == k)[0][0])
    err = float(res["err"][IU1])

    # Precision set by the ERROR, not fixed at the real figures' 3 decimals:
    # the layered toys close at 1e-5 - 1e-3 with a 8e-4 error, so 3 decimals
    # prints "+0.000 +- 0.000" and hides both the value and its uncertainty.
    nd = int(np.clip(1 - np.floor(np.log10(max(err, 1e-12))), 3, 6))
    note = (r"closure $\langle e^{-uz^2}\rangle_{\rm data}-"
            r"\langle e^{-uz^2}\rangle_{\rm model}$ at $u=1$" "\n"
            f"{cl[kk]:+.{nd}f} here  ·  {nplane}-plane mean "
            f"{cl.mean():+.{nd}f} ± {err:.{nd}f}")
    title = (f"{LABELS[func]} at $r={r:.0f}$ cm — {WHAT[func]}\n"
             f"{geomlabel}, {nplane} planes · "
             r"$\mu$, $p_T=3$ GeV, $\eta=0.30$, $\phi=0.70$")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
    msf._panel(func, k, sim, legs, a1, a2, title, norm="fisher", note=note)
    fig.tight_layout()
    fs = msf._fit_title(fig, a1, a2)
    msf.save(fig, f"result_{TAGS[func]}_{tag}_pt3_fisher{suffix}")
    return cl, err, r, k, fs


def copy_real():
    """Uniformly-named COPIES of the existing real-geometry figures.

    Copies only.  `result_qop.*` / `result_ms.*` are the deck's assets and are
    left exactly as they are -- not rewritten, not renamed, not removed.
    """
    n = 0
    for d in (OUT, DATED):
        for tag in ("qop", "ms"):
            for ext in ("png", "pdf"):
                src = os.path.join(d, f"result_{tag}.{ext}")
                dst = os.path.join(d, f"result_{tag}_real_pt3_fisher.{ext}")
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                    n += 1
                else:
                    print(f"  MISSING (not copied): {src}")
    print(f"  copied {n} real-geometry files to *_real_pt3_fisher.*")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="premise checks against NOTES_GEOMCLOSURE, no figures")
    ap.add_argument("--geoms", nargs="+", default=[c[0] for c in CONFIGS])
    ap.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    ap.add_argument("--extra-plane", type=int, default=None,
                    help="also emit the same figures at this plane index, "
                         "suffixed _k<NN> (3 = the homogeneous locx peak)")
    ap.add_argument("--real-copies", action="store_true", default=True)
    ap.add_argument("--no-real-copies", dest="real_copies", action="store_false")
    args = ap.parse_args()

    if args.check:
        cmd_check(args)
        return

    print(f"assets  -> {OUT}")
    print(f"dated   -> {DATED}\n")

    # One flat pool over (geometry, functional, plane) before anything else, so
    # the four toys x two functionals x 14 planes are built together rather than
    # eight pools of 14 in sequence. Everything below then hits the cache.
    fn.prewarm_scales([(gc.geom_model(key), func, key)
                       for tag, key, _lbl in CONFIGS if tag in args.geoms
                       for func in args.funcs])

    summary, dump = [], {}
    for tag, key, geomlabel in CONFIGS:
        if tag not in args.geoms:
            continue
        v = verify_pair(key)
        # A mis-paired model/sim yields a plausible number silently, so this is
        # a hard stop rather than a warning.
        assert v["ok"], (f"{tag}: model/sim pair failed validation -- "
                         f"{v['nlegs']} legs vs {v['nplanes']} planes, "
                         f"detid seq {v['seqok']}, max|dr| {v['dr']:.4f} cm, "
                         f"dE gap {v['gap']:+.3f} MeV")
        print(f"--- {tag}  ({v['model']} / {v['sim']}, {v['nplanes']} planes, "
              f"{v['nev']} events)")
        for func in args.funcs:
            res = measure(key, func)
            trunc = cf_truncation(res["legs"], len(res["legs"]) - 1, func,
                                  res["sc"]["sF"][len(res["legs"]) - 1])
            if trunc > 1e-3:
                print(f"    WARNING {func}: |phi| = {trunc:.2e} at the top of "
                      f"the inversion grid -- the density overlay may ring")
            cl, err, r, k, fs = fig_for(tag, key, geomlabel, func, res)
            if args.extra_plane is not None:
                fig_for(tag, key, geomlabel, func, res, k=args.extra_plane,
                        suffix=f"_k{args.extra_plane:02d}")
            if fs < 17:
                print(f"    (title shrunk to {fs} pt to clear the closure "
                      f"annotation)")
            rm, re = REFERENCE[tag][func]
            print(f"    {func:<5} closure(u=1) per plane: "
                  + " ".join(f"{v_:+.4f}" for v_ in cl))
            print(f"    {func:<5} outermost {cl[-1]:+.5f}   "
                  f"{len(cl)}-plane mean {cl.mean():+.5f} +- {err:.5f}   "
                  f"[reference {rm:+.5f} +- {re:.5f}, "
                  f"pull {(cl.mean() - rm) / max(err, 1e-12):+.2f}]")
            summary.append((tag, func, cl[-1], cl.mean(), err, rm, re))
            dump[f"{tag}|{func}|rows"] = res["rows"]
            dump[f"{tag}|{func}|err"] = res["err"]
            dump[f"{tag}|{func}|ks"] = res["ks"]
            dump[f"{tag}|{func}|sF"] = res["sc"]["sF"]
            dump[f"{tag}|{func}|sigma"] = res["sc"]["sigma"]
            dump[f"{tag}|{func}|invI"] = res["sc"]["invI"]
        print()

    if args.real_copies:
        print("--- real-geometry copies")
        copy_real()

    if dump:
        os.makedirs(DATED, exist_ok=True)
        p = os.path.join(DATED, "result_closure_toys_fisher.npz")
        np.savez(p, u=fn.UCURVE, **dump)
        print(f"  wrote {p}")

    print("\n=== SUMMARY: closure at u = 1, Fisher normalization")
    print(f"  {'geom':<7} {'func':<5} {'outermost':>11} {'plane mean':>12} "
          f"{'+-':>9} {'reference':>11} {'pull':>7}")
    for tag, func, out_, mean, err, rm, re in summary:
        print(f"  {tag:<7} {func:<5} {out_:+11.5f} {mean:+12.5f} {err:9.5f} "
              f"{rm:+11.5f} {(mean - rm) / max(err, 1e-12):+7.2f}")


if __name__ == "__main__":
    main()

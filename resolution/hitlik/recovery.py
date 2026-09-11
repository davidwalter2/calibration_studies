#!/usr/bin/env python3
"""Injection recovery and the certified fitted values.

For each (baseline, injected) pair of rabbit fitresults, prints

    value, error, EDM               for both fits
    shift = injected - baseline     against the injected truth
    leakage                         rms and worst of the OTHER parameters

The SHIFT is the observable, not the injected fit's absolute value: on this
gun the model has a known offset from the fit's own error (the Rossi-vs-
Moliere gap), so a material amount does not sit at zero even with no
injection -- see the `truth` table this also prints.

Everything is printed in PHYSICAL units: `k`, the log material amount of the
group, and `eps`, the linear variance scale of a hit class.  The card may
float a rescaled variable; the factor is read from the card's own
`group_units` (`groups.card_units`) and divided out here, so the values, the
errors, the injected truth and the prior are all directly comparable.

usage:
    recovery.py --pairs cf=fits/cf:fits/inj_cf gaussq=fits/gaussq:fits/inj_gaussq \\
        --card cards/inj_cf.hdf5 --param material_tib_support
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def read_fit(path):
    from rabbit import io_tools
    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    val = np.asarray(h.values(), np.float64)
    err = np.sqrt(np.asarray(h.variances(), np.float64))
    extra = {}
    for k in ("edmval", "nllvalfull", "nllvalreduced", "ndfsat"):
        if k in fr:
            try:
                v = fr[k]
                extra[k] = float(np.asarray(v.get() if hasattr(v, "get") else v))
            except Exception:  # noqa: BLE001
                pass
    cov = np.asarray(fr["cov"].get().values(), np.float64) if "cov" in fr else None
    return dict(names=names, val=val, err=err, cov=cov, **extra)


def read_card(path):
    import h5py
    out = {}
    if not path or not os.path.exists(path):
        return out
    with h5py.File(path, "r") as f:
        g = f.get("auxiliary/global_index_map")
        if g is None:
            return out
        for k in g.keys():
            try:
                v = g[k][...]
            except OSError:
                # a filter plugin the image does not ship (the string datasets
                # are written with one); the numeric ones we need read fine.
                continue
            if v.dtype.kind in "SO":
                out[k] = [s.decode() if isinstance(s, bytes) else str(s)
                          for s in v]
            else:
                out[k] = np.asarray(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", nargs="+", required=True,
                   metavar="LABEL=BASEDIR:INJDIR")
    p.add_argument("--card", default=None)
    p.add_argument("--param", default="material_tib_support")
    p.add_argument("--truth", type=float, default=None,
                   help="the value of --param given to make_*_card.py "
                        "--inject, for when the card's own string datasets "
                        "cannot be read; card units, as typed there")
    p.add_argument("--truth-only", nargs="*", default=[],
                   metavar="LABEL=DIR", help="fits to tabulate values for")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt",
                   help="the tier file; the card's own string datasets are "
                        "written with an HDF5 filter the image does not ship, "
                        "so the names and units are rebuilt from here")
    a = p.parse_args()

    card = read_card(a.card)
    inj = None
    if "params" in card and "injected" in card:
        inj = dict(zip(card["params"], np.asarray(card["injected"])))
    import groups as G
    gnames, gpri = G.group_param_names(42, a.groups)
    units = dict(zip(gnames, G.card_group_units(len(gnames), a.groups)))
    if "group_units" in card and len(card["group_units"]) == len(gnames):
        units = dict(zip(gnames, np.asarray(card["group_units"])))
    if inj is None and "injected" in card:
        inj = {}     # names unreadable; fall back to the CLI truth


    print("=" * 96)
    print("A. FITTED VALUES AT MC TRUTH (no injection) -- value +- error, "
          "and the physical amount")
    print("=" * 96)
    tabs = {}
    for spec in a.pairs:
        lab, paths = spec.split("=", 1)
        base, _ = paths.split(":", 1)
        tabs[lab] = read_fit(os.path.join(base, "fitresults.hdf5"))
    for spec in a.truth_only:
        lab, d = spec.split("=", 1)
        tabs[lab] = read_fit(os.path.join(d, "fitresults.hdf5"))
    labs = list(tabs)
    print(f"{'fit':<14}{'EDM':>12}{'NLL':>16}{'params':>9}")
    for lab in labs:
        t = tabs[lab]
        print(f"{lab:<14}{t.get('edmval', np.nan):>12.3e}"
              f"{t.get('nllvalfull', np.nan):>16.4f}{len(t['names']):>9}")
    ref = tabs[labs[0]]
    pri = dict(zip(gnames, gpri))
    mats = [i for i, n in enumerate(ref["names"]) if n.startswith("material_")]
    # order by how much the term CONSTRAINS the group relative to its prior;
    # a group at exactly 1.00 is unmeasured (the prior is the whole error).
    con = {i: (ref["err"][i] * units.get(ref["names"][i], 1.0)
               / max(pri.get(ref["names"][i], np.nan), 1e-300)) for i in mats}
    order = sorted(mats, key=lambda i: con[i])[: a.top]
    print()
    hdr = f"{'parameter':<28}{'prior':>8}"
    for lab in labs:
        hdr += f"{lab + ' k':>13}{'+-':>9}{'%mat':>8}{'/prior':>8}"
    print(hdr)
    for i in order:
        nm = ref["names"][i]
        u = units.get(nm, 1.0)
        row = f"{nm:<28}{pri.get(nm, np.nan):>8.3f}"
        for lab in labs:
            t = tabs[lab]
            if nm not in t["names"]:
                row += f"{'--':>13}{'--':>9}{'--':>8}{'--':>8}"
                continue
            j = t["names"].index(nm)
            k = t["val"][j] * u
            e = t["err"][j] * u
            row += (f"{k:>13.4f}{e:>9.4f}{100*(np.exp(k)-1):>8.2f}"
                    f"{e/max(pri.get(nm, np.nan),1e-30):>8.3f}")
        print(row)
    for extra in ("alpha",):
        if extra in ref["names"]:
            row = f"{extra:<28}{np.nan:>8.3f}"
            for lab in labs:
                t = tabs[lab]
                j = t["names"].index(extra) if extra in t["names"] else None
                row += (f"{t['val'][j]:>13.4f}{t['err'][j]:>9.4f}"
                        f"{np.nan:>8.2f}{np.nan:>8.3f}" if j is not None
                        else f"{'--':>13}{'--':>9}{'--':>8}{'--':>8}")
            print(row)

    print()
    print("=" * 96)
    print(f"B. INJECTION RECOVERY for {a.param}  (physical units)")
    upar = units.get(a.param, 1.0)
    _t = (a.truth if a.truth is not None
          else (inj.get(a.param, np.nan) if inj else np.nan))
    if np.isfinite(_t):
        print(f"   injected card value {_t:+.6g}  = k "
              f"{_t*upar:+.6f}  = {100*(np.exp(_t*upar)-1):+.3f} % material")
    print("=" * 96)
    print("   SIGN: the injection scales that group's exponents by exp(+k) in "
          "the CARD, i.e.\n   it declares the model at k = 0 to already have "
          "exp(k) times the material, so\n   the MLE moves by -k.  What is "
          "tested is |shift| against |truth|.")
    print("   PRIOR SHRINKAGE: the group carries its tier prior, and the "
          "injection is about\n   one prior sigma, so the POSTERIOR only moves "
          "by a fraction\n   f = sigma_post^2 / sigma_lik^2 of it, with "
          "1/sigma_lik^2 = 1/sigma_post^2 - 1/sigma_pri^2.\n   The "
          "prior-corrected column divides the shift by that f.")
    print(f"{'arm':<12}{'baseline':>11}{'+-':>9}{'injected':>11}{'+-':>9}"
          f"{'shift':>11}{'/truth':>8}{'f_pri':>7}{'corr/truth':>11}"
          f"{'pull_lik':>9}{'leak':>7}")
    for spec in a.pairs:
        lab, paths = spec.split("=", 1)
        base, injd = paths.split(":", 1)
        b = read_fit(os.path.join(base, "fitresults.hdf5"))
        q = read_fit(os.path.join(injd, "fitresults.hdf5"))
        if a.param not in b["names"] or a.param not in q["names"]:
            print(f"{lab:<14} parameter not in the fit")
            continue
        ib, iq = b["names"].index(a.param), q["names"].index(a.param)
        vb, vq = b["val"][ib] * upar, q["val"][iq] * upar
        sh = vq - vb
        e = q["err"][iq] * upar
        tru = (a.truth if a.truth is not None
               else (inj.get(a.param, np.nan) if inj else np.nan))
        tru = tru * upar
        # leakage: shift of every OTHER parameter, in units of its own error
        common = [n for n in b["names"] if n in q["names"] and n != a.param]
        d_ = np.array([(q["val"][q["names"].index(n)]
                        - b["val"][b["names"].index(n)])
                       / max(q["err"][q["names"].index(n)], 1e-300)
                       for n in common])
        # the prior on this parameter, PHYSICAL: the group's tier prior, or
        # the card's own prior vector converted with the same units
        spri = dict(zip(gnames, gpri)).get(a.param, np.nan)
        if not np.isfinite(spri) and "prior_sigmas" in card:
            # the card's prior vector is in the same order as its params list,
            # which is (materials, hit classes) for a residual-only card
            gp_ = list(gnames) + [n for n in b["names"]
                                  if n.startswith("hitres_")]
            if a.param in gp_ and len(card["prior_sigmas"]) == len(gp_):
                spri = float(card["prior_sigmas"][gp_.index(a.param)]) * upar
        f_pri = np.nan
        if np.isfinite(spri) and spri > 0 and e < spri:
            # posterior precision = likelihood + prior, so a shift of the
            # likelihood centre by `tru` moves the posterior mean by
            #   f = sigma_post^2 / sigma_lik^2 = 1 - sigma_post^2/sigma_pri^2
            f_pri = 1.0 - (e ** 2) / (spri ** 2)
        corr = sh / f_pri if np.isfinite(f_pri) and f_pri else np.nan
        slik = e / np.sqrt(max(f_pri, 1e-300)) if np.isfinite(f_pri) else e
        print(f"{lab:<12}{vb:>11.5f}{b['err'][ib]*upar:>9.5f}"
              f"{vq:>11.5f}{e:>9.5f}{sh:>11.5f}"
              f"{sh/tru if tru else np.nan:>8.3f}{f_pri:>7.3f}"
              f"{corr/tru if tru else np.nan:>11.3f}"
              f"{(abs(corr) - abs(tru))/slik if tru else np.nan:>9.2f}"
              f"{np.sqrt(np.mean(d_**2)):>7.3f}")
        w = np.argsort(-np.abs(d_))[:3]
        print("      largest leakage: " + ", ".join(
            f"{common[j]} {d_[j]:+.2f} sigma" for j in w))


if __name__ == "__main__":
    main()

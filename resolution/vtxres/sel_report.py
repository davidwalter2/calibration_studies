#!/usr/bin/env python3
"""The STANDARD-SELECTION gates, read off the six fits of `run_sel.sh`.

Three arms per channel, all on the J/psi gun (pure signal), all EDM-certified,
all in PHYSICAL units (`k` = the log material amount of a group, `eps` = the
linear variance scale of a hit class, `alpha` in units of 1e-3):

  a  no |z_v| cut, untruncated likelihood     -- the REFERENCE
  b  |z_v| < 5 AND the truncated normalisation -- the STANDARD
  c  |z_v| < 5 with NO truncated normalisation -- the BIAS

What is quoted
  (a) vs (b)   they must agree within statistics: the cut plus the matching
               normalisation is a consistent likelihood for a truncated
               sample.  The comparison is of the SAME parameter measured on
               nested samples, so the error on the DIFFERENCE is
               sqrt(|sigma_a^2 - sigma_b^2|) (Cochran / Durbin: for nested
               samples the covariance of the two estimates is the variance of
               the more precise one), not sqrt(sigma_a^2 + sigma_b^2).
  (c) vs (b)   the same shift with the normalisation left out -- the number
               that says how big the mistake is.
  mass a vs b  the effect of the CUT on the mass term: alpha, the material
               groups, the hit classes.

usage:
  python3 sel_report.py --fits <R>/fits --groups <materialGroups50.txt>
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres"),
           os.path.join(_RES, "hitlik")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recovery as RC  # noqa: E402


def physical(fit, unit_of):
    """(name -> (value, error)) in physical units."""
    out = {}
    for nm, v, e in zip(fit["names"], fit["val"], fit["err"]):
        u = unit_of.get(nm, 1.0)
        out[nm] = (v * u, e * u)
    return out


def nested_sigma(sa, sb):
    """Error on (b - a) for NESTED samples (b subset of a).

    cov(a, b) = Var(b) when b is the subset and both are efficient, so
    Var(b - a) = Var(b) - Var(a).  The absolute value guards the case where
    the subset is the more precise one (it can be: a truncated likelihood is
    not the same estimator), and is stated rather than hidden.
    """
    return np.sqrt(np.abs(sb * sb - sa * sa))


def compare(A, B, names, lab, unit_of, sigma_mode="nested", top=8):
    """Shifts b - a, in TWO denominators, because they answer two questions.

    `n_own` = (b - a) / sigma_b, the shift in units of the parameter's OWN
    statistical error: "how much of my error bar does this move me?", which is
    the systematic-vs-statistical comparison and the one the 0.2 sigma
    threshold is about.

    `n_dif` = (b - a) / sigma(b - a), the significance of the shift itself.
    For NESTED samples sigma(b-a) = sqrt(|sigma_b^2 - sigma_a^2|); for the
    SAME data fitted two ways the two estimates are so correlated that no
    honest denominator exists, so `n_dif` is not printed there -- a
    normalisation left out is a BIAS, not a fluctuation, and `n_own` is the
    number that matters.
    """
    rows = []
    for nm in names:
        if nm not in A or nm not in B:
            continue
        va, ea = A[nm]
        vb, eb = B[nm]
        own = eb if eb > 0 else ea
        dif = nested_sigma(ea, eb) if sigma_mode == "nested" else np.nan
        rows.append((nm, vb - va, own,
                     (vb - va) / own if own > 0 else np.nan,
                     va, ea, vb, eb, dif,
                     (vb - va) / dif if (dif > 0) else np.nan))
    rows.sort(key=lambda r: -abs(r[3]))
    print(f"\n=== {lab}   ({len(rows)} parameters)")
    print(f"  {'parameter':28s} {'a':>12s} {'b':>12s} "
          f"{'b-a':>12s} {'sigma_b':>11s} {'n_own':>8s} {'n_dif':>8s}")
    for r in rows[:top] if top else rows:
        print(f"  {r[0]:28s} {r[4]:12.5f} {r[6]:12.5f} "
              f"{r[1]:12.5f} {r[2]:11.5f} {r[3]:8.2f} {r[9]:8.2f}")
    ns = np.array([abs(r[3]) for r in rows])
    ns = ns[np.isfinite(ns)]
    if len(ns):
        print(f"  |shift| / sigma_own:  max {ns.max():.2f}   median "
              f"{np.median(ns):.2f}   rms {np.sqrt((ns**2).mean()):.2f}   "
              f"> 0.2 sigma: {int((ns > 0.2).sum())}/{len(ns)}")
    nd = np.array([abs(r[9]) for r in rows])
    nd = nd[np.isfinite(nd)]
    if len(nd):
        print(f"  |shift| / sigma_dif:  max {nd.max():.2f}   median "
              f"{np.median(nd):.2f}   rms {np.sqrt((nd**2).mean()):.2f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--max-edm", type=float, default=1e-3)
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args()

    import groups as G
    gnames, _ = G.group_param_names(42, a.groups)
    unit_of = dict(zip(gnames, G.card_group_units(len(gnames), a.groups)))

    fits, ok = {}, True
    print(f"{'fit':<22}{'npar':>6}{'NLL(reduced)':>18}{'EDM':>12}{'cert':>6}")
    for nm in ("vtx_a_nocut", "vtx_b_cut_norm", "vtx_c_cut_nonorm",
               "mass_a_nocut", "mass_b_cut_norm", "mass_c_cut_nonorm"):
        path = os.path.join(a.fits, nm, "fitresults.hdf5")
        if not os.path.exists(path):
            print(f"{nm:<22}{'--':>6}{'(missing)':>18}")
            continue
        try:
            f = RC.read_fit(path)
        except (OSError, BlockingIOError) as e:      # a fit still writing
            print(f"{nm:<22}{'--':>6}{'(locked: ' + type(e).__name__ + ')':>18}")
            ok = False
            continue
        fits[nm] = f
        edm = f.get("edmval", np.nan)
        cert = "OK" if (np.isfinite(edm) and edm < a.max_edm) else "FAIL"
        ok &= cert == "OK"
        # `nllvalfull` is not written by this rabbit; `nllvalreduced` is the
        # one that is, and it is the quantity the certification compares
        # BETWEEN arms of the same data anyway.
        nll = f.get("nllvalfull", f.get("nllvalreduced", np.nan))
        print(f"{nm:<22}{len(f['names']):6d}{nll:18.6f}{edm:12.3e}{cert:>6}")
    if not ok:
        print("\n!! at least one fit is NOT certified (EDM >= "
              f"{a.max_edm:g}); the numbers below are not quotable")

    P = {k: physical(v, unit_of) for k, v in fits.items()}
    mats = [n for n in gnames]
    hits = sorted({n for f in fits.values() for n in f["names"]
                   if n.startswith("hitres_")})

    if "vtx_a_nocut" in P and "vtx_b_cut_norm" in P:
        compare(P["vtx_a_nocut"], P["vtx_b_cut_norm"], mats + hits,
                "GATE (a) vs (b): the cut WITH the truncated normalisation "
                "-- must agree", unit_of, top=a.top)
    if "vtx_b_cut_norm" in P and "vtx_c_cut_nonorm" in P:
        compare(P["vtx_b_cut_norm"], P["vtx_c_cut_nonorm"], mats + hits,
                "GATE (c) vs (b): the SAME data with the normalisation LEFT "
                "OUT -- the bias", unit_of, sigma_mode="same", top=a.top)
    if "mass_a_nocut" in P and "mass_b_cut_norm" in P:
        compare(P["mass_a_nocut"], P["mass_b_cut_norm"],
                ["alpha"] + mats + hits,
                "THE MASS TERM under the |z_v| < 5 cut (a -> b)",
                unit_of, top=a.top)
        for nm in ("alpha",):
            if nm in P["mass_a_nocut"] and nm in P["mass_b_cut_norm"]:
                va, ea = P["mass_a_nocut"][nm]
                vb, eb = P["mass_b_cut_norm"][nm]
                s = nested_sigma(ea, eb)
                print(f"\n  {nm} [1e-3]: {va:.4f} +- {ea:.4f} (no cut) -> "
                      f"{vb:.4f} +- {eb:.4f} (cut);  shift {vb-va:+.4f} = "
                      f"{(vb-va)/eb:+.2f} sigma_own"
                      + (f", {(vb-va)/s:+.2f} sigma_dif" if s > 0 else ""))


if __name__ == "__main__":
    main()

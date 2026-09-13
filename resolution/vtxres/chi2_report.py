#!/usr/bin/env python3
"""WHAT THE `chi2/ndof` CUT DOES TO EVERY TERM, read off the fits of
`run_chi2.sh`.

The method is section 15.4's, applied to the other acceptance cut of the
standard selection.  One extraction per sample with the cut OFF; cards that
differ ONLY by the cut value (0 = off, 3 = the standard, 5, 10); the shift of
every parameter between arms, in units of its OWN error, with 0.2 sigma as the
threshold at which a selection becomes a systematic of the measurement.

Two denominators, because they answer two questions (`sel_report.py`):
  n_own = (b - a) / sigma_b        how much of my error bar does this move me
  n_dif = (b - a) / sigma(b - a)   is the shift itself significant.  The
                                   samples are NESTED (the cut sample is a
                                   subset), so sigma(b-a) = sqrt(|s_b^2-s_a^2|).

usage:
  python3 chi2_report.py --fits <R>/fits --groups <materialGroups50.txt>
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

import recovery as RC            # noqa: E402
from sel_report import physical, nested_sigma, compare   # noqa: E402

MJPSI = 3.0969
MZ = 91.1876
#: reference mass of each channel, for turning `alpha` [1e-3] into MeV
MREF = {"gun": MJPSI, "dy": MZ}

ARMS = ("off", "c3", "c5", "c10")
ARMLAB = {"off": "no chi2 cut", "c3": "chi2/ndof < 3",
          "c5": "chi2/ndof < 5", "c10": "chi2/ndof < 10"}


def read_all(fitdir, max_edm):
    fits, cert = {}, {}
    rows = []
    for nm in sorted(os.listdir(fitdir)):
        path = os.path.join(fitdir, nm, "fitresults.hdf5")
        if not os.path.exists(path):
            continue
        try:
            f = RC.read_fit(path)
        except (OSError, BlockingIOError) as e:
            rows.append((nm, -1, np.nan, np.nan, f"LOCKED {type(e).__name__}"))
            continue
        fits[nm] = f
        edm = float(f.get("edmval", np.nan))
        nll = float(f.get("nllvalfull", f.get("nllvalreduced", np.nan)))
        ok = np.isfinite(edm) and edm < max_edm
        cert[nm] = ok
        rows.append((nm, len(f["names"]), nll, edm, "OK" if ok else "FAIL"))
    print(f"\n{'fit':<26}{'npar':>6}{'NLL(reduced)':>18}{'EDM':>12}{'cert':>6}")
    for nm, npar, nll, edm, c in rows:
        print(f"{nm:<26}{npar:6d}{nll:18.6f}{edm:12.3e}{c:>6}")
    bad = [nm for nm, ok in cert.items() if not ok]
    if bad:
        print(f"\n!! NOT CERTIFIED (EDM >= {max_edm:g}): {bad} -- "
              "those numbers are not quotable")
    return fits


def alpha_line(P, a, b, mref, laba, labb):
    if "alpha" not in P.get(a, {}) or "alpha" not in P.get(b, {}):
        return
    va, ea = P[a]["alpha"]
    vb, eb = P[b]["alpha"]
    s = nested_sigma(ea, eb)
    print(f"\n  alpha [1e-3]: {va:+.4f} +- {ea:.4f} ({laba})  ->  "
          f"{vb:+.4f} +- {eb:.4f} ({labb})")
    print(f"    shift {vb-va:+.4f} = {(vb-va)/eb:+.2f} sigma_own"
          + (f", {(vb-va)/s:+.2f} sigma_dif" if s > 0 else ""))
    print(f"    IN MASS UNITS at m_ref = {mref:.4f} GeV: "
          f"{mref*va:+.3f} -> {mref*vb:+.3f} MeV, "
          f"shift {mref*(vb-va):+.3f} MeV "
          f"(the arm's own error is {mref*eb:.3f} MeV)")


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

    fits = read_all(a.fits, a.max_edm)
    P = {k: physical(v, unit_of) for k, v in fits.items()}
    mats = list(gnames)
    hits = sorted({n for f in fits.values() for n in f["names"]
                   if n.startswith("hitres_")})

    for samp in ("gun", "dy"):
        for chan in ("mass", "vtx"):
            base = f"{samp}_{chan}_off"
            if base not in P:
                continue
            print("\n" + "=" * 78)
            print(f"### {samp.upper()} {chan.upper()}: the chi2 cut against "
                  f"NO chi2 cut")
            print("=" * 78)
            names = (["alpha"] if chan == "mass" else []) + mats + hits
            for arm in ("c3", "c5", "c10"):
                nm = f"{samp}_{chan}_{arm}"
                if nm not in P:
                    continue
                compare(P[base], P[nm], names,
                        f"{samp} {chan}: {ARMLAB['off']} -> {ARMLAB[arm]}",
                        unit_of, top=a.top)
                if chan == "mass":
                    alpha_line(P, base, nm, MREF[samp],
                               ARMLAB["off"], ARMLAB[arm])
            # the three-way of the VERTEX term: the chi2 cut interacts with
            # the |z_v| truncation, so the arm with BOTH off is quoted too
            bo = f"{samp}_vtx_bothoff"
            if chan == "vtx" and bo in P:
                compare(P[bo], P[base], mats + hits,
                        f"{samp} vtx: (c) BOTH cuts off, untruncated -> "
                        f"(b) |z_v| < 5 + truncated norm, no chi2 cut",
                        unit_of, top=a.top)
                compare(P[bo], P.get(f"{samp}_vtx_c3", {}), mats + hits,
                        f"{samp} vtx: (c) BOTH cuts off, untruncated -> "
                        f"(a) |z_v| < 5 + truncated norm + chi2/ndof < 3",
                        unit_of, top=a.top)


if __name__ == "__main__":
    main()

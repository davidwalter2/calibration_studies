#!/usr/bin/env python3
"""The gate-4 table: the m formulation against the v formulation, side by side.

Reads whatever is in `results/` and `results/eng/` and prints the comparison
the coordinator asked for -- `m_Z` and `Gamma_Z` against the generator,
inclusive and per `eta` band, plus the `K(m)` 5/6/7 ladder for `Gamma_Z` --
with the fit-free prediction alongside so the two can be read together.
"""
import glob
import json
import os
import sys

FS = os.path.dirname(os.path.abspath(__file__))

# tag -> (label, m-form tag, v-form tag)
ROWS = [
    ("inclusive, K 5 terms", "f380fl_base", "V_full"),
    ("K 6 terms",            "f380fl_s6",   "V_s6"),
    ("K 7 terms",            "f380fl_s7",   "V_s7"),
    ("|eta_lead| < 0.9",     "loc_etaB",    "V_etaB"),
    ("0.9 - 1.6",            "loc_etaT",    "V_etaT"),
    ("1.6 - 3.0",            "loc_etaE",    "V_etaE"),
    ("the assembly toy",     "F_toy",       "V_toy"),
]

# the eta bands fitted with the PER-BAND FSR kernel measured on that band's own
# selected candidates -- the one surviving hypothesis for the eta pattern
KROWS = [("|eta_lead| < 0.9", "V_etaB", "VK_etaB"),
         ("0.9 - 1.6", "V_etaT", "VK_etaT"),
         ("1.6 - 3.0", "V_etaE", "VK_etaE")]

# the fit-free prediction of sec. 0f.1, MeV: conditioning on sigma vs on k
PREDICT = {"inclusive, K 5 terms": (-15.33, +0.30),
           "|eta_lead| < 0.9": (-12.04, -0.23),
           "0.9 - 1.6": (-16.50, +0.44),
           "1.6 - 3.0": (-25.61, +0.20)}


def load(tag):
    for pat in (f"{FS}/results/eng/fit_{tag}.json", f"{FS}/results/fit_{tag}.json"):
        if os.path.exists(pat):
            d = json.load(open(pat))
            p = d["params"]
            x, e = d["fitted"], (d.get("sandwich_err") or d["err"])
            out = {q: (x[p.index(q)], e[p.index(q)])
                   for q in ("m_Z", "Gamma_Z") if q in p}
            out["n"] = d["n"]
            out["nit"] = d.get("nit")
            out["converged"] = d.get("converged")
            if out["converged"] is None:
                # a verdict from checkconv.py, run after the fact
                cv = f"{FS}/results/conv_" + os.path.basename(pat)[4:]
                if os.path.exists(cv):
                    out["converged"] = json.load(open(cv)).get("converged")
                    out["worst_poi_step_sigma"] = json.load(
                        open(cv)).get("worst_poi_sigma")
            out["worst_poi_step_sigma"] = d.get("worst_poi_step_sigma")
            out["gradmax"] = d.get("gradmax")
            return out
    return None


def cell(r, q):
    """One table cell, with the convergence verdict attached.

    A `!` means the fit stopped with a POI more than `conv_tol` of its own
    error from the minimum and MUST NOT BE QUOTED; a `?` means the result
    predates the gate and its convergence is unknown. `|grad|inf` is not the
    test -- see `checkconv.py`.
    """
    if r is None or q not in r:
        return f"{'--':>20s}"
    flag = " " if r.get("converged") else ("!" if r.get("converged") is False
                                           else "?")
    return f"{r[q][0]:+9.2f} +- {r[q][1]:5.2f}{flag} "


def main():
    print("GATE 4: the m formulation against the v formulation (p = 1.264)")
    print("  a trailing `!` = the fit did NOT converge in the POI directions "
          "and must not be quoted;\n  `?` = it predates the convergence gate "
          "and its status is unknown.\n")
    print(f"{'':22s} {'m_Z  (m form)':>18s} {'m_Z  (v form)':>18s}"
          f" {'Gamma_Z (m)':>18s} {'Gamma_Z (v)':>18s}")
    for lab, mt, vt in ROWS:
        rm, rv = load(mt), load(vt)
        print(f"{lab:22s} {cell(rm,'m_Z')} {cell(rv,'m_Z')}"
              f" {cell(rm,'Gamma_Z')} {cell(rv,'Gamma_Z')}")
    print("\nthe fit-free prediction of the same thing (sec. 0f.1), MeV:")
    print(f"{'':22s} {'cond. on sigma':>18s} {'cond. on k':>18s}")
    for lab, (a, b) in PREDICT.items():
        print(f"{lab:22s} {a:+18.2f} {b:+18.2f}")
    print("\nthe eta bands with the PER-BAND FSR kernel (v form both columns):")
    print(f"{'':22s} {'inclusive kernel':>18s} {'per-band kernel':>18s}")
    for lab, a_, b_ in KROWS:
        print(f"{lab:22s} {cell(load(a_),'m_Z')} {cell(load(b_),'m_Z')}")

    missing = [v for _, _, v in ROWS + KROWS if load(v) is None]
    if missing:
        print("\nstill missing: " + " ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

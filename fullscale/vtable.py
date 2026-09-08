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
            return {q: (x[p.index(q)], e[p.index(q)]) for q in ("m_Z", "Gamma_Z")
                    if q in p} | {"n": d["n"], "nit": d.get("nit")}
    return None


def cell(r, q):
    if r is None or q not in r:
        return f"{'--':>18s}"
    return f"{r[q][0]:+9.2f} +- {r[q][1]:5.2f}"


def main():
    print("GATE 4: the m formulation against the v formulation (p = 1.264)\n")
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
    missing = [v for _, _, v in ROWS if load(v) is None]
    if missing:
        print("\nstill missing: " + " ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

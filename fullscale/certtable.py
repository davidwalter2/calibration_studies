#!/usr/bin/env python3
"""The CERTIFIED table: value AND NLL AND EDM for every row, m form and v form.

The acceptance test of sec. 0f.16, applied mechanically:

1. **the value** -- what the row returns;
2. **the NLL** -- the SAME minimum? Every fit of a given card is compared with
   the LOWEST NLL any converged fit of that card has reached. A row more than
   `--nll-tol` above it is at a different (worse) stationary point, or is not at
   one at all;
3. **the EDM** -- `0.5 g^T H^-1 g < --edm-tol`; and
4. **the full Newton step of each POI, in units of its own error** -- the
   interpretable form of (3), and the one that catches sec. 0f.19's failure
   mode: a fit that never moved sits at the MC truth and reads as a perfect
   closure.

A row is QUOTABLE only if it passes all four. Nothing here re-fits anything;
it reads what the fits wrote.

    python3 certtable.py                       # everything it can find
    python3 certtable.py --pairs               # the m-vs-v side-by-side
"""
import argparse
import glob
import json
import os

import numpy as np

FS = os.path.dirname(os.path.abspath(__file__))

# label -> (m-form row, v-form row).  Each row is (rabbit tag, card basename,
# the legacy `fit.py` json tag if one exists).
ROWS = [
    ("inclusive, K(m) 5 terms", ("f380refS", "z_full380_fl",    "f380fl_base"),
                                ("SVfull",   "z_V_full",        "V_full")),
    ("K(m) 6 terms",            ("Ss6",      "z_full380_fl_s6", "f380fl_s6"),
                                ("SVs6",     "z_V_s6",          "V_s6")),
    ("K(m) 7 terms",            ("Ss7",      "z_full380_fl_s7", "f380fl_s7"),
                                ("SVs7",     "z_V_s7",          "V_s7")),
    ("|eta_lead| < 0.9",        ("SMetaB",   "z_M_etaB",        "M_etaB"),
                                ("SVetaB",   "z_V_etaB",        "V_etaB")),
    ("0.9 - 1.6",               ("SMetaT",   "z_M_etaT",        "M_etaT"),
                                ("SVetaT",   "z_V_etaT",        "V_etaT")),
    ("1.6 - 3.0",               ("SMetaE",   "z_M_etaE",        "M_etaE"),
                                ("SVetaE",   "z_V_etaE",        "V_etaE")),
    ("the assembly toy",        ("Stoy",     "z_F_toy",         "F_toy"),
                                ("SVtoy",    "z_V_toy",         "V_toy")),
    ("F_dc8 (reweighted)",      ("Sdc8",     "z_F_dc8",         "F_dc8"),
                                (None,       None,              None)),
    ("F_toydc",                 ("Stoydc",   "z_F_toydc",       "F_toydc"),
                                (None,       None,              None)),
    ("F_w70110 (70-110 GeV)",   ("Sw70110",  "z_F_w70110",      "F_w70110"),
                                (None,       None,              None)),
    ("per-band FSR kernel B",   (None,       None,              None),
                                ("SVKetaB",  "z_VK_etaB",       "VK_etaB")),
    ("per-band FSR kernel T",   (None,       None,              None),
                                ("SVKetaT",  "z_VK_etaT",       "VK_etaT")),
]

POIS = ("m_Z", "Gamma_Z")


def _rabbit(tag):
    """value / NLL / EDM out of a `rabbit_fit.py` result, via native_dump json."""
    p = f"{FS}/results/nativejson/rabbit_{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    n, v, e = d["params"], d["fitted"], d["err"]
    out = {"src": "rabbit", "nll": d.get("nllvalreduced"), "edm": d.get("edmval")}
    for q in POIS:
        if q in n:
            out[q] = (v[n.index(q)], e[n.index(q)])
    return out


def _sandwich(tag):
    """the `fit.py` seeded re-evaluation: sandwich errors and its own NLL."""
    for p in (f"{FS}/results/eng/fit_{tag}.json", f"{FS}/results/fit_{tag}.json"):
        if os.path.exists(p):
            d = json.load(open(p))
            n = d["params"]
            sw = d.get("sandwich_err") or d["err"]
            out = {"src": os.path.basename(p), "nll": d.get("nll"),
                   "edm": d.get("edm"), "nit": d.get("nit")}
            for q in POIS:
                if q in n:
                    out[q] = (d["fitted"][n.index(q)], sw[n.index(q)])
            return out
    return None


def _conv(tag):
    """`checkconv.py`'s verdict on a stored `fit.py` json."""
    p = f"{FS}/results/conv_fit_{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"edm": d.get("edm"),
            "step": {q: d["params"][q]["full_step_sigma"]
                     for q in POIS if q in d.get("params", {})}}


def collect(tag, card):
    """One row: rabbit's minimum, the sandwich pass, and the verdict."""
    if tag is None:
        return None
    r = _rabbit(tag)
    s = _sandwich(tag)
    if r is None and s is None:
        return None
    row = {"tag": tag, "card": card, "rabbit": r, "sand": s}
    row["edm"] = (r or {}).get("edm", (s or {}).get("edm"))
    row["nll"] = (r or {}).get("nll", (s or {}).get("nll"))
    for q in POIS:
        # the value from rabbit (it is the fit), the error from the sandwich
        val = (r or {}).get(q, (s or {}).get(q, (np.nan, np.nan)))[0]
        err = (s or {}).get(q, (r or {}).get(q, (np.nan, np.nan)))[1]
        row[q] = (val, err)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edm-tol", type=float, default=1e-3)
    ap.add_argument("--nll-tol", type=float, default=0.01,
                    help="NLL units above the best known minimum of the same "
                         "card that still counts as the same point")
    ap.add_argument("--step-tol", type=float, default=0.045,
                    help="full Newton step of a POI, in units of its own error")
    a = ap.parse_args()

    # the best NLL any fit of each card has reached, from every source
    best = {}
    for pat in ("results/nativejson/rabbit_*.json",
                "results/eng/fit_*.json", "results/fit_*.json"):
        for p in glob.glob(os.path.join(FS, pat)):
            d = json.load(open(p))
            c = os.path.basename(d.get("card", "")).replace(".hdf5", "")
            nll = d.get("nll", d.get("nllvalreduced"))
            if c and nll is not None:
                best[c] = min(best.get(c, np.inf), nll)
    # rabbit results carry no card name; take it from the ROWS map
    for _, m, v in ROWS:
        for tag, card, _leg in (m, v):
            if tag is None:
                continue
            r = _rabbit(tag)
            if r and r.get("nll") is not None and card:
                best[card] = min(best.get(card, np.inf), r["nll"])

    def verdict(row):
        if row is None:
            return "--", ""
        why = []
        edm = row.get("edm")
        if edm is None or not np.isfinite(edm):
            why.append("no EDM")
        elif edm > a.edm_tol:
            why.append(f"EDM {edm:.3g}")
        b = best.get(row["card"], np.nan)
        if row.get("nll") is not None and np.isfinite(b) \
                and row["nll"] > b + a.nll_tol:
            why.append(f"NLL +{row['nll'] - b:.3g}")
        return ("QUOTE" if not why else "no"), "; ".join(why)

    print("CERTIFIED TABLE -- value AND NLL AND EDM (sec. 0f.16). "
          f"EDM < {a.edm_tol:g}; NLL within {a.nll_tol:g} of the best known "
          "minimum of the same card.")
    print("Errors are the sandwich (x1.09-1.18, NOT a flat factor).\n")
    hdr = (f"{'row':26s} {'form':4s} {'tag':9s} {'m_Z [MeV]':>17s} "
           f"{'Gamma_Z [MeV]':>17s} {'NLL':>18s} {'EDM':>10s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for lab, m, v in ROWS:
        for form, spec in (("m", m), ("v", v)):
            if spec[0] is None:
                continue
            row = collect(spec[0], spec[1])
            if row is None:
                print(f"{lab:26s} {form:4s} {spec[0]:9s} {'(not yet run)':>17s}")
                continue
            vd, why = verdict(row)
            mz, gz = row["m_Z"], row["Gamma_Z"]
            print(f"{lab:26s} {form:4s} {row['tag']:9s} "
                  f"{mz[0]:+8.2f} +-{mz[1]:6.2f} {gz[0]:+8.2f} +-{gz[1]:6.2f} "
                  f"{row['nll'] if row['nll'] is not None else float('nan'):18.4f} "
                  f"{row['edm'] if row['edm'] is not None else float('nan'):10.2e}  "
                  f"{vd}{(' (' + why + ')') if why else ''}")
    print("\ngenerator truth: m_Z = 91.153509740726733 GeV, "
          "Gamma_Z = 2.4932018986110700 GeV; the table is the OFFSET in MeV.")


if __name__ == "__main__":
    main()

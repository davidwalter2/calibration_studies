#!/usr/bin/env python3
"""The J/psi FSR-kernel comparison table, from `native_dump.py` JSONs.

Rows are `(tag, json)` pairs; columns are what the comparison is about:
`m_Z`, `Gamma_Z`, `bfield_mode0`, the mass-coherent projection of the whole
field-mode vector, the largest material pulls, NLL and EDM.  Differences are
taken against the row named by `--ref`.

`bfield_mode0` alone is not the momentum scale.  The scale the J/psi term
actually moves along is `<D_card>`, the mean of the card's own Jacobian over
its candidates, so the table also reports

    dm_pred = <D_card> . theta          [MeV]

which is the mean predicted mass shift the fitted calibration vector produces
on the J/psi leg -- the quantity a delta kernel forces to equal `-<dm>` and a
kernelled term does not.

usage::

    python3 jpsi_fsr_table.py --ref P2XP \\
        P2XP=runs/engaging_260913/rabbit_P2XP.json \\
        P2K=runs/engaging_260916/rabbit_P2K.json \\
        J0=runs/engaging_260916/rabbit_J0.json \\
        JK=runs/engaging_260916/rabbit_JK.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDM_TOL = 1e-3


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    p = list(d["params"])
    return {"path": path, "p": p, "v": np.asarray(d["fitted"], float),
            "e": np.asarray(d["err"], float),
            "nll": d.get("nllvalreduced", float("nan")),
            "edm": d.get("edmval", float("nan")),
            "idx": {n: i for i, n in enumerate(p)}}


def get(r, name):
    i = r["idx"].get(name)
    return (float("nan"), float("nan")) if i is None else (r["v"][i], r["e"][i])


def dmean(rows, dbar, names):
    """`<D_card> . theta` per row, in MeV."""
    out = {}
    for tag, r in rows.items():
        th = np.array([get(r, n)[0] for n in names])
        th = np.where(np.isfinite(th), th, 0.0)
        out[tag] = float(dbar @ th)          # dbar is already MeV per unit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", metavar="TAG=JSON")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--dbar", default=os.path.join(HERE, "runs",
                                                   "jpsi_Dbar.npz"),
                    help="npz with `names` and `dbar` (MeV per card unit), "
                         "written by `--make-dbar`")
    ap.add_argument("--nmat", type=int, default=6)
    a = ap.parse_args()

    rows = {}
    for spec in a.rows:
        tag, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  !! {tag}: {path} is not there")
            continue
        rows[tag] = load(path)
    if not rows:
        sys.exit("no results")
    ref = a.ref if a.ref in rows else list(rows)[0]

    print(f"{'row':8s} {'cert':>5s} {'m_Z [MeV]':>20s} {'Gamma_Z [MeV]':>20s} "
          f"{'bfield_mode0':>22s} {'NLL':>18s} {'EDM':>10s}")
    print("-" * 110)
    for tag, r in rows.items():
        m, gz, b0 = get(r, "m_Z"), get(r, "Gamma_Z"), get(r, "bfield_mode0")
        cert = "OK" if r["edm"] < EDM_TOL else "NO"
        print(f"{tag:8s} {cert:>5s} {m[0]:+11.3f} +-{m[1]:6.3f} "
              f"{gz[0]:+11.3f} +-{gz[1]:6.3f} "
              f"{b0[0] * 1e3:+13.5f} +-{b0[1] * 1e3:6.5f} "
              f"{r['nll']:18.4f} {r['edm']:10.2e}")
    print("\n(bfield_mode0 in 1e-3 card units; `cert` is EDM < 1e-3 -- the "
          "positive-definite Hessian is the fit's own covariance step, check "
          "the log)")

    print(f"\ndifferences against {ref}:")
    R = rows[ref]
    for tag, r in rows.items():
        if tag == ref:
            continue
        m, gz, b0 = get(r, "m_Z"), get(r, "Gamma_Z"), get(r, "bfield_mode0")
        mr, gr, br = get(R, "m_Z"), get(R, "Gamma_Z"), get(R, "bfield_mode0")
        print(f"  {tag} - {ref}:  d m_Z = {m[0] - mr[0]:+8.3f} MeV   "
              f"d Gamma_Z = {gz[0] - gr[0]:+8.3f} MeV   "
              f"d bfield_mode0 = {(b0[0] - br[0]) * 1e3:+9.5f} e-3")

    if os.path.exists(a.dbar):
        z = np.load(a.dbar, allow_pickle=True)
        names = [str(s) for s in z["names"]]
        dbar = np.asarray(z["dbar"], float)
        dm = dmean(rows, dbar, names)
        print("\nthe mean predicted J/psi mass shift the fitted calibration "
              "vector produces, <D_card> . theta [MeV]")
        print("  (a delta kernel has to make this equal -<dm> = +7.1969 MeV; "
              "a kernelled term does not)")
        for tag in rows:
            print(f"  {tag:8s} {dm[tag]:+9.4f} MeV = {dm[tag] / 3096.9:+.4e} "
                  "of the momentum scale")
    else:
        print(f"\n(no {a.dbar}: run `make_dbar.py` for the <D_card> column)")

    # the material pulls
    print("\nlargest material pulls (value / error):")
    for tag, r in rows.items():
        mats = [(n, r["v"][i], r["e"][i]) for n, i in r["idx"].items()
                if n.startswith("material_") and r["e"][i] > 0]
        mats.sort(key=lambda t: -abs(t[1] / t[2]))
        s = "  ".join(f"{n[9:]} {v * 100:+.2f}% ({v / e:+.0f}s)"
                      for n, v, e in mats[:a.nmat])
        print(f"  {tag:8s} {s}")


if __name__ == "__main__":
    main()

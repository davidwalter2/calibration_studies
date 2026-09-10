#!/usr/bin/env python3
"""Certify every fit of the ladder: value, NLL and rabbit's own EDM.

A fit is quoted only if all three are there:
  * the VALUE (and its error) of the parameter the fit is about,
  * the NLL at the minimum (`nllvalfull`, the full one, priors included),
  * rabbit's EDM = 1/2 g^T H^-1 g with the FULL Hessian -- the only
    convergence statement that means anything (NOTES: |g|_inf and diagonal
    proxies sit 1-5 sigma off).

usage: certify.py --fits runs/perhit/fits [--param material_tib_support]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HL = os.path.dirname(_HERE)
for _p in (_HERE, _HL, os.path.dirname(_HL), os.path.join(os.path.dirname(_HL), "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recovery as RC  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", required=True)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--params", nargs="+",
                   default=["material_tib_support", "hitres_str_N3_lo"])
    p.add_argument("--max-edm", type=float, default=1e-3)
    a = p.parse_args()

    names = a.names or sorted(d for d in os.listdir(a.fits)
                              if os.path.exists(os.path.join(a.fits, d,
                                                             "fitresults.hdf5")))
    hdr = (f"{'fit':<20}{'npar':>6}{'NLLred(min)':>16}{'EDM':>11}{'cert':>6}")
    for q in a.params:
        hdr += f"{q.split('_', 1)[1][:12]:>26}"
    print(hdr)
    print("-" * len(hdr))
    nbad = 0
    for nm in names:
        try:
            f = RC.read_fit(os.path.join(a.fits, nm, "fitresults.hdf5"))
        except Exception as e:  # noqa: BLE001
            print(f"{nm:<20}  UNREADABLE: {type(e).__name__}: {e}")
            nbad += 1
            continue
        edm = f.get("edmval", np.nan)
        ok = np.isfinite(edm) and abs(edm) < a.max_edm
        nbad += 0 if ok else 1
        row = (f"{nm:<20}{len(f['names']):>6}{(f.get('nllvalfull', f.get('nllvalreduced', np.nan))):>16.4f}"
               f"{edm:>11.2e}{'PASS' if ok else 'FAIL':>6}")
        idx = {q: i for i, q in enumerate(f["names"])}
        for q in a.params:
            if q in idx:
                i = idx[q]
                row += f"{f['val'][i]:>+14.5f} +- {f['err'][i]:<9.5f}"
            else:
                row += f"{'--':>26}"
        print(row)
    print(f"\n{len(names) - nbad}/{len(names)} certified at EDM < {a.max_edm:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

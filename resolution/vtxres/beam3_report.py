#!/usr/bin/env python3
"""THE `--beam3` CLOSURE -- every floated luminous-region parameter against
the SIMULATED luminous region, and what floating them does to the mass term.

Reads rabbit fit results, refuses to quote an uncertified one (rabbit's EDM,
not a gradient proxy), and reports

  1. the seven beam parameters: fitted value, error, and the pull against the
     GENERATOR's closed form and against the same quantity measured on this
     sample's own gen vertices (`beam3_gen.py`);
  2. the mass-channel parameters (`alpha`, and the hit-class scales) between
     two fits of the SAME candidates, one with `--beam3` and one without --
     the Delta is what floating the luminous region costs or buys the mass;
  3. the correlation matrix of the beam block, which is what says whether the
     seven are separately measured or one direction.

usage:
  ./run_tf.sh python3 beam3_report.py --fits A=dirA B=dirB [--genref f.npz]
      [--delta A,B] [--edm-max 1e-3]
"""
import argparse
import os
import sys

import numpy as np

BEAM = ["beamwidth_x", "beamwidth_y", "beamcorr_xy",
        "beamtilt_x", "beamtilt_y", "beamcentre_x", "beamcentre_y"]
UNIT = {"beamwidth_x": "eps", "beamwidth_y": "eps", "beamcorr_xy": "atanh(rho)",
        "beamtilt_x": "1e-5", "beamtilt_y": "1e-5",
        "beamcentre_x": "um", "beamcentre_y": "um"}


def read_fit(path):
    from rabbit import io_tools
    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    out = dict(names=names,
               val=np.asarray(h.values(), np.float64),
               err=np.sqrt(np.asarray(h.variances(), np.float64)))
    for k in ("edmval", "nllvalfull", "nllvalreduced"):
        if k in fr:
            try:
                v = fr[k]
                out[k] = float(np.asarray(v.get() if hasattr(v, "get") else v))
            except Exception:  # noqa: BLE001
                pass
    if "cov" in fr:
        out["cov"] = np.asarray(fr["cov"].get().values(), np.float64)
    return out


def get(f, nm):
    if nm not in f["names"]:
        return np.nan, np.nan
    i = f["names"].index(nm)
    return float(f["val"][i]), float(f["err"][i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", required=True,
                    help="LABEL=dir (or LABEL=path/fitresults.hdf5)")
    ap.add_argument("--genref", default=None,
                    help="the npz beam3_gen.py wrote (the closure reference)")
    ap.add_argument("--delta", default=None,
                    help="A,B -- report B minus A on the shared parameters")
    ap.add_argument("--edm-max", type=float, default=1e-3)
    ap.add_argument("--delta-params", default="alpha,hitres_",
                    help="comma-separated names / prefixes for the Delta table")
    a = ap.parse_args()

    fits = {}
    for spec in a.fits:
        lab, _, d = spec.partition("=")
        fp = d if d.endswith(".hdf5") else os.path.join(d, "fitresults.hdf5")
        if not os.path.exists(fp):
            print(f"[missing] {lab}: {fp}")
            continue
        fits[lab] = read_fit(fp)

    ref = np.load(a.genref, allow_pickle=False) if a.genref else None

    print(f"{'fit':18s} {'EDM':>11s} {'NLL':>16s} {'npar':>5s}  certified")
    bad = 0
    for lab, f in fits.items():
        edm = f.get("edmval", np.nan)
        cert = np.isfinite(edm) and edm < a.edm_max
        bad += (not cert)
        print(f"{lab:18s} {edm:11.2e} {f.get('nllvalfull', np.nan):16.6f} "
              f"{len(f['names']):5d}  {'YES' if cert else '*** NO ***'}")

    print("\n=== THE LUMINOUS-REGION PARAMETERS ===")
    if ref is not None:
        print(f"{'parameter':14s} {'unit':11s} {'generator':>10s} "
              f"{'gen-vtx':>18s}", end="")
    else:
        print(f"{'parameter':14s} {'unit':11s}", end="")
    for lab in fits:
        print(f" | {lab:>22s}", end="")
    print()
    for nm in BEAM:
        if not any(nm in f["names"] for f in fits.values()):
            continue
        line = f"{nm:14s} {UNIT[nm]:11s}"
        tgt = gm = ge = np.nan
        if ref is not None:
            tgt = float(ref[f"target_{nm}"])
            gm = float(ref[f"genmeas_{nm}"])
            ge = float(ref[f"generr_{nm}"])
            line += f" {tgt:+10.4f} {gm:+10.4f}+-{ge:5.4f}"
        for lab, f in fits.items():
            v, e = get(f, nm)
            if not np.isfinite(v):
                line += f" | {'--':>22s}"
            else:
                line += f" | {v:+9.4f} +- {e:6.4f}"
        print(line)
        if ref is not None:
            pl = f"{'':14s} {'pull':11s} {'':10s} {'':18s}"
            for lab, f in fits.items():
                v, e = get(f, nm)
                if not np.isfinite(v) or not (e > 0):
                    pl += f" | {'--':>22s}"
                else:
                    # the reference has its own error; the pull uses both
                    sg = np.sqrt(e * e + ge * ge)
                    pl += (f" | gen {(v-tgt)/e:+6.2f}  vtx {(v-gm)/sg:+6.2f}")
            print(pl)

    # correlations inside the beam block
    for lab, f in fits.items():
        if "cov" not in f:
            continue
        idx = [(nm, f["names"].index(nm)) for nm in BEAM if nm in f["names"]]
        if len(idx) < 2:
            continue
        c = f["cov"]
        print(f"\n--- {lab}: correlation of the beam block")
        print(f"{'':14s}" + "".join(f"{n[4:12]:>10s}" for n, _ in idx))
        for n1, i1 in idx:
            row = f"{n1:14s}"
            for _n2, i2 in idx:
                r = c[i1, i2] / np.sqrt(max(c[i1, i1] * c[i2, i2], 1e-300))
                row += f"{r:+10.3f}"
            print(row)

    if a.delta:
        la, lb = a.delta.split(",")
        if la in fits and lb in fits:
            A, B = fits[la], fits[lb]
            keys = [k for k in a.delta_params.split(",") if k]
            names = [n for n in A["names"] if n in B["names"]
                     and any(n == k or n.startswith(k) for k in keys)]
            print(f"\n=== THE MASS TERM: {lb} minus {la}, same candidates ===")
            print(f"{'parameter':26s} {la:>20s} {lb:>20s} {'Delta':>12s} "
                  f"{'Delta/sigma':>11s}")
            worst = 0.0
            for n in names:
                va, ea = get(A, n)
                vb, eb = get(B, n)
                d = vb - va
                s = d / ea if ea > 0 else np.nan
                worst = max(worst, abs(s) if np.isfinite(s) else 0.0)
                print(f"{n:26s} {va:+11.5f}+-{ea:7.5f} "
                      f"{vb:+11.5f}+-{eb:7.5f} {d:+12.5f} {s:+11.2f}")
            print(f"  largest |Delta/sigma| over {len(names)} parameters: "
                  f"{worst:.2f}")

    print(f"\n{bad} fit(s) not EDM-certified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

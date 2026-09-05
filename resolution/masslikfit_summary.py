#!/usr/bin/env python3
"""Combine the cf_masslik_fit.py result npz files into one comparison table.

    python3 masslikfit_summary.py runs/masslikfit_*.npz -o <outdir>

Written by the step-1 chain (chain_masslikfit_260903.sh); runs in any python
with numpy (no TensorFlow needed).
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("-o", "--outdir", default=None)
    a = ap.parse_args()
    outdir = a.outdir or os.path.expanduser(
        "~/public_html/cvh/%s_masslikfit/" % datetime.date.today().strftime("%y%m%d")
    )
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for pat in a.npz for f in glob.glob(pat))
    # the phiK tabulation cache lives in the same directory with a matching
    # prefix; keep only files that actually carry a fit result
    files = [f for f in files if "fit_x" in np.load(f, allow_pickle=True).files]
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("# cf_masslik_fit.py summary   %s" % datetime.datetime.now().isoformat(" ", "seconds"))
    P()
    hdr = (f"{'run':<22s} {'n':>8s} {'model':<9s} {'fbkg':<6s} "
           f"{'alpha[1e-3]':>20s} {'k_hit':>17s} {'k_ms':>17s} {'k_ioni':>17s} "
           f"{'k_rad':>17s} "
           f"{'f_bkg[1e-3]':>17s} {'NLL':>14s} {'cond(H)':>10s} {'wall[s]':>8s} "
           f"{'nhev':>5s} {'maxdScan':>10s}")
    P(hdr)
    P("-" * len(hdr))
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        pn = meta.get("parnames", [])
        x, e = z["fit_x"], z["fit_err"]
        val = dict(zip(pn, zip(x, e)))
        ev = z["fit_eig"]

        def cell(nm):
            key = next((k for k in val if k.split("[")[0] == nm), None)
            if key is None:
                return f"{'-':>17s}"
            v, s = val[key]
            return f"{v:10.5f} +-{s:7.5f}"

        ds = (f"{np.max(np.abs(z['scan_check_diff'])):10.2e}"
              if "scan_check_diff" in z.files else f"{'-':>10s}")
        name = os.path.basename(f).replace("masslikfit_", "").replace(".npz", "")
        model = meta.get("model", "?")
        if model == "r":
            kh = km = ki = "r"
            row_k = f"{'  (r) ' + f'{x[1]:8.5f} +-{e[1]:7.5f}':>17s}"
            P(f"{name:<22s} {int(z['n']):8d} {model:<9s} "
              f"{str(meta.get('float_bkg', False)):<6s} "
              f"{x[0]:10.5f} +-{e[0]:8.5f} "
              f"{row_k} {'':>17s} {'':>17s} {cell('f_bkg')} "
              f"{float(z['fit_nll']):14.4f} {ev.max()/ev.min():10.2e} "
              f"{float(z['fit_wall']):8.1f} {int(z['fit_nhev']):5d} {ds}")
        else:
            P(f"{name:<22s} {int(z['n']):8d} {model:<9s} "
              f"{str(meta.get('float_bkg', False)):<6s} "
              f"{x[0]:10.5f} +-{e[0]:8.5f} "
              f"{cell('k_hit')} {cell('k_ms')} {cell('k_ioni')} {cell('k_rad')} "
              f"{cell('f_bkg')} "
              f"{float(z['fit_nll']):14.4f} {ev.max()/ev.min():10.2e} "
              f"{float(z['fit_wall']):8.1f} {int(z['fit_nhev']):5d} {ds}")
    P()
    P("correlation matrices")
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        pn = list(meta.get("parnames", []))
        R = z["fit_corr"]
        P("  " + os.path.basename(f))
        w = max(len(n) for n in pn)
        P("    " + " " * (w + 2) + " ".join(f"{n:>11s}" for n in pn))
        for i, n in enumerate(pn):
            P("    " + f"{n:>{w}s}  " + " ".join(f"{R[i,j]:11.4f}" for j in range(len(pn))))
        P("    Hessian eigenvalues: " + "  ".join(f"{v:.4e}" for v in z["fit_eig"]))
    out = os.path.join(outdir, "masslikfit_summary.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()

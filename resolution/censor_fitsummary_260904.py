#!/usr/bin/env python3
"""One table of every mass-likelihood fit relevant to the 2026-09-04
censoring test.  Reads `meta` (json) + fit_x/fit_err from the npz."""
import glob, json, os, sys
import numpy as np

PAT = sys.argv[1:] or ["runs/masslikfit_*260903x*.npz",
                       "runs/masslikfit_*260904*.npz",
                       "runs/masslikfit_gun_260904*.npz",
                       "runs/masslikfit_v3_260904*.npz"]
HDR = ["alpha[1e-3]", "k_hit", "k_ms", "k_ioni", "k_rad", "r"]
rows = []
seen = set()
for pat in PAT:
    for f in sorted(glob.glob(pat)):
        if "phiKtab" in f or f in seen:
            continue
        seen.add(f)
        d = np.load(f, allow_pickle=True)
        try:
            mt = json.loads(str(d["meta"]))
        except Exception:
            continue
        P = dict(zip(mt["parnames"], zip(d["fit_x"], d["fit_err"])))
        win = mt.get("window_norm", False)
        w = mt.get("window", [0, 0])
        rows.append((os.path.basename(f)[len("masslikfit_"):-4],
                     int(mt["n"]),
                     os.path.basename(mt.get("subset") or "-").replace("mask_", "").replace(".npz", ""),
                     (f"{w[0]:.4g}-{w[1]:.4g}" if win else "."),
                     P, float(d["fit_nll"]),
                     float(np.max(np.abs(d["fit_x"] - d["fit_x_clip"])))))
print(f"{'fit':<38}{'n':>8}{'subset':>16}{'window':>13}"
      + "".join(f"{h:>21}" for h in HDR) + f"{'NLL':>13}{'floorshift':>11}")
for nm, n, sub, w, P, nll, fs in rows:
    s = f"{nm:<38}{n:>8}{sub[:16]:>16}{w:>13}"
    for h in HDR:
        if h in P:
            v, e = P[h]
            s += f"{v:>+12.5f}+-{e:<7.4f}"
        else:
            s += f"{'-':>21}"
    print(s + f"{nll:>13.2f}{fs:>11.4f}")

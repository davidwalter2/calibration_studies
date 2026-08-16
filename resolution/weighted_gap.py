#!/usr/bin/env python3
"""The mean-median gap of the fit's INFLUENCE-WEIGHTED energy loss.

Closes the last open item of NOTES_TRANSMISSION.md. Established there:

    ds_req = 1 - median(W)/mean(W),   W = sum_i w_i delta_i

with w_i the CVH fit's per-step influence weights, while the reference
fractions in `gate_refs.json` are the same gap for the UNWEIGHTED total loss
sum_i delta_i. The two coincide only if w is uniform, and it is not: the global
material-weighted mean is T_sys = 0.61 and the nested-cylinder probe measures
w running from 1 at the beam pipe to 0 at the outermost layer.

This computes both gaps on the SAME Geant4 tracks, so their ratio

    kappa = f(weighted) / f(unweighted)

is the correction the gate comparison needs, with the acceptance and
reference-definition systematics largely cancelling.

Everything here is Geant4 truth: per-plane true momentum `pabs` from the
cleanprop sim, and the model's own reference momentum `refp` per plane. No
saddlepoint, no CF inversion, no truncation convention.

usage:
  python weighted_gap.py --wprofile runs/cyl_weights.json
  python weighted_gap.py --wprofile runs/cyl_weights.json --labels pt3,pt40
"""
import argparse
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MMU = 0.1056583745
RMAX, ZMAX = 120.0, 300.0     # the cylinder family's normalisation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wprofile", default="runs/cyl_weights.json")
    p.add_argument("--labels", default="pt3,pt10e10,pt10e16,pt40,pt40b,pt100")
    p.add_argument("--nboot", type=int, default=400)
    p.add_argument("--out", default="runs/weighted_gap.json")
    return p.parse_args()


def plane_positions(mpath):
    """(refglobr, refglobz, refp, detid) per plane -- load_model drops the
    global position, so read the tree directly."""
    f = uproot.open(mpath)
    tkey = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    a = f[tkey].arrays(["detid", "refglobr", "refglobz", "refp"], library="np")
    return (np.asarray(a["refglobr"], float), np.asarray(a["refglobz"], float),
            np.asarray(a["refp"], float), np.asarray(a["detid"], np.int64))


def w_of_u(prof, u):
    """Interpolate the measured differential influence weight at cylinder
    coordinate u. Below the first knot w is pinned to the innermost measured
    value (the beam pipe measures 1.000, i.e. everything is downstream);
    above the last knot it is pinned to the outermost (0)."""
    prof = [p for p in prof if np.isfinite(p["w"]) and p["dM_frac"] > 1e-6]
    uu = np.array([p["u_mid"] for p in prof])
    ww = np.array([p["w"] for p in prof])
    o = np.argsort(uu)
    return np.interp(u, uu[o], ww[o], left=ww[o][0], right=ww[o][-1])


def gap_stats(x, nboot, rng):
    """(mean - median)/mean with a bootstrap error, plus the raw median."""
    m = np.mean(x)
    med = np.median(x)
    f = (m - med) / m if m != 0 else np.nan
    idx = rng.integers(0, len(x), (nboot, len(x)))
    s = x[idx]
    fb = (s.mean(axis=1) - np.median(s, axis=1)) / s.mean(axis=1)
    return float(f), float(np.std(fb)), float(m), float(med)


def main():
    args = parse_args()
    rng = np.random.default_rng(17)
    from cf_propagation_test import load_model, load_sim          # noqa: F401
    from gate_checks import SAMPLES

    with open(args.wprofile) as fh:
        prof = json.load(fh)
    prof = [p for p in prof if np.isfinite(p["w"]) and p["dM_frac"] > 1e-6]
    print("influence-weight profile (nested cylinders, u = max(r/120, |z|/300)):")
    for p in prof:
        print(f"  u = {p['u_lo']:.3f} - {p['u_hi']:.3f}  (r < {RMAX*p['u_hi']:6.1f} cm, "
              f"|z| < {ZMAX*p['u_hi']:6.1f} cm)   w = {p['w']:+.4f} +- {p['w_err']:.4f}")
    print()

    rows = []
    for lab in args.labels.split(","):
        if lab not in SAMPLES:
            print(f"[skip] {lab}: not in SAMPLES")
            continue
        mpath, spath = SAMPLES[lab]
        r, z, refp, mdet = plane_positions(mpath)
        try:
            sim = load_sim(spath, acceptance="perplane")
        except Exception as exc:                              # noqa: BLE001
            print(f"[skip] {lab}: no sim ({exc})")
            continue
        # MODEL/SIM CORRESPONDENCE -- asserted on the detid sequence, never
        # assumed. model_pt10_eta0.30_phi0.20.root shares not one module with
        # its supposed sim and is still live in the model directory.
        sdet = np.asarray(sim["detid"], np.int64)
        if len(sdet) != len(mdet) or not np.all(sdet == mdet):
            print(f"[skip] {lab}: model/sim detid sequence MISMATCH "
                  f"({len(sdet)} vs {len(mdet)})")
            continue

        nplane = len(refp)
        pabs = sim["pabs"]                     # (nev, nplane)
        valid = sim["valid"]
        allok = valid.all(axis=1) & np.isfinite(pabs).all(axis=1)
        nev = int(allok.sum())
        if nev < 500:
            print(f"[skip] {lab}: only {nev} tracks valid on all planes")
            continue
        P = pabs[allok]                        # (nev, nplane)
        E = np.sqrt(P ** 2 + MMU ** 2)
        Eref = np.sqrt(refp ** 2 + MMU ** 2)

        # per-block TRUE loss and the model's REFERENCE loss.  Block k is the
        # material between plane k-1 and plane k; block 0 is everything before
        # plane 0, for which the true starting energy is the model's own refp
        # at the entrance -- unavailable, so block 0 is dropped from BOTH sums
        # (it is the beam pipe + first pixel, ~1 MeV of ~25-100).
        dtrue = E[:, :-1] - E[:, 1:]                     # (nev, nplane-1)
        dref = (Eref[:-1] - Eref[1:])                    # (nplane-1,)
        u = np.maximum(np.abs(r) / RMAX, np.abs(z) / ZMAX)
        umid = 0.5 * (u[:-1] + u[1:])
        w = w_of_u(prof, umid)

        # deviation of the reference from the truth, unweighted and weighted
        U = (dref[None, :] - dtrue).sum(axis=1) * 1e3            # MeV
        W = ((dref[None, :] - dtrue) * w[None, :]).sum(axis=1) * 1e3
        mu_un = float(dref.sum() * 1e3)
        mu_w = float((dref * w).sum() * 1e3)

        # the gap as a FRACTION of the corresponding mean loss: this is the
        # quantity ds_req is compared with
        f_un = float(np.median(U)) / mu_un
        f_w = float(np.median(W)) / mu_w
        bs_un, bs_w = [], []
        for _ in range(args.nboot):
            k = rng.integers(0, nev, nev)
            bs_un.append(np.median(U[k]) / mu_un)
            bs_w.append(np.median(W[k]) / mu_w)
        e_un, e_w = float(np.std(bs_un)), float(np.std(bs_w))
        kappa = f_w / f_un if f_un != 0 else np.nan
        # error on the ratio from the same replicas (correlated: same tracks)
        bk = np.array(bs_w) / np.array(bs_un)
        e_k = float(np.std(bk))

        p0 = float(refp[0])
        Tsys_ray = mu_w / mu_un
        rows.append(dict(label=lab, p=p0, nev=nev, dE=mu_un, dE_w=mu_w,
                         T_ray=Tsys_ray, f_un=f_un, f_un_err=e_un,
                         f_w=f_w, f_w_err=e_w, kappa=kappa, kappa_err=e_k))
        print(f"{lab:9s} p={p0:7.2f} n={nev:6d}  dE={mu_un:7.2f} MeV  "
              f"dE_w={mu_w:7.2f}  T_ray={Tsys_ray:.3f} | "
              f"f_un={f_un:.4f}+-{e_un:.4f}  f_w={f_w:.4f}+-{e_w:.4f}  "
              f"kappa={kappa:.3f}+-{e_k:.3f}")

    if not rows:
        raise SystemExit("nothing computed")
    ks = np.array([r["kappa"] for r in rows])
    es = np.array([r["kappa_err"] for r in rows])
    wgt = 1.0 / es ** 2
    kbar = float(np.sum(wgt * ks) / np.sum(wgt))
    kerr = float(1.0 / np.sqrt(np.sum(wgt)))
    chi2 = float(np.sum(wgt * (ks - kbar) ** 2))
    print(f"\nkappa (weighted/unweighted gap fraction) = {kbar:.3f} +- {kerr:.3f}"
          f"   chi2/ndf = {chi2:.1f}/{len(ks)-1}")
    print("ds_req should be compared with kappa * f_ref, not f_ref.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(kappa=kbar, kappa_err=kerr, rows=rows), fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

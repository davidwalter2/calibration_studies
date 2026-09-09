#!/usr/bin/env python3
"""The two-component decomposition on the REAL discriminator, Z and J/psi legs.

STATE sec. 0f.50 tried this with `chi2/ndof` as a proxy and it did NOT
reproduce: the IN population it selects is `eta`-flat but so is its FRACTION,
where the hit-class agent's grows 2.4 -> 21.2 %. The actual discriminator is
`|seed -> final dq/p|` per leg, which `oddmoment/aux_seed.py` now extracts from
the two-track trees and aligns to the pairs caches.

Two levels, both against the same split:

PART A -- LEG level. The per-leg pull is
    z_l = (q/p final - q/p gen) / (sigrel_l |q/p final|)
and the TRUTH-REFERENCED pull (mandatory for anything charge-split, else
`<q z> = -a` is all one measures) is `x = z/(1 - a q z)` with
`a = sigrel_l (1 - vgf)`. `vgf` is the MASS-level Gaussian variance fraction --
the per-leg one is not exported -- so the two extremes `a = sigrel_l` (vgf = 0)
and `a = sigrel_l(1 - 2 vgf)` are printed as the systematic on it.
The statistic is the charge-EVEN mean, `0.5(<x>_+ + <x>_-)`, per `|eta|` band.

PART B -- MASS level, kernel-free, `data - model` with the FULL per-candidate
CF (`model_odd_mass.model_odd`), IN vs OUT per band. The candidate-level
discriminator is `max(dq_p, dq_m)` by default (a candidate is IN if EITHER leg
took a big step); `--leg lead` uses the leading-pT leg instead.

Binning: `|eta|` from the GEN legs (`corr(|eta| lead, z) = -0.008` on reco, and
gen is safer still) and the ABSOLUTE step only -- the SIGNED step has
`corr(., x) = +0.113`, a worse trap than reco pT. `corr(|dq|, |x|)` is measured
and printed for THIS sample before anything is binned on it.

    python3 mixture_legs.py --tag dyv2 [--pct 80 90 95]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "resolution"))
from model_odd_mass import model_odd, PROBES  # noqa: E402

BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"), (1.6, 2.4, "1.6-2.4")]
CFG = {
    "dyv2": dict(pairs="runs/zpairs_dyv2_full.npz", seed="runs/auxseed_dyv2.npz",
                 aux="runs/auxgen_dyv2.npz"),
    "jpsiv2": dict(pairs="runs/jpairs_v2_n600.npz", seed="runs/auxseed_jpsiv2.npz",
                   aux="runs/auxgen_jpsiv2.npz"),
}


def odd(x, u, w):
    return np.average(x * np.exp(-u * x * x), weights=w)


def boot_odd(x, u, w, nb, rng):
    n = len(x)
    if n < 50:
        return np.nan
    return float(np.std([odd(x[k], u, w[k])
                         for k in (rng.integers(0, n, n) for _ in range(nb))]))


def flatness(v, e):
    v, e = np.asarray(v, float), np.asarray(e, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return np.nan, np.nan, 0
    w = 1.0 / e[ok] ** 2
    mu = float((v[ok] * w).sum() / w.sum())
    return mu, float((w * (v[ok] - mu) ** 2).sum()), int(ok.sum() - 1)


def part_a(P, S, args, rng):
    """Leg-level charge-even skew, IN vs OUT on |seed->final dq/p|."""
    vgf = P["vgf"].astype(np.float64)
    w1 = P["w"].astype(np.float64) if "w" in P.files else np.ones(len(vgf))
    zl = np.concatenate([S["zleg_p"], S["zleg_m"]])
    q = np.concatenate([S["q_p"], S["q_m"]])
    sr = np.concatenate([S["sigrel_p"], S["sigrel_m"]])
    dq = np.concatenate([S["dq_p"], S["dq_m"]])
    eta = np.abs(np.concatenate([S["geta_p"], S["geta_m"]]))
    vg = np.concatenate([vgf, vgf])
    w = np.concatenate([w1, w1])

    print(f"\n{'='*78}\nPART A -- LEG level, charge-even <x>, u = {args.u}\n{'='*78}")
    variants = (("a = sigrel(1-vgf)  NOMINAL", sr * (1.0 - vg)),
                ("a = sigrel         vgf->0", sr),
                ("a = sigrel(1-2vgf) vgf->2vgf", sr * (1.0 - 2.0 * vg)))
    for vlab, a_l in variants:
        den = 1.0 - a_l * q * zl
        x = np.where(np.abs(den) > 1e-3, zl / den, np.nan)
        ok = np.isfinite(x) & (np.abs(x) < args.xmax) & np.isfinite(dq) & (dq >= 0)
        r = np.corrcoef(dq[ok], np.abs(x[ok]))[0, 1]
        ce = 0.5 * (odd(x[ok & (q > 0)], args.u, w[ok & (q > 0)])
                    + odd(x[ok & (q < 0)], args.u, w[ok & (q < 0)]))
        print(f"\n  {vlab}:  n = {int(ok.sum())}, corr(|dq|,|x|) = {r:+.4f}, "
              f"inclusive charge-even = {1e3*ce:+.2f} e-3")
        if vlab != variants[0][0] and not args.all_variants:
            continue
        thr_all = {p: np.percentile(dq[ok], p) for p in args.pct}
        for p in args.pct:
            thr = thr_all[p]
            print(f"    --- split at |dq/p| p{p:.0f} = {thr:.4e}")
            print(f"      {'band':14s} {'n':>9s} {'f_IN':>7s} "
                  f"{'IN':>18s} {'OUT':>18s}")
            rows = {"IN": ([], []), "OUT": ([], [])}
            for lo, hi, blab in BANDS:
                b = ok & (eta >= lo) & (eta < hi)
                f = np.average((dq > thr)[b], weights=w[b])
                cells = []
                for nm, sel in (("IN", dq > thr), ("OUT", dq <= thr)):
                    mk = b & sel
                    v = 0.5 * (odd(x[mk & (q > 0)], args.u, w[mk & (q > 0)])
                               + odd(x[mk & (q < 0)], args.u, w[mk & (q < 0)]))
                    e = boot_odd(x[mk], args.u, w[mk], args.nboot, rng)
                    rows[nm][0].append(1e3 * v)
                    rows[nm][1].append(1e3 * e)
                    cells.append(f"{1e3*v:+9.2f} +-{1e3*e:5.2f}")
                print(f"      {blab:14s} {int(b.sum()):9d} {100*f:6.2f}% "
                      f"{cells[0]:>18s} {cells[1]:>18s}")
            for nm in ("IN", "OUT"):
                mu, c2, nd = flatness(*rows[nm])
                print(f"      {nm:3s}: weighted mean {mu:+7.2f} e-3, "
                      f"chi2 vs eta-flat {c2:5.2f} / {nd}")
            # the mixture identity
            print(f"      {'mixture check':14s} "
                  f"{'f*IN+(1-f)*OUT':>18s} {'measured':>18s}")
            for i, (lo, hi, blab) in enumerate(BANDS):
                b = ok & (eta >= lo) & (eta < hi)
                f = np.average((dq > thr)[b], weights=w[b])
                pred = f * rows["IN"][0][i] + (1 - f) * rows["OUT"][0][i]
                meas = 0.5 * (odd(x[b & (q > 0)], args.u, w[b & (q > 0)])
                              + odd(x[b & (q < 0)], args.u, w[b & (q < 0)]))
                e = boot_odd(x[b], args.u, w[b], args.nboot, rng)
                print(f"      {blab:14s} {pred:+18.2f} "
                      f"{1e3*meas:+12.2f} +-{1e3*e:4.2f}")


def part_b(P, S, args, rng):
    """Mass level, data - model with the full CF, IN vs OUT."""
    d = dict(P)
    if "mgen" not in P.files and "eta" in P.files:
        d["mgen"] = P["eta"]          # this cache stores m_gen under `eta`

    class _D:                          # model_odd wants a `.files` mapping
        def __init__(self, dd):
            self._d = dd
            self.files = list(dd.keys())

        def __getitem__(self, k):
            return self._d[k]

    D = _D(d)
    z = np.asarray(d["z"], float)
    sig = np.asarray(d["sigma"], float)
    mg = np.asarray(d["mgen"], float)
    m = z * sig + mg
    w = np.asarray(d["w"], float) if "w" in d else np.ones(len(z))
    etal = np.maximum(np.abs(np.asarray(d["etap"], float)),
                      np.abs(np.asarray(d["etam"], float)))
    a_i = args.acoef * sig / m
    den = 1.0 - a_i * z
    x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
    if args.leg == "max":
        dq = np.maximum(S["dq_p"], S["dq_m"])
    elif args.leg == "lead":
        dq = np.where(np.asarray(d["ptp"], float) >= np.asarray(d["ptm"], float),
                      S["dq_p"], S["dq_m"])
    else:
        dq = 0.5 * (S["dq_p"] + S["dq_m"])
    ok = (np.isfinite(x) & (np.abs(x) < 8) & (sig / m < 0.10)
          & np.isfinite(dq) & (dq >= 0))
    MOD = model_odd(D, None, jensen=True)
    u = PROBES[0]
    r = np.corrcoef(dq[ok], np.abs(x[ok]))[0, 1]
    print(f"\n{'='*78}\nPART B -- MASS level, data - model (full CF), u = {u}, "
          f"discriminator = {args.leg}(dq_p, dq_m)\n{'='*78}")
    print(f"  {int(ok.sum())} candidates; corr(|dq|, |x|) = {r:+.4f}, "
          f"corr(|dq|, x) = {np.corrcoef(dq[ok], x[ok])[0,1]:+.4f}")
    for p in args.pct:
        thr = np.percentile(dq[ok], p)
        print(f"\n  === split at |dq/p| p{p:.0f} = {thr:.4e}")
        print(f"    {'band':14s} {'n':>9s} {'f_IN':>7s} {'IN':>19s} {'OUT':>19s}")
        rows = {"IN": ([], []), "OUT": ([], [])}
        for lo, hi, blab in BANDS:
            b = ok & (etal >= lo) & (etal < hi)
            f = np.average((dq > thr)[b], weights=w[b])
            cells = []
            for nm, sel in (("IN", dq > thr), ("OUT", dq <= thr)):
                mk = b & sel
                dv = odd(x[mk], u, w[mk])
                e = boot_odd(x[mk], u, w[mk], args.nboot, rng)
                mv = np.average(MOD[mk, 0], weights=w[mk])
                rows[nm][0].append(1e3 * (dv - mv))
                rows[nm][1].append(1e3 * e)
                cells.append(f"{1e3*(dv-mv):+10.2f} +-{1e3*e:5.2f}")
            print(f"    {blab:14s} {int(b.sum()):9d} {100*f:6.2f}% "
                  f"{cells[0]:>19s} {cells[1]:>19s}")
        for nm in ("IN", "OUT"):
            mu, c2, nd = flatness(*rows[nm])
            print(f"    {nm:3s}: weighted mean {mu:+7.2f} e-3, "
                  f"chi2 vs eta-flat {c2:5.2f} / {nd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dyv2", choices=sorted(CFG))
    ap.add_argument("--pct", type=float, nargs="*", default=[80, 90, 95])
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--xmax", type=float, default=30.0)
    ap.add_argument("--acoef", type=float, default=1.211,
                    help="the MEASURED a/(sigma/m) used to truth-reference the "
                         "mass pull (measure_a.py)")
    ap.add_argument("--leg", default="max", choices=["max", "lead", "mean"])
    ap.add_argument("--nboot", type=int, default=150)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20260908)
    c = CFG[args.tag]
    P = np.load(os.path.join(HERE, c["pairs"]))
    S = np.load(os.path.join(HERE, c["seed"]))
    assert np.array_equal(np.asarray(S["z"], float),
                          np.asarray(P["z"], float)), "seed cache misaligned"
    print(f"{args.tag}: {len(P['z'])} candidates, seed cache aligned "
          f"(z bit-identical)")
    if not args.skip_a:
        part_a(P, S, args, rng)
    if not args.skip_b:
        part_b(P, S, args, rng)


if __name__ == "__main__":
    main()

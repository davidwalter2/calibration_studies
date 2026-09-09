#!/usr/bin/env python3
"""THE NO-FREE-PARAMETER PREDICTION OF THE PHI HARMONICS from the measured
per-class CPE LOCATION bias.

WHY THIS IS THE DECISIVE TEST OF PART 1's NULL RESULT.
PART 1 measured the hit location bias (BPix local x +0.1193 sigma_CPE, BPix-1
alone +0.227 = +2.7 um; every strip subdetector null), propagated it through
the per-hit influence weights with no free parameter, and got
+0.86 / +1.59 / +1.39 e-3 against a measured -4.89 / -3.57 / +0.12 -- wrong
sign, wrong size, and >= 5.3 sigma away at every join key. That was a
PHI-AVERAGED comparison.

The propagation is `delta z = sum_b s_b a_b mu_b` with `s_b` the BENDING SENSE
read from `resinfbv`. `s_b` is a property of the MODULE'S ORIENTATION: PART 1
measured the influence-weighted `<|<s>_group|>` running 0.016 (subdetector) ->
0.072 (+layer) -> 0.159 (+z side) -> 0.761 (orientation group) -> 0.905
(module). In the Phase-0 pixel barrel the ladders alternate inner/outer, so a
location bias that is FIXED in the module's local x frame enters the bending
coordinate with a sign that alternates from ladder to ladder -- period 2
ladders. BPix-1 has 20 ladders, i.e. 36 deg, i.e. **n = 10**; the TEC has 8
petals per disk face, i.e. **n = 8**. The phi-AVERAGE of an alternating sign
is nearly zero, which is exactly why PART 1's prediction came out at 1e-3
while the harmonic content -- which does not average away -- is +20e-3.

So this script runs PART 1's propagation UNCHANGED and asks for its HARMONIC
content instead of its mean:

    A_n^cos = 2 <dz cos(n phi)>,   A_n^sin = 2 <dz sin(n phi)>

the same estimator applied to the data. `dz` is charge-INDEPENDENT by
construction (`s_b` is geometric), so its charge-even projection is just its
mean and the comparison is like for like.

THE ERROR ON THE PREDICTION is the `hitres` arm's finite statistics on each
class location: `A_n = sum_k L_k mu_k` with
`L_k = 2 <sum_{b in k} s_b a_b f(n phi)>`, so `var(A_n) = sum_k L_k^2 var(mu_k)`
exactly, as in `t3_keys.py`.

THE SIGN OF THE INFLUENCE FUNCTIONAL -- a bug in PART 1, fixed here from the
code and confirmed by the data.
`resinfbv` / `resinfv` are built from `W5 = VinvF * Cinvd.solve(E5)`
(`ResidualGlobalCorrectionMakerG4e.cc:4619`), i.e. `W5 = V^-1 F C^-1 E5`,
while the Gauss-Newton step the fit actually takes is
`dxfree = -Cinvd.solve(VinvF.transpose()*rfull)`  (:4098), i.e.
`dx = -C^-1 F^T V^-1 r`, and the state is updated by ADDING it
(`qbpupd = qbp + dxref[0]`, :4269). With C and V symmetric,

    W5^T r = E5^T C^-1 F^T V^-1 r = -dx  .

**The exported influence functional therefore has the OPPOSITE sign to the
fit's response**: a positive residual `r_b` moves q/p by `-wqop_b r_b`, not
`+wqop_b r_b`. (`F` is the RESIDUAL Jacobian, not the prediction one: the
constraint is `dy0 = hitx - lxcor` at :3349, measured minus predicted, and the
global-parameter block carries the matching minus,
`Jfull.block(...) = -Hm*dStateDparams` at :3035.) The VARIANCE uses of these
branches are unaffected -- `v_b = w^T dV w` is quadratic and sign-blind, which
is why nothing else in the campaign moves; only the LOCATION (odd) channel
PART 1 introduced does.

So the propagation is `delta z = -sum_b s_b a_b mu_b`, and `--sign +1` (the
default) applies that minus. `--sign -1` reproduces PART 1's published numbers
for comparison.

THE JOIN. `blocks_mugun_ul16_260903x.npz` and `conv_ref903x_full.npz` are two
extractions of the SAME production with the same three cuts in the same file
order, so they line up element by element -- asserted here on `z`, `sigma`,
`eta`, `pt` and `q` being bit-identical, not assumed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t3_keys as T3                                              # noqa: E402

BANDS = [(0.0, 0.9, "barrel"), (0.9, 1.6, "middle"), (1.6, 2.4, "endcap")]


def bank(h, b, key_h, key_b, nmin=150):
    """(mu, var(mu)) per block, from the hitres arm's class locations."""
    mu = np.zeros(len(key_b))
    ve = np.zeros(len(key_b))
    for isy, pk in ((0, "pullx"), (1, "pully")):
        p = h[pk].astype(np.float64)
        ok = np.isfinite(p) & (np.abs(p) < 10)
        kk = key_h[ok] * 2 + isy
        u, inv = np.unique(kk, return_inverse=True)
        n = np.bincount(inv)
        m1 = np.bincount(inv, weights=p[ok]) / np.maximum(n, 1)
        c = p[ok] - m1[inv]
        m2 = np.bincount(inv, weights=c ** 2) / np.maximum(n, 1)
        good = n >= nmin
        bm = dict(zip(u[good], m1[good]))
        bv = dict(zip(u[good], (m2 / np.maximum(n, 1))[good]))
        sel = (b["b_isy"] == isy)
        kb = key_b[sel] * 2 + isy
        mu[sel] = [bm.get(int(x), 0.) for x in kb]
        ve[sel] = [bv.get(int(x), 0.) for x in kb]
    return mu, ve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+",
                    default=["sd+lay", "og", "og+cls", "mod"])
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--sign", type=float, default=+1.,
                    help="+1 = the corrected sign (delta z = -sum s a mu); "
                         "-1 reproduces PART 1's published, wrong-signed numbers")
    a = ap.parse_args()

    h = np.load("data/hits_mugun_ul16_h2.npz")
    b = np.load("data/blocks_mugun_ul16_260903x.npz")
    c = np.load("data/conv_ref903x_full.npz")
    for k1, k2 in (("t_z", "z"), ("t_sigma", "sigma"), ("t_eta", "eta"),
                   ("t_pt", "pt"), ("t_q", "q")):
        if not np.array_equal(np.asarray(b[k1], float), np.asarray(c[k2], float)):
            raise SystemExit(f"caches do NOT line up on {k1} -- re-extract")
    ph = c["genphi"].astype(np.float64)
    npx = c["npixhit"]
    aeta = np.abs(b["t_eta"].astype(np.float64))
    band = np.digitize(aeta, [0.9, 1.6])
    x = np.where(np.abs(1 - b["t_sigma"] * b["t_pt"] * np.cosh(b["t_eta"])
                        * (1 - b["t_vgf"]) * b["t_q"] * b["t_z"]) > 1e-3,
                 b["t_z"] / (1 - b["t_sigma"] * b["t_pt"] * np.cosh(b["t_eta"])
                             * (1 - b["t_vgf"]) * b["t_q"] * b["t_z"]), np.nan)
    q = b["t_q"].astype(np.float64)
    good = np.isfinite(x) & (np.abs(x) < 10)

    s = b["b_s"].astype(np.float64)
    amp = np.sqrt(b["b_a2"].astype(np.float64))
    ntr = len(b["nblk"])
    trk = np.repeat(np.arange(ntr), b["nblk"])
    print(f"tracks {ntr}, hit blocks {len(s)}, "
          f"hitres hits {len(h['pullx'])}\n")

    # --- the MEASURED harmonics, for the comparison ------------------------
    print("=== MEASURED charge-even harmonics (the target) ===")
    meas = {}
    for n in range(1, a.nmax + 1):
        row = []
        for f in (np.cos, np.sin):
            vv, ee = [], []
            for sg in (+1, -1):
                m = good & (q == sg)
                vv.append(2. * np.mean(x[m] * f(n * ph[m])))
                ee.append(2. * np.std(x[m] * f(n * ph[m])) / np.sqrt(m.sum()))
            row.append((0.5 * (vv[0] + vv[1]), 0.5 * float(np.hypot(*ee))))
        meas[n] = row
    for n in (8, 10):
        print(f"  n={n:2d}  cos {meas[n][0][0]*1e3:+7.2f}+-{meas[n][0][1]*1e3:4.2f}"
              f"   sin {meas[n][1][0]*1e3:+7.2f}+-{meas[n][1][1]*1e3:4.2f}")

    for spec in a.keys:
        key_h = T3.make_key(h, spec, "")
        key_b = T3.make_key(b, spec, "b_")
        mu, ve = bank(h, b, key_h, key_b)
        # THE SIGN: dz = -sum_b s_b a_b mu_b (see the module docstring)
        sa = -a.sign * s * amp
        w = sa * mu                           # per-block contribution to dz
        dz = np.bincount(trk, weights=w, minlength=ntr)
        # exact propagation of the hitres statistical error onto every number
        uk, ik = np.unique(key_b * 2 + b["b_isy"].astype(np.int64),
                           return_inverse=True)
        vk = np.zeros(len(uk))
        vk[ik] = ve                            # var(mu) is a per-key constant

        def pred(weight):
            """sum_k L_k mu_k and its error, for a per-TRACK weight."""
            wt = weight[trk]
            val = float((sa * mu * wt).sum() / ntr)
            L = np.bincount(ik, weights=sa * wt, minlength=len(uk)) / ntr
            return val, float(np.sqrt((L ** 2 * vk).sum()))

        print(f"\n{'='*74}\n=== KEY = {spec}   "
              f"({len(uk)//2} cells with a measured location)\n{'='*74}")
        print("  phi-AVERAGED mean per band, prediction vs the MEASURED "
              "charge-even <x>, 1e-3:")
        for i, (lo, hi, nm) in enumerate(BANDS):
            sel = (band == i).astype(float) * ntr / max((band == i).sum(), 1)
            v, e = pred(sel)
            mv, me = [], []
            for sg in (+1, -1):
                m = good & (band == i) & (q == sg)
                mv.append(np.mean(x[m]))
                me.append(np.std(x[m]) / np.sqrt(m.sum()))
            mm = 0.5 * (mv[0] + mv[1])
            mev = 0.5 * float(np.hypot(*me))
            print(f"    {nm:8s} pred {v*1e3:+7.2f}+-{e*1e3:4.2f}   meas "
                  f"{mm*1e3:+7.2f}+-{mev*1e3:4.2f}   pull "
                  f"{(mm-v)/np.hypot(e, mev):+5.2f}")
        print("\n  HARMONICS: predicted vs measured, 1e-3")
        print(f"  {'n':>3s}{'pred cos':>18s}{'meas cos':>18s}"
              f"{'pred sin':>18s}{'meas sin':>18s}")
        for n in range(1, a.nmax + 1):
            pc, ec = pred(2. * np.cos(n * ph))
            ps, es = pred(2. * np.sin(n * ph))
            mark = "  <---" if n in (8, 10) else ""
            if n in (8, 10) or max(abs(pc), abs(ps)) * 1e3 > 3.:
                print(f"  {n:3d}{pc*1e3:+12.2f}+-{ec*1e3:4.2f}"
                      f"{meas[n][0][0]*1e3:+12.2f}+-{meas[n][0][1]*1e3:4.2f}"
                      f"{ps*1e3:+12.2f}+-{es*1e3:4.2f}"
                      f"{meas[n][1][0]*1e3:+12.2f}+-{meas[n][1][1]*1e3:4.2f}{mark}")
        # global goodness of the no-free-parameter harmonic prediction
        c2p = c2n = 0.
        for n in range(1, a.nmax + 1):
            for f in (np.cos, np.sin):
                pv, pe = pred(2. * f(n * ph))
                mv, me = (meas[n][0] if f is np.cos else meas[n][1])
                c2p += (mv - pv) ** 2 / (pe ** 2 + me ** 2)
                c2n += mv ** 2 / me ** 2
        print(f"\n  GLOBAL over n = 1..{a.nmax} (cos and sin, {2*a.nmax} dof):"
              f"  chi2(measured == PREDICTED) = {c2p:6.1f}"
              f"   chi2(measured == 0) = {c2n:6.1f}")

        print("\n  DOSE RESPONSE in nValidPixelHits, n=10 sine, 1e-3")
        print(f"  {'npix':>5s}{'pred':>16s}{'measured':>18s}")
        for k in range(5):
            sel = (npx == k)
            if sel.sum() < 3000:
                continue
            wt = np.where(sel, 2. * np.sin(10 * ph) * ntr / sel.sum(), 0.)
            p10, e10 = pred(wt)
            mv, me = [], []
            for sg in (+1, -1):
                m = good & sel & (q == sg)
                mv.append(2. * np.mean(x[m] * np.sin(10 * ph[m])))
                me.append(2. * np.std(x[m] * np.sin(10 * ph[m])) / np.sqrt(m.sum()))
            print(f"  {k:5d}{p10*1e3:+11.2f}+-{e10*1e3:4.2f}"
                  f"{0.5*(mv[0]+mv[1])*1e3:+12.2f}+-"
                  f"{0.5*float(np.hypot(*me))*1e3:4.2f}")


if __name__ == "__main__":
    main()

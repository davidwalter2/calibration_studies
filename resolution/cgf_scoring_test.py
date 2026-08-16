#!/usr/bin/env python3
"""Does the Fisher-scoring surrogate actually WORK as an iteration?

Everything before this was one block, one observation, solved by inspection.
This asks the three questions that decide whether the scheme can go into a fit:

  0. Is the saddlepoint SOLVE itself sound? (It was not: see below. Any
     conclusion about the iteration drawn with the old solver is a conclusion
     about the solver.)
  1. Does a SAFEGUARDED iteration converge from every start, to the right root?
  2. Is a Newton/Fisher HYBRID better than pure Fisher, or just more code?
  3. MULTI-BLOCK: 19 blocks sharing one parameter c, r_k = z_k - c. Does it
     converge, and does it land where a direct 1-D scan of the same objective
     says it should? The scan is built by interpolating the theta curve, so it
     shares no root-finding code with the iteration and cannot check itself.

usage: python cgf_scoring_test.py --model M.root [--part 0123]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_propagation_test import (load_model, model_variance, step_transports,
                                 FUNCTIONALS)
from cgf_phase0_validate import collect_ioni_steps
from cgf_saddlepoint import ioni_cgf_derivs, solve_saddlepoint
from cgf_irls import (make_block, solve_theta, block_eval, objective,
                      fisher_scoring, block_curve, block_mode, c_domain)


def collect_ioni_steps_per_leg(legs, k, avec, sigma):
    """collect_ioni_steps, but returned per leg instead of concatenated.

    These blocks are INDEPENDENT (disjoint sets of Geant4 steps), unlike the
    cumulative blocks used elsewhere in this study, so summing -ln p over them
    is a genuine likelihood rather than a repeated count of the same steps.
    All are transported to the same reference plane k, which is exactly how the
    real fit couples them: one shared track parameter, many independent noise
    blocks with different weights.
    """
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return out


def part0(blk18):
    print("=" * 100)
    print("0. THE SADDLEPOINT SOLVE. Old damped Newton vs the new bracketed "
          "solve, plane 18.")
    print("   K'(theta) is monotone, so a maintained bracket cannot fail; the "
          "old one had none.")
    print("=" * 100)
    st = blk18["steps"]
    print(f"   trusted domain: theta in [{blk18['th_lo']:.4e}, "
          f"{blk18['th_hi']:.4e}] -> r in [{blk18['r_lo']:.4g}, "
          f"{blk18['r_hi']:.6f}]")
    print(f"{'r':>12} | {'theta OLD':>13} {'K1-r OLD':>13} | "
          f"{'theta NEW':>13} {'K1-r NEW':>12}")
    print("-" * 100)
    for r in (0., 5., 11., 20., 24.0, -1., -10., -50., -300., -1e4, -1e6):
        to = solve_saddlepoint(st, r)
        if np.isfinite(to):
            _, k1, _ = ioni_cgf_derivs(st, to)
            eo = float(k1[0]) - r
            so = f"{to:13.5e} {eo:13.3e}"
        else:
            so = f"{'nan':>13} {'--':>13}"
        tn = solve_theta(blk18, r)
        if np.isfinite(tn):
            _, k1, _ = ioni_cgf_derivs(st, tn)
            en = float(k1[0]) - r
            sn = f"{tn:13.5e} {en:12.3e}"
        else:
            sn = f"{'nan':>13} {'out of dom':>12}"
        print(f"{r:12.1f} | {so} | {sn}")
    print("   (the old solver's error at r = -300 and r = -1e4 is the SAME "
          "theta = -0.397, i.e.\n    it had stopped responding to r at all; "
          "K'(that theta) = -1.7e96.)")


def part1(blk18):
    print()
    print("=" * 100)
    print("1. START DEPENDENCE, single block (plane 18). "
          "mu_{n+1} = mu_n + psi(z-mu_n)/I, safeguarded.")
    print("=" * 100)
    zs = (0.0, 5.0, 20.0, -50.0, -300.0)
    print(f"{'z':>8} {'mu0':>9} | {'raw: mu':>10} {'it':>5} {'ok':>3} | "
          f"{'safeguarded: mu':>16} {'it':>5} {'nfev':>6} {'ok':>3} | "
          f"{'scan mu*':>10} {'d':>10}")
    print("-" * 100)
    for z in zs:
        # ground truth for this observation: minimize -ln p(z - c) over c by
        # scanning the theta curve (no root finding, no iteration)
        rg, lpg = block_curve(blk18)
        m = np.isfinite(lpg)
        rg, lpg = rg[m], lpg[m]
        i = int(np.argmax(lpg))
        r_star = rg[i]
        if 0 < i < len(rg) - 1:
            y0, y1, y2 = lpg[i - 1], lpg[i], lpg[i + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                h = 0.5 * (rg[i + 1] - rg[i - 1])
                r_star = rg[i] - 0.5 * h * (y2 - y0) / den
        mu_star = z - r_star
        for mu0 in (0.0, z, z - 20.0, z + 20.0, 1e3):
            for b in (blk18,):
                b["nfev"] = 0
            raw = fisher_scoring([blk18], [z], mu0, mode="fisher", itmax=500,
                                 maxstep=np.inf, nback=1, expand=False)
            for b in (blk18,):
                b["nfev"] = 0
            saf = fisher_scoring([blk18], [z], mu0, mode="fisher", itmax=500)
            d = saf["c"] - mu_star
            print(f"{z:8.1f} {mu0:9.1f} | {raw['c']:10.4f} "
                  f"{raw['iters']:5d} {str(raw['converged'])[0]:>3} | "
                  f"{saf['c']:16.4f} {saf['iters']:5d} {saf['nfev']:6d} "
                  f"{str(saf['converged'])[0]:>3} | {mu_star:10.4f} {d:10.2e}")
    print("   raw = no step cap, no backtracking, no expansion (what was "
          "tested before).")
    print("   scan mu* = z - argmax(ln p) from the theta curve, independent of "
          "both.")


def part2(blk18):
    print()
    print("=" * 100)
    print("2. HYBRID (Newton where the curvature is positive, Fisher "
          "elsewhere) vs pure FISHER")
    print("=" * 100)
    print(f"{'z':>8} {'mu0':>9} | {'fisher it':>10} {'nfev':>7} {'mu':>10} | "
          f"{'hybrid it':>10} {'nfev':>7} {'mu':>10} | {'d(mu)':>10}")
    print("-" * 100)
    tf = th = 0.0
    nf = nh = 0
    for z in (0.0, 5.0, 20.0, -50.0, -300.0):
        for mu0 in (0.0, z, z + 20.0):
            blk18["nfev"] = 0
            t0 = time.time()
            a = fisher_scoring([blk18], [z], mu0, mode="fisher")
            tf += time.time() - t0
            na = blk18["nfev"]
            blk18["nfev"] = 0
            t0 = time.time()
            b = fisher_scoring([blk18], [z], mu0, mode="hybrid")
            th += time.time() - t0
            nb = blk18["nfev"]
            nf += a["iters"]
            nh += b["iters"]
            print(f"{z:8.1f} {mu0:9.1f} | {a['iters']:10d} {na:7d} "
                  f"{a['c']:10.4f} | {b['iters']:10d} {nb:7d} {b['c']:10.4f} | "
                  f"{b['c']-a['c']:10.2e}")
    print(f"   total iterations: fisher {nf}, hybrid {nh};  "
          f"wall {tf:.2f}s vs {th:.2f}s")


def pool_legs(per, nmin=8):
    """Merge consecutive legs until each block has at least nmin steps.

    NOT cosmetic. A leg with ONE Geant4 step has a loss distribution that is a
    near-delta at its mean loss with a 1/E^2 tail to -infinity: measured, its
    hard edge is at r = +4.3e-5 while its kappa2 is 1.1e-2 (sigma = 0.10, i.e.
    2400x the edge). Its ln p_SPA has NO interior maximum -- it rises
    monotonically into the edge -- so the joint minimum is pinned on the
    support boundary and no interior stationary point exists. Pooling is also
    what the real fit does: its blocks are per module / per global parameter
    index, never per Geant4 step.
    """
    groups, cur = [], []
    for s in per:
        cur.append(s)
        if sum(len(x) for x in cur) >= nmin:
            groups.append(np.concatenate(cur))
            cur = []
    if cur:
        if groups:
            groups[-1] = np.concatenate([groups[-1]] + cur)
        else:
            groups.append(np.concatenate(cur))
    return groups


def scan_ground_truth(blocks, curves, cz, npts=4001):
    """Minimum of F(c) by a MATCHED scan: the grid is the allowed c domain
    itself, not an arbitrary +-40 window.

    The first version of this used linspace(min z - 40, max z + 40, 4001),
    spacing 0.02, on a problem whose entire allowed domain was 4.3e-5 wide. It
    reported a 'minimum' that was simply its leftmost feasible grid point, and
    the iteration then beat it by 0.22 in F. Matched grids, again.
    """
    lo, hi = c_domain(blocks, cz)
    if not (hi > lo):
        return None
    span = hi - lo
    if not np.isfinite(span) or span > 1e6:
        hi = min(hi, lo + 4.0 * (max(cz) - min(cz) + 10.0))
        span = hi - lo

    def F(c):
        tot = 0.0
        for (r, lp), z in zip(curves, cz):
            rr = z - c
            if rr <= r[0] or rr >= r[-1]:
                return np.inf
            tot += -np.interp(rr, r, lp)
        return tot

    eps = 1e-9 * span
    cg = np.linspace(lo + eps, hi - eps, npts)
    Fg = np.array([F(c) for c in cg])
    fin = np.isfinite(Fg)
    i = int(np.nanargmin(np.where(fin, Fg, np.inf)))
    loc = [j for j in range(1, len(cg) - 1)
           if fin[j] and fin[j - 1] and fin[j + 1]
           and Fg[j] < Fg[j - 1] and Fg[j] < Fg[j + 1]]
    # refine twice, each time on the bracketing cells
    for _ in range(3):
        a = cg[max(i - 1, 0)]
        b = cg[min(i + 1, len(cg) - 1)]
        cg = np.linspace(a, b, 2001)
        Fg = np.array([F(c) for c in cg])
        i = int(np.nanargmin(np.where(np.isfinite(Fg), Fg, np.inf)))
    at_edge = (i <= 1) or (i >= len(cg) - 2)
    return dict(c=float(cg[i]), F=float(Fg[i]), lo=lo, hi=hi,
                nloc=len(loc), at_edge=at_edge, Ffun=F)


def part3(blocks, zsets, seedinfo, tag):
    print()
    print("=" * 104)
    print(f"3{tag}. MULTI-BLOCK: {len(blocks)} independent blocks sharing ONE "
          f"parameter c, r_k = z_k - c")
    print("=" * 104)
    Itot = sum(b["I"] for b in blocks)
    print(f"   {len(blocks)} blocks, "
          f"{sum(len(b['steps']) for b in blocks)} steps total; "
          f"1/I_tot = {1.0/Itot:.3e};  per-block 1/I in "
          f"[{min(1/b['I'] for b in blocks):.2e}, "
          f"{max(1/b['I'] for b in blocks):.2e}];  hard edges r_hi in "
          f"[{min(b['r_hi'] for b in blocks):.3e}, "
          f"{max(b['r_hi'] for b in blocks):.3e}]")
    curves = [block_curve(b) for b in blocks]

    for name, cz in zsets:
        print(f"\n   --- observations: {name}   ({seedinfo})")
        gt = scan_ground_truth(blocks, curves, cz)
        if gt is None:
            print("       EMPTY c domain -- no c makes every block finite. "
                  "SKIP")
            continue
        c_star, F_star = gt["c"], gt["F"]
        Fd, Gd, _, _, _ = objective(blocks, cz, c_star)
        print(f"       allowed c domain (hard support bound): "
              f"({gt['lo']:+.6g}, {gt['hi']:+.6g})")
        print(f"       scan: c* = {c_star:+.8g}  F* = {F_star:.6f}  "
              f"{gt['nloc']} interior local min;  "
              f"{'*** MINIMUM IS ON THE DOMAIN BOUNDARY ***' if gt['at_edge'] else 'interior minimum'}")
        print(f"       cross-check at c*: scan F {F_star:.6f} vs direct "
              f"block_eval {Fd:.6f} (d = {Fd-F_star:+.2e}), dF/dc = {Gd:+.4g}")
        print(f"{'':7} {'start c0':>12} | {'mode':>7} {'c':>14} "
              f"{'c - c*':>11} {'F - F*':>11} {'it':>5} {'nfev':>7} "
              f"{'reason':>12}")
        w = gt["hi"] - gt["lo"]
        starts = [c_star, c_star + 0.2 * w, c_star - 0.2 * w,
                  gt["lo"] + 1e-3 * w, c_star + 5.0, c_star - 5.0, 0.0]
        for c0 in starts:
            for mode in ("fisher", "hybrid"):
                for b in blocks:
                    b["nfev"] = 0
                res = fisher_scoring(blocks, cz, c0, mode=mode, itmax=500)
                nf = sum(b["nfev"] for b in blocks)
                dF = res["F"] - F_star if np.isfinite(res["F"]) else np.nan
                dc = res["c"] - c_star if np.isfinite(res["c"]) else np.nan
                print(f"{'':7} {c0:12.6g} | {mode:>7} {res['c']:14.8g} "
                      f"{dc:11.2e} {dF:11.2e} {res['iters']:5d} "
                      f"{nf:7d} {res['reason']:>12}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--func", default="qop")
    ap.add_argument("--part", default="0123")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    legs = load_model(args.model)
    avec = FUNCTIONALS[args.func]
    K = len(legs) - 1
    var, _, _ = model_variance(legs, K, avec)
    sig = float(np.sqrt(var))

    print(f"model {os.path.basename(args.model)}, {len(legs)} planes, "
          f"reference plane {K}, sigma {sig:.4e}")
    blk18 = make_block(collect_ioni_steps(legs, 18, avec, sig))
    print(f"plane-18 cumulative block: 1/I = {1.0/blk18['I']:.4f}, "
          f"r in [{blk18['r_lo']:.4g}, {blk18['r_hi']:.6f}]")

    if "0" in args.part:
        part0(blk18)
    if "1" in args.part:
        part1(blk18)
    if "2" in args.part:
        part2(blk18)
    if "3" in args.part:
        per = collect_ioni_steps_per_leg(legs, K, avec, sig)
        print(f"\nper-leg step counts: {[len(s) for s in per]}")

        # (a) the naive construction: one block per leg
        blocks = [make_block(s) for s in per]
        part3(blocks, [("all z_k = 0", np.zeros(len(blocks)))],
              f"seed {args.seed}", "a")

        # (b) pooled so that no block is a single Geant4 step
        pooled = pool_legs(per, nmin=8)
        print(f"\npooled step counts: {[len(s) for s in pooled]}")
        pb = [make_block(s) for s in pooled]
        n = len(pb)
        rng = np.random.default_rng(args.seed)
        modes = np.array([block_mode(b)[0] for b in pb])
        sds = np.array([1.0 / np.sqrt(b["I"]) for b in pb])
        print("   per-block modes: " +
              ", ".join(f"{m:+.3f}" for m in modes))
        zsets = [
            ("all z_k = 0", np.zeros(n)),
            ("z_k = mode_k (every block already at its own optimum, so the "
             "answer must be c = 0)", modes.copy()),
            ("z_k = mode_k + N(0, 1/sqrt(I_k)) -- a realistic draw",
             modes + rng.normal(0.0, sds, n)),
            ("as above but with a true common offset c = +0.5",
             modes + 0.5 + rng.normal(0.0, sds, n)),
            ("one block dragged 50 sigma into its tail",
             np.concatenate([[modes[0] - 50.0],
                             (modes + rng.normal(0.0, sds, n))[1:]])),
        ]
        part3(pb, zsets, f"seed {args.seed}", "b")


if __name__ == "__main__":
    main()

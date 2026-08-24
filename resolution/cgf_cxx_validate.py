#!/usr/bin/env python3
"""Reference dump + comparison for the C++ block-CGF evaluator.

The in-fit prototype (`TrackPropagation/Geant4e/src/CGFQoPBlock.cc`) is a
port of the IONIZATION channel of `cf_track_resolution.ioni_step_exponent`
plus the exact-inversion Fisher route of `cgf_fisher.fisher_exact`. This
script is the like-for-like reference for it:

  dump   -- write the per-step ionization records (transport weight folded
            into column 10, exactly as `collect_ioni_steps` does) plus the
            standardization sigma, for a set of blocks, in a plain text
            format the C++ test main reads.
  ref    -- compute S(t) on a fixed t list and 1/I by the VALIDATED python
            route for the same blocks, and write them next to the dump.
  cmp    -- read the C++ output and report the relative differences.

Blocks come in two flavours, and they are NOT the same object:
  * `--blocks cum`  : the CUMULATIVE block to plane k (all steps from the
    track start), which is what every published 1/I number in
    NOTES_XXII / NOTES_FISHERNORM / NOTES_EXPORTS refers to.
  * `--blocks leg`  : the PER-LEG block (one propagation, surface to
    surface), which is what the CVH fit actually uses as a process-noise
    block and therefore what the in-fit prototype computes.

Nothing in this directory is modified by this script.
"""
import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_brems_exact  # noqa: E402
import cf_propagation_test as cpt  # noqa: E402
import cgf_channels as cgc  # noqa: E402
import cgf_fisher  # noqa: E402
from cf_track_resolution import ioni_step_exponent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = ("/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/"
                 "model/model_mu_pt3_eta0.30.root")
# q/p functional
AVEC = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def leg_steps(legs, k, avec, sigma, only_last):
    """Ionization steps with the transport weight in column 10.

    only_last keeps just leg k's own steps (the per-leg process-noise block);
    otherwise every step from the track start to the end of leg k.
    """
    _, A_ioni, _ = cgc.step_transports(legs, k)
    out = []
    lo = k if only_last else 0
    for j in range(lo, k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


def block_sigma(legs, k, avec):
    var, _, _ = cpt.model_variance(legs, k, avec)
    return float(np.sqrt(var))


def leg_sigma(legs, k, avec):
    """Gaussian sigma of leg k's OWN noise, transported to the end of leg k."""
    Q = legs[k]["Q"]
    return float(np.sqrt(avec @ Q @ avec))


def leg_rad(legs, k, avec, sigma, only_last):
    """Radiative records with their transport weights, mirroring `leg_steps`.

    `cf_brems_exact.rad_exponent` takes the weight separately (it multiplies
    the record's own `cs`), so this returns (records, shapes, weights) rather
    than folding the weight into a column."""
    _, A_ioni, _ = cgc.step_transports(legs, k)
    recs, spec, wts = [], [], []
    lo = k if only_last else 0
    for j in range(lo, k + 1):
        leg = legs[j]
        if leg.get("rad") is None or not len(leg["rad"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        # The radiative log has one entry per STEP and the ionization log one
        # per step that produced a fluctuation record, so the two need not be
        # the same length. This is the convention cgf_channels uses, kept
        # deliberately so the C++ is compared against the SAME object the
        # offline closure numbers were produced with -- the in-fit path
        # attaches the exact per-step weight instead, which is strictly better
        # and would not be a like-for-like test.
        wr = w if len(w) == len(leg["rad"]) else np.full(len(leg["rad"]), np.mean(w))
        recs.append(np.asarray(leg["rad"], dtype=float))
        spec.append(np.asarray(leg["radspec"], dtype=float))
        wts.append(wr)
    if not recs:
        return None
    return np.concatenate(recs), np.concatenate(spec), np.concatenate(wts)


def python_ref(steps, tlist, rad=None, vgrid=None):
    """S(t) on tlist and 1/I by the validated exact-inversion route, for a
    step array whose column 10 already carries the transport weight."""
    def _S(tau):
        tau = np.asarray(tau, dtype=float)
        out = ioni_step_exponent(steps, 1.0, tau)
        if rad is not None:
            out = out + cf_brems_exact.rad_exponent(tau, rad[0], rad[1], vgrid,
                                                    weights=rad[2])
        return out

    S = _S(tlist)
    # t grid matched to THIS block: bisect Re S = lncut on a coarse log scan,
    # exactly as cgf_channels.tau_reach does, but on the step array directly
    # so no `legs` bookkeeping is involved.
    lo, hi, n, lncut = 1e-3, 1e6, 400, -60.0
    tg = np.geomspace(lo, hi, n)
    Sg = _S(tg).real
    below = np.where(Sg < lncut)[0]
    if not len(below):
        top = hi
    elif int(below[0]) == 0:
        top = lo
    else:
        i = int(below[0])
        a, b = tg[i - 1], tg[i]
        for _ in range(24):
            m = np.sqrt(a * b)
            if _S(np.array([m]))[0].real < lncut:
                b = m
            else:
                a = m
        top = float(np.sqrt(a * b))
    top *= 1.3
    tau = np.concatenate([[0.0], np.geomspace(1e-4, top, 8000)])
    Sfull = _S(tau)
    z, p, dp = cgc.invert_cf(Sfull, tau, npad=32, nt=1 << 17, deriv=True)
    invI, info = cgf_fisher.fisher_exact(z, p, dp)
    # EXACT score psi = -p'/p on the contiguous support, from the same pair of
    # transforms. Never the saddlepoint form (NOTES_EXPORTS 5.2), and never
    # obtained by differencing ln p.
    sl = cgf_fisher._support(p, 1e-8)
    zs, ps, dps = z[sl], p[sl], dp[sl]
    psi = -dps / ps
    # Sign changes ONLY where the density carries mass. Over the full support
    # psi is pure inversion noise in the deep tail and flips sign thousands of
    # times; an unrestricted count is meaningless (measured 4322 at plane 0).
    sig = ps > 1e-4 * np.nanmax(p)
    nzc = int(np.sum(np.diff(np.signbit(psi[sig])) != 0))
    zsig = (float(zs[sig][0]), float(zs[sig][-1])) if sig.any() else (0., 0.)
    def psi_at(r):
        if r <= zs[0] or r >= zs[-1]:
            return np.nan
        return float(np.interp(r, zs, psi))
    return (S, 1.0 / invI if invI else np.nan, top, info,
            dict(psi_at=psi_at, zlo=float(zs[0]), zhi=float(zs[-1]), nzc=nzc,
                 zmode=float(z[int(np.nanargmax(p))]), zsig=zsig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "cmp"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--planes", type=int, nargs="+", default=[0, 9, 18])
    ap.add_argument("--blocks", choices=["cum", "leg"], default="cum")
    ap.add_argument("--out", default="/tmp/cgf_ref")
    ap.add_argument("--cxx", default=None, help="C++ output file for cmp")
    ap.add_argument("--rad", action="store_true",
                    help="include the RADIATIVE (brems + pair) channel, on both "
                         "sides: the python reference adds "
                         "cf_brems_exact.rad_exponent and the dump grows a "
                         "RADGRID/RAD section for the C++ to read")
    args = ap.parse_args()

    legs = cpt.load_model(args.model)
    tlist = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    refs = {}
    for k in args.planes:
        sig = (leg_sigma(legs, k, AVEC) if args.blocks == "leg"
               else block_sigma(legs, k, AVEC))
        st = leg_steps(legs, k, AVEC, sig, only_last=(args.blocks == "leg"))
        if not len(st):
            logger.warning(f"plane {k}: no ionization steps, skipped")
            continue
        rad = leg_rad(legs, k, AVEC, sig, only_last=(args.blocks == "leg")) \
            if args.rad else None
        vg = np.asarray(legs[k]["radvgrid"], dtype=float) \
            if (args.rad and legs[k].get("radvgrid") is not None) else None
        if args.rad and (rad is None or vg is None):
            logger.warning(f"plane {k}: --rad asked for but the model carries no "
                           f"radiative records; the arm is IONIZATION ONLY")
            rad = None
        S, invI, top, info, sc = python_ref(st, tlist, rad=rad, vgrid=vg)
        refs[k] = dict(sigma=sig, nstep=len(st), S=S, invI=invI, top=top,
                       info=info, sc=sc, rad=rad, vg=vg)
        logger.info(f"plane {k:2d} [{args.blocks}]  nstep {len(st):4d}  "
                    f"sigma {sig:.6e}  t_max {top:.4f}  "
                    f"1/I(z) {invI:.6f}  1/I(phys) {invI*sig*sig:.6e}  "
                    f"mass {info.get('mass', float('nan')):.6f}  "
                    f"zmode {sc['zmode']:+.5f}  psi zeros {sc['nzc']}  "
                    f"support [{sc['zlo']:.3f},{sc['zhi']:.3f}]  "
                    f"mass region [{sc['zsig'][0]:.3f},{sc['zsig'][1]:.3f}]")

    if args.cmd == "dump":
        fn = f"{args.out}_{args.blocks}.txt"
        with open(fn, "w") as f:
            f.write(f"# nblocks {len(refs)}\n")
            f.write(f"# tlist {' '.join(repr(t) for t in tlist)}\n")
            for k, r in refs.items():
                st = leg_steps(legs, k, AVEC, r["sigma"],
                               only_last=(args.blocks == "leg"))
                f.write(f"BLOCK {k} {len(st)} {float(r['sigma'])!r} "
                        f"{float(r['invI'])!r} {float(r['top'])!r}\n")
                for row in st:
                    # regime gsig2 a1 e1 a2 e2 a3 e0r tmaxr scaling gs
                    f.write(" ".join(f"{float(v)!r}" for v in row) + "\n")
                rad, vg = r.get("rad"), r.get("vg")
                if rad is not None and vg is not None:
                    recs, spec, wts = rad
                    f.write(f"RADGRID {len(vg)} "
                            + " ".join(f"{float(x)!r}" for x in vg) + "\n")
                    f.write(f"RAD {len(recs)}\n")
                    nv = len(vg)
                    for rec, sp, w in zip(recs, spec, wts):
                        # etot [MeV], cs*w, dE_brem [MeV], dE_pair [MeV], then
                        # the two RAW shapes. Energies go out in MeV because
                        # that is cvhcgf's convention for every energy in a
                        # block; the records are natively GeV.
                        etot = float(rec[cf_brems_exact.R_ETOT]) * 1e3
                        L = float(rec[cf_brems_exact.R_STEPCM])
                        deb = float(rec[cf_brems_exact.R_DEDXBREM]) * L * 1e3
                        dep = float(rec[cf_brems_exact.R_DEDXPAIR]) * L * 1e3
                        csw = float(rec[cf_brems_exact.R_CS]) * float(w)
                        vals = [etot, csw, deb, dep]
                        vals += [float(x) for x in sp[:nv]]
                        vals += [float(x) for x in sp[nv:2 * nv]]
                        f.write(" ".join(f"{x!r}" for x in vals) + "\n")
                f.write("SREF\n")
                for t, s in zip(tlist, r["S"]):
                    f.write(f"{float(t)!r} {float(s.real)!r} {float(s.imag)!r}\n")
        logger.info(f"wrote {fn}")
        return

    # cmp
    if not args.cxx:
        logger.error("--cxx required for cmp")
        return 1
    cx = {}
    for line in open(args.cxx):
        p = line.split()
        if not p:
            continue
        if p[0] == "IVAL":
            cx.setdefault(int(p[1]), {})["invI"] = float(p[2])
        elif p[0] == "MVAL":
            d = cx.setdefault(int(p[1]), {})
            d["zmode"], d["nzc"] = float(p[2]), int(p[3])
        elif p[0] == "PVAL":
            cx.setdefault(int(p[1]), {}).setdefault("psi", []).append(
                (float(p[2]), float(p[3]), int(p[4])))
        elif p[0] == "SVAL":
            cx.setdefault(int(p[1]), {}).setdefault("S", []).append(
                (float(p[2]), float(p[3]), float(p[4])))
    print(f"{'plane':>5} {'t':>10} {'ReS py':>16} {'ReS c++':>16} {'rel':>10} "
          f"{'ImS py':>16} {'ImS c++':>16} {'rel':>10}")
    worst = 0.
    for k, r in refs.items():
        for (t, sre, sim), spy in zip(cx.get(k, {}).get("S", []), r["S"]):
            dr = abs(sre - spy.real) / max(abs(spy.real), 1e-300)
            di = abs(sim - spy.imag) / max(abs(spy.imag), 1e-300)
            worst = max(worst, dr if abs(spy.real) > 1e-14 else 0.,
                        di if abs(spy.imag) > 1e-14 else 0.)
            print(f"{k:>5} {t:>10.4g} {spy.real:>16.8e} {sre:>16.8e} "
                  f"{dr:>10.2e} {spy.imag:>16.8e} {sim:>16.8e} {di:>10.2e}")
    print()
    print(f"{'plane':>5} {'1/I python':>14} {'1/I c++':>14} {'rel':>10}")
    for k, r in refs.items():
        c = cx.get(k, {}).get("invI", float("nan"))
        print(f"{k:>5} {r['invI']:>14.8f} {c:>14.8f} "
              f"{abs(c - r['invI'])/r['invI']:>10.2e}")
    print(f"\nworst relative S difference: {worst:.2e}")

    print()
    print(f"{'plane':>5} {'zmode py':>12} {'zmode c++':>12} {'d':>10} "
          f"{'nzc py':>7} {'nzc c++':>8}")
    for k, r in refs.items():
        c = cx.get(k, {})
        print(f"{k:>5} {r['sc']['zmode']:>12.6f} {c.get('zmode', float('nan')):>12.6f} "
              f"{abs(c.get('zmode', float('nan')) - r['sc']['zmode']):>10.2e} "
              f"{r['sc']['nzc']:>7} {c.get('nzc', -1):>8}")

    print()
    print(f"{'plane':>5} {'r':>7} {'psi python':>15} {'psi c++':>15} "
          f"{'abs diff':>10} {'rel':>10} {'clamped':>8}")
    wpsi = 0.
    for k, r in refs.items():
        for (z, pc, cl) in cx.get(k, {}).get("psi", []):
            pp = r["sc"]["psi_at"](z)
            if np.isnan(pp):
                print(f"{k:>5} {z:>7.1f} {'outside support':>15} "
                      f"{pc:>15.7e} {'--':>10} {'--':>10} {cl:>8}")
                continue
            d = abs(pc - pp)
            rel = d / max(abs(pp), 1e-30)
            wpsi = max(wpsi, rel)
            print(f"{k:>5} {z:>7.1f} {pp:>15.7e} {pc:>15.7e} {d:>10.2e} "
                  f"{rel:>10.2e} {cl:>8}")
    print(f"\nworst relative psi difference (inside support): {wpsi:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

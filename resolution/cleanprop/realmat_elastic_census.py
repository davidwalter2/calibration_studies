"""The elastic arm of one ditrack leg, collision by collision: does the sim
scatter where and as often as the nuclear-elastic table says, and are the
tracks WITHOUT a large kick the clean arm's?

  kicks     per interval between consecutive planes, the fraction of tracks
            whose direction turns by more than a cut beyond the field's
            bending (elastic arm minus clean arm; the bending rotation about
            the field axis is the reference's, scaled by the track's own
            p_T), against the table: sum over the interval's rows of
            mu(p) x_g P(theta > cut), P from the joint (theta, dE) bins of the
            row's two momentum nodes, theta rescaled to fixed momentum
            transfer (`nucel_tables.Table`).  Outer intervals of a looping leg
            are contaminated by earlier kicks (the crossing geometry of a
            kicked track moves), so they bound the rate from above only.
  unkicked  the core of the tracks with no turn > 20 mrad up to the plane,
            elastic against clean arm: <e^{-u z^2}>, z in the clean arm's
            robust width per plane, ladder mean over planes 1..last with
            bootstrap errors.  A side effect of enabling hadElastic on
            tracks it does not scatter would show here; sub-threshold kicks
            widen the elastic arm (negative).

    python cleanprop/realmat_elastic_census.py [--xstat] <decay> <leg> kicks|unkicked
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ksclosure", "nucel"))

import realmat_ditrack as rd                          # noqa: E402
import realmat_closure as rc                          # noqa: E402
import cf_propagation_test as cpt                     # noqa: E402
import cf_nucel_exact as cne                          # noqa: E402
from toy_loader import plane_frames                   # noqa: E402

CUTS = (0.02, 0.05, 0.1)       # rad
FLAG = 0.02                    # rad, `unkicked`
US = (0.01, 0.1, 1.0, 3.0)
NBOOT = 100


def sample(leg, arm):
    rc.ARM = arm
    return rd.load_sim(leg)


def directions(sim, R):
    """Unit global direction and p_T per (event, plane) from the local slopes."""
    ld = np.stack([sim["dxdz"], sim["dydz"], np.ones_like(sim["dxdz"])], axis=-1)
    d = np.einsum("epi,pij->epj", ld, R)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d, sim["pabs"] * np.hypot(d[..., 0], d[..., 1])


def turns(d, pt, phi_ref, pt_ref, dz_ref):
    """Angle between the direction at plane k and the one at k-1 carried by
    the field's bending (the reference's rotation about z, scaled by
    pT_ref/pT) and the reference's change of polar direction."""
    out = np.full(d.shape[:2], np.nan)
    for k in range(1, d.shape[1]):
        ptm = 0.5 * (pt[:, k - 1] + pt[:, k])
        a = (phi_ref[k] - phi_ref[k - 1]) * 0.5 * (pt_ref[k - 1] + pt_ref[k]) / ptm
        c, s = np.cos(a), np.sin(a)
        p = d[:, k - 1]
        rot = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1],
                        p[:, 2] + dz_ref[k]], axis=1)
        rot /= np.linalg.norm(rot, axis=1, keepdims=True)
        out[:, k] = np.arccos(np.clip((d[:, k] * rot).sum(1), -1.0, 1.0))
    return out


def arm_turns(leg):
    ns = {}
    exec(open(rd.planes_path(leg)).read(), ns)
    _, R = plane_frames(ns["origin"], ns["normal"], ns["uaxis"])
    dirs = {arm: directions(sample(leg, arm), R) for arm in ("off", "elonly")}
    d0, pt0 = dirs["off"]
    ref = np.nanmean(d0, axis=0)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    phi_ref = np.arctan2(ref[:, 1], ref[:, 0])
    dz_ref = np.concatenate([[0.0], np.diff(ref[:, 2])])
    pt_ref = np.nanmedian(pt0, axis=0)
    return {arm: turns(d, pt, phi_ref, pt_ref, dz_ref) for arm, (d, pt) in dirs.items()}


def table_tail(leg):
    """(interval, cut) expected collisions with theta > cut, from the table."""
    legs = cpt.load_model(rd.model_path(leg))
    T = cne.table()
    isp = T.species_index(rd.LEGS[leg]["pdg"])
    out = np.zeros((len(legs), len(CUTS)))
    for k, lg in enumerate(legs):
        ms = np.asarray(lg["ms"])
        im = cne.leg_materials(T, lg)
        for s in range(len(ms)):
            if im[s] < 0 or ms[s, 2] <= 0.0:
                continue
            mx = T.mixture(isp, int(im[s]))
            p = float(ms[s, 3])
            n = float(T.rate(mx, p)) * ms[s, 2]
            j0, j1, f = T.nodes(isp, p)
            for j, w in ((int(j0[0]), 1.0 - float(f[0])), (int(j1[0]), float(f[0]))):
                th, _, W = mx["joint"][j]
                th = th * mx["nodes"][j] / p
                out[k] += n * w * np.array([W[th > c].sum() for c in CUTS])
    return out


def cmd_kicks(leg):
    K = arm_turns(leg)
    M = table_tail(leg)
    ne, no = len(K["elonly"]), len(K["off"])
    print(f"{rd.DECAY} {leg}: {ne} elastic / {no} clean tracks; per 1e3 tracks, "
          "sim (elastic - clean) / table / ratio")
    for k in range(1, M.shape[0]):
        cells = []
        for i, c in enumerate(CUTS):
            pe = np.nanmean(K["elonly"][:, k] > c)
            po = np.nanmean(K["off"][:, k] > c)
            err = np.sqrt(pe / ne + po / no)
            cells.append(f">{c * 1e3:3.0f} mrad {1e3 * (pe - po):7.3f} +- {1e3 * err:.3f} "
                         f"{1e3 * M[k, i]:7.3f} {(pe - po) / M[k, i]:5.2f}")
        print(f"  {k - 1:2d}->{k:2d}  " + "   ".join(cells))


def cmd_unkicked(leg):
    K = arm_turns(leg)
    S = {arm: sample(leg, arm) for arm in ("off", "elonly")}
    us = np.asarray(US)
    rng = np.random.default_rng(1)
    for func in ("qop", "dxdz", "dydz", "locx", "locy"):
        xo = S["off"][func]
        med = np.nanmedian(xo, axis=0)
        sig = 1.4826 * np.nanmedian(np.abs(xo - med), axis=0)
        per = {}
        for arm in S:
            z = (S[arm][func] - med) / sig
            kk = np.where(np.isfinite(K[arm]), K[arm], 0.0)
            ok = np.isfinite(z) & ~np.maximum.accumulate(kk > FLAG, axis=1)
            per[arm] = (np.where(ok[None], np.exp(-us[:, None, None] * z[None] ** 2), 0.0), ok)

        def ladder(arm, idx):
            v, ok = per[arm]
            return (v[:, idx, 1:].sum(1) / ok[idx, 1:].sum(0)[None]).mean(1)

        n = {arm: per[arm][1].shape[0] for arm in per}
        d0 = ladder("elonly", np.arange(n["elonly"])) - ladder("off", np.arange(n["off"]))
        bs = np.array([ladder("elonly", rng.integers(0, n["elonly"], n["elonly"]))
                       - ladder("off", rng.integers(0, n["off"], n["off"]))
                       for _ in range(NBOOT)])
        print(f"{func:5s} unkicked elastic - clean, ladder, 1e-3:  " + "  ".join(
            f"u={u:g} {1e3 * d0[i]:+.2f} +- {1e3 * bs[:, i].std():.2f}" for i, u in enumerate(us)),
            flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xstat", action="store_true", help="the extended sample")
    ap.add_argument("decay")
    ap.add_argument("leg")
    ap.add_argument("what", choices=("kicks", "unkicked"))
    a = ap.parse_args()
    rd.configure(a.decay)
    rc.MODEL_TAG = "_mat"
    rd.XSTAT = a.xstat
    (cmd_kicks if a.what == "kicks" else cmd_unkicked)(a.leg)


if __name__ == "__main__":
    main()

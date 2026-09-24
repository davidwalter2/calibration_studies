#!/usr/bin/env python3
"""Per-plane residual pulls of a charged PION on the REAL tracker, with and
without the nuclear-elastic channel -- the track-level check before the mass.

Clean-propagation ground truth (one fixed pi- shot many times through the real
CMS tracker, `cleanprop/run_campaign.sh`) against the model CF of
`cf_propagation_test.model_phi`, per sensor plane, in the BENDING (local x)
and NON-BENDING (local y) projections.  The a-vectors are LOCAL, pushed to the
curvilinear basis by the production Jacobian H (`curv2local.leg_H`, pion
mass), so the numerator (the SIM's local residual) and the model's width
describe the same variable on every plane, stereo modules included.

Per plane the report gives, data against model with the channel OFF and ON:
  * the tail fractions P(|z| > 3) and P(|z| > 5), the model's from the exact
    inversion of its CF: P(|z|>c) = 1 - (2/pi) int_0^inf Re phi(t) sin(ct)/t dt;
  * the bounded averages <exp(-u z^2)> at u = 1e-3, 1e-2 (tail) and 1 (core).
z is the residual over the propagator's own (truncated) sigma, as in
`cf_propagation_test.compare`.  Acceptance is `perplane` with the inelastic
veto 0.2 (hadronic INELASTIC is not modelled; elastic costs no momentum and is
not caught by it).  The residual acceptance loss is tail-first (a ray kicked
off a module is lost), so the data tails at the OUTER planes are a lower bound.

usage:
  python3 perplane_pulls.py --points pt3:<simglob>:<model> ... --out rows.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RES)
os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import cf_nucel_exact                                        # noqa: E402
import cf_propagation_test as cpt                            # noqa: E402
import curv2local as c2l                                     # noqa: E402

M_PI = 0.13957039
DRIVER = os.environ.get(
    'NUCEL_DRIVER',
    '/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/nucel/nucel_g4driver.sh')
CACHE = os.environ.get('NUCEL_KCACHE',
                       '/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/nucel/kcache')
# the offline channel's driver/cache live in a scratch that no longer exists;
# point them at this session's build (same source, same build script)
cf_nucel_exact._driver = lambda: DRIVER
cf_nucel_exact._cachedir = lambda: (os.makedirs(CACHE, exist_ok=True) or CACHE)
# numerics only (`cf_nucel_exact._NOT_PHYSICS`): 2e4 log-theta bins for the J0
# quadrature of the sampled kernel instead of 1e5 -- the residual phase error
# per bin, u theta dln(theta) ~ 1e-3 x u theta, matters only where g has already
# decayed to its MC floor; the J0 quadrature is what dominates the run time
cf_nucel_exact.NUCEL_NBIN = int(os.environ.get('NUCEL_NBIN', '20000'))

TAU = np.concatenate([[0.0], np.geomspace(1e-3, 60.0, 1200)])
PROBES = (1e-3, 1e-2, 1.0)
CUTS = (3.0, 5.0)


def tail_model(phi, c, tau=TAU):
    """P(|z| > c) from the CF: 1 - (2/pi) int_0^inf Re phi(t) sin(ct)/t dt."""
    f = np.empty(len(tau))
    f[0] = c * phi.real[0]
    f[1:] = phi.real[1:] * np.sin(c * tau[1:]) / tau[1:]
    return 1.0 - (2.0 / np.pi) * np.trapezoid(f, tau)


_W = {}


def _plane(k):
    legs, sim, label = _W["legs"], _W["sim"], _W["label"]
    rows = []
    H, _ = c2l.leg_H(legs, k, mass=M_PI)
    m = sim["valid"][:, k]
    r = float(np.nanmedian(sim["globr"][:, k]))
    for name, i, br, rb in (("locx", 3, "locx", "reflocx"),
                            ("locy", 4, "locy", "reflocy")):
        a = H.T @ np.eye(5)[i]
        var = cpt.model_variance(legs, k, a)[0]
        if var <= 0:
            continue
        sigma = np.sqrt(var)
        z = (sim[br][m, k] - legs[k][rb]) / sigma
        row = dict(label=label, layer=k, r=r, func=name, n=int(m.sum()),
                   std=float(z.std()),
                   rob=float(0.5 * np.diff(np.percentile(z, [15.865, 84.135]))[0]))
        for c in CUTS:
            row[f"d{c:g}"] = float(np.mean(np.abs(z) > c))
        for u in PROBES:
            row[f"fd{u:g}"] = float(np.mean(np.exp(-u * z * z)))
        for chan in (0, 1):
            cf_nucel_exact.NUCEL_CHANNEL = bool(chan)
            phi = cpt._model_phi_uncached(legs, k, a, sigma, TAU)
            tg = "on" if chan else "off"
            for c in CUTS:
                row[f"m{c:g}_{tg}"] = float(tail_model(phi, c))
            for u in PROBES:
                row[f"fm{u:g}_{tg}"] = cpt.weier_scalar(phi, u, TAU)
        rows.append(row)
        print(f"{label} {name} L{k:2d} r={r:6.1f} n={row['n']:6d} "
              f"P(|z|>3) data {100*row['d3']:.3f}% model off {100*row['m3_off']:.3f}% "
              f"on {100*row['m3_on']:.3f}% | P(|z|>5) {100*row['d5']:.3f}% "
              f"off {100*row['m5_off']:.3f}% on {100*row['m5_on']:.3f}%", flush=True)
    return rows


def run_point(label, simglob, model, veto, nproc):
    legs = cpt.load_model(model)
    c2l.attach_extras(legs, model)
    sim = cpt.load_sim(simglob, acceptance="perplane", veto_eloss=veto)
    assert len(legs) == len(sim["detid"])
    for k, leg in enumerate(legs):
        assert leg["detid"] == sim["detid"][k] and leg["ok"]
    # warm the kernels in the parent so the forked workers inherit them
    cf_nucel_exact.NUCEL_CHANNEL = True
    cf_nucel_exact.warm(legs)
    _W.update(legs=legs, sim=sim, label=label)
    from multiprocessing import Pool
    with Pool(min(nproc, len(legs))) as pool:
        out = pool.map(_plane, range(len(legs)))
    return [r for rr in out for r in rr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True, help="label:simglob:model")
    ap.add_argument("--veto", type=float, default=0.2)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(f"NUCEL_RECOIL={int(cf_nucel_exact.NUCEL_RECOIL)} MS_NSUB={cpt.MS_NSUB}")
    rows = []
    for p in args.points:
        label, sg, mdl = p.split(":")
        rows += run_point(label, sg, mdl, args.veto, args.nproc)
    keys = sorted({k for r in rows for k in r})
    out = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

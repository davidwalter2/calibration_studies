#!/usr/bin/env python3
"""What the two corrections actually DO, per candidate, on a given card.

The fluctuation form is a deterministic map `u_i(x) = sigma_i x + c_i x^2 + d_i`
of the resolution fluctuation, so its whole content is three per-candidate
numbers and the mean they imply,

    E[Delta_i] = Var(x) (c_i - a_i sigma_i) + d_i    (Var(x) = 1 in the model)

against which the residual form's `delta`-valued argument is measured. This
prints both, on any `make_card.py` / `make_joint_card.py` card, so the J/psi and
the Z can be compared on the same footing.

    ./run_tf.sh python3 corr_census.py --card cards/z_full380_fl.hdf5
"""
import argparse
import os

import numpy as np


def q(x, name, unit=1.0, fmt="{:+10.4f}"):
    x = np.asarray(x, dtype=np.float64) * unit
    qs = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
    print(f"  {name:<26s}" + "".join(fmt.format(v) for v in qs)
          + f"   mean {fmt.format(x.mean())}  |max| {fmt.format(np.abs(x).max())}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None, help="term name (default: each)")
    a = p.parse_args()

    import h5py
    from rabbit import unbinned

    with h5py.File(a.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    for t in terms:
        if a.term and t.name != a.term:
            continue
        cfg = t.config()
        sig = t.sigma.numpy()
        m = t.mobs.numpy() + t.m_ref
        print(f"\n=== {t.name}  ({t.n} candidates, m_ref {t.m_ref:.4f}, "
              f"corr_form {cfg.get('corr_form', 'residual')}, "
              f"jensen {cfg.get('jensen_mode')}, clip {cfg.get('corr_clip')}) ===")
        print(f"  {'quantiles [MeV]':<26s}" + "".join(
            f"{s:>10s}" for s in ("q01", "q25", "med", "q75", "q99")))
        q(sig, "sigma_i", 1e3)
        q(sig / m, "sigma_i/m_i", 1e3, "{:+10.4f}")
        if getattr(t, "a_res", None) is not None:
            q(t.a_res.numpy(), "a_i (dimensionless)", 1e3)
        if getattr(t, "_fluct_active", False):
            c = t._fl_g.numpy() * sig
            d = np.zeros(t.n) if t._fl_d is None else t._fl_d.numpy()
            asig = t._fl_a.numpy() * sig
            q(c, "c_i", 1e3)
            q(d, "d_i", 1e3)
            q(-asig, "-a_i sigma_i", 1e3)
            q(c - asig + d, "E[Delta_i] (the shift)", 1e3)
        # what the RESIDUAL form would have been fed, on the same candidates
        delta = t.mobs.numpy()
        q(delta, "delta_i = m_i - m_ref", 1e3)
        q(np.abs(delta) / sig, "|delta_i|/sigma_i", 1.0, "{:+10.3f}")
        if getattr(t, "jensen_s2", None) is not None:
            s2 = t.jensen_s2.numpy()
            r = delta / m
            disc = np.maximum(1.0 + 4.0 * (r - 0.5 * s2), 0.1)
            u = 0.5 * (np.sqrt(disc) - 1.0)
            q(np.abs(u * m - delta), "residual form MOVES delta by", 1e3)


if __name__ == "__main__":
    main()

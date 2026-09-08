#!/usr/bin/env python3
"""Is a stored fit result actually at its minimum, in the POI directions?

`|grad|inf` is not the answer. This objective's Hessian eigenvalues span 0.05
to 1e5 -- the soft eigenvalues are `1/sigma^2` for `m_Z` and `Gamma_Z`, the
stiff ones are the `K(m)` shape coefficients -- so a stopping rule on the
UNSCALED gradient infinity norm stops when the STIFFEST direction is converged
and says nothing about the softest. `V_full` stopped with `|grad|inf = 1.42`
(all of it `shape5`, converged to 0.003 sigma) while `m_Z` was 0.98 sigma and
`Gamma_Z` 2.06 sigma from their minimum.

What this reports instead is the per-parameter gradient at the stored point and
the diagonal Newton step it implies IN UNITS OF THAT PARAMETER'S OWN ERROR,
`g_i sigma_i` (since `H_ii ~ 1/sigma_i^2`). A fit is usable if that is small
for the parameters being quoted. It is a diagonal approximation and the true
step is larger where the POIs correlate with the shapes, so it is a LOWER bound
on the displacement -- which is the safe direction for a gate.

    RABBIT=... ./run_tf.sh python3 -u checkconv.py \\
        --card cards/z_F_dc8.hdf5 --result results/eng/fit_F_dc8.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--pois", nargs="*", default=["m_Z", "Gamma_Z"])
    ap.add_argument("--tol", type=float, default=0.1,
                    help="required |g_i sigma_i| for the POIs, in sigma")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    import h5py
    from rabbit import unbinned
    from chunkfit import ChunkedObjective

    with h5py.File(a.card, "r") as f:
        term = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
    names = list(term.param_names)
    free = [i for i, p in enumerate(names) if not p.startswith("k_")]
    obj = ChunkedObjective([term], free, chunk=a.chunk)
    fn = list(obj.freenames)

    d = json.load(open(a.result))
    x = np.array(obj.x0[free], dtype=np.float64, copy=True)
    x0 = x.copy()
    for q, v in zip(d["params"], d["fitted"]):
        if q in fn:
            x[fn.index(q)] = v
    err = dict(zip(d["params"], d.get("sandwich_err") or d["err"]))

    f_, g = obj.value_grad(x)
    print(f"[checkconv] {os.path.basename(a.card)} / "
          f"{os.path.basename(a.result)}")
    print(f"    n = {term.n}, vpow = {getattr(term, 'vpow', None)}, "
          f"stored |grad|inf = {d.get('gradmax'):.4g}, nit = {d.get('nit')}")
    print(f"    NLL {f_:.6f}")
    print(f"\n  {'param':10s} {'fitted':>13s} {'start':>10s} {'grad':>12s} "
          f"{'sigma':>9s} {'step [sigma]':>13s}")
    worst = 0.0
    out = {}
    for i, q in enumerate(fn):
        s = err.get(q, np.nan)
        step = g[i] * s
        out[q] = {"grad": float(g[i]), "sigma": float(s), "step": float(step)}
        if q in a.pois:
            worst = max(worst, abs(step))
        print(f"  {q:10s} {x[i]:+13.6f} {x0[i]:+10.4f} {g[i]:+12.5g} "
              f"{s:9.4f} {step:+13.4f}")
    ok = worst < a.tol
    print(f"\n  worst POI displacement {worst:.3f} sigma "
          f"(requirement < {a.tol:g}) -> {'CONVERGED' if ok else 'NOT CONVERGED'}")
    if a.output:
        json.dump({"card": a.card, "result": a.result, "params": out,
                   "worst_poi_sigma": float(worst), "converged": bool(ok)},
                  open(a.output, "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

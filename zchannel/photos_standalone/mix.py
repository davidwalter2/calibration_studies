#!/usr/bin/env python3
"""Combine standalone runs into one accumulation with per-band mixing weights.

Photos applies its exact Z matrix-element correction only when it finds two
opposite-sign fermion mothers for the decaying particle
(``HEPEVT_struct::check_ME_channel``, reached through
``PhotosParticle::findProductionMothers`` which walks up through Z self-copies).
In a POWHEG ``Zj`` + MiNNLO sample the underlying Born is Z + 1 parton, so the
gluon-initiated events - ``q g -> Z q`` - have a gluon among those mothers and
the correction does not fire, while ``q qbar -> Z g`` events get it.  The
sample's kernel is therefore a mixture of the two, with the weight fixed by the
initial-state composition of the sample and not by any fit.

    mix.py --run gen_mcB.npz:0.53 --run gen_pair.npz:0.47 -o gen_mcMix.npz

Each run is normalised inside every generated band before mixing, so the
weights are event fractions and the runs need not have equal statistics.
"""
import argparse
import os

import numpy as np

SUMS = ("n", "n_nophot", "n_noemit", "n_norad", "mpre_s1", "over", "nphot_s1",
        "npair", "nbad", "w2")
ARRS = ("mom", "fine_s0", "fine_s1", "fine_s2", "tail_s0", "tail_s1",
        "tail_s2", "logu_h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="path:weight (weights are renormalised to 1)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frac-file", default=None,
                    help="json with 'm' and 'f_me': the first run's weight as a "
                         "function of m_pre, linearly interpolated at the band "
                         "centre and held flat outside (two runs only)")
    a = ap.parse_args()

    specs = []
    for s in a.run:
        if ":" in s:
            p, _, w = s.rpartition(":")
            specs.append((p, float(w)))
        else:
            specs.append((s, 1.0))
    tot = sum(w for _, w in specs)
    specs = [(p, w / tot) for p, w in specs]

    fvec = None
    if a.frac_file:
        import json
        with open(a.frac_file) as fh:
            j = json.load(fh)
        if len(specs) != 2:
            raise SystemExit("--frac-file needs exactly two --run arguments")
        d0 = np.load(specs[0][0], allow_pickle=True)
        mc = 0.5 * (d0["bands_lo"] +
                    np.where(np.isfinite(d0["bands_hi"]),
                             d0["bands_hi"], d0["bands_lo"] + 2.0))
        fvec = np.interp(mc, j["m"], j["f_me"])

    out, ref = None, None
    for iz, (p, w) in enumerate(specs):
        d = np.load(p, allow_pickle=True)
        n = d["n"].astype(float)
        wv = (np.full(len(n), w) if fvec is None
              else (fvec if iz == 0 else 1.0 - fvec))
        scale = np.where(n > 0, wv / np.maximum(n, 1.0), 0.0)
        if out is None:
            out = {}
            ref = d
            for k in ("bands_lo", "bands_hi"):
                out[k] = d[k]
            for k in SUMS:
                if k in d.files:
                    out[k] = np.zeros_like(d[k], dtype=float)
            for k in ARRS:
                out[k] = np.zeros_like(d[k], dtype=float)
        for k in SUMS:
            if k in d.files and k in out:
                out[k] += scale * d[k]
        for k in ARRS:
            out[k] += scale[:, None] * d[k]
    # `n` is now 1 per populated band; keep the raw generated count so that
    # Poisson errors downstream reflect the statistics actually generated
    ngen = np.zeros_like(out["n"])
    for p, w in specs:
        d = np.load(p, allow_pickle=True)
        ngen += d["n"].astype(float)
    out["ngen"] = ngen
    out["meta"] = np.array(
        "mix: " + ", ".join(f"{p}:{w:.6f}" for p, w in specs)
        + (f" [per-band weights from {a.frac_file}]" if fvec is not None else ""))
    if fvec is not None:
        out["mix_frac"] = fvec
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez_compressed(a.output, **out)
    n = out["n"].sum()
    print(f"{a.output}: {len(specs)} runs, "
          + (", ".join(f"{os.path.basename(p)} x {w:.4f}" for p, w in specs)
             if fvec is None else
             f"{os.path.basename(specs[0][0])} x f_ME(m) "
             f"[{fvec.min():.4f}..{fvec.max():.4f}], "
             f"{os.path.basename(specs[1][0])} x 1-f_ME(m)"))
    print(f"  inclusive-over-bands <u> = {out['mom'][:,1].sum()/n:.6e}, "
          f"P(none) = {out['n_noemit'].sum()/n:.6f}, N_gen = {ngen.sum():.4e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H and J of the vertex / mass / joint CF term, in the layout
`hitlik/efficiency.py` reads (`H_<arm>_<channel>`, `J_<arm>_<channel>`,
`G_<arm>_<channel>`, `params`).

The terms are built exactly as `make_vtx_card.py` builds them, card units and
all, but everything WRITTEN here is in PHYSICAL units -- `k_g` for a material
group, `eps_c` for a hit class -- converted with the terms' own
`groups.term_units`.

`H` is the curvature the term CLAIMS; `J` is the score covariance, so the
SANDWICH `(H+P)^-1 J (H+P)^-1` is the variance the estimator actually has.
The two `hessian` / `score_cov` routines are imported from
`hitlik/fisher_cmp.py` -- the same code that produced the per-hit numbers.

For the JOINT channel the per-batch gradients of the two terms are SUMMED
batch by batch over the SAME candidates, so `J` carries the within-candidate
correlation between the vertex and the mass score.  That is the only way the
over-counting can show up (a `sandwich/quoted` above 1 on the joint that
neither term shows alone).

usage:
  ./run_tf.sh python3 fisher_vtx.py --vtx-npz v.npz --mass-npz m.npz \
      --groups .../materialGroups50.txt --channels vtx mass joint \
      --arms cf gauss gaussq --maxn 8000 -o runs/vtxres/fisherHJ.npz
"""
import argparse, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres"), os.path.join(_RES, "hitlik"),
           os.path.join(_RES, "globalfit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fisher_cmp as FC        # noqa: E402  (hessian, score_cov)
import selection              # noqa: E402
import make_vtx_card as MVC    # noqa: E402
import groups as G             # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class _A:
    """The subset of make_vtx_card's argparse namespace `build_term` uses."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtx-npz", default=None)
    p.add_argument("--mass-npz", default=None)
    p.add_argument("--bsx-npz", default=None)
    p.add_argument("--bsy-npz", default=None)
    p.add_argument("--groups", required=True)
    p.add_argument("--beam3", action="store_true",
                   help="float the luminous region as the full 3x3 covariance "
                        "(make_vtx_card.py --beam3)")
    p.add_argument("--channels", nargs="+", default=["vtx", "mass", "joint"])
    p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
    p.add_argument("--maxn", type=int, default=8000)
    # `--max-chi2-ndof` is the standard selection's (`_sel.add_args`, below)
    p.add_argument("--prune-frac", type=float, default=1e-3)
    p.add_argument("--no-hits", action="store_true")
    p.add_argument("--no-beamwidth", action="store_true")
    p.add_argument("--nbatch", type=int, default=200)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--no-hessian", action="store_true")
    p.add_argument("--max-abs-z", type=float, default=40.0)
    p.add_argument("--vtx-norm-window", type=float, default=None)
    p.add_argument("--norm-classes", type=int, default=8)
    p.add_argument("--norm-tpoints", type=int, default=2048)
    p.add_argument("--alpha", action="store_true")
    import selection as _sel
    _sel.add_args(p)
    p.add_argument("--m-ref", type=float, default=None,
                   help="the mass channel's reference mass (GeV); 91.1876 for Z")
    p.add_argument("--m-window", type=float, default=0.5)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()

    import tensorflow as tf
    tf.config.optimizer.set_jit(False)

    if a.m_ref is not None:
        # the card builder's module-level reference mass, so a Z sample's mass
        # channel is built the same way here as in `make_vtx_card`
        MVC.MREF[0], MVC.MREF[1] = float(a.m_ref), float(a.m_window)
    gmap, _ = G.read_groups(a.groups)
    ngroups = (max(gmap) + 1) if gmap else 0
    gparams, _ = G.group_param_names(ngroups, a.groups)
    group_units = G.card_group_units(ngroups, a.groups)   # as the cards
    import hitres_classes
    hparams = [f"hitres_{c}" for c in hitres_classes.CLASSES]

    # EVERY attribute `make_vtx_card.build_term` reads has to be here: the
    # namespace is a stub, so a new option in the card builder shows up as an
    # AttributeError in the middle of a Fisher run rather than at parse time.
    # `vtx_norm_window` / `norm_classes` / `norm_tpoints` are the truncated
    # normalisation's; `alpha` the floor's.
    ba = _A(maxn=a.maxn, prune_frac=a.prune_frac,
            amount_mode="exp", hit_mode="linear", no_hits=a.no_hits,
            floor="softplus", chunk=a.chunk, vtx_window=0.0,
            max_abs_z=a.max_abs_z,
            vtx_norm_window=a.vtx_norm_window, norm_classes=a.norm_classes,
            norm_tpoints=a.norm_tpoints, alpha=a.alpha,
            no_standard_selection=a.no_standard_selection,
            max_abs_vtxz=a.max_abs_vtxz, min_leg_hits=a.min_leg_hits,
            max_chi2_ndof=a.max_chi2_ndof,
            # the two LUMINOUS-REGION WIDTH scales float here too, so the
            # sandwich is computed over the SAME parameter vector the cards fit
            no_beamwidth=a.no_beamwidth,
            # `--beam3` replaces those two classes by the FULL 3x3 covariance
            # plus the two centre offsets; the sandwich has to be able to see
            # the new parameters or it is measuring a different model from the
            # one that was fitted
            beam3=a.beam3, mass_corrections=False, corr_form="residual",
            m_ref=a.m_ref, m_window=a.m_window, keep_mask=None, inject=[])

    res, params = {}, None
    idx = None
    for ch in a.channels:
        # A channel is a SET of terms fitted over ONE parameter vector on the
        # SAME candidates: that is what makes the joint `J` carry the
        # within-candidate correlation, and it is the only place over-counting
        # can show up.  The two BEAM-LINE residuals join as two more terms of
        # the vertex kind.
        _T = {"vtx": ("vtx",), "mass": ("mass",), "bsx": ("bsx",),
              "bsy": ("bsy",), "bs": ("bsx", "bsy"),
              "joint": ("vtx", "mass"),
              "vtxbs": ("vtx", "bsx", "bsy"),
              "bsmass": ("bsx", "bsy", "mass"),
              "vtxbsmass": ("vtx", "bsx", "bsy", "mass")}
        _NPZ = {"vtx": a.vtx_npz, "mass": a.mass_npz,
                "bsx": a.bsx_npz, "bsy": a.bsy_npz}
        if ch not in _T:
            sys.exit(f"unknown channel '{ch}'; known: {sorted(_T)}")
        need = [(nm, _NPZ[nm]) for nm in _T[ch]]
        if any(x[1] is None for x in need):
            log(f"skip channel {ch}: missing npz")
            continue
        for arm in a.arms:
            terms = []
            lidx = idx
            for nm, npz in need:
                t, _, lidx = MVC.build_term(nm, npz, arm, ba, group_units,
                                            gparams, hparams, ngroups, {}, {},
                                            idx=lidx)
                terms.append(t)
            if idx is None:
                idx = lidx
            names = terms[0].param_names
            for t in terms[1:]:
                if list(t.param_names) != list(names):
                    sys.exit("the two terms do not share a parameter vector")
            if params is None:
                params = list(names)
            npar = len(names)
            x0 = np.zeros(npar)
            # ---- H -------------------------------------------------------
            H = np.zeros((npar, npar))
            if not a.no_hessian:
                t0 = time.time()
                for t in terms:
                    Hi, v, g = FC.hessian(t, x0)
                    H += Hi
                log(f"{ch}/{arm}: H in {time.time()-t0:.0f} s, "
                    f"diag {np.diag(H).min():.3g} .. {np.diag(H).max():.3g}")
            # ---- J, batch means over the SAME candidate batches ----------
            Gsum = None
            gtot = np.zeros(npar)
            for t in terms:
                _, gt, M, Gm = FC.score_cov(t, x0, nbatch=a.nbatch)
                Gsum = Gm if Gsum is None else Gsum + Gm
                gtot = gtot + gt
            M = Gsum.shape[0]
            gb = Gsum.mean(axis=0)
            D = Gsum - gb
            J = (M / (M - 1.0)) * (D.T @ D)
            log(f"{ch}/{arm}: J from {M} batches, "
                f"|sum_m g_m - g| = {np.abs(Gsum.sum(axis=0)-gtot).max():.3g}")
            u = G.term_units(terms[0])
            res["param_units"] = u
            res[f"H_{arm}_{ch}"] = G.matrix_to_physical(H, u)
            res[f"J_{arm}_{ch}"] = G.matrix_to_physical(J, u)
            res[f"G_{arm}_{ch}"] = np.asarray(
                [G.gradient_to_physical(gb, u) for gb in Gsum], np.float64)
            res[f"n_{ch}"] = terms[0].n
    res["params"] = np.array(params)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez_compressed(a.output, **res)
    log(f"-> {a.output}")


if __name__ == "__main__":
    main()

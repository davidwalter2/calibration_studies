#!/usr/bin/env python3
"""PHYSICS dictionary vs data-driven PCA, at equal number of scalars.

The exponents are not arbitrary functions of t.  Per Geant4 step:

  ionization / radiative -- a compound Poisson, so the exponent is a LEVY
      integral over a POSITIVE measure nu_i (`cf_track_resolution.
      ioni_step_exponent`, `cf_ioni_exact.block_exponent`, `cf_brems_exact`):
          S(t) = Int ( e^{i v t} - 1 - i v t ) nu_i(dv)
      -- one measure gives BOTH the real and the imaginary part.
  multiple scattering -- the same structure with Moliere's universal shape
      (`cf_ms_exact.block_exponent`):
          S(t) = Int G(b t ; ymax) mu_i(db),   G = `cf_ms_exact.gshape`.

So a k-node quadrature of that measure is a candidate representation with k
scalars per candidate and a FIXED, model-derived, candidate-independent table
-- no SVD over the data, and nothing to re-derive when the sample changes.

Implementation of "k quadrature nodes": a fine dictionary of LOG-BINNED
profiles (each column is the bin integral of the universal kernel with dv/v
weight, which stays bounded where a discrete node oscillates), from which k
columns are chosen GREEDILY to minimise the population residual on a training
split -- i.e. the best k-node quadrature for this population, not an arbitrary
log-spaced grid.  Errors are quoted on a disjoint test split.

Also reported: `poly`, the cumulant/Taylor baseline (S ~ sum_p kappa_p (it)^p),
and `pca`, the empirical optimum on the same block.

Usage
  python3 physbasis.py --sub <sub_*.npz> --tag jpsigun --out phys_jpsigun.npz --phik
"""
import argparse, time
import numpy as np
import cfbasis as CB

KLIST = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]


def errs(R, W, nt, nfam):
    aR = np.abs(R)
    ea = aR.max(axis=1)
    ew = np.zeros(len(R))
    for j in range(nfam):
        ew = np.maximum(ew, (aR[:, j * nt:(j + 1) * nt] * W).max(axis=1))
    return ea, ew


# ------------------------------------------------------ fine dictionaries --
def levy_bins(TG, lo, hi, nb, nq=64, signed=True):
    """Column j = Int_{bin j} (e^{ivt} - 1 - ivt) dv/v, [Re ; Im] stacked."""
    nt = len(TG)
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.logspace(lo, hi, nb + 1)
    cols = []
    for a, b in zip(e[:-1], e[1:]):
        lv = 0.5 * (np.log(b) - np.log(a)) * x + 0.5 * (np.log(b) + np.log(a))
        v = np.exp(lv)
        wt = w * 0.5 * (np.log(b) - np.log(a))
        th = np.outer(TG, v)
        cols.append(np.concatenate([(np.cos(th) - 1.) @ wt,
                                    (np.sin(th) - th) @ wt]))
    D = np.stack(cols, axis=1)
    if signed:   # a pair carries one leg of each charge -> both signs of v
        D = np.concatenate([D, np.concatenate([D[:nt], -D[nt:]], axis=0)],
                           axis=1)
    return D


def moliere_bins(TG, lo, hi, nb, nq=32, ymax=1e4):
    """Column j = Int_{bin j} G(b t ; ymax) db/b."""
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.logspace(lo, hi, nb + 1)
    cols = []
    for a, b in zip(e[:-1], e[1:]):
        lb = 0.5 * (np.log(b) - np.log(a)) * x + 0.5 * (np.log(b) + np.log(a))
        bb = np.exp(lb)
        wt = w * 0.5 * (np.log(b) - np.log(a))
        cols.append(CB.gshape(np.outer(TG, bb).ravel(), ymax=ymax)
                    .reshape(len(TG), len(bb)) @ wt)
    return np.stack(cols, axis=1)


def greedy_columns(D, Y, k):
    """Pick k columns of D minimising ||Y - Y P_sel||_F, via the m x m Gram.

    gain(c) = c~^T (Y^T Y) c~ / (c~^T c~) with c~ the column orthogonalised
    against the ones already picked."""
    C = Y.T @ Y
    Dn = D / np.maximum(np.linalg.norm(D, axis=0), 1e-300)
    sel, Q = [], np.zeros((D.shape[0], 0))
    for _ in range(k):
        Dt = Dn - Q @ (Q.T @ Dn) if Q.shape[1] else Dn
        nn = np.linalg.norm(Dt, axis=0)
        ok = nn > 1e-8
        g = np.full(D.shape[1], -np.inf)
        g[ok] = np.einsum("ij,ij->j", Dt[:, ok], C @ Dt[:, ok]) / nn[ok] ** 2
        g[sel] = -np.inf
        j = int(np.argmax(g))
        sel.append(j)
        q = Dt[:, j] / nn[j]
        Q = np.concatenate([Q, q[:, None]], axis=1)
        Dn = Dn.copy()
    return sel


def ls_resid(Y, D):
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    keep = s > 1e-12 * s[0]
    P = (Vt[keep].T / s[keep]) @ U[:, keep].T
    return Y - (Y @ P.T) @ D.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntrain", type=int, default=20000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--nsel", type=int, default=4000)
    ap.add_argument("--phik", action="store_true")
    a = ap.parse_args()

    import spectral as SP
    d = np.load(a.sub)
    TG = d["tgrid"].astype(np.float64)
    nt = len(TG)
    ntr, nte = a.ntrain, a.ntest
    sl_tr, sl_te = slice(0, ntr), slice(ntr, ntr + nte)
    fams = ["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"]
    M = {f: d[f][: ntr + nte].astype(np.float64) for f in fams}
    vgf = d["vgf"][: ntr + nte].astype(np.float64)
    sig = d["sigma"][: ntr + nte].astype(np.float64)
    Sre = -0.5 * vgf[:, None] * TG[None, :] ** 2
    for f in ("Sms", "Sio_re", "Srad_re"):
        Sre = Sre + M[f]
    W = np.exp(Sre)
    if a.phik:
        W = W * SP.phik_abs(a.tag, TG, sig)
    Wte = W[sl_te]
    res = {"tag": a.tag, "TG": TG, "klist": np.array(KLIST)}

    print("building the fine physics dictionaries ...", flush=True)
    DL = levy_bins(TG, -4, 2, 96)               # 192 columns (both signs)
    DM = moliere_bins(TG, -1, 3, 128)           # 128 columns
    print(f"  levy fine {DL.shape}  moliere fine {DM.shape}", flush=True)

    blocks = [("IO", ["Sio_re", "Sio_im"], DL),
              ("RAD", ["Srad_re", "Srad_im"], DL),
              ("MS", ["Sms"], DM)]
    for bname, bfams, Dfine in blocks:
        Y = np.concatenate([M[f] for f in bfams], axis=1)
        Ytr, Yte = Y[sl_tr], Y[sl_te]
        nfam = len(bfams)
        print(f"== block {bname} (fine dict K={Dfine.shape[1]}) ==", flush=True)
        # fine-dictionary floor (all columns): how well the PHYSICS FORM itself
        # reproduces the cache
        R = ls_resid(Yte, Dfine)
        ea, ew = errs(R, Wte, nt, nfam)
        res[f"{bname}/fine/ewgt_max"] = ew.max()
        res[f"{bname}/fine/ewgt_q999"] = np.quantile(ew, .999)
        res[f"{bname}/fine/eabs_max"] = ea.max()
        res[f"{bname}/fine/K"] = Dfine.shape[1]
        print(f"  FULL fine dictionary K={Dfine.shape[1]}: "
              f"ewgt max={ew.max():.2e} q999={np.quantile(ew,.999):.2e} "
              f"eabs max={ea.max():.2e}", flush=True)
        for meth in ("phys", "poly", "pca"):
            for tag in ("max", "q999", "med"):
                res[f"{bname}/{meth}/ewgt_{tag}"] = np.full(len(KLIST), np.nan)
                res[f"{bname}/{meth}/eabs_{tag}"] = np.full(len(KLIST), np.nan)
        mu, V, sv = CB.pca_fit(Ytr, max(KLIST))
        sel_all = greedy_columns(Dfine, Ytr[: a.nsel], max(KLIST))
        res[f"{bname}/phys/sel"] = np.array(sel_all)
        for ik, k in enumerate(KLIST):
            # PCA
            R = (Yte - mu) - ((Yte - mu) @ V[:, :k]) @ V[:, :k].T
            ea, ew = errs(R, Wte, nt, nfam)
            for tag, fn in (("max", np.max),
                            ("q999", lambda x: np.quantile(x, .999)),
                            ("med", np.median)):
                res[f"{bname}/pca/eabs_{tag}"][ik] = fn(ea)
                res[f"{bname}/pca/ewgt_{tag}"][ik] = fn(ew)
            # physics dictionary, k greedily chosen quadrature nodes
            R = ls_resid(Yte, Dfine[:, sel_all[:k]])
            ea, ew = errs(R, Wte, nt, nfam)
            for tag, fn in (("max", np.max),
                            ("q999", lambda x: np.quantile(x, .999)),
                            ("med", np.median)):
                res[f"{bname}/phys/eabs_{tag}"][ik] = fn(ea)
                res[f"{bname}/phys/ewgt_{tag}"][ik] = fn(ew)
            # cumulant / Taylor baseline
            if nfam == 2:
                kre = (k + 1) // 2
                kim = k - kre
                D = np.zeros((2 * nt, kre + kim))
                D[:nt, :kre] = CB.poly_design(TG, kre, 0)
                for j in range(kim):
                    D[nt:, kre + j] = TG ** (2 * j + 3)
            else:
                D = CB.poly_design(TG, k, 0)
            R = ls_resid(Yte, D)
            ea, ew = errs(R, Wte, nt, nfam)
            for tag, fn in (("max", np.max),
                            ("q999", lambda x: np.quantile(x, .999)),
                            ("med", np.median)):
                res[f"{bname}/poly/eabs_{tag}"][ik] = fn(ea)
                res[f"{bname}/poly/ewgt_{tag}"][ik] = fn(ew)
            print(f"  k={k:3d}  ewgt q999  phys={res[f'{bname}/phys/ewgt_q999'][ik]:.2e}"
                  f"  poly={res[f'{bname}/poly/ewgt_q999'][ik]:.2e}"
                  f"  pca={res[f'{bname}/pca/ewgt_q999'][ik]:.2e}", flush=True)
    np.savez(a.out, **{k: np.asarray(v) for k, v in res.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()

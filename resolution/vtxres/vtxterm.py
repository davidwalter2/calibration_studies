#!/usr/bin/env python3
"""The vertex / mass CF term offline: load an `extract_vtx.py` npz, build the
per-row exponents of an arm, and evaluate the predicted density.

Same three arms as `hitlik_term.py`:
  cf      the exported log-CF exponents
  gauss   -1/2 kappa2 tau^2 with kappa2 read off the SAME arrays
  gaussq  the variance the FIT used (`Gvqms`, `Gvqio`), nothing radiative
"""
import numpy as np

FAMS = ("ms", "io_re", "io_im", "rad_re", "rad_im")
ARMS = ("cf", "gauss", "gaussq")


def kappa2_from_grid(S, tgrid):
    t1 = float(tgrid[1])
    return -(16.0 * S[:, 1] - S[:, 2]) / (6.0 * t1 * t1)


def load(npz, maxn=None, max_chi2_ndof=3.0, keys=None):
    d = np.load(npz, allow_pickle=False)
    n_all = len(d["sigma"])
    keep = np.ones(n_all, bool)
    if max_chi2_ndof and max_chi2_ndof > 0:
        keep &= d["chi2ndof"] < max_chi2_ndof
    idx = np.flatnonzero(keep)
    if maxn and maxn < len(idx):
        idx = idx[:maxn]
    n = len(idx)
    ptr = d["grp_ptr"].astype(np.int64)
    cnt = np.diff(ptr)[idx]
    rows = (np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in idx])
            if n else np.zeros(0, np.int64))
    out = {"n": n, "idx": idx, "tgrid": np.asarray(d["tgrid"], np.float64),
           "grp_seg": np.repeat(np.arange(n), cnt),
           "grp_id": d["grp_id"][rows].astype(np.int64),
           "grp_ptr": np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64),
           "group_names": [str(x) for x in d["group_names"]],
           "hit_classes": [str(x) for x in d["hit_classes"]],
           "functional": str(d["functional"])}
    for f in FAMS:
        out["S" + f] = d["S" + f][rows].astype(np.float64)
    for k in ("Gvqms", "Gvqio"):
        if k in d.files:
            out[k] = d[k][rows].astype(np.float64)
    hptr = d["hit_ptr"].astype(np.int64)
    hcnt = np.diff(hptr)[idx]
    hrows = (np.concatenate([np.arange(hptr[i], hptr[i + 1]) for i in idx])
             if n else np.zeros(0, np.int64))
    out["hit_seg"] = np.repeat(np.arange(n), hcnt)
    out["hit_cls"] = d["hit_cls"][hrows].astype(np.int64)
    out["hit_v"] = d["hit_v"][hrows].astype(np.float64)
    for k in d.files:
        if k in ("tgrid", "grp_ptr", "grp_id", "hit_ptr", "hit_cls", "hit_v",
                 "families", "group_names", "hit_classes", "functional",
                 "amount_convention", "Gvqms", "Gvqio") or k.startswith("S"):
            continue
        v = d[k]
        if v.ndim == 1 and len(v) == n_all:
            out[k] = v[idx]
    mref = 0.0 if out["functional"] == "vtx" else 3.0969
    out["z"] = (out["m0"] - mref) / out["sigma"]
    return out


def arm_arrays(sel, arm):
    """Per-GROUP-ROW exponents of one arm, as a dict family -> (nnz, nt)."""
    tg = sel["tgrid"]
    nt = len(tg)
    nnz = len(sel["grp_id"])
    if arm == "cf":
        return {f: sel["S" + f] for f in FAMS}
    if arm == "gauss":
        t2 = -0.5 * tg ** 2
        out = {}
        for f in FAMS:
            out[f] = (np.zeros((nnz, nt)) if f.endswith("_im")
                      else np.outer(kappa2_from_grid(sel["S" + f], tg), t2))
        return out
    if arm == "gaussq":
        t2 = -0.5 * tg ** 2
        out = {f: np.zeros((nnz, nt)) for f in FAMS}
        out["ms"] = np.outer(sel["Gvqms"], t2)
        out["io_re"] = np.outer(sel["Gvqio"], t2)
        return out
    raise ValueError(arm)


def row_exponents(sel, arm):
    """(Sre, Sim) per CANDIDATE, including the Gaussian -1/2 vgf tau^2."""
    tg = sel["tgrid"]
    n, nt = sel["n"], len(tg)
    a = arm_arrays(sel, arm)
    Sre = np.zeros((n, nt))
    Sim = np.zeros((n, nt))
    seg = sel["grp_seg"]
    for f in FAMS:
        (Sim if f.endswith("_im") else Sre)
        if f.endswith("_im"):
            np.add.at(Sim, seg, a[f])
        else:
            np.add.at(Sre, seg, a[f])
    Sre = Sre - 0.5 * np.outer(sel["vgf"], tg ** 2)
    return Sre, Sim


def upsample_matrix(tg, k):
    tf = np.linspace(tg[0], tg[-1], (len(tg) - 1) * k + 1)
    U = np.zeros((len(tf), len(tg)))
    for j in range(len(tf)):
        i = min(int(np.searchsorted(tg, tf[j], "right") - 1), len(tg) - 2)
        w = (tf[j] - tg[i]) / (tg[i + 1] - tg[i])
        U[j, i] = 1 - w
        U[j, i + 1] = w
    return tf, U


def mean_density(sel, arm, rows, zgrid, upsample=8, chunk=4000):
    """Row-averaged predicted density of z over `rows`, by the same inverse
    Fourier transform the term itself uses."""
    tg = sel["tgrid"]
    tf, U = upsample_matrix(tg, upsample)
    Sre, Sim = row_exponents(sel, arm)
    rows = np.asarray(rows)
    Sre = Sre[rows] @ U.T
    Sim = Sim[rows] @ U.T
    E = np.exp(Sre)
    cre, cim = E * np.cos(Sim), E * np.sin(Sim)
    w = np.gradient(tf)
    w[0] *= 0.5
    w[-1] *= 0.5
    C = np.cos(np.outer(tf, zgrid))
    S = np.sin(np.outer(tf, zgrid))
    dens = np.zeros(len(zgrid))
    for i0 in range(0, len(rows), chunk):
        sl = slice(i0, i0 + chunk)
        dens += ((cre[sl] * w) @ C + (cim[sl] * w) @ S).sum(axis=0)
    return dens / (np.pi * len(rows))


def model_moments(sel, arm, rows=None):
    """Model variance and skew of z from the exponent's derivatives at 0."""
    tg = sel["tgrid"]
    Sre, Sim = row_exponents(sel, arm)
    if rows is not None:
        Sre, Sim = Sre[rows], Sim[rows]
    var = kappa2_from_grid(Sre, tg)
    # kappa3 from the imaginary part: Sim = -kappa3 tau^3/6 + ...
    t1 = float(tg[1])
    k3 = -(32.0 * Sim[:, 1] - Sim[:, 2]) / (4.0 * t1 ** 3)  # tau^5 eliminated
    return var, k3

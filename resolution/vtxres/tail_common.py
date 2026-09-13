"""Shared pieces of the TAIL study: the published baseline, and small stats.

The baseline is `extract_vtx.py`'s `keep` mask reproduced EXACTLY, so that
every number quoted here sits on the same 10 254 candidates the beam-line
study published (`STATE.md` section 14.12) and not on a superset.
"""
import numpy as np

MAX_CHI2_NDOF = 3.0     # extract_vtx.py --max-chi2-ndof
MAX_VCHK = 1e-4         # extract_vtx.py --max-vchk
MIN_LEG_HITS = 8        # resolution/selection.py
MAX_ABS_VTXZ = 5.0


def baseline(d, need_bs=True, chi2=MAX_CHI2_NDOF, vchk=MAX_VCHK,
             min_leg_hits=MIN_LEG_HITS, max_abs_vtxz=MAX_ABS_VTXZ,
             verbose=True):
    """`extract_vtx.py`'s selection, on a `tail_extract.py` npz."""
    n = len(d["run"])
    ndof = np.asarray(d["ndof"], float)
    chi2n = np.where(ndof > 0, np.asarray(d["chisq"], float) /
                     np.maximum(ndof, 1), 1e9)
    steps = []
    k = np.isfinite(d["vtxsig"]) & (d["vtxsig"] > 0)
    steps.append(("finite sigma_v > 0", k.copy()))
    if need_bs and "bsok" in d:
        k &= d["bsok"].astype(bool); steps.append(("Jpsi_bsok", k.copy()))
    if "vtxok" in d:
        k &= d["vtxok"].astype(bool); steps.append(("Jpsi_vtxok", k.copy()))
    if "cfmass_ok" in d:
        k &= d["cfmass_ok"].astype(bool); steps.append(("cfmass_ok", k.copy()))
    k &= np.asarray(d["sigmamass"], float) > 0
    steps.append(("sigma_m > 0", k.copy()))
    if chi2 > 0:
        k &= chi2n < chi2; steps.append((f"chi2/ndof < {chi2:g}", k.copy()))
    if vchk > 0 and "vtxvchk" in d:
        k &= np.abs(np.asarray(d["vtxvchk"], float)) < vchk
        steps.append((f"|vtxvchk| < {vchk:g}", k.copy()))
    nl = np.minimum(d["nvalid_plus"], d["nvalid_minus"])
    if min_leg_hits > 0:
        k &= nl >= min_leg_hits
        steps.append((f"weaker leg >= {min_leg_hits}", k.copy()))
    if max_abs_vtxz > 0:
        k &= np.abs(np.asarray(d["vtxz"], float)) < max_abs_vtxz
        steps.append((f"|z_v| < {max_abs_vtxz:g}", k.copy()))
    if verbose:
        print(f"# baseline flow on {n} candidates")
        for nm, m in steps:
            print(f"#   {nm:<24s} {int(m.sum()):7d}")
    return k, chi2n, nl


def binom(kk, nn):
    if nn <= 0:
        return float("nan"), float("nan")
    p = kk / nn
    return p, float(np.sqrt(max(p * (1 - p), 0.0) / nn))


def pm(kk, nn):
    p, e = binom(kk, nn)
    return f"{p:.5f} +- {e:.5f} ({kk}/{nn})"


def moments(x, name, fmt=True):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m, v = x.mean(), x.var()
    s = ((x - m) ** 3).mean() / v ** 1.5
    ku = ((x - m) ** 4).mean() / v ** 2
    if fmt:
        print(f"  {name:<26s} n={n:6d}  mean {m:+8.4f} +- {np.sqrt(v/n):.4f}"
              f"  Var {v:9.4f}  skew {s:+9.3f}  kurt {ku:9.2f}"
              f"  P>3 {(np.abs(x)>3).mean():.5f}  P>4 {(np.abs(x)>4).mean():.5f}"
              f"  P>5 {(np.abs(x)>5).mean():.5f}")
    return dict(n=n, mean=m, var=v, skew=s, kurt=ku)


def robust_line(z, v):
    """Median-of-slopes free straight-line fit: offset and slope, robust to
    the displaced-vertex tail that makes a least-squares fit meaningless."""
    ok = np.isfinite(z) & np.isfinite(v)
    z, v = z[ok], v[ok]
    # iteratively reweighted least squares with a Huber-like 3-MAD clip
    A = np.vstack([np.ones(len(z)), z]).T
    c = np.array([np.median(v), 0.0])
    for _ in range(20):
        r = v - A @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = (np.abs(r) < 3 * s).astype(float)
        if w.sum() < 10:
            break
        Aw = A * w[:, None]
        c = np.linalg.solve(Aw.T @ A, Aw.T @ v)
    r = v - A @ c
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    w = np.abs(r) < 3 * s
    cov = np.linalg.inv(A[w].T @ A[w]) * (r[w] @ r[w]) / max(w.sum() - 2, 1)
    return c, np.sqrt(np.diag(cov)), int(w.sum())

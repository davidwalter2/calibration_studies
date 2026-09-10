#!/usr/bin/env python3
"""Build the residual-vector CF likelihood term from an ``extract_res5`` npz.

ONE function, :func:`build`, returns a ready ``rabbit.unbinned.MaterialCFTerm``
plus the datasets a ``tensorwriter`` card needs, in three ARMS that differ only
in what replaces the per-(row, group) log-CF exponent:

``cf``
    the exponents as extracted -- the full non-Gaussian densities.
``gauss``
    every family replaced by its own variance-matched Gaussian,
    ``S_g(tau) -> -1/2 kappa2_g tau^2``, ``kappa2_g`` read off the SAME arrays
    at small tau with the tau^4 term eliminated,
    ``kappa2 = -(16 S(t1) - S(2 t1)) / (6 t1^2)``.
    The imaginary (skew) parts are dropped, as a Gaussian has none.
    This arm has the SAME variance as ``cf``, so the comparison is purely
    about SHAPE.
``gaussq``
    every family replaced by the variance the FIT ITSELF used -- the
    Q-matrix (``thp2``) convention for MS and ``ioni_sq2`` for ionization,
    with no radiative and no delta variance (the fit's ``Q`` has neither).
    By construction ``sum_g kappa2_g + vgf == 1`` exactly for every row, so
    this arm IS the fit's own pull model, i.e. the Gaussian chi2.
    It is NARROWER than ``gauss``: ``Q``'s MS is Rossi's core scattering power
    and understates the modelled (full Moliere second moment) MS variance by
    14 % (NOTES 2026-08-16).

Both Gaussian arms keep the per-class Gaussian hit share untouched -- the hit
term is Gaussian in every arm, and it is the material families that differ.

The observable is the whitened residual component ``z`` with ``sigma = 1``,
``m_ref = 0``, a delta kernel, no background and no ``D`` rows: field and
alignment are NOT parameters of this term (they stay in the quadratic one).
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_MAT = os.path.join(_RES, "matres")
for _p in (_HERE, _RES, _MAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import groups as G  # noqa: E402

FAMS = ("ms", "del", "io_re", "io_im", "rad_re", "rad_im")
ARMS = ("cf", "gauss", "gaussq")


def kappa2_from_grid(S, tgrid):
    """-S''(0) from the tabulated exponent, with the tau^4 term eliminated.

    ``S(tau) = -kappa2 tau^2 / 2 + c tau^4 + O(tau^6)``; using ``t1`` and
    ``2 t1`` (grid points 1 and 2 of a uniform grid),
    ``16 S(t1) - S(2 t1) = -6 kappa2 t1^2``.
    """
    t1 = float(tgrid[1])
    return -(16.0 * S[:, 1] - S[:, 2]) / (6.0 * t1 * t1)


def load(npz, max_tracks=0, comps=None, max_chi2_ndof=0.0, max_inflat=0.0,
         track_offset=0):
    """Read the extraction and select rows.

    Returns a dict of the selected per-row arrays with the CSR blocks
    renumbered, plus ``ntrk`` and the row -> track map.
    """
    d = np.load(npz, allow_pickle=True)
    keys = set(d.files)
    ncomp = int(np.max(d["comp"])) + 1
    ntrk_all = len(d["eta"])
    tkeep = np.ones(ntrk_all, bool)
    if max_chi2_ndof > 0.0:
        tkeep &= d["chi2ndof"] < max_chi2_ndof
    if max_inflat > 0.0 and "inflat" in keys:
        # a guard on the CONDITIONING of the whitening, i.e. on the fit's own
        # covariance -- NOT on the residual being measured.
        cs_ = list(range(ncomp)) if comps is None else sorted(set(comps))
        tkeep &= np.asarray(d["inflat"])[:, cs_].max(axis=1) < max_inflat
    tidx = np.where(tkeep)[0]
    if track_offset:
        tidx = tidx[int(track_offset):]
    if max_tracks and max_tracks < len(tidx):
        tidx = tidx[:max_tracks]
    cset = list(range(ncomp)) if comps is None else sorted(set(comps))
    rows = np.concatenate([tidx * ncomp + k for k in cset]).reshape(len(cset), -1)
    rows = rows.T.ravel()                      # track-major, component-minor
    out = {"ntrk": len(tidx), "ncomp_used": len(cset), "comps": cset,
           "ncomp_file": ncomp, "tidx": tidx, "rows": rows}
    for k in ("z", "sigma", "vgf", "vg_other", "comp", "trk"):
        out[k] = np.asarray(d[k])[rows]
    for k in ("eta", "phi", "charge", "chi2ndof", "nvhit", "trackPt",
              "genPt", "covdev", "sigqop", "xcum", "rres", "inflat"):
        if k in keys:
            out[k] = np.asarray(d[k])[tidx]
    out["tgrid"] = np.asarray(d["tgrid"], np.float64)
    out["hit_classes"] = [str(s) for s in d["hit_classes"]]
    out["group_names"] = ([str(s) for s in d["group_names"]]
                          if "group_names" in keys else None)
    out["groups_file"] = str(d["groups_file"]) if "groups_file" in keys else ""
    out["provenance"] = str(d["provenance"]) if "provenance" in keys else ""

    gp = np.asarray(d["grp_ptr"], np.int64)
    cnt = np.diff(gp)[rows]
    grows = (np.concatenate([np.arange(gp[i], gp[i + 1]) for i in rows])
             if len(rows) else np.zeros(0, np.int64))
    out["grp_ptr"] = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    out["grp_id"] = np.asarray(d["grp_id"], np.int64)[grows]
    out["grp_seg"] = np.repeat(np.arange(len(rows)), cnt)
    for f in FAMS:
        out["S" + f] = np.asarray(d["S" + f])[grows]
    out["vQms"] = np.asarray(d["vQms"])[grows]
    out["vQio"] = np.asarray(d["vQio"])[grows]

    hp = np.asarray(d["hit_ptr"], np.int64)
    hcnt = np.diff(hp)[rows]
    hrows = (np.concatenate([np.arange(hp[i], hp[i + 1]) for i in rows])
             if len(rows) else np.zeros(0, np.int64))
    out["hit_ptr"] = np.concatenate([[0], np.cumsum(hcnt)]).astype(np.int64)
    out["hit_cls"] = np.asarray(d["hit_cls"], np.int64)[hrows]
    out["hit_v"] = np.asarray(d["hit_v"], np.float64)[hrows]
    return out


def arm_families(sel, arm, prune_frac=0.0, freeze_dead=True, ngroups=None,
                 inject=None, amount_mode="exp"):
    """The MaterialCFTerm ``group_families`` for one arm, with pruning.

    ``inject`` is ``{group_index: k}``: the DATA side is given ``exp(k)`` times
    that group's exponents, so a fit that recovers ``k`` recovers the injected
    material.  It is applied to whichever arm is being built, so the CF and
    Gaussian recoveries are comparable.
    """
    tg = sel["tgrid"]
    nt = len(tg)
    n = len(sel["z"])
    gid = sel["grp_id"]
    seg = sel["grp_seg"]
    ng = int(gid.max()) + 1 if len(gid) else 0
    if ngroups is not None:
        ng = max(ng, ngroups)

    arrs = {}
    if arm == "cf":
        for f in FAMS:
            arrs[f] = sel["S" + f].astype(np.float64)
    elif arm == "gauss":
        t2 = -0.5 * tg ** 2
        for f in FAMS:
            if f.endswith("_im"):
                arrs[f] = np.zeros((len(gid), nt))
                continue
            k2 = kappa2_from_grid(sel["S" + f].astype(np.float64), tg)
            arrs[f] = np.outer(k2, t2)
    elif arm == "gaussq":
        t2 = -0.5 * tg ** 2
        for f in FAMS:
            if f == "ms":
                arrs[f] = np.outer(sel["vQms"].astype(np.float64), t2)
            elif f == "io_re":
                arrs[f] = np.outer(sel["vQio"].astype(np.float64), t2)
            else:
                arrs[f] = np.zeros((len(gid), nt))
    else:
        raise ValueError(arm)

    # amplitude for pruning
    amp = np.zeros(len(gid))
    for f in FAMS:
        amp = np.maximum(amp, np.abs(arrs[f]).max(axis=1))
    top = np.zeros(n)
    if len(gid):
        np.maximum.at(top, seg, amp)
    drop = np.zeros(len(gid), bool)
    if prune_frac > 0.0:
        drop = amp < prune_frac * top[seg]
    if freeze_dead and len(gid):
        live = np.zeros(ng, bool)
        live[np.unique(gid[~drop])] = True
        dead = np.where(~live)[0]
        if len(dead):
            drop = drop | np.isin(gid, dead)

    # injection on the data side
    if inject:
        w = np.ones(ng)
        for g, k in inject.items():
            w[g] = np.exp(k) if amount_mode == "exp" else 1.0 + k
        for f in FAMS:
            arrs[f] = arrs[f] * w[gid][:, None]

    fam_out, fix = [], {}
    for f in FAMS:
        nm = f[:-3] if f.endswith(("_re", "_im")) else f
        comp = "im" if f.endswith("_im") else "re"
        fx = np.zeros((n, nt))
        if drop.any():
            np.add.at(fx, seg[drop], arrs[f][drop])
        e = fix.setdefault(nm, {"name": nm})
        e[comp] = arrs[f][~drop].astype(np.float32)
        if np.any(fx):
            e["fix_" + comp] = fx.astype(np.float32)
    fam_out = [fix[k] for k in sorted(fix)]

    kcnt = np.zeros(n, np.int64)
    if len(gid):
        np.add.at(kcnt, seg[~drop], 1)
    ptr = np.concatenate([[0], np.cumsum(kcnt)]).astype(np.int64)
    return fam_out, ptr, gid[~drop], int(drop.sum()), ng, amp, drop


def build(sel, arm="cf", prune_frac=0.0, groups_file=None, ngroups=None,
          inject=None, hit_inject=None, amount_mode="exp", hit_mode="linear",
          chunk=4096, no_hits=False, floor="softplus", name="hitres"):
    """(term, data, meta).  ``inject``/``hit_inject`` are {index: value}."""
    from rabbit import unbinned

    fam, ptr, gid, ndrop, ng, amp, drop = arm_families(
        sel, arm, prune_frac=prune_frac, ngroups=ngroups, inject=inject,
        amount_mode=amount_mode)
    gparams, gpriors = G.group_param_names(ng, groups_file)
    hclasses = sel["hit_classes"]
    hparams = [f"hitres_{c}" for c in hclasses]

    n = len(sel["z"])
    hit_v = sel["hit_v"].copy()
    if hit_inject:
        for c, e in hit_inject.items():
            m = sel["hit_cls"] == c
            hit_v[m] *= (np.exp(e) if hit_mode == "exp" else 1.0 + e)
    # vgf must track the injected hit variance, or the model is inconsistent
    vgf = sel["vgf"].astype(np.float64).copy()
    if hit_inject:
        add = np.zeros(n)
        hseg = np.repeat(np.arange(n), np.diff(sel["hit_ptr"]))
        np.add.at(add, hseg, hit_v - sel["hit_v"])
        vgf = vgf + add
    vg_other = sel["vg_other"].astype(np.float64)

    if no_hits:
        hparams = []
        share = (np.zeros(n + 1, np.int64), np.zeros(0, np.int64),
                 np.zeros(0), vgf)
    else:
        share = (sel["hit_ptr"], sel["hit_cls"], hit_v, vg_other)

    data = {
        "sigma": sel["sigma"].astype(np.float64),
        "mobs": sel["z"].astype(np.float64),
        "tgrid": sel["tgrid"],
        "grp_ptr": ptr,
        "grp_id": gid,
        "group_units": np.ones(ng),
        "hit_ptr": share[0], "hit_cls": share[1], "hit_v": share[2],
        "vg_other": share[3], "vgf": vgf,
        "hit_units": np.ones(len(hparams)),
    }
    for m in fam:
        for c in ("re", "im"):
            if c in m:
                data[f"Sg_{c}_{m['name']}"] = m[c]
            if "fix_" + c in m:
                data[f"Sgfix_{c}_{m['name']}"] = m["fix_" + c]

    term = unbinned.MaterialCFTerm(
        name,
        sigma=data["sigma"], mobs=data["mobs"], tgrid=data["tgrid"],
        families=[], vgf=vgf,
        group_params=gparams,
        group_families=[{k: v for k, v in m.items()
                         if k in ("name", "re", "im", "fix_re", "fix_im")}
                        for m in fam],
        grp_ptr=ptr, grp_id=gid, group_units=np.ones(ng),
        hit_params=hparams, hit_share=share,
        amount_mode=amount_mode, hit_mode=hit_mode,
        kernel=unbinned.DeltaKernel(),
        background=None, m_ref=0.0, scale_param=None, bkg_frac=0.0,
        floor=floor, chunk=chunk, channel="hitres",
    )
    meta = dict(arm=arm, ngroups=ng, ndrop=int(ndrop), nrows=len(gid),
                group_params=gparams, group_priors=gpriors,
                hit_params=hparams, n=n, amp=amp, drop=drop)
    return term, data, meta

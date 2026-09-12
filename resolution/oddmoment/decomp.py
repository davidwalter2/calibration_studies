#!/usr/bin/env python3
"""CHARGE DECOMPOSITION of the single-track q/p odd moment.

THE OPEN ITEM.  The odd moment <z e^{-uz^2}> of
z = (q/p_fit - q/p_gen)/sigma_pred is 6-10x the CF model at pT 20-60 and ~2.6x
at pT 2-20, and its ABSOLUTE size does not fall with momentum the way any
energy-loss channel does.

THE DECOMPOSITION.  Write any odd statistic O measured separately on the two
charges as

    A = (O(q+) - O(q-)) / 2      "odd-in-q"   : O flips sign with the charge
    S = (O(q+) + O(q-)) / 2      "even-in-q"  : O has the same sign for both

and note what each one IS in terms of the underlying track:

  * z = q * d(1/p) / sigma.  A bias of the MAGNITUDE |q/p| (energy loss, a
    field-scale error, anything that makes the fit think the track is stiffer
    or softer than it is) enters z with a factor q -> it lives ENTIRELY in A.
    The CF energy-loss model is of this type: its A is the whole model and its
    S is zero by construction.
  * A curvature OFFSET d(q/p) = const independent of the charge (a sagitta
    bias: misalignment, a hit-position shift coherent in the bending plane,
    a transverse field component) enters z with the SAME sign for both charges
    -> it lives ENTIRELY in S.

So A is the "momentum scale" channel and S is the "sagitta bias" channel, and
they are exactly orthogonal on a sample with both charges.

STATISTICS reported per charge and as A/S:
  mean, median, trimmed mean <z> over |z|<T (T = 1,2,3,5,10), the bounded odd
  moment <z e^{-uz^2}> on a u grid, the tail fractions P(z>+T)/P(z<-T) and the
  tail means.  Bootstrap errors throughout; the A/S errors follow from the two
  charge arms being independent samples.

The model side is the same CF model as every published closure
(cf_track_resolution.model_phi, hitmode=gauss, k = 0, krad from --krad),
evaluated per track, averaged into the pooled characteristic function of each
charge and inverted to a density so that the SAME trimmed estimators can be
run on it.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module("cft", os.path.join(_RES, "cf_track_resolution.py"))
TG = m.TG

PROBES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
TRIMS = (1.0, 2.0, 3.0, 5.0, 10.0)

NEED = ("z", "sigma", "eta", "phi", "charge", "genpt", "trackpt",
        "normchi2", "nvalidhits", "vgf", "ndof", "chisqval")
NEED_BIG = ("Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im")


class Args:
    khit = 0.0
    kms = 0.0
    kioni = 0.0
    krad = 1.0
    kdel = None
    hitmode = "gauss"


def load(cache, need_model=True, max_tracks=0):
    d = np.load(cache)
    out = {k: np.asarray(d[k]) for k in NEED if k in d.files}
    if need_model:
        for k in NEED_BIG:
            if k in d.files:
                out[k] = np.asarray(d[k])
    if "ioni_charge_signed" not in d.files:
        raise SystemExit(f"{cache} predates the charge-sign fix (2026-09-03)")
    n = len(out["z"])
    if max_tracks and max_tracks < n:
        rng = np.random.default_rng(1234)
        idx = np.sort(rng.choice(n, max_tracks, replace=False))
        out = {k: v[idx] for k, v in out.items()}
    return out


# ------------------------------------------------------------------ weights
def odd_weight_matrix(probes, tg=TG):
    """W with <z e^{-uz^2}> = (Im phi) @ W[:, iu]  (cf_skew_closure.odd_weights)."""
    tw = np.gradient(tg)
    tw[0] *= 0.5
    tw[-1] *= 0.5
    W = np.stack([(tg / (2. * u)) * np.exp(-tg ** 2 / (4. * u))
                  / np.sqrt(np.pi * u) for u in probes], axis=1)
    return W * tw[:, None]


def even_weight_matrix(probes, tg=TG):
    tw = np.gradient(tg)
    tw[0] *= 0.5
    tw[-1] *= 0.5
    W = np.stack([np.exp(-tg ** 2 / (4. * u)) / np.sqrt(np.pi * u)
                  for u in probes], axis=1)
    return W * tw[:, None]


def model_per_track(d, args, probes, chunk=20000, pool_id=None, npool=0):
    """Per-track model <z e^{-uz^2}> and <e^{-uz^2}>; pooled phi per group."""
    n = len(d["z"])
    Wo = odd_weight_matrix(probes)
    We = even_weight_matrix(probes)
    Om = np.empty((n, len(probes)))
    Em = np.empty((n, len(probes)))
    phis = np.zeros((max(npool, 1), len(TG)), dtype=np.complex128)
    cnt = np.zeros(max(npool, 1))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sub = {k: d[k][lo:hi] for k in NEED_BIG if k in d}
        sub["vgf"] = d["vgf"][lo:hi]
        phi = m.model_phi(sub, args)
        Om[lo:hi] = phi.imag @ Wo
        Em[lo:hi] = phi.real @ We
        if npool:
            b = pool_id[lo:hi]
            for k in range(npool):
                msk = b == k
                if msk.any():
                    phis[k] += phi[msk].sum(axis=0)
                    cnt[k] += msk.sum()
    if npool:
        phis /= np.maximum(cnt, 1)[:, None]
    return Om, Em, phis, cnt


# ---------------------------------------------------------------- estimators
def stats_of(z, trims=TRIMS, tailT=(3., 5., 10.)):
    """Dict of scalar statistics of a sample of z."""
    o = {}
    o["mean"] = float(z.mean())
    o["median"] = float(np.median(z))
    for T in trims:
        s = z[np.abs(z) < T]
        o[f"trim{T:g}"] = float(s.mean()) if len(s) else np.nan
    for T in tailT:
        o[f"fpos{T:g}"] = float((z > T).mean())
        o[f"fneg{T:g}"] = float((z < -T).mean())
        o[f"asym{T:g}"] = o[f"fpos{T:g}"] - o[f"fneg{T:g}"]
        sp, sn = z[z > T], z[z < -T]
        o[f"mpos{T:g}"] = float(sp.mean()) if len(sp) else np.nan
        o[f"mneg{T:g}"] = float(sn.mean()) if len(sn) else np.nan
        # the tail's contribution to the mean
        o[f"cont{T:g}"] = float(sp.sum() + sn.sum()) / len(z)
    return o


def odd_of(z, probes=PROBES):
    return np.array([float((z * np.exp(-u * z ** 2)).mean()) for u in probes])


def boot_stats(z, fn, nboot, rng):
    n = len(z)
    keys = None
    acc = []
    for _ in range(nboot):
        s = z[rng.integers(0, n, n)]
        v = fn(s)
        if isinstance(v, dict):
            if keys is None:
                keys = list(v)
            acc.append([v[k] for k in keys])
        else:
            acc.append(np.asarray(v))
    acc = np.asarray(acc)
    sd = acc.std(axis=0, ddof=1)
    return (dict(zip(keys, sd)) if keys is not None else sd)


# ------------------------------------------------------------ model density
def density(phibar, zg, tg=TG):
    w = np.gradient(tg)
    w[0] *= 0.5
    w[-1] *= 0.5
    ph = tg[None, :] * zg[:, None]
    return ((np.cos(ph) * phibar.real[None, :]
             + np.sin(ph) * phibar.imag[None, :]) @ w) / np.pi


def density_stats(phibar, trims=TRIMS, zmax=12., nz=4801):
    zg = np.linspace(-zmax, zmax, nz)
    p = density(phibar, zg)
    dz = zg[1] - zg[0]
    o = {}
    o["mean"] = float(np.trapezoid(zg * p, zg))
    c = np.cumsum(p) * dz
    c /= c[-1]
    o["median"] = float(np.interp(0.5, c, zg))
    for T in trims:
        s = np.abs(zg) < T
        num = np.trapezoid(zg[s] * p[s], zg[s])
        den = np.trapezoid(p[s], zg[s])
        o[f"trim{T:g}"] = float(num / den) if den > 0 else np.nan
    return o, zg, p


# ------------------------------------------------------------------- report
def AS(vp, vm):
    """(odd-in-q, even-in-q) of a per-charge pair."""
    return 0.5 * (vp - vm), 0.5 * (vp + vm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()

    args = Args()
    args.krad = a.krad
    d = load(a.cache, need_model=not a.no_model, max_tracks=a.max_tracks)
    z = d["z"]
    q = np.sign(d["charge"]).astype(np.int64)
    n = len(z)
    ip, im = q > 0, q < 0
    rng = np.random.default_rng(20260905)

    lab = a.label or os.path.basename(a.cache)
    L = []
    P = L.append
    P(f"# {lab}")
    P(f"# n = {n}  (q+ {int(ip.sum())}, q- {int(im.sum())})   krad = {a.krad}")
    srel = d["sigma"] * d["genpt"] * np.cosh(d["eta"].astype(np.float64))
    P(f"# median sigma_rel = sigma(q/p)/(q/p) = {np.median(srel):.5f}"
      f"   [1 pull unit = {1e4*np.median(srel):.2f}e-4 in dp/p]")

    # ---- model per track
    if not a.no_model:
        pool = np.where(ip, 0, 1)
        Om, Em, phis, cnt = model_per_track(d, args, PROBES,
                                            pool_id=pool, npool=2)
    # ---- odd moment table
    P("")
    P("## <z e^{-u z^2}>  per charge and the charge decomposition")
    P("| u | data q+ | data q- | A=odd-in-q | S=even-in-q | mod q+ | mod q- | A_mod | A_d-A_m |")
    P("|---|---|---|---|---|---|---|---|---|")
    Od_p = odd_of(z[ip]); Od_m = odd_of(z[im])
    ep = boot_stats(z[ip], odd_of, a.nboot, rng)
    em = boot_stats(z[im], odd_of, a.nboot, rng)
    for i, u in enumerate(PROBES):
        A_, S_ = AS(Od_p[i], Od_m[i])
        eA = 0.5 * np.hypot(ep[i], em[i])
        if not a.no_model:
            mp, mm = Om[ip, i].mean(), Om[im, i].mean()
            Am, _ = AS(mp, mm)
            P(f"| {u:g} | {Od_p[i]:+.5f} | {Od_m[i]:+.5f} | {A_:+.5f} +- {eA:.5f} "
              f"| {S_:+.5f} +- {eA:.5f} | {mp:+.5f} | {mm:+.5f} | {Am:+.5f} | {A_-Am:+.5f} |")
        else:
            P(f"| {u:g} | {Od_p[i]:+.5f} | {Od_m[i]:+.5f} | {A_:+.5f} +- {eA:.5f} "
              f"| {S_:+.5f} +- {eA:.5f} | - | - | - | - |")

    # ---- location statistics
    P("")
    P("## location statistics of z (data), and the model density")
    sp = stats_of(z[ip]); sm = stats_of(z[im])
    bp = boot_stats(z[ip], stats_of, a.nboot, rng)
    bm = boot_stats(z[im], stats_of, a.nboot, rng)
    if not a.no_model:
        mdp, zg, pdp = density_stats(phis[0])
        mdm, _, pdm = density_stats(phis[1])
    keys = ["mean", "median"] + [f"trim{T:g}" for T in TRIMS]
    P("| stat | data q+ | data q- | A_data | S_data | mod q+ | mod q- | A_mod | S_mod |")
    P("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        A_, S_ = AS(sp[k], sm[k])
        eA = 0.5 * np.hypot(bp[k], bm[k])
        if not a.no_model:
            Am, Sm = AS(mdp[k], mdm[k])
            P(f"| {k} | {sp[k]:+.5f} | {sm[k]:+.5f} | {A_:+.5f} +- {eA:.5f} "
              f"| {S_:+.5f} +- {eA:.5f} | {mdp[k]:+.5f} | {mdm[k]:+.5f} "
              f"| {Am:+.5f} | {Sm:+.5f} |")
        else:
            P(f"| {k} | {sp[k]:+.5f} | {sm[k]:+.5f} | {A_:+.5f} +- {eA:.5f} "
              f"| {S_:+.5f} +- {eA:.5f} | - | - | - | - |")

    # ---- tails
    P("")
    P("## tails of z (data only): fraction and mean beyond |z| = T")
    P("| T | q | P(z>T) | P(z<-T) | asym | <z>_{z>T} | <z>_{z<-T} | tail contrib to <z> |")
    P("|---|---|---|---|---|---|---|---|")
    for T in (3., 5., 10.):
        for nm, s in (("+", sp), ("-", sm)):
            P(f"| {T:g} | {nm} | {s[f'fpos{T:g}']:.5f} | {s[f'fneg{T:g}']:.5f} "
              f"| {s[f'asym{T:g}']:+.5f} | {s[f'mpos{T:g}']:+.2f} "
              f"| {s[f'mneg{T:g}']:+.2f} | {s[f'cont{T:g}']:+.5f} |")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt + "\n")
        # keep the arrays for the differential/figure steps
        np.savez_compressed(a.out.replace(".txt", ".npz"),
                            probes=np.array(PROBES),
                            Od_p=Od_p, Od_m=Od_m, eOd_p=ep, eOd_m=em,
                            **({"phi_p": phis[0], "phi_m": phis[1],
                                "Om": Om.astype(np.float32)}
                               if not a.no_model else {}))


if __name__ == "__main__":
    main()

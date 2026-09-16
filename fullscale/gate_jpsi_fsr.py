#!/usr/bin/env python3
"""GATE -- the FSR kernel CF under the J/psi mass term, against exact quadrature.

Three checks of `rabbit.unbinned.MassCFTerm`'s ``phik`` path on a J/psi-shaped
term (`make_joint_card.build_jpsi`'s configuration, with the two corrections
OFF so the density is a pure convolution and the check is of the kernel and
nothing else):

1. **the density IS the convolution.**  With a kernel that is a finite set of
   atoms ``(dm_j, w_j)`` the exact post-kernel density is
   ``L_K(m) = sum_j w_j L_delta(m - dm_j)``, and ``L_delta`` is the SAME term
   with ``phik=None``.  Feeding the atoms' exact CF ``sum_j w_j e^{i t dm_j}``
   as ``phik`` must reproduce it.  Run at three ``t`` steps: the residual has
   to fall as ``dt^2``, which identifies it as the tabulation's linear
   interpolation and nothing else.

2. **the real kernel.**  The same identity with the `jpsi_fsr_kernel.py`
   tabulation and the empirical ``dm`` sample behind it, which is the object
   the card will carry.

3. **the truncation normalisation.**  ``_norm_z`` (Gil-Pelaez, in Fourier
   space) against a direct Simpson quadrature of the term's own density over
   the window, with the kernel ON.  One resolution class per candidate, so the
   class density IS the candidate density and the two are comparable row by
   row.

The cache is read ONCE (it is 12 GB and compressed; re-opening it per term
costs 8 GB of decompression each time) and every term is built from the
in-memory slice.

usage::

    ./run_tf.sh python3 -u gate_jpsi_fsr.py \\
        --pairs runs/jpairs_v2_n600.npz --kernel ../zchannel/data/jpsi_kern_mc.npz
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(os.path.dirname(HERE), "resolution")
for _p in (HERE, _RES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MJPSI = 3.0969
WINDOW = 0.35


def _quiet(*_a, **_k):
    pass


class Slice:
    """The candidate columns one term needs, already in memory."""

    def __init__(self, pairs, idx, log):
        import make_card

        d = np.load(pairs, allow_pickle=True)
        self.tgrid = np.asarray(d["tgrid"], dtype=np.float64)
        self.nt = len(self.tgrid)
        self.sigma = d["sigma"].astype(np.float64)[idx]
        self.vgf = d["vgf"].astype(np.float64)[idx]
        mgen = d["eta"].astype(np.float64)[idx]
        z = d["z"].astype(np.float64)[idx]
        self.mobs = z * self.sigma + mgen - MJPSI
        self.fams = make_card.discover_families(set(d.files), False)
        self.arrays = {}
        for name, re_k, im_k in self.fams:
            a = {"re": np.asarray(d[re_k], dtype=np.float64)[idx]}
            if im_k:
                a["im"] = np.asarray(d[im_k], dtype=np.float64)[idx]
            self.arrays[name] = a
        log(f"  cache read once: {len(idx)} candidates, nt = {self.nt}, "
            f"families {[f[0] for f in self.fams]}")


def build_term(sl, phik, nclass=0, mobs=None, tile=1, norm_window=False):
    """A J/psi `MassCFTerm` on the slice, corrections OFF, optional ``phik``.

    ``norm_window`` is OFF by default: the truncation block is a
    ``(nclass, norm_tpoints)`` cubic resample per family, which on a term
    tiled to 10^5 rows with one class per row is tens of gigabytes -- and
    checks 1 and 2 only ever look at the density.  Check 3 turns it on.
    """
    import make_joint_card as mjc
    from rabbit import unbinned

    sigma, vgf = sl.sigma, sl.vgf
    arrays = sl.arrays
    m = sl.mobs if mobs is None else np.asarray(mobs, dtype=np.float64)
    if tile > 1:
        sigma = np.repeat(sigma, tile)
        vgf = np.repeat(vgf, tile)
        arrays = {nm: {c: np.repeat(a, tile, axis=0) for c, a in arr.items()}
                  for nm, arr in arrays.items()}
    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    for name, _, _ in sl.fams:
        families.append({"name": name, "param": f"k_{name}", "kind": "tab"})
    norm = (mjc.norm_classes(sigma, vgf, arrays, sl.nt, nclass, _quiet)[0]
            if norm_window else None)
    return unbinned.MassCFTerm(
        "jpsi",
        families=[dict(f, **arrays.get(f["name"], {})) for f in families],
        norm=norm, sigma=sigma, mobs=m, tgrid=sl.tgrid, vgf=vgf,
        phik=phik, kernel=unbinned.DeltaKernel(), background=None,
        m_ref=MJPSI, scale_param=None, bkg_frac=0.0, weights=None,
        a_res=None, self_consistent_sigma=False, jensen_s2=None,
        norm_window=((MJPSI - WINDOW, MJPSI + WINDOW) if norm_window else None),
        norm_tpoints=8192,
        upsample=1, floor_scale=1e-7, chunk=1 << 20, channel="jpsi")


def density(term):
    import tensorflow as tf

    vals = {p: tf.constant(1.0, tf.float64) for p in term.param_names}
    return np.concatenate([term._chunk_li(vals, ci).numpy()
                           for ci in range(len(term._chunks))])


def norm_z(term):
    import tensorflow as tf

    vals = {p: tf.constant(1.0, tf.float64) for p in term.param_names}
    return term._norm_z(vals).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=os.path.join(HERE, "runs",
                                                   "jpairs_v2_n600.npz"))
    p.add_argument("--kernel", default=os.path.join(
        os.path.dirname(HERE), "zchannel", "data", "jpsi_kern_mc.npz"))
    p.add_argument("--ncand", type=int, default=32)
    p.add_argument("--natom", type=int, default=48)
    p.add_argument("--ndraw", type=int, default=200000)
    p.add_argument("--block", type=int, default=4000)
    p.add_argument("--ngrid", type=int, default=4001)
    p.add_argument("--nclass-check", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    log = print
    rng = np.random.default_rng(a.seed)

    with np.load(a.pairs, allow_pickle=True) as d:
        sig = d["sigma"].astype(np.float64)
        mg = d["eta"].astype(np.float64)
        z = d["z"].astype(np.float64)
        c2 = (d["chisqval"].astype(np.float64)
              / np.maximum(d["ndof"].astype(np.float64), 1.0))
    m = z * sig + mg
    sel = (np.isfinite(m) & (sig > 0) & (c2 < 3.0)
           & (sig / np.maximum(np.abs(m), 1e-9) < 0.10))
    ok = np.where(sel & (np.abs(m - MJPSI) <= WINDOW))[0]
    idx = np.sort(rng.choice(ok, a.ncand, replace=False))
    log(f"[gate] {a.ncand} candidates from {os.path.basename(a.pairs)}")
    sl = Slice(a.pairs, idx, log)
    log(f"  sigma [MeV]: min {sl.sigma.min()*1e3:.2f} med "
        f"{np.median(sl.sigma)*1e3:.2f} max {sl.sigma.max()*1e3:.2f}")

    # ---- 1. the atom identity -------------------------------------------
    log("\n=== 1. L_K == sum_j w_j L_delta(m - dm_j), exact atoms ===")
    dm_at = np.sort(np.clip(rng.normal(-0.01, 0.06, a.natom), -0.34, 0.34))
    w_at = rng.random(a.natom)
    w_at /= w_at.sum()
    log(f"  {a.natom} atoms, dm in [{dm_at.min():+.4f}, {dm_at.max():+.4f}] "
        f"GeV, <dm> = {np.sum(w_at * dm_at) * 1e3:+.3f} MeV")
    mob = (sl.mobs[:, None] - dm_at[None, :]).ravel()
    L = density(build_term(sl, None, tile=a.natom, mobs=mob))
    ref = L.reshape(len(idx), a.natom) @ w_at
    for dt in (0.02, 0.01, 0.005):
        tt = np.arange(0.0, 1000.0 + 0.5 * dt, dt)
        cf = np.exp(1j * np.outer(tt, dm_at)) @ w_at
        got = density(build_term(sl, (tt, cf.real.copy(), cf.imag.copy())))
        r = np.abs(got / ref - 1.0)
        log(f"  dt = {dt:<6g}: max |L_K/L_conv - 1| = {r.max():.3e}, "
            f"median {np.median(r):.3e}")
    log("  (the residual must fall as dt^2: it is the linear interpolation of "
        "the CF tabulation, which is the ONLY approximation in the path)")

    # ---- 2. the real kernel ---------------------------------------------
    log("\n=== 2. the same identity with the jpsi_fsr_kernel.py tabulation ===")
    K = np.load(a.kernel, allow_pickle=True)
    phik = (np.asarray(K["phik_t"], np.float64),
            np.asarray(K["phik_re"], np.float64),
            np.asarray(K["phik_im"], np.float64))
    log(f"  {os.path.basename(a.kernel)}: {len(phik[0])} points to "
        f"t = {phik[0][-1]:g} 1/GeV")
    dm_all = (mg[sel] - MJPSI).astype(np.float64)
    draw = rng.choice(dm_all, a.ndraw, replace=False)
    log(f"  {len(dm_all)} dm samples; {a.ndraw} drawn for the direct sum "
        f"(<dm> {draw.mean()*1e3:+.4f} vs {dm_all.mean()*1e3:+.4f} MeV)")
    acc = np.zeros(len(idx))
    for j0 in range(0, a.ndraw, a.block):
        blk = draw[j0:j0 + a.block]
        mob = (sl.mobs[:, None] - blk[None, :]).ravel()
        acc += density(build_term(sl, None, tile=len(blk), mobs=mob)
                       ).reshape(len(idx), len(blk)).sum(axis=1)
    acc /= a.ndraw
    got = density(build_term(sl, phik))
    r = got / acc - 1.0
    log(f"  max |L_K/L_MCsum - 1| = {np.abs(r).max():.3e}, median "
        f"{np.median(np.abs(r)):.3e}")
    log(f"  the direct sum is itself a {a.ndraw}-sample estimate of the same "
        f"integral the tabulation does over {len(dm_all)}, so its own noise "
        f"(~1/sqrt({a.ndraw}) = {1/np.sqrt(a.ndraw):.1e} of the density's own "
        "spread) is the floor here, not the path")

    # ---- 3. the truncation normalisation --------------------------------
    log("\n=== 3. _norm_z (Gil-Pelaez) vs direct quadrature, kernel ON ===")
    nc = min(a.nclass_check, len(idx))
    sub = Slice.__new__(Slice)
    sub.__dict__.update({k: v for k, v in sl.__dict__.items()})
    sub.sigma = sl.sigma[:nc]
    sub.vgf = sl.vgf[:nc]
    sub.mobs = sl.mobs[:nc]
    sub.arrays = {nm: {c: v[:nc] for c, v in arr.items()}
                  for nm, arr in sl.arrays.items()}
    Z = norm_z(build_term(sub, phik, nclass=0, norm_window=True))
    lo, hi = MJPSI - WINDOW, MJPSI + WINDOW
    for ng in (a.ngrid, 2 * (a.ngrid - 1) + 1):
        mgrid = np.linspace(lo, hi, ng)
        mob = np.tile(mgrid - MJPSI, nc)
        L = density(build_term(sub, phik, tile=ng, mobs=mob)).reshape(nc, ng)
        wS = np.ones(ng)
        wS[1:-1:2] = 4.0
        wS[2:-1:2] = 2.0
        Zq = (L * wS).sum(axis=1) * (mgrid[1] - mgrid[0]) / 3.0
        log(f"  Simpson on {ng} mass points: worst |Z/Z_quad - 1| = "
            f"{np.abs(Z / Zq - 1).max():.3e}")
    log("   c   sigma [MeV]       Z(_norm_z)       Z(Simpson)      rel diff")
    for c in range(nc):
        log(f"  {c:2d}  {sub.sigma[c]*1e3:9.3f}   {Z[c]:.12f}   {Zq[c]:.12f}  "
            f"{Z[c] / Zq[c] - 1:+.3e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where the J/psi mass term WANTS the momentum scale, term by term, no fit.

The calibration vector enters the J/psi term only through the per-candidate
residual, `delta_i = m_i - M - (D theta)_i`, so a common momentum-scale error
is one number: a common shift `s` of every candidate's predicted mass. This
scans the truncated `-sum log(L_i/Z_i)` in `s` and reports the minimum -- the
term's own preferred scale, with truth at `s = 0` -- for each configuration of
the two corrections and the kernel.

That makes the pieces of the residual separable without refitting anything:

* `kernel off -> on` reproduces the FSR step the fits measure;
* `corrections on -> off` is what the self-consistent width and the exact
  Jensen map contribute to the preferred scale, which is the leading named
  candidate for what is left;
* the difference between the two correction FORMS cannot be scanned here for
  the same reason the `JKF` fit failed -- the fluctuation form's first-order
  truncation goes negative on this term -- and the script says so rather than
  returning a number.

usage::

    ./run_tf.sh python3 -u jpsi_scale_pref.py --n 400000
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(os.path.dirname(HERE), "resolution")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MJPSI = 3.0969
WINDOW = 0.35


def _quiet(*_a, **_k):
    pass


def build(sl, phik, corrections, nclass, chunk, corr_form="residual"):
    """The card's own J/psi term on the slice, with the two corrections
    switched as a pair (they are what `fit.py --ares/--jensen` toggles)."""
    import make_joint_card as mjc
    from rabbit import unbinned

    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    for name, _, _ in sl["fams"]:
        families.append({"name": name, "param": f"k_{name}", "kind": "tab"})
    norm, _ = mjc.norm_classes(sl["sigma"], sl["vgf"], sl["arrays"],
                               sl["nt"], nclass, _quiet)
    return unbinned.MassCFTerm(
        "jpsi",
        families=[dict(f, **sl["arrays"].get(f["name"], {})) for f in families],
        norm=norm, sigma=sl["sigma"], mobs=sl["mobs"], tgrid=sl["tgrid"],
        vgf=sl["vgf"], phik=phik, kernel=unbinned.DeltaKernel(),
        background=None, m_ref=MJPSI, scale_param=None, bkg_frac=0.0,
        weights=None,
        a_res=(sl["a_res"] if corrections else None),
        self_consistent_sigma=bool(corrections),
        jensen_s2=(sl["jensen_s2"] if corrections else None),
        jensen_mode=("exact" if corrections else "off"),
        corr_form=corr_form,
        norm_window=(MJPSI - WINDOW, MJPSI + WINDOW), norm_tpoints=8192,
        upsample=1, floor_scale=1e-7, chunk=chunk, channel="jpsi")


def nll_at(sl, phik, corrections, shifts, nclass, chunk, log):
    """``-sum log(L_i/Z_i)`` at each common predicted-mass shift in ``shifts``."""
    import tensorflow as tf

    out = []
    base = sl["mobs"].copy()
    for s in shifts:
        sl["mobs"] = base - s
        t = build(sl, phik, corrections, nclass, chunk)
        vals = {p: tf.constant(1.0, tf.float64) for p in t.param_names}
        L = np.concatenate([t._chunk_li(vals, ci).numpy()
                            for ci in range(len(t._chunks))])
        Z = t._norm_z(vals).numpy()[t._norm_class.numpy()]
        out.append(-float(np.sum(np.log(np.maximum(L, 1e-300)) - np.log(Z))))
        del t
    sl["mobs"] = base
    return np.asarray(out)


def vertex(x, y):
    """The parabola vertex through the three points nearest the minimum."""
    i = int(np.argmin(y))
    i = min(max(i, 1), len(x) - 2)
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    return (-b / (2.0 * a), a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=os.path.join(HERE, "runs",
                                                   "jpairs_v2_n600.npz"))
    p.add_argument("--kernel", default=os.path.join(
        os.path.dirname(HERE), "zchannel", "data", "jpsi_kern_mc.npz"))
    p.add_argument("--n", type=int, default=400000)
    p.add_argument("--nclass", type=int, default=64)
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--span", type=float, default=6.0, help="scan +-span [MeV]")
    p.add_argument("--npoint", type=int, default=13)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    log = print
    rng = np.random.default_rng(a.seed)

    import make_card

    d = np.load(a.pairs, allow_pickle=True)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mg = d["eta"].astype(np.float64)
    c2 = (d["chisqval"].astype(np.float64)
          / np.maximum(d["ndof"].astype(np.float64), 1.0))
    m = z * sig + mg
    sel = (np.isfinite(m) & (sig > 0) & (c2 < 3.0)
           & (sig / np.maximum(np.abs(m), 1e-9) < 0.10)
           & (np.abs(m - MJPSI) <= WINDOW))
    idx = np.sort(rng.choice(np.where(sel)[0], a.n, replace=False))
    sl = {"tgrid": np.asarray(d["tgrid"], dtype=np.float64)}
    sl["nt"] = len(sl["tgrid"])
    sl["sigma"] = sig[idx]
    sl["vgf"] = d["vgf"].astype(np.float64)[idx]
    sl["mobs"] = (m - MJPSI)[idx]
    mreco = m[idx]
    sl["a_res"] = np.clip((1.0 + sl["vgf"]) * sl["sigma"]
                          / np.maximum(np.abs(mreco), 1e-9), -0.5, 0.5)
    s2 = (sl["sigma"] / np.maximum(np.abs(mreco), 1e-9)) ** 2
    fang = np.clip(np.asarray(d["fang"], dtype=np.float64)[idx], -0.5, 1.0)
    sl["jensen_s2"] = s2 * (1.5 - fang) / 1.5
    sl["fams"] = make_card.discover_families(set(d.files), False)
    sl["arrays"] = {}
    for name, re_k, im_k in sl["fams"]:
        e = {"re": np.asarray(d[re_k], dtype=np.float64)[idx]}
        if im_k:
            e["im"] = np.asarray(d[im_k], dtype=np.float64)[idx]
        sl["arrays"][name] = e
    del d
    log(f"[scale-pref] {a.n} candidates, {a.nclass} norm classes, "
        f"a_res median {np.median(sl['a_res']):.5f}")

    K = np.load(a.kernel, allow_pickle=True)
    phik = (np.asarray(K["phik_t"], np.float64),
            np.asarray(K["phik_re"], np.float64),
            np.asarray(K["phik_im"], np.float64))
    shifts = np.linspace(-a.span * 1e-3, a.span * 1e-3, a.npoint)

    res = {}
    for tag, pk, corr in (("delta,   corrections ON ", None, True),
                          ("kernel,  corrections ON ", phik, True),
                          ("delta,   corrections OFF", None, False),
                          ("kernel,  corrections OFF", phik, False)):
        y = nll_at(sl, pk, corr, shifts, a.nclass, a.chunk, log)
        s0, curv = vertex(shifts, y)
        res[tag] = s0
        log(f"  {tag}: preferred shift = {s0*1e3:+8.4f} MeV "
            f"= {s0/MJPSI:+.4e}   (curvature {curv:.3e}/GeV^2)")

    log("\n  the pieces, as differences of the preferred scale:")
    k_on = res["kernel,  corrections ON "]
    k_off = res["delta,   corrections ON "]
    log(f"    the FSR kernel, corrections ON : "
        f"{(k_on-k_off)*1e3:+8.4f} MeV = {(k_on-k_off)/MJPSI:+.4e}")
    c_on = res["kernel,  corrections ON "]
    c_off = res["kernel,  corrections OFF"]
    log(f"    the two corrections, kernel ON : "
        f"{(c_on-c_off)*1e3:+8.4f} MeV = {(c_on-c_off)/MJPSI:+.4e}")
    log(f"\n  the term's own residual with the kernel and both corrections: "
        f"{k_on*1e3:+8.4f} MeV = {k_on/MJPSI:+.4e}")
    log("  (the FLUCTUATION form cannot be scanned: its first-order Fourier "
        "truncation goes negative on this term -- the `JKF` fit died on a "
        "non-positive-definite Hessian, rc=1)")


if __name__ == "__main__":
    main()

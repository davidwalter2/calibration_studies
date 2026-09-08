#!/usr/bin/env python3
"""GATE 1 of the fluctuation reformulation, on the REAL J/psi gun candidates.

`MASSCFTERM_SPEC.md` measured both corrections on the J/psi gun with the
offline numpy implementation `oddmoment/masslik_np.py`:

    naive                       alpha = -0.0047  e-3
    + self-consistent sigma            +0.1410  e-3   (i.e. +0.146 e-3)
    + Jensen (EXACT) on top            +0.0512 +- 0.0167 e-3

At the J/psi `delta_i` IS the resolution fluctuation, so the RESIDUAL form
(the historical one) is exact there and this is the one place the two forms
must agree.  This script builds the SAME candidates as three rabbit
`MassCFTerm`s -- uncorrected, `corr_form="residual"` at `corr_clip = 0`, and
`corr_form="fluctuation"` -- and scans `alpha`.

What is compared is the SHIFT each form applies (corrected minus uncorrected),
because that is what the spec measured and it carries no statistical error:
the three fits share the candidates.

    THREADS=32 ./run_tf.sh python3 -u gate_fluct_gun.py \\
        --pairs ../resolution/runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz \\
        --kernel ../resolution/runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MJPSI = 3.0969
FBKG = 0.005
WIN = 0.7


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--krad", type=float, default=1.0)
    p.add_argument("--vpow", type=float, default=0.0,
                   help="also build the v-formulation term (`sigma -> "
                        "k = sigma/m^p`, `mobs -> v(m) - v(M)`, the FSR kernel "
                        "mapped to v). At a DELTA kernel the true mass is a "
                        "single point, so the m and v formulations must "
                        "coincide up to the few-per-cent spread the FSR kernel "
                        "itself puts on `m_true` -- which is exactly what "
                        "makes this the gate: the v form must reproduce the "
                        "spec's alpha shifts, not improve on them.")
    p.add_argument("--fbkg", type=float, default=FBKG)
    p.add_argument("--halfwidth", type=float, default=2.0,
                   help="alpha scan half-width, in 1e-3")
    p.add_argument("-o", "--output", default=None)
    return p.parse_args(argv)


def build_terms(args, log=print):
    import tensorflow as tf
    from rabbit import unbinned

    d = np.load(args.pairs)
    tg = np.asarray(d["tgrid"], dtype=np.float64)
    n = len(d["z"])
    if args.maxn and args.maxn < n:
        n = args.maxn
    sl = slice(0, n)
    z = d["z"].astype(np.float64)[sl]
    sig = d["sigma"].astype(np.float64)[sl]
    mgen = d["eta"].astype(np.float64)[sl]          # `eta` IS the gen mass
    vgf = d["vgf"].astype(np.float64)[sl]
    mobs = z * sig + (mgen - MJPSI)                 # m_i - M_Jpsi
    log(f"  {n} candidates, nt = {len(tg)}, sigma median {np.median(sig):.5f}, "
        f"sigma/m median {np.median(sig/mgen):.5f}")

    has_rad = ("Srad_re" in d.files) and int(d["rad_model"]) == 1
    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"},
                {"name": "ms", "param": "k_ms", "kind": "tab",
                 "re": d["Sms"][sl]},
                {"name": "ioni", "param": "k_ioni", "kind": "tab",
                 "re": d["Sio_re"][sl], "im": d["Sio_im"][sl]}]
    if has_rad:
        families.append({"name": "rad", "param": "k_rad", "kind": "tab",
                         "re": d["Srad_re"][sl], "im": d["Srad_im"][sl]})

    # the FSR kernel CF on the SAME absolute-t tabulation masslik_np uses
    k = np.load(args.kernel)
    dm = np.asarray(k["dm"], dtype=np.float64)
    tabs = np.linspace(0.0, tg[-1] / sig.min(), 8192)
    t0 = time.time()
    blk = 1024
    phi = np.concatenate([np.mean(np.exp(1j * np.outer(tabs[i:i + blk], dm)),
                                  axis=1) for i in range(0, len(tabs), blk)])
    log(f"  phiK table ({len(dm)} kernel samples, tmax {tabs[-1]:.1f}) in "
        f"{time.time()-t0:.1f} s")

    # a_i and s^2 exactly as `masslik_np` defines them (on m_gen)
    a_res = (1.0 + vgf) * sig / mgen
    s2 = (sig / mgen) ** 2
    log(f"  a_res median {np.median(a_res):.5f}, "
        f"1.5 s^2 median {1.5*np.median(s2):.4e} "
        f"-> {1.5*np.median(s2)*MJPSI*1e3:.3f} MeV")

    common = dict(sigma=sig, mobs=mobs, tgrid=tg, families=families, vgf=vgf,
                  phik=(tabs, phi.real.copy(), phi.imag.copy()),
                  kernel=unbinned.DeltaKernel(),
                  background=unbinned.UniformBackground(
                      (MJPSI - 0.5 * WIN, MJPSI + 0.5 * WIN)),
                  m_ref=MJPSI, scale_param="alpha", bkg_frac=args.fbkg,
                  floor="clip", chunk=args.chunk)

    terms = {}
    terms["uncorrected"] = unbinned.MassCFTerm("gun_none", **common)
    if args.vpow:
        pw = float(args.vpow)
        vmap = (lambda x: np.log(x)) if pw == 1.0 else \
            (lambda x: x ** (1.0 - pw) / (1.0 - pw))
        m_phys = mobs + MJPSI
        vref = float(vmap(np.array(MJPSI)))
        # the FSR kernel is a distribution of the mass SHIFT; in v it is the
        # distribution of v(M + dm) - v(M), which is dm/M^p to first order
        dmv = vmap(MJPSI + dm) - vref
        phiv = np.concatenate(
            [np.mean(np.exp(1j * np.outer(tabs[i:i + blk] * MJPSI ** pw, dmv)),
                     axis=1) for i in range(0, len(tabs), blk)])
        wv_lo = float(vmap(np.array(MJPSI - 0.5 * WIN))) - vref
        wv_hi = float(vmap(np.array(MJPSI + 0.5 * WIN))) - vref
        mref_v = MJPSI ** (1.0 - pw)
        cv = dict(common)
        cv.update(sigma=sig / m_phys ** pw,
                  mobs=vmap(m_phys) - vref,
                  phik=(tabs * MJPSI ** pw, phiv.real.copy(), phiv.imag.copy()),
                  background=unbinned.UniformBackground(
                      (wv_lo + mref_v, wv_hi + mref_v)),
                  m_ref=mref_v, corr_mass=m_phys)
        terms["v_fluctuation"] = unbinned.MassCFTerm(
            "gun_v", a_res=a_res, jensen_s2=s2, jensen_mode="exact",
            corr_form="fluctuation", vpow=pw, **cv)
        terms["v_ares_only"] = unbinned.MassCFTerm(
            "gun_v_a", a_res=a_res, jensen_mode="off",
            corr_form="fluctuation", vpow=pw, **cv)
        terms["v_uncorrected"] = unbinned.MassCFTerm("gun_v_none", **cv)
        log(f"  --vpow {pw:g}: k median {np.median(sig/m_phys**pw):.6g}, "
            f"v window [{wv_lo:.6g}, {wv_hi:.6g}]")
    terms["residual"] = unbinned.MassCFTerm(
        "gun_res", a_res=a_res, jensen_s2=s2, jensen_mode="exact",
        corr_clip=0.0, corr_form="residual", **common)
    terms["res_ares_only"] = unbinned.MassCFTerm(
        "gun_resa", a_res=a_res, jensen_mode="off",
        corr_clip=0.0, corr_form="residual", **common)
    terms["fluctuation"] = unbinned.MassCFTerm(
        "gun_flu", a_res=a_res, jensen_s2=s2, jensen_mode="exact",
        corr_form="fluctuation", **common)
    terms["flu_ares_only"] = unbinned.MassCFTerm(
        "gun_flua", a_res=a_res, jensen_mode="off",
        corr_form="fluctuation", **common)
    for t in terms.values():
        decl = {p: (1.0 if p.startswith("k_") else 0.0, np.nan,
                    1.0 if p.startswith("k_") else 0.0,
                    1 if p == "alpha" else 0) for p in t.param_names}
        if "k_rad" in decl:
            decl["k_rad"] = (args.krad, np.nan, args.krad, 0)
        unbinned.declare_params(t, decl)
    return terms, n


def fit_alpha(term, halfwidth, log=print):
    """masslik_np's parabolic minimiser, on the rabbit term."""
    import tensorflow as tf

    names = list(term.param_names)
    base = np.asarray(term.param_defaults, dtype=np.float64)
    ia = names.index("alpha")

    def nll(a):
        x = base.copy()
        x[ia] = a
        return float(term.nll(tf.constant(x, tf.float64)).numpy())

    lo, hi = -halfwidth, halfwidth
    c = None
    for _ in range(3):
        g = np.linspace(lo, hi, 7)
        v = np.array([nll(x) for x in g])
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"non-finite NLL on the alpha grid: {v}")
        i = int(np.clip(np.argmin(v), 1, len(g) - 2))
        c = np.polyfit(g[i - 1:i + 2] - g[i], v[i - 1:i + 2], 2)
        ahat = g[i] - 0.5 * c[1] / c[0]
        step = g[1] - g[0]
        lo, hi = ahat - step, ahat + step
    err = float(np.sqrt(0.5 / c[0])) if c[0] > 0 else float("nan")
    return float(ahat), err, nll(ahat)


def main(argv=None):
    args = parse_args(argv)
    print(f"=== gate 1: the fluctuation form on the J/psi gun ===")
    terms, n = build_terms(args)
    out = {}
    names = ["uncorrected", "res_ares_only", "flu_ares_only",
             "residual", "fluctuation"]
    if args.vpow:
        names += ["v_uncorrected", "v_ares_only", "v_fluctuation"]
    for name in names:
        t0 = time.time()
        a, e, v = fit_alpha(terms[name], args.halfwidth)
        out[name] = {"alpha": a, "err": e, "nll": v, "seconds": time.time() - t0}
        print(f"  {name:16s} alpha = {a:+.5f} +- {e:.5f} e-3   "
              f"NLL {v:.4f}   ({time.time()-t0:.0f} s)")

    ref = out["uncorrected"]["alpha"]
    print("\n  --- the SHIFT each form applies (no statistical error: same "
          "candidates) ---")
    print(f"  {'form':22s} {'shift [e-3]':>12s}  {'spec':>10s}")
    rows = [
        ("a_res only, residual", out["res_ares_only"]["alpha"] - ref, +0.1457),
        ("a_res only, fluctuation", out["flu_ares_only"]["alpha"] - ref, +0.1457),
        ("both, residual", out["residual"]["alpha"] - ref, +0.0559),
        ("both, fluctuation", out["fluctuation"]["alpha"] - ref, +0.0559),
    ]
    for lab, sh, spec in rows:
        print(f"  {lab:22s} {sh:+12.5f}  {spec:+10.4f}")
    d_a = (out["flu_ares_only"]["alpha"] - out["res_ares_only"]["alpha"])
    d_b = (out["fluctuation"]["alpha"] - out["residual"]["alpha"])
    print(f"\n  GATE: |fluctuation - residual|  a_res only {abs(d_a):.5f} e-3, "
          f"both {abs(d_b):.5f} e-3   (requirement < 0.01 e-3)")
    ok = abs(d_a) < 0.01 and abs(d_b) < 0.01
    if args.vpow:
        vref_a = out["v_uncorrected"]["alpha"]
        d_va = out["v_ares_only"]["alpha"] - vref_a
        d_vb = out["v_fluctuation"]["alpha"] - vref_a
        print(f"  {'a_res only, v form':22s} {d_va:+12.5f}  {+0.1457:+10.4f}")
        print(f"  {'both, v form':22s} {d_vb:+12.5f}  {+0.0559:+10.4f}")
        e_a = abs(d_va - (out["flu_ares_only"]["alpha"] - ref))
        e_b = abs(d_vb - (out["fluctuation"]["alpha"] - ref))
        print(f"\n  GATE (v): |v - m| shift  a_res only {e_a:.5f} e-3, "
              f"both {e_b:.5f} e-3   (requirement < 0.01 e-3)")
        # the ABSOLUTE alpha may legitimately move: at a delta kernel the true
        # mass is one point, but the FSR kernel gives it a few-per-cent spread
        # and the v form treats the width across THAT correctly
        print(f"  absolute alpha: m form {out['fluctuation']['alpha']:+.5f}, "
              f"v form {out['v_fluctuation']['alpha']:+.5f} e-3 "
              f"(difference {out['v_fluctuation']['alpha'] - out['fluctuation']['alpha']:+.5f})")
        ok = ok and e_a < 0.01 and e_b < 0.01
        out["_vgate"] = {"d_ares": d_va, "d_both": d_vb,
                         "e_ares": e_a, "e_both": e_b}
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    out["_gate"] = {"n": n, "d_ares": d_a, "d_both": d_b, "pass": bool(ok)}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-end impact of compressing the CF exponents on the mass fit.

Refits the SAME 4-parameter unbinned mass likelihood
(`cf_masslik_fit.py --model families`, k_rad fixed at 1, window off) with the
per-candidate exponents replaced by a rank-r / k-scalar reconstruction, and
reports the shift of alpha and of the k_f against the uncompressed fit.

Bases (fitted on a TRAINING split, applied to the fit sample):
  pca_all   ONE coefficient vector per candidate, joint over all 5 families
  pca_fam   an independent rank-r basis per family (5r scalars)
  levy      the physics dictionaries of `cfbasis` (no data pass for the basis)

Usage
  python3 nll_impact.py --tag jpsigun --mode validate           # full cache
  python3 nll_impact.py --tag jpsigun --mode compress --nfit 20000
"""
import argparse, json, os, time
import numpy as np
import massnll_np as MN
import cfbasis as CB

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
RUNS = MN.RUNS
FAMS = ["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"]
FULL = {"jpsigun": f"{RUNS}/cf_masspairs_jpsigun_ul16_260903x_m0.npz",
        "btojpsix": f"{RUNS}/cf_masspairs_btojpsix_v3_260903x_m0.npz"}


def load(tag, src, n0, n1, log=print):
    t0 = time.time()
    d = np.load(src)
    TG = d["tgrid"].astype(np.float64)
    sl = slice(n0, n1)
    sig = d["sigma"][sl].astype(np.float64)
    z = d["z"][sl].astype(np.float64)
    eta = d["eta"][sl].astype(np.float64)
    vgf = d["vgf"][sl].astype(np.float64)
    mobs = z * sig + (eta - MN.MJPSI)
    M = {f: np.ascontiguousarray(d[f][sl]) for f in FAMS}
    pKre, pKim = MN.phik(tag, TG, sig)
    log(f"  loaded {len(sig)} candidates in {time.time()-t0:.1f} s")
    return dict(TG=TG, sig=sig, vgf=vgf, mobs=mobs, M=M, pKre=pKre, pKim=pKim)


def make_model(inp, M, nproc, chunk=8192):
    return MN.Model(inp["sig"], inp["mobs"], inp["vgf"], M["Sms"], M["Sio_re"],
                    M["Sio_im"], M["Srad_re"], M["Srad_im"], inp["pKre"],
                    inp["pKim"], inp["TG"], nproc=nproc, chunk=chunk)


# ------------------------------------------------------------ bases -------
def fit_basis(Mtr, kind, r, TG):
    """Return (name, apply_fn, bytes_per_candidate)."""
    nt = len(TG)
    if kind == "pca_all":
        X = np.concatenate([Mtr[f].astype(np.float64) for f in FAMS], axis=1)
        mu, V, sv = CB.pca_fit(X, r)

        def ap(M):
            Y = np.concatenate([M[f].astype(np.float64) for f in FAMS], axis=1)
            C = ((Y - mu) @ V).astype(np.float32).astype(np.float64)  # f32 store
            R = mu + C @ V.T
            return {f: np.ascontiguousarray(R[:, j * nt:(j + 1) * nt],
                                            dtype=np.float32)
                    for j, f in enumerate(FAMS)}
        return f"pca_all r={r}", ap, 4 * r
    if kind == "pca_fam":
        B = {}
        for f in FAMS:
            B[f] = CB.pca_fit(Mtr[f].astype(np.float64), r)[:2]

        def ap(M):
            o = {}
            for f in FAMS:
                mu, V = B[f]
                Y = M[f].astype(np.float64)
                C = ((Y - mu) @ V).astype(np.float32).astype(np.float64)
                o[f] = np.ascontiguousarray(mu + C @ V.T, dtype=np.float32)
            return o
        return f"pca_fam r={r}/fam", ap, 4 * r * len(FAMS)
    if kind == "levy":
        # r = total scalars, split ms : io : rad
        kms = max(2, r // 2)
        kio = max(2, (r - kms) * 2 // 3)
        krad = max(2, r - kms - kio)
        Dms = CB.moliere_design(TG, np.logspace(-0.5, 2.5, kms))
        vio = np.logspace(-1.5, 1.5, max(1, kio // 2))
        Dio = CB.levy_design(TG, np.concatenate([vio, -vio]))
        vrd = np.logspace(-1.5, 1.5, max(1, krad // 2))
        Drd = CB.levy_design(TG, np.concatenate([vrd, -vrd]))

        def piv(D):
            U, s, Vt = np.linalg.svd(D, full_matrices=False)
            k = s > 1e-12 * s[0]
            return (Vt[k].T / s[k]) @ U[:, k].T

        Pms, Pio, Prd = piv(Dms), piv(Dio), piv(Drd)

        def ap(M):
            o = {}
            Y = M["Sms"].astype(np.float64)
            C = (Y @ Pms.T).astype(np.float32).astype(np.float64)
            o["Sms"] = np.ascontiguousarray(C @ Dms.T, dtype=np.float32)
            for pre, D, P in (("Sio", Dio, Pio), ("Srad", Drd, Prd)):
                Y = np.concatenate([M[pre + "_re"].astype(np.float64),
                                    M[pre + "_im"].astype(np.float64)], axis=1)
                C = (Y @ P.T).astype(np.float32).astype(np.float64)
                R = C @ D.T
                o[pre + "_re"] = np.ascontiguousarray(R[:, :nt], np.float32)
                o[pre + "_im"] = np.ascontiguousarray(R[:, nt:], np.float32)
            return o
        return (f"levy k={kms}+{2*(kio//2)}+{2*(krad//2)}", ap,
                4 * (kms + 2 * (kio // 2) + 2 * (krad // 2)))
    raise ValueError(kind)


def report(name, fit, ref, nbytes=0, log=print):
    if ref is None:
        log(f"{name:22s} NLL={fit['nll']:.6f}  " +
            "  ".join(f"{p}={v:+.6f}+-{e:.6f}"
                      for p, v, e in zip(MN.PARNAMES, fit["x"], fit["err"])))
        return
    d = fit["x"] - ref["x"]
    log(f"{name:22s} dNLL={fit['nll']-ref['nll']:+.4e}  bytes/cand={nbytes:5d}  "
        + "  ".join(f"d{p}={dv:+.3e}({dv/e:+.3f}s)"
                    for p, dv, e in zip(MN.PARNAMES, d, ref["err"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jpsigun")
    ap.add_argument("--mode", default="compress",
                    choices=["validate", "compress", "fullshift"])
    ap.add_argument("--nfit", type=int, default=20000)
    ap.add_argument("--ntrain", type=int, default=40000)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--ranks", default="4,8,16,32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    sub = f"{SCRATCH}/sub_{a.tag}_n60000_s1234.npz"

    if a.mode == "validate":
        print(f"== VALIDATION: full cache {FULL[a.tag]} ==")
        inp = load(a.tag, FULL[a.tag], 0, None)
        m = make_model(inp, inp["M"], a.nproc)
        t0 = time.time()
        f, g = m.nll_grad([0.2447, 0.9426, 1.0108, 0.6710])
        print(f"  NLL at the published minimum = {f:.6f}  |grad|inf={np.abs(g).max():.3e}"
              f"  ({time.time()-t0:.1f} s/eval, nproc={a.nproc})")
        r = m.fit(log=print)
        report("full", r, None)
        print(f"  nit={r['nit']} |g|inf={r['gnorm']:.2e} success={r['success']}")
        print("  published (rad260903x/masslikfit_summary.txt, gun k_rad=1):")
        print("    alpha=+0.2447+-0.0177 k_hit=0.9426+-0.0354 "
              "k_ms=1.0108+-0.0061 k_ioni=0.6710+-0.0545 NLL=-588375.47")
        return

    if a.mode == "fullshift":
        # the basis is fitted on the 60k shuffled subsample and applied to the
        # WHOLE cache; the shift of the minimum is evaluated to first order,
        # dtheta = -H^-1 g_compressed(theta*_full), which needs one gradient
        # instead of a full refit (and is not limited by optimiser tolerance).
        print(f"== FULL-SAMPLE SHIFT [{a.tag}] ==")
        tr = load(a.tag, sub, 0, 60000)
        ev = load(a.tag, FULL[a.tag], 0, None)
        m0 = make_model(ev, ev["M"], a.nproc)
        ref = m0.fit()
        report("full (uncompressed)", ref, None)
        print(f"  nit={ref['nit']} |g|inf={ref['gnorm']:.2e}")
        rows = [dict(name="full", nll=ref["nll"], x=ref["x"].tolist(),
                     err=ref["err"].tolist(), bytes=5 * 448 * 4)]
        for kind in ("pca_all",):
            for r in ranks:
                name, apf, nb = fit_basis(tr["M"], kind, r, ev["TG"])
                Mr = apf(ev["M"])
                e = max(np.abs(Mr[f].astype(np.float64)
                               - ev["M"][f].astype(np.float64)).max() for f in FAMS)
                mm = make_model(ev, Mr, a.nproc)
                nllc, gc = mm.nll_grad(ref["x"])
                d1 = -ref["cov"] @ gc
                print(f"{name:22s} bytes={nb:5d} max|dS|={e:.2e} "
                      f"dNLL(th*)={nllc-ref['nll']:+.4e}  "
                      + "  ".join(f"d{p}={v:+.3e}({v/er:+.3f}s)" for p, v, er
                                  in zip(MN.PARNAMES, d1, ref["err"])))
                rows.append(dict(name=name, kind=kind, r=r, bytes=nb,
                                 maxdS=float(e), d1=d1.tolist(),
                                 dnll_at_ref=float(nllc - ref["nll"])))
                del Mr, mm
        out = a.out or f"{SCRATCH}/nllfullshift_{a.tag}.json"
        json.dump(rows, open(out, "w"), indent=1)
        print("wrote", out)
        return

    print(f"== COMPRESSION IMPACT [{a.tag}]  nfit={a.nfit} "
          f"(rows {a.ntrain}..{a.ntrain+a.nfit}), basis trained on rows 0..{a.ntrain} ==")
    tr = load(a.tag, sub, 0, a.ntrain)
    ev = load(a.tag, sub, a.ntrain, a.ntrain + a.nfit)
    m0 = make_model(ev, ev["M"], a.nproc)
    t0 = time.time()
    ref = m0.fit()
    print(f"  reference fit in {time.time()-t0:.1f} s, nit={ref['nit']}")
    report("full (uncompressed)", ref, None)
    print(f"  bytes/candidate (5 x 448 x float32) = {5*448*4}")
    rows = [dict(name="full", nll=ref["nll"], x=ref["x"].tolist(),
                 err=ref["err"].tolist(), bytes=5 * 448 * 4)]
    for kind in ("pca_all", "pca_fam", "levy"):
        for r in ranks:
            name, apf, nb = fit_basis(tr["M"], kind, r, ev["TG"])
            Mr = apf(ev["M"])
            # exponent reconstruction error on the fit sample
            e = max(np.abs(Mr[f].astype(np.float64)
                           - ev["M"][f].astype(np.float64)).max() for f in FAMS)
            mm = make_model(ev, Mr, a.nproc)
            nllref, gref = mm.nll_grad(ref["x"])
            # first-order shift of the minimum: dtheta = -H^-1 g_compressed(th*)
            # (g_full(th*) = 0), which is what actually moves the scale
            d1 = -ref["cov"] @ gref
            fr = mm.fit(th0=ref["x"])
            report(name, fr, ref, nb)
            print(f"{'':22s}   max|dS|={e:.2e}  dNLL(theta*_full)="
                  f"{nllref-ref['nll']:+.4e}  storage x{5*448*4/nb:.1f}"
                  f"  1st-order dalpha={d1[0]*1e-3:+.3e}")
            rows.append(dict(name=name, kind=kind, r=r, nll=fr["nll"],
                             x=fr["x"].tolist(), err=fr["err"].tolist(),
                             bytes=nb, maxdS=float(e), d1=d1.tolist(),
                             dnll_at_ref=float(nllref - ref["nll"])))
            del Mr, mm
    out = a.out or f"{SCRATCH}/nllimpact_{a.tag}_n{a.nfit}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

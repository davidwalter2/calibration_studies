#!/usr/bin/env python3
"""SVD / PCA compressibility of the per-candidate CF exponents.

For every family f the cache stores an (n x 448) matrix S_f.  The question is
how many numbers per candidate are actually needed: the exponent is, per
family, a sum over Geant4 steps of ONE universal function of a scaled
argument, so the row space should be far smaller than 448.

Blocks analysed (a "block" is what gets ONE coefficient vector per candidate):
  Sms, Sio_re, Sio_im, Srad_re, Srad_im, Sdel   -- single families
  IO   = [Sio_re | Sio_im]                      -- the complex ioni exponent
  RAD  = [Srad_re | Srad_im]
  ALL  = every family concatenated               -- ONE vector per candidate
                                                    that still lets every k_f
                                                    float independently

Metrics, per candidate i, for the rank-r reconstruction R = S - (mu + A_r V_r^T):
  eabs_i = max_t |R_i(t)|                       -- absolute error on the exponent
  ewgt_i = max_t W_i(t) |R_i(t)|,  W = exp(Re S_tot)  [x |phi_K| if --phik]
because the mass integrand is e^{Re S} (phi_K cos psi - ...) and an error in
the exponent enters the integrand multiplied by exactly that envelope.

Usage
  python3 spectral.py --sub <sub_*.npz> --tag jpsigun --out spec_jpsigun.npz
"""
import argparse, json, os, time
import numpy as np

RLIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def block_defs(keys):
    fams = [k for k in ("Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im")
            if k in keys]
    blocks = [(k, [k]) for k in fams]
    if "Sio_re" in fams and "Sio_im" in fams:
        blocks.append(("IO", ["Sio_re", "Sio_im"]))
    if "Srad_re" in fams and "Srad_im" in fams:
        blocks.append(("RAD", ["Srad_re", "Srad_im"]))
    blocks.append(("ALL", fams))
    return fams, blocks


RUNS = "/work/submit/david_w/ZMass/calibration_studies/resolution/runs"
# the phi_K tables cf_masslik_fit.py already built on the FULL caches (its own
# cache-key convention: kernel name, n kernel samples, tmax_abs = 14/sig.min()
# over the full cache).  Reused verbatim so the weight here is the SAME phi_K
# the published fits used.
PHIK_TAB = {
    "jpsigun": f"{RUNS}/masslikfit_phiKtab_cf_masskernel_jpsigun_ul16_260903x"
               f"_m0_300243_1.635557262348e+03.npz",
    "btojpsix": f"{RUNS}/masslikfit_phiKtab_cf_masskernel_btojpsix_v3_260903x"
                f"_m0_128687_1.271935440271e+03.npz",
}


def phik_abs(tag, TG, sig):
    """|phi_K(t/sigma_i)| on the standardized grid, from the cached table."""
    z = np.load(PHIK_TAB[tag])
    tabs, tab = z["tabs"], z["phiK_tab"]
    tgi = TG[None, :] / sig[:, None]
    return np.abs(np.interp(tgi, tabs, tab.real)
                  + 1j * np.interp(tgi, tabs, tab.imag))


def analyse(X, W, fams, nt, rlist, scaled=False, log=print):
    """SVD of the centred block; per-rank error tables.

    Returns dict with sv, and per-rank per-family error quantiles.
    X : (n, nfam*nt) float64.  W : (n, nt) weight.  scaled: divide each row by
    its own L2 norm first (then r+1 scalars are stored instead of r).
    """
    n, m = X.shape
    s_i = np.ones(n)
    if scaled:
        s_i = np.linalg.norm(X, axis=1)
        s_i[s_i == 0] = 1.0
        X = X / s_i[:, None]
    mu = X.mean(axis=0)
    Xc = X - mu
    t0 = time.time()
    G = Xc.T @ Xc
    ev, V = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1]
    ev, V = np.clip(ev[o], 0, None), V[:, o]
    sv = np.sqrt(ev)
    log(f"    gram+eigh {m}x{m} in {time.time()-t0:.1f} s; "
        f"sv[0:4]={sv[:4]}")
    A = Xc @ V[:, :max(rlist)]                       # (n, rmax)
    out = dict(sv=sv, nfam=len(fams), nt=nt, ranks=np.array(rlist, int))
    for tag in ("eabs", "ewgt"):
        for f in fams:
            out[f"{tag}_{f}_max"] = np.zeros(len(rlist))
            out[f"{tag}_{f}_q999"] = np.zeros(len(rlist))
            out[f"{tag}_{f}_med"] = np.zeros(len(rlist))
    for ir, r in enumerate(rlist):
        R = Xc - A[:, :r] @ V[:, :r].T
        if scaled:
            R = R * s_i[:, None]
        aR = np.abs(R)
        for j, f in enumerate(fams):
            sl = aR[:, j * nt:(j + 1) * nt]
            ea = sl.max(axis=1)
            ew = (sl * W).max(axis=1)
            for tag, v in (("eabs", ea), ("ewgt", ew)):
                out[f"{tag}_{f}_max"][ir] = v.max()
                out[f"{tag}_{f}_q999"][ir] = np.quantile(v, 0.999)
                out[f"{tag}_{f}_med"][ir] = np.median(v)
        del R, aR
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfit", type=int, default=40000)
    ap.add_argument("--phik", action="store_true",
                    help="fold |phi_K| into the weight (mass caches only)")
    a = ap.parse_args()
    d = np.load(a.sub)
    TG = d["tgrid"].astype(np.float64)
    nt = len(TG)
    keys = set(d.files)
    fams, blocks = block_defs(keys)
    n = min(a.nfit, len(d["z"]))
    print(f"[{a.tag}] n={n} families={fams}")
    sig = d["sigma"][:n].astype(np.float64)
    vgf = d["vgf"][:n].astype(np.float64)
    M = {f: d[f][:n].astype(np.float64) for f in fams}
    # total real exponent at the nominal k_f = 1 (Sdel is a DIAGNOSTIC family
    # of the track cache, not part of the published model, so it is excluded
    # from the envelope)
    Sre = -0.5 * vgf[:, None] * TG[None, :] ** 2
    for f in ("Sms", "Sio_re", "Srad_re"):
        if f in M:
            Sre = Sre + M[f]
    W = np.exp(Sre)
    if a.phik:
        W = W * phik_abs(a.tag, TG, sig)
    print(f"  weight W: median over candidates at t=14 -> {np.median(W[:, -1]):.3e}; "
          f"t where median W = 1e-3 -> "
          f"{TG[np.searchsorted(-np.median(W, axis=0), -1e-3)]:.2f}")
    # float32 quantisation of the cache itself, for reference
    # the cache is float32, so the exponent it stores is itself only known to
    # one ulp of float32 at its own magnitude -- the floor of any compression
    # study on these files
    q32 = max(float(np.spacing(np.float32(np.abs(M[f]).max()))) for f in fams)
    print(f"  float32 cache ulp at max|S|: {q32:.3e}   "
          f"(max|S| = {max(np.abs(M[f]).max() for f in fams):.1f})")

    res = dict(tag=a.tag, TG=TG, fams=np.array(fams), n=n, q32=q32,
               Wmed=np.median(W, axis=0))
    for bname, bfams in blocks:
        print(f"  block {bname} ({len(bfams)} fam)")
        X = np.concatenate([M[f] for f in bfams], axis=1)
        rl = [r for r in RLIST if r <= X.shape[1]]
        o = analyse(X, W, bfams, nt, rl, scaled=False)
        for k, v in o.items():
            res[f"{bname}/{k}"] = v
        if bname == "ALL":
            o2 = analyse(X, W, bfams, nt, rl, scaled=True)
            for k, v in o2.items():
                res[f"ALLSC/{k}"] = v
        del X
    np.savez(a.out, **{k: np.asarray(v) for k, v in res.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robustness of the sigma-self-consistency explanation, and the candidate arm.

(1) does the T2 answer depend on how sigma_bar is defined?  Three cell
    definitions, one of them TRUTH-ONLY (no reconstructed quantity at all).
(2) the residual charge-odd RELATIVE momentum bias per eta bin, three trims.
(3) the even-in-q part S: constant in pull units (pull-side) or constant in
    absolute q/p (a sagitta bias)?  Its two representations, both samples.
(4) the candidate (dimuon-mass) arm: the same self-consistency coefficient,
    Cov(z, sigma)/<sigma>, on the mass-pair caches.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import sigma_pull as SP                                       # noqa: E402

MJ = 3.0969


def qbin(v, nb):
    return np.clip(np.digitize(v, np.quantile(v, np.linspace(0, 1, nb + 1))[1:-1]),
                   0, nb - 1).astype(np.int64)


def ubin(v, lo, hi, nb):
    return np.clip(np.digitize(v, np.linspace(lo, hi, nb + 1)[1:-1]), 0, nb - 1
                   ).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    L = []
    P = L.append
    samples = [("mu gun pT 20-60", "runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz"),
               ("mu gun pT 2-20", "runs/cf_trackres_mugun_lowpt_260905d_m0_k0.npz")]
    P("## 1  does T2 depend on the sigma_bar cell definition?")
    P("| sample | cells | ncell | A(z) | A(z_bar) | err |")
    P("|---|---|---|---|---|---|")
    for tag, p in samples:
        d = np.load(p)
        z = d["z"]; sig = d["sigma"]; q = np.sign(d["charge"]).astype(float)
        eta = d["eta"].astype(float); phi = d["phi"].astype(float)
        pgen = d["genpt"] * np.cosh(eta); nvh = d["nvalidhits"]
        eps = z * sig
        m5 = np.abs(z) < 5.
        defs = [("gen p (20q) x |eta| (20q) x nhits",
                 [qbin(np.log(pgen), 20), qbin(np.abs(eta), 20),
                  np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30)]),
                ("gen p (20q) x eta (40 uniform)  TRUTH ONLY",
                 [qbin(np.log(pgen), 20), ubin(eta, -2.4, 2.4, 40)]),
                ("gen p (20q) x eta (20) x phi (12)  TRUTH ONLY",
                 [qbin(np.log(pgen), 20), ubin(eta, -2.4, 2.4, 20),
                  ubin(phi, -np.pi, np.pi, 12)])]
        for nm, cols in defs:
            cid = SP.cellid(*cols); nc = int(cid.max()) + 1
            sb = SP.cellmean(sig, cid, nc, leave_one_out=True)
            zb = eps / sb
            A = float((q * z)[m5].mean()); Ab = float((q * zb)[m5].mean())
            e = float((q * zb)[m5].std(ddof=1) / np.sqrt(m5.sum()))
            P(f"| {tag} | {nm} | {nc} | {A:+.5f} | {Ab:+.5f} | {e:.5f} |")

    P("")
    P("## 2  residual charge-odd RELATIVE momentum bias  <p_gen/p_fit> - 1  [1e-4]")
    P("The raw quantity, no 1/sigma anywhere. The CF model's own mean is zero "
      "by construction, so any nonzero entry is a genuine momentum bias.")
    for tag, p in samples:
        d = np.load(p)
        z = d["z"]; sig = d["sigma"]; q = np.sign(d["charge"]).astype(float)
        eta = d["eta"].astype(float); pgen = d["genpt"] * np.cosh(eta)
        rel = q * z * sig * pgen
        P("")
        P(f"### {tag}")
        P("| eta | |z|<3 | |z|<5 | |z|<10 | no trim |")
        P("|---|---|---|---|---|")
        e_ = np.linspace(-2.4, 2.4, 13)
        b = np.clip(np.digitize(eta, e_[1:-1]), 0, 11)
        for k in range(12):
            row = []
            for T in (3., 5., 10., 1e9):
                m = (b == k) & (np.abs(z) < T)
                v = rel[m]
                row.append(f"{1e4*v.mean():+.2f}+-{1e4*v.std(ddof=1)/np.sqrt(len(v)):.2f}")
            P(f"| {e_[k]:+.1f}..{e_[k+1]:+.1f} | " + " | ".join(row) + " |")
        row = []
        for T in (3., 5., 10., 1e9):
            m = np.abs(z) < T
            v = rel[m]
            row.append(f"{1e4*v.mean():+.2f}+-{1e4*v.std(ddof=1)/np.sqrt(len(v)):.2f}")
        P("| **all** | " + " | ".join(row) + " |")

    P("")
    P("## 3  the even-in-q part S, in the two representations")
    P("A sagitta bias is a CONSTANT d(q/p); a pull-side effect is constant in "
      "pull units. sigma differs by 3x between the two samples, so the two "
      "readings are distinguishable in principle.")
    P("| sample | S = <z>_{|z|<5} | <eps>_{|z|<5} [1/GeV] |")
    P("|---|---|---|")
    for tag, p in samples:
        d = np.load(p)
        z = d["z"]; sig = d["sigma"]; eps = z * sig
        m = np.abs(z) < 5.
        P(f"| {tag} | {z[m].mean():+.5f} +- {z[m].std(ddof=1)/np.sqrt(m.sum()):.5f} "
          f"| {eps[m].mean():+.3e} +- {eps[m].std(ddof=1)/np.sqrt(m.sum()):.3e} |")

    P("")
    P("## 4  the candidate (dimuon mass) arm")
    P("z_m = (Jpsi_mass - Jpsigen_mass)/Jpsi_sigmamass, and Jpsi_sigmamass is "
      "again the fit's OWN error, so the same conditioning defect exists. Its "
      "coefficient is the inclusive Cov(z, sigma)/<sigma> (an upper bound: it "
      "also carries between-cell terms).")
    P("| cache | n | <z> | <dm>/m [1e-4] | Cov(z,sigma)/<sigma> | med sigma/m |")
    P("|---|---|---|---|---|---|")
    for tag, p in (("jpsigun 260903x", "runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz"),
                   ("jpsigun 260905d", "runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz"),
                   ("btojpsix v3", "runs/cf_masspairs_btojpsix_v3_260903x_m0.npz")):
        if not os.path.exists(p):
            continue
        d = np.load(p); z = d["z"]; s = d["sigma"]
        # the gun mass caches carry a handful of pathological sigma (up to
        # 1e3 x m) which own the inclusive covariance; window at 5x the median
        m = (np.abs(z) < 5.) & (s < 5. * np.median(s))
        P(f"| {tag} | {int(m.sum())} | {z[m].mean():+.5f} +- "
          f"{z[m].std(ddof=1)/np.sqrt(m.sum()):.5f} | "
          f"{1e4*(z*s)[m].mean()/MJ:+.2f} +- "
          f"{1e4*(z*s)[m].std(ddof=1)/np.sqrt(m.sum())/MJ:.2f} | "
          f"{np.cov(z[m], s[m])[0,1]/s[m].mean():+.5f} | "
          f"{np.median(s[m])/MJ:.5f} |")

    P("")
    P("## 5  the prediction A = -a + A_model, BIN BY BIN, no free parameter")
    P("a is re-measured inside each bin as the in-cell (gen p x |eta| x nhits) "
      "regression slope of ln sigma on q z; A_model is the model's own trimmed "
      "mean in the same bin (from oddmoment/diff.py). No fit anywhere.")
    for tag, p_, dfile in (
            ("mu gun pT 20-60", "runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz",
             "oddmoment/out/diff_ul16.npz"),
            ("mu gun pT 2-20", "runs/cf_trackres_mugun_lowpt_260905d_m0_k0.npz",
             "oddmoment/out/diff_lowpt.npz")):
        if not os.path.exists(dfile):
            P(f"(skip {tag}: run oddmoment/diff.py first)")
            continue
        d = np.load(p_); dm = np.load(dfile)
        z = d["z"]; sig = d["sigma"]; q = np.sign(d["charge"]).astype(float)
        eta = d["eta"].astype(float); pgen = d["genpt"] * np.cosh(eta)
        nvh = d["nvalidhits"]
        cid = SP.cellid(qbin(np.log(pgen), 20), qbin(np.abs(eta), 20),
                        np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30))
        nc = int(cid.max()) + 1
        ls = np.log(sig)
        lsc = ls - SP.cellmean(ls, cid, nc)
        yc = (q * z) - SP.cellmean(q * z, cid, nc)
        m5 = np.abs(z) < 5.
        for vname, v in (("abseta", np.abs(eta)), ("genpt", d["genpt"])):
            rows = dm[vname]; edges = dm[vname + "_edges"]
            nb = len(edges) - 1
            b = np.clip(np.digitize(v, edges[1:-1]), 0, nb - 1)
            P("")
            P(f"### {tag}, bins of {vname}")
            P("| bin | n | a_bin | A_model | pred = -a+A_mod | A_data | "
              "data-pred | err |")
            P("|---|---|---|---|---|---|---|---|")
            for k in range(nb):
                m = (b == k) & m5
                if m.sum() < 500:
                    continue
                ab = float((lsc[m] * yc[m]).sum() / (yc[m] ** 2).sum())
                # A_model(trim) column of diff.py rows: (ctr,n,Adu,Amu,Adt,Amt,eA,Sdt)
                row = rows[k]
                Am = float(row[5])
                Ad = float((q * z)[m].mean())
                e = float((q * z)[m].std(ddof=1) / np.sqrt(m.sum()))
                P(f"| {edges[k]:.3g}..{edges[k+1]:.3g} | {int(m.sum())} | "
                  f"{ab:+.5f} | {Am:+.5f} | {-ab+Am:+.5f} | {Ad:+.5f} | "
                  f"{Ad-(-ab+Am):+.5f} | {e:.5f} |")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()

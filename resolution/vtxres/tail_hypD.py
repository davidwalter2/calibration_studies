#!/usr/bin/env python3
"""TAIL study, hypothesis D: the PIXEL HIT-NOISE tail, and a dump of the
tail candidates.

The per-hit study found a non-Gaussian hit-noise tail that neither arm models.
Both beam residuals and the vertex residual are pixel-hit dominated at Z
momenta (`hit` carries 0.60 / 0.44 / 0.66 of their variance), so if that tail
is the cause the excess must scale with how much of a candidate's variance
sits in the pixel classes -- and the J/psi gun, which has the same hit model,
must show it too unless its kinematics avoid those classes.

usage:
  python3 tail_hypD.py --npz <tail/dy_bs_final.npz> [--gun <tail/gun_vtxon.npz>]
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402

CLS = ['pix_x_q0', 'pix_x_q1', 'pix_x_q2', 'pix_x_q3',
       'pix_y_q0', 'pix_y_q1', 'pix_y_q2', 'pix_y_q3',
       'str_N1_lo', 'str_N1_hi', 'str_N2_lo', 'str_N2_hi', 'str_N3_lo',
       'str_N3_hi', 'str_N4_lo', 'str_N4_hi', 'str_N5_lo', 'str_N5_hi']
PIX = list(range(8))
PIXX = list(range(4))


def shares(M, sel):
    v = M[sel]
    tot = v.sum(axis=1, keepdims=True)
    tot[tot <= 0] = np.nan
    return np.nanmedian(v / tot, axis=0), np.nanmean(v / tot, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", default="")
    ap.add_argument("--dump", type=int, default=30)
    a = ap.parse_args()

    d = dict(np.load(a.npz, allow_pickle=False))
    base, chi2n, nl = TC.baseline(d, verbose=False)
    basenoz, _, _ = TC.baseline(d, max_abs_vtxz=0.0, verbose=False)
    cls, names, aux = GB.classify(d)
    isig = names.index("signal")
    z1, z2 = d["bsz"][:, 0], d["bsz"][:, 1]
    zv = np.asarray(d["vtxz"], float)
    sig = base & (cls == isig)
    signoz = basenoz & (cls == isig)

    for tag, M, z in (("beam z_1", d["bsx_hitv"], z1),
                      ("beam z_2", d["bsy_hitv"], z2),
                      ("vertex z_v", d["vtx_hitv"], zv)):
        selall = sig if tag != "vertex z_v" else signoz
        tot = M.sum(axis=1)
        pixfrac = np.where(tot > 0, M[:, PIX].sum(axis=1) / np.maximum(tot, 1e-30), np.nan)
        print(f"\n=== {tag}: the hit-class composition, core against tail")
        med_all, _ = shares(M, selall)
        t = selall & (np.abs(z) > 3)
        med_t, _ = shares(M, t)
        print(f"  {'class':<12s}{'share, all':>14s}{'share, |z|>3':>16s}{'ratio':>9s}")
        order = np.argsort(-med_all)
        for i in order[:8]:
            print(f"  {CLS[i]:<12s}{med_all[i]:>14.4f}{med_t[i]:>16.4f}"
                  f"{med_t[i]/max(med_all[i],1e-12):>9.3f}")
        print(f"  {'PIXEL total':<12s}{med_all[PIX].sum():>14.4f}"
              f"{med_t[PIX].sum():>16.4f}"
              f"{med_t[PIX].sum()/max(med_all[PIX].sum(),1e-12):>9.3f}")
        print(f"  {'pix local x':<12s}{med_all[PIXX].sum():>14.4f}"
              f"{med_t[PIXX].sum():>16.4f}")

        print(f"  -- does the tail follow the PIXEL share of the variance?")
        e = np.nanpercentile(pixfrac[selall], [0, 20, 40, 60, 80, 100])
        for i in range(len(e) - 1):
            b = selall & (pixfrac >= e[i]) & (pixfrac <= e[i + 1])
            if b.sum() < 50:
                continue
            print(f"     pix share {e[i]:.3f}..{e[i+1]:.3f}  N {int(b.sum()):6d}"
                  f"   P(|z|>3) {TC.pm(int((np.abs(z[b])>3).sum()), int(b.sum()))}"
                  f"   P(|z|>5) {TC.pm(int((np.abs(z[b])>5).sum()), int(b.sum()))}"
                  f"   Var {np.var(z[b]):.4f}")

    if a.gun and os.path.exists(a.gun):
        g = dict(np.load(a.gun, allow_pickle=False))
        gb, _, _ = TC.baseline(g, need_bs=False, max_abs_vtxz=0.0, verbose=False)
        gz = np.asarray(g["vtxz"], float)
        M = g["vtx_hitv"]
        tot = M.sum(axis=1)
        pf = np.where(tot > 0, M[:, PIX].sum(axis=1) / np.maximum(tot, 1e-30), np.nan)
        med_g, _ = shares(M, gb)
        print(f"\n=== the J/psi GUN, vertex residual ({int(gb.sum())} candidates)"
              f"   -- the same hit model, no tail")
        TC.moments(gz[gb], "z_v (gun)")
        print(f"  median PIXEL share of Var(z_v): gun {np.nanmedian(pf[gb]):.4f}"
              f"   DY {np.nanmedian(np.where(d['vtx_hitv'].sum(1)>0, d['vtx_hitv'][:, PIX].sum(1)/np.maximum(d['vtx_hitv'].sum(1),1e-30), np.nan)[signoz]):.4f}")
        print(f"  {'class':<12s}{'gun share':>12s}{'DY share':>12s}")
        dm, _ = shares(d["vtx_hitv"], signoz)
        for i in np.argsort(-med_g)[:8]:
            print(f"  {CLS[i]:<12s}{med_g[i]:>12.4f}{dm[i]:>12.4f}")
        print(f"  -- the gun's tail against its own pixel share")
        e = np.nanpercentile(pf[gb], [0, 25, 50, 75, 100])
        for i in range(len(e) - 1):
            b = gb & (pf >= e[i]) & (pf <= e[i + 1])
            if b.sum() < 50:
                continue
            print(f"     pix share {e[i]:.3f}..{e[i+1]:.3f}  N {int(b.sum()):7d}"
                  f"   P(|z_v|>3) {TC.pm(int((np.abs(gz[b])>3).sum()), int(b.sum()))}"
                  f"   P(|z_v|>5) {TC.pm(int((np.abs(gz[b])>5).sum()), int(b.sum()))}")

    # ---------------- the dump ----------------
    print("\n=== THE TAIL CANDIDATES (diagnostic selection |z_1|>5 or |z_2|>5 "
          "or |z_v|>5, gen SIGNAL)")
    t = np.flatnonzero(signoz & ((np.abs(z1) > 5) | (np.abs(z2) > 5) |
                                 (np.abs(zv) > 5)))
    order = t[np.argsort(-np.maximum(np.abs(z1[t]), np.abs(z2[t])))]
    hdr = (f"  {'evt':>12s}{'z_1':>8s}{'z_2':>8s}{'z_v':>8s}{'chi2/n':>8s}"
           f"{'nv+':>5s}{'nv-':>5s}{'npx+':>5s}{'npx-':>5s}{'pT+':>8s}{'pT-':>8s}"
           f"{'eta+':>7s}{'eta-':>7s}{'dphi':>7s}{'mass':>8s}{'sig_v um':>10s}"
           f"{'mult':>5s}{'pixsh':>7s}")
    print(hdr)
    pf1 = np.where(d["bsx_hitv"].sum(1) > 0,
                   d["bsx_hitv"][:, PIX].sum(1) / np.maximum(d["bsx_hitv"].sum(1), 1e-30), np.nan)
    dphi = np.abs(np.mod(d["phi_plus"] - d["phi_minus"] + np.pi, 2*np.pi) - np.pi)
    for i in order[:a.dump]:
        print(f"  {int(d['event'][i]):>12d}{z1[i]:>8.2f}{z2[i]:>8.2f}{zv[i]:>8.2f}"
              f"{chi2n[i]:>8.2f}{int(d['nvalid_plus'][i]):>5d}{int(d['nvalid_minus'][i]):>5d}"
              f"{int(d['nvalidpixel_plus'][i]):>5d}{int(d['nvalidpixel_minus'][i]):>5d}"
              f"{d['pt_plus'][i]:>8.1f}{d['pt_minus'][i]:>8.1f}"
              f"{d['eta_plus'][i]:>7.2f}{d['eta_minus'][i]:>7.2f}{dphi[i]:>7.3f}"
              f"{d['mass'][i]:>8.2f}{d['vtxsig'][i]*1e4:>10.2f}"
              f"{int(aux['mult'][i]):>5d}{pf1[i]:>7.3f}")
    print(f"  ({len(t)} candidates in the set)")
    print("\n  -- and how the tail set compares with the core")
    for nm, q in (("chi2/ndof", chi2n), ("sigma_v um", d["vtxsig"] * 1e4),
                  ("nvalid weak", nl), ("pT soft", np.minimum(d["pt_plus"], d["pt_minus"])),
                  ("|eta| max", np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"]))),
                  ("dphi", dphi), ("mass", d["mass"]),
                  ("bsmeig", d["bsmeig"]), ("pixel share", pf1),
                  ("event multiplicity", aux["mult"].astype(float)),
                  ("nTrueInt", d["ntrueint"]),
                  ("sigma_m GeV", d["sigmamass"])):
        sel = signoz
        print(f"  {nm:<20s} core median {np.nanmedian(q[sel]):10.4f}   "
              f"tail median {np.nanmedian(q[t]):10.4f}   "
              f"tail p90 {np.nanpercentile(q[t],90):10.4f}")


if __name__ == "__main__":
    main()

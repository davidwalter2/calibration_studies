#!/usr/bin/env python3
"""TAIL study, hypothesis A: is the tail the LUMINOUS REGION itself?

The beam residual compares the FITTED vertex with the beam-spot RECORD.  If
the TRUE (gen) vertex is far from the record's line, the pull is large with no
tracking problem at all.  This splits the residual into the two pieces it is
made of,

    r_bs = (x_gen - x_line(z_gen))  +  (x_fit - x_gen)
         =      delta_lum          +      delta_reco

and asks which of them carries the tail.  Nothing is fitted.

usage: python3 tail_hypA.py --npz <tail/dy_bs_final.npz> [--tag dy]
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default="dy")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = dict(np.load(a.npz, allow_pickle=False))
    n0 = len(d["run"])
    print(f"# {a.tag}: {n0} candidates from {a.npz}")
    base, chi2n, nl = TC.baseline(d)

    bsz, bsres, bscov = d["bsz"], d["bsres"], d["bscov"]
    bsvtx, spot, slope, wid = d["bsvtx"], d["bsspot"], d["bsslope"], d["bswidth"]
    z1, z2, zv = bsz[:, 0], bsz[:, 1], np.asarray(d["vtxz"], float)

    print("\n=== gate: the exported pull IS the residual over its covariance")
    ok = np.isfinite(bsres[:, 0]) & (np.abs(bsres[:, 0]) > 1e-6)
    z1chk = bsres[:, 0] / np.sqrt(bscov[:, 0])
    print(f"  max |z_1 - r_x/sqrt(Cov_xx)|  {np.nanmax(np.abs(z1[ok]-z1chk[ok])):.3e}")
    line_x = spot[:, 0] + slope[:, 0] * (bsvtx[:, 2] - spot[:, 2])
    dd = np.abs(bsvtx[:, 0] - line_x - bsres[:, 0])
    print(f"  max |(x_v - line_x) - r_x|    {np.nanmax(dd[ok])*1e4:.4f} um "
          f"  median {np.nanmedian(dd[ok])*1e4:.6f} um")

    print("\n=== the beam-spot RECORD the maker used")
    for i, nm in enumerate(("x0", "y0", "z0")):
        u = np.unique(np.round(spot[:, i], 9))
        print(f"  {nm:<8s} {u[0]*1e4 if i<2 else u[0]:+12.4f} "
              f"{'um' if i < 2 else 'cm'}  ({len(u)} distinct)")
    for i, nm in enumerate(("sigma_x", "sigma_y", "sigma_z")):
        u = np.unique(np.round(wid[:, i], 9))
        print(f"  {nm:<8s} {u[0]*1e4 if i<2 else u[0]:12.4f} "
              f"{'um' if i < 2 else 'cm'}  ({len(u)} distinct)")
    for i, nm in enumerate(("dxdz", "dydz")):
        u = np.unique(np.round(slope[:, i], 12))
        print(f"  {nm:<8s} {u[0]:+12.4e}       ({len(u)} distinct)")

    gx, gy, gz = d["genvx"], d["genvy"], d["genvz"]
    hasgen = (gx > -90) & (gy > -90) & (gz > -90)
    gline_x = spot[:, 0] + slope[:, 0] * (gz - spot[:, 2])
    gline_y = spot[:, 1] + slope[:, 1] * (gz - spot[:, 2])
    dlum_x, dlum_y = gx - gline_x, gy - gline_y
    glum_x, glum_y = dlum_x / wid[:, 0], dlum_y / wid[:, 1]
    # THE LEAVE-ONE-OUT VERTEX.  `Jpsi_bsvtx` is the CONSTRAINED vertex; the
    # residual is the innovation e_B = (I + A M^-1) rho_B, so the vertex the
    # residual is built from is x_line + r_bs and NOT x_v.  (The gate above
    # measures the difference: median 13 um.)
    loo_x = line_x + bsres[:, 0]
    loo_y = spot[:, 1] + slope[:, 1] * (bsvtx[:, 2] - spot[:, 2]) + bsres[:, 1]
    dreco_x, dreco_y = loo_x - gx, loo_y - gy

    cls, names, aux = GB.classify(d)
    isig = names.index("signal")
    m = base & hasgen
    sig = m & (cls == isig)
    print(f"\n# baseline {int(base.sum())}, with a gen vertex {int(m.sum())}, "
          f"gen SIGNAL {int(sig.sum())}")
    print("# classes on the baseline: " + ", ".join(
        f"{c} {int((base & (cls == i)).sum())}" for i, c in enumerate(names)))

    for lbl, sel in (("baseline", m), ("gen SIGNAL only", sig)):
        print(f"\n=== the two pieces of r_bs -- {lbl} ({int(sel.sum())} cand)")
        TC.moments(glum_x[sel], "gen offset x / sigBS_rec")
        TC.moments(glum_y[sel], "gen offset y / sigBS_rec")
        TC.moments(dlum_x[sel] / np.sqrt(bscov[sel, 0]), "delta_lum,x / sqrt(Cov)")
        TC.moments(dreco_x[sel] / np.sqrt(bscov[sel, 0]), "delta_reco,x / sqrt(Cov)")
        print(f"  corr(delta_lum,x, delta_reco,x) = "
              f"{np.corrcoef(dlum_x[sel], dreco_x[sel])[0,1]:+.4f}"
              f"   ;  sum of the two variances "
              f"{(np.var(dlum_x[sel])+np.var(dreco_x[sel]))/np.mean(bscov[sel,0]):.4f}"
              f"  against Var(z_1) {np.var(z1[sel]):.4f}")
        TC.moments(z1[sel], "z_1 (global x pull)")
        TC.moments(z2[sel], "z_2")
        TC.moments(zv[sel], "z_v")
        for nm, q in (("delta_lum,x", dlum_x), ("delta_lum,y", dlum_y),
                      ("delta_reco,x", dreco_x), ("delta_reco,y", dreco_y),
                      ("r_bs,x", dlum_x + dreco_x)):
            v = q[sel] * 1e4
            print(f"  {nm:<14s} rms {v.std():9.3f} um   "
                  f"MAD {1.4826*np.median(np.abs(v-np.median(v))):8.3f} um   "
                  f"P(|.|>50um) {(np.abs(v)>50).mean():.5f}")

    print("\n=== is the SIMULATED luminous region Gaussian?  gen offsets over "
          "their OWN robust width, gen SIGNAL only")
    for nm, v in (("x", dlum_x[sig]), ("y", dlum_y[sig])):
        s = 1.4826 * np.median(np.abs(v - np.median(v)))
        TC.moments((v - np.median(v)) / s, f"gen offset {nm} / MAD")
    print(f"  record sigma_x / simulated MAD = "
          f"{wid[0,0]/(1.4826*np.median(np.abs(dlum_x[sig]-np.median(dlum_x[sig])))):.4f}")
    print(f"  record sigma_y / simulated MAD = "
          f"{wid[0,1]/(1.4826*np.median(np.abs(dlum_y[sig]-np.median(dlum_y[sig])))):.4f}")

    print("\n=== THE TAIL CANDIDATES (a DIAGNOSTIC selection, not a cut)")
    for nm, z in (("z_1", z1), ("z_2", z2), ("z_v", zv)):
        for thr in (5.0,):
            t = m & (np.abs(z) > thr)
            print(f"\n  -- |{nm}| > {thr:g} : {int(t.sum())} candidates "
                  f"({int((t & (cls == isig)).sum())} gen signal)")
            if t.sum() == 0:
                continue
            for lbl, q in (("|gen pull| max(x,y)",
                            np.maximum(np.abs(glum_x), np.abs(glum_y))),
                           ("|delta_lum,x| um", np.abs(dlum_x) * 1e4),
                           ("|delta_lum,y| um", np.abs(dlum_y) * 1e4),
                           ("|delta_reco,x| um", np.abs(dreco_x) * 1e4),
                           ("|delta_reco,y| um", np.abs(dreco_y) * 1e4)):
                print(f"     {lbl:<22s} median {np.median(q[t]):10.3f}   "
                      f"(baseline median {np.median(q[m]):10.3f})")
            for gt in (3, 5):
                g = np.maximum(np.abs(glum_x), np.abs(glum_y))
                print(f"     gen vertex beyond {gt} sigma of the line: "
                      f"{TC.pm(int((g[t] > gt).sum()), int(t.sum()))}"
                      f"   [baseline {TC.pm(int((g[m] > gt).sum()), int(m.sum()))}]")

    print("\n=== A z-dependent offset would be a SLOPE mismatch "
          "(gen SIGNAL, robust IRLS fit)")
    for nm, v, w in (("dxdz", dlum_x, 0), ("dydz", dlum_y, 1)):
        c, e, nfit = TC.robust_line(gz[sig] - spot[sig, 2], v[sig])
        print(f"  simulated {nm} - record {nm} = {c[1]:+.3e} +- {e[1]:.3e}"
              f"   (record {slope[0,w]:+.3e}; {nfit} of {int(sig.sum())} in fit)")
        print(f"     -> centroid offset {c[0]*1e4:+.4f} +- {e[0]*1e4:.4f} um"
              f"   ;  the slope residual over |z|<3.6cm is "
              f"{abs(c[1])*3.6/wid[0,w]:.4f} sigma_BS")
    print("\n  <z_1> and <gen pull> against the gen vertex z (gen SIGNAL)")
    edges = np.percentile(gz[sig], np.linspace(0, 100, 9))
    print(f"  {'z bin (cm)':<20s}{'N':>7s}{'<z_1>':>20s}{'<gen pull x>':>22s}")
    for i in range(len(edges) - 1):
        b = sig & (gz >= edges[i]) & (gz < edges[i + 1])
        if b.sum() < 5:
            continue
        # trimmed means, so one displaced vertex does not own the bin
        def tmean(v):
            v = np.sort(v[b]); k = max(int(0.01 * len(v)), 1)
            v = v[k:-k]
            return v.mean(), v.std() / np.sqrt(len(v))
        m1, e1 = tmean(z1); m2, e2 = tmean(glum_x)
        print(f"  {edges[i]:+7.2f}..{edges[i+1]:+7.2f}{int(b.sum()):>7d}"
              f"{m1:>+14.4f} +- {e1:.4f}{m2:>+16.4f} +- {e2:.4f}")

    if a.out:
        np.savez_compressed(a.out, glum_x=glum_x, glum_y=glum_y,
                            dlum_x=dlum_x, dlum_y=dlum_y, dreco_x=dreco_x,
                            dreco_y=dreco_y, base=base, hasgen=hasgen,
                            cls=cls, z1=z1, z2=z2, zv=zv, gz=gz)
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()

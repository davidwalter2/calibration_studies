#!/usr/bin/env python3
"""TAIL study, hypotheses B and C.

B -- ALIGNMENT.  `tail_geom.py` showed the DY refit runs on the UL16 MC
tracker alignment while the J/psi gun runs on the ideal geometry, and that in
MC the Geant4 tracker IS the ideal one, so the aligned geometry displaces
every reconstructed hit from where the particle crossed.  Two signatures are
tested here, both of which a resolution effect cannot produce:
  (i)  STRUCTURE: the MEAN of each pull in bins of the legs' phi and eta;
  (ii) EXPOSURE: whether the tail candidates' legs actually cross the badly
       displaced modules, read off `globalidxv` against the `runtree`.

C -- THIN LEGS.  The same per-leg module list gives the first-hit radius, the
BPix-L1 presence and the module count actually used by the fit, which is what
distinguishes a MiniAOD-truncated hit list from a genuinely short track; the
reco/gen pT ratio separates a mis-measured leg from a short one.

usage:
  python3 tail_hypBC.py --npz <tail/dy_bs_final.npz> \
      --files '<prod>/task_*/globalcor_*.root' --geom <geom_dy_vs_gun.npz>
"""
import argparse, glob, os, sys
import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402

SUBDET = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}


def module_map(fn, geomnpz):
    """iidx -> module row, and the module's displacement from IDEAL."""
    rt = uproot.open(fn + ":runtree")
    a = rt.arrays(["iidx", "parmtype", "rawdetid", "subdet", "layer",
                   "rho", "phi", "z"], library="np")
    g = np.load(geomnpz)
    disp = np.sqrt(g["dx"] ** 2 + g["dy"] ** 2 + g["dz"] ** 2)
    dmap = dict(zip(g["rawdetid"].tolist(), disp.tolist()))
    rmap = dict(zip(g["rawdetid"].tolist(), g["drphi"].tolist()))
    nmax = int(a["iidx"].max()) + 1
    out = dict(det=np.full(nmax, -1, np.int64),
               sub=np.full(nmax, -1, np.int8),
               lay=np.full(nmax, -99, np.int16),
               rho=np.full(nmax, np.nan),
               phi=np.full(nmax, np.nan),
               zz=np.full(nmax, np.nan),
               disp=np.full(nmax, np.nan),
               drphi=np.full(nmax, np.nan),
               align=np.zeros(nmax, bool))
    al = a["parmtype"] <= 5
    ii = a["iidx"]
    out["det"][ii] = a["rawdetid"]
    out["sub"][ii] = a["subdet"]
    out["lay"][ii] = a["layer"]
    out["rho"][ii] = a["rho"]
    out["phi"][ii] = a["phi"]
    out["zz"][ii] = a["z"]
    out["align"][ii[al]] = True
    out["disp"][ii] = [dmap.get(int(x), np.nan) for x in a["rawdetid"]]
    out["drphi"][ii] = [rmap.get(int(x), np.nan) for x in a["rawdetid"]]
    return out


def per_leg(files, M, nmax=0):
    """Per candidate: the modules the fit attaches an alignment parameter to.

    CAVEAT, measured and not worked around: in the TWO-TRACK fit both legs'
    `jacRef` are non-zero on EVERY module of the pair, because the common
    vertex couples them -- 41 modules for a 15 + 17 hit pair -- so the "plus"
    and "minus" columns below are NOT a per-leg split.  They are the PAIR's
    module list, computed twice.  The per-leg quantities quoted in section 16
    (`nvalid`, `nvalidpixel`, `nhits`, the reco/gen `pT` ratio) come from the
    tree's own per-leg branches, which are exact; only `rhomin`, `bpixL1` and
    the displacement summaries are pair-level.  A genuine split would need the
    `parmtype == 0` entries, which ARE ordered leg by leg in increasing radius
    (`Jpsi_vtxfirstplus` says which leg comes first) -- see `tail_align.py`.
    """
    cols = ["run", "lumi", "event", "nParms", "globalidxv",
            "Muplus_jacRef", "Muminus_jacRef"]
    rows = []
    for fn in sorted(glob.glob(files)):
        t = uproot.open(fn + ":tree")
        d = t.arrays(cols, library="np")
        n = len(d["run"])
        for i in range(n):
            gi = np.asarray(d["globalidxv"][i], np.int64)
            npar = len(gi)
            jp = np.asarray(d["Muplus_jacRef"][i], float).reshape(3, npar)
            jm = np.asarray(d["Muminus_jacRef"][i], float).reshape(3, npar)
            ap = np.abs(jp).max(axis=0) > 0
            am = np.abs(jm).max(axis=0) > 0
            r = {"run": d["run"][i], "lumi": d["lumi"][i],
                 "event": d["event"][i]}
            for tag, sel in (("plus", ap), ("minus", am)):
                idx = gi[sel & M["align"][gi]]
                det = M["det"][idx]
                u, first = np.unique(det, return_index=True)
                rho = M["rho"][idx][first]
                sub = M["sub"][idx][first]
                lay = M["lay"][idx][first]
                dsp = M["disp"][idx][first]
                dpz = M["drphi"][idx][first]
                pix = sub <= 1
                r[f"nmod_{tag}"] = len(u)
                r[f"npixmod_{tag}"] = int(pix.sum())
                r[f"rhomin_{tag}"] = float(rho.min()) if len(rho) else np.nan
                r[f"bpixL1_{tag}"] = bool(((sub == 0) & (lay == 1)).any())
                r[f"dmax_{tag}"] = float(np.nanmax(dsp)) if len(dsp) else np.nan
                r[f"dmaxpix_{tag}"] = (float(np.nanmax(dsp[pix]))
                                       if pix.any() else np.nan)
                r[f"drms_{tag}"] = (float(np.sqrt(np.nanmean(dsp ** 2)))
                                    if len(dsp) else np.nan)
                # the r-phi displacement seen at the innermost module, which is
                # the lever arm a d0 / vertex residual actually feels
                r[f"drphi_in_{tag}"] = (float(dpz[np.argmin(rho)])
                                        if len(rho) else np.nan)
            rows.append(r)
        if nmax and len(rows) >= nmax:
            break
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def prof(sel, x, ys, edges, label, trim=0.01):
    print(f"\n  -- {label}")
    hdr = f"  {'bin':<20s}{'N':>7s}" + "".join(f"{nm:>22s}" for nm, _ in ys)
    print(hdr)
    for i in range(len(edges) - 1):
        b = sel & (x >= edges[i]) & (x < edges[i + 1])
        if b.sum() < 5:
            continue
        line = f"  {edges[i]:+8.3f}..{edges[i+1]:+8.3f}{int(b.sum()):>7d}"
        for nm, y in ys:
            v = np.sort(y[b]); k = max(int(trim * len(v)), 1)
            v = v[k:-k]
            line += f"{v.mean():>+14.4f}+-{v.std()/np.sqrt(len(v)):.4f}"
        print(line)
    # a chi2 against a constant, which is what "structure" means
    for nm, y in ys:
        mus, ers = [], []
        for i in range(len(edges) - 1):
            b = sel & (x >= edges[i]) & (x < edges[i + 1])
            if b.sum() < 5:
                continue
            v = np.sort(y[b]); k = max(int(trim * len(v)), 1); v = v[k:-k]
            mus.append(v.mean()); ers.append(v.std() / np.sqrt(len(v)))
        mus, ers = np.array(mus), np.array(ers)
        w = 1 / ers ** 2
        mu0 = (mus * w).sum() / w.sum()
        chi2 = (((mus - mu0) / ers) ** 2).sum()
        print(f"     {nm:<22s} chi2/ndof vs a constant = "
              f"{chi2:.1f}/{len(mus)-1} = {chi2/max(len(mus)-1,1):.2f}"
              f"   (amplitude p2p {mus.max()-mus.min():+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="dy")
    a = ap.parse_args()

    d = dict(np.load(a.npz, allow_pickle=False))
    base, chi2n, nl = TC.baseline(d, verbose=False)
    cls, names, aux = GB.classify(d)
    isig = names.index("signal")
    z1, z2 = d["bsz"][:, 0], d["bsz"][:, 1]
    zv = np.asarray(d["vtxz"], float)
    spot, slope, bsvtx, bsres = d["bsspot"], d["bsslope"], d["bsvtx"], d["bsres"]
    gx, gy, gz = d["genvx"], d["genvy"], d["genvz"]
    hasgen = (gx > -90) & (gy > -90) & (gz > -90)
    loo_x = spot[:, 0] + slope[:, 0] * (bsvtx[:, 2] - spot[:, 2]) + bsres[:, 0]
    loo_y = spot[:, 1] + slope[:, 1] * (bsvtx[:, 2] - spot[:, 2]) + bsres[:, 1]
    dreco_x = (loo_x - gx) * 1e4
    dreco_y = (loo_y - gy) * 1e4
    sig = base & hasgen & (cls == isig)
    print(f"# {a.tag}: baseline {int(base.sum())}, gen signal with a gen "
          f"vertex {int(sig.sum())}")

    files = sorted(glob.glob(a.files))
    M = module_map(files[0], a.geom)
    L = per_leg(a.files, M)
    # align the two tables on (run, lumi, event) ORDER -- both are written in
    # tree order over the same files, so a straight check suffices
    assert len(L["run"]) == len(d["run"]), (len(L["run"]), len(d["run"]))
    assert np.array_equal(L["event"], d["event"])

    ys = [("z_1", z1), ("z_2", z2), ("z_v", zv),
          ("dreco_x [um]", dreco_x), ("dreco_y [um]", dreco_y)]

    print("\n=== B(iii) STRUCTURE: the MEAN of each pull in bins of the LEG "
          "kinematics (1 %-trimmed; misalignment gives structure, resolution "
          "does not)")
    prof(sig, d["phi_plus"], ys, np.linspace(-np.pi, np.pi, 13),
         "phi of the POSITIVE leg")
    prof(sig, d["phi_minus"], ys, np.linspace(-np.pi, np.pi, 13),
         "phi of the NEGATIVE leg")
    prof(sig, d["eta_plus"], ys, np.linspace(-2.4, 2.4, 9),
         "eta of the POSITIVE leg")
    prof(sig, d["phi"], ys, np.linspace(-np.pi, np.pi, 13), "phi of the PAIR")

    print("\n=== B(ii) EXPOSURE: do the tail candidates cross the displaced "
          "modules?  (per-module |r_aligned - r_ideal|, um)")
    dmax = np.maximum(L["dmax_plus"], L["dmax_minus"]) * 1e4
    dmaxpix = np.maximum(L["dmaxpix_plus"], L["dmaxpix_minus"]) * 1e4
    drmsw = np.maximum(L["drms_plus"], L["drms_minus"]) * 1e4
    hdr = (f"  {'set':<22s}{'N':>7s}{'med dmax':>11s}{'p90 dmax':>11s}"
           f"{'med dmaxpix':>13s}{'med drms':>11s}")
    print(hdr)
    for lbl, s in (("gen signal (all)", sig),
                   ("|z_1| > 3", sig & (np.abs(z1) > 3)),
                   ("|z_1| > 5", sig & (np.abs(z1) > 5)),
                   ("|z_2| > 5", sig & (np.abs(z2) > 5)),
                   ("|z_v| > 3", sig & (np.abs(zv) > 3))):
        if s.sum() == 0:
            continue
        print(f"  {lbl:<22s}{int(s.sum()):>7d}{np.nanmedian(dmax[s]):>11.2f}"
              f"{np.nanpercentile(dmax[s],90):>11.2f}"
              f"{np.nanmedian(dmaxpix[s]):>13.2f}{np.nanmedian(drmsw[s]):>11.2f}")

    print("\n=== C THIN LEGS: what the WEAKER leg is made of")
    weak_is_plus = d["nvalid_plus"] <= d["nvalid_minus"]
    def wk(name):
        return np.where(weak_is_plus, L[f"{name}_plus"], L[f"{name}_minus"])
    nmod_w = wk("nmod"); npix_w = wk("npixmod")
    rho_w = wk("rhomin") ; bl1_w = wk("bpixL1")
    nvw = np.minimum(d["nvalid_plus"], d["nvalid_minus"])
    npixw = np.where(weak_is_plus, d["nvalidpixel_plus"], d["nvalidpixel_minus"])
    nhw = np.where(weak_is_plus, d["nhits_plus"], d["nhits_minus"])
    ptw = np.where(weak_is_plus, d["pt_plus"], d["pt_minus"])
    gptw = np.where(weak_is_plus, d["genpt_plus"], d["genpt_minus"])
    ratio = np.where(gptw > 0, ptw / np.maximum(gptw, 1e-9), np.nan)
    hdr = (f"  {'set':<22s}{'N':>7s}{'nvalid':>9s}{'nmod':>8s}{'npix':>7s}"
           f"{'rho_in cm':>11s}{'BPixL1':>9s}{'pT/gen p50':>12s}"
           f"{'p1':>8s}{'p99':>8s}")
    print(hdr)
    for lbl, s in (("gen signal (all)", sig),
                   ("|z_1| > 5", sig & (np.abs(z1) > 5)),
                   ("|z_2| > 5", sig & (np.abs(z2) > 5)),
                   ("|z_v| > 3", sig & (np.abs(zv) > 3))):
        if s.sum() == 0:
            continue
        print(f"  {lbl:<22s}{int(s.sum()):>7d}{np.median(nvw[s]):>9.1f}"
              f"{np.median(nmod_w[s]):>8.1f}{np.median(npixw[s]):>7.1f}"
              f"{np.nanmedian(rho_w[s]):>11.3f}{bl1_w[s].mean():>9.3f}"
              f"{np.nanmedian(ratio[s]):>12.4f}"
              f"{np.nanpercentile(ratio[s],1):>8.3f}"
              f"{np.nanpercentile(ratio[s],99):>8.3f}")
    print("\n  -- is the stored hit list TRUNCATED?  nvalid against the "
          "modules the fit used, and against nhits")
    print(f"  median nvalid(weak) {np.median(nvw[sig]):.1f}   "
          f"median modules used {np.median(nmod_w[sig]):.1f}   "
          f"median nhits {np.median(nhw[sig]):.1f}")
    print(f"  P(nmod < nvalid) = {TC.pm(int((nmod_w[sig] < nvw[sig]).sum()), int(sig.sum()))}")
    print(f"  P(nvalid < nhits) = {TC.pm(int((nvw[sig] < nhw[sig]).sum()), int(sig.sum()))}")
    print("\n  -- P(tail) against the WEAKER leg's hit count and inner radius")
    for lbl, cut in (("nvalid <= 9", nvw <= 9), ("nvalid 10-11", (nvw >= 10) & (nvw <= 11)),
                     ("nvalid >= 12", nvw >= 12),
                     ("no BPix L1", ~bl1_w.astype(bool)), ("BPix L1", bl1_w.astype(bool)),
                     ("npixel <= 2", npixw <= 2), ("npixel >= 3", npixw >= 3)):
        s = sig & cut
        if s.sum() < 5:
            continue
        print(f"  {lbl:<16s} N {int(s.sum()):6d}   "
              f"P(|z_1|>5) {TC.pm(int((np.abs(z1[s])>5).sum()), int(s.sum()))}   "
              f"P(|z_1|>3) {TC.pm(int((np.abs(z1[s])>3).sum()), int(s.sum()))}   "
              f"P(|z_v|>3) {TC.pm(int((np.abs(zv[s])>3).sum()), int(s.sum()))}")

    if a.out:
        np.savez_compressed(a.out, base=base, sig=sig, cls=cls, z1=z1, z2=z2,
                            zv=zv, dreco_x=dreco_x, dreco_y=dreco_y,
                            dmax=dmax, dmaxpix=dmaxpix, drms=drmsw,
                            nvw=nvw, npixw=npixw, rho_w=rho_w, bl1_w=bl1_w,
                            ratio=ratio, nmod_w=nmod_w,
                            **{k: v for k, v in L.items()})
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()

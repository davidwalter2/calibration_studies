#!/usr/bin/env python3
"""Masks for the 2026-09-04 censoring test.

Two families.

 A. SANITY masks (remove the candidates for which a 0.3 GeV window is not a
    window at all): sigma_m < SIGMAX.  With sigma_m ~ 0.3 GeV the accepted
    window holds only 38 % of the model probability (window_norm_validate
    gate 3, max 1-P = 0.62), so 1/P_i is a large, strongly alpha-dependent
    weight fitted on a handful of rows.

 B. CONTROLLED TRUNCATION on the J/psi GUN, which has NO reco-mass window in
    its production chain (legacy pair loop over generalTracks + requireGen
    only, ResidualGlobalCorrectionMakerTwoTrackG4e.cc ~1355-1470).  Cutting
    m_reco by hand and refitting with and without the truncation term is the
    only place in this study where the truncation is EXACTLY of the form the
    correction assumes (a sharp window in the SAME variable the likelihood
    models), so it measures both (i) that the correction works and (ii) how
    many 1e-3 of alpha a censoring of a given width costs.

m_reco = z*sigma + m_gen and the cache stores `eta` = Jpsigen_mass, so the
mass is reconstructed from the cache alone -- no tree access, no alignment
assumption.
"""
import numpy as np, os, sys

MJ = 3.0969
OUT = "runs/censoring260904"
os.makedirs(OUT, exist_ok=True)
SIGMAX = 0.15

CACHES = {
    "gun": "runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz",
    "v3": "runs/cf_masspairs_btojpsix_v3_260903x_m0.npz",
}

def save(name, mask, note, sigfull):
    p = f"{OUT}/mask_{name}.npz"
    np.savez_compressed(p, mask=mask)
    print(f"{name:<28s} {int(mask.sum()):8d} / {len(mask):8d} "
          f"({100.*mask.mean():6.3f} %)  sig.min {sigfull[mask].min():.6f}  {note}")

for tag, c in CACHES.items():
    d = np.load(c)
    z = d["z"].astype(np.float64); s = d["sigma"].astype(np.float64)
    mg = d["eta"].astype(np.float64)
    m = z * s + mg
    fin = np.isfinite(m) & np.isfinite(s) & (s > 0)
    print(f"\n== {tag}  n={len(z)}  sig.min(all)={s[fin].min():.6f} ==")
    print(f"   m_gen: min {mg[fin].min():.4f} med {np.median(mg[fin]):.4f} "
          f"max {mg[fin].max():.4f}   |m_gen-MJ|>1e-6 on {100.*np.mean(np.abs(mg[fin]-MJ)>1e-6):.2f} %")
    print(f"   m_reco quantiles: "
          + "  ".join(f"q{q}={np.quantile(m[fin],q/100.):.4f}"
                      for q in (0.1, 1, 50, 99, 99.9)))
    print(f"   sigma_m: med {np.median(s[fin]):.4f} "
          f"q99 {np.quantile(s[fin],.99):.4f} max {s[fin].max():.4f};  "
          f"sigma>{SIGMAX}: {int((s>SIGMAX).sum())}")
    sane = fin & (s < SIGMAX)
    save(f"{tag}_sane", sane, f"finite & sigma<{SIGMAX}", s)
    if tag == "gun":
        for hw in (0.15, 0.10, 0.05):
            sel = sane & (np.abs(m - MJ) < hw)
            save(f"gun_cut{int(hw*100):03d}", sel,
                 f"sane & |m-MJ|<{hw} ({hw/np.median(s[sane]):.1f} median sigma)", s)
    else:
        for lo, hi in ((2.95, 3.25), (2.90, 3.30), (2.85, 3.35)):
            sel = sane & (m > lo) & (m < hi)
            save(f"v3_post{int(lo*100)}_{int(hi*100)}", sel,
                 f"sane & {lo}<m_reco<{hi}", s)

# ---------------------------------------------------------------------------
# 2026-09-04 (later): QUALITY masks, from the aux columns (alignment proved by
# censoring_aux.py).  The gun's two-track fit has a 12 % normchi2 > 3 tail and
# a 7 % `frozen` population (the Gauss-Newton momentum-floor clamp at 2 GeV
# scaled the step to zero, so the refit mass IS the seed mass); for those
# candidates the CF model -- which describes the ideal linear estimator given
# the noise record -- is not the estimator that produced the number.
# ---------------------------------------------------------------------------
AUX = {"gun": f"{OUT}/aux_jpsigun.npz", "v3": f"{OUT}/aux_btojpsix.npz"}
for tag, c in CACHES.items():
    d = np.load(c); a = np.load(AUX[tag])
    s = d["sigma"].astype(np.float64)
    z = d["z"].astype(np.float64)
    assert np.array_equal(a["z"].astype(np.float64), z), f"{tag}: aux not aligned"
    sane = np.isfinite(s) & (s > 0) & (s < SIGMAX) & np.isfinite(z)
    c2 = a["normchi2"].astype(np.float64)
    it = a["niter"].astype(np.float64)
    fz = a["frozen"].astype(np.float64) > 0.5
    pt = a["ptmin"].astype(np.float64)
    print(f"\n== {tag} quality masks ==")
    for nm, sel in (
            (f"{tag}_q3", sane & (c2 < 3.)),
            (f"{tag}_q3nf", sane & (c2 < 3.) & ~fz & (it < 10)),
            (f"{tag}_nofrozen", sane & ~fz),
            (f"{tag}_pt2", sane & (pt > 2.)),
            (f"{tag}_pt3", sane & (pt > 3.)),
            (f"{tag}_pt3q3", sane & (pt > 3.) & (c2 < 3.)),
    ):
        save(nm, sel, "", s)

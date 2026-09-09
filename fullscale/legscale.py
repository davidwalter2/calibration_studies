#!/usr/bin/env python3
"""Is the mass-level `sigma/m` pattern a per-leg MOMENTUM bias, or the model?

The fixed-`eta` `sigma/m` split (sec. 0f.44) gives `m_Z` = -4.56 (barrel LOW),
-40.90 (barrel HIGH), +65.87 (endcap HIGH) MeV. Only a per-leg bias that is the
SAME for both charges survives into the pair mass: with
`m ~ 2 sqrt(p1 p2) sin(theta/2)`, a relative momentum bias `d_l` on each leg
gives `dm/m = (d_1 + d_2)/2`, so the charge-EVEN part adds and the charge-ODD
part cancels. In the SIGNED `q/p` pull the roles are swapped (a scale bias is
charge-ODD there, because `x ~ q (1/p_reco - 1/p_gen)`); this script works in
the MOMENTUM variable throughout, so:

    d_l = q/p(reco) / q/p(gen) - 1 = p_gen / p_reco - 1     (positive => reco p too LOW)
    A = 1/2 (<d>_+ + <d>_-)   charge-EVEN, SCALE-like, ADDS in the pair mass
    M = 1/2 (<d>_+ - <d>_-)   charge-ODD, misalignment-like, CANCELS

`A` and `dm/m` have OPPOSITE signs: reco `p` too low => reco mass too low =>
fitted `m_Z` low, so a more negative fitted `m_Z` needs a more positive `A`.
Sec. 0b's inclusive `A = -0.68e-4` is +6.2 MeV on the mass, and it checks out.

WHAT THE MASS SPLIT NEEDS. Barrel HIGH - LOW is -36.34 MeV on 91.19 GeV, so
`A(high) - A(low) = +3.99e-04`; endcap HIGH - barrel LOW is +70.43 MeV, so
`-7.72e-04`. Those are 6x and 11x the inclusive `A` itself, and 10x its whole
`eta` spread (0.41e-4), so if the legs carry them it is unmissable.

THE CONDITIONING, which is the whole difficulty. `sigma_m/m` is the fit's own
error and `sigma = sigma_bar (1 + a z)`, so splitting on it is a cut on the
mass residual -- standing rule 3, and it is what the SPLIT CARDS THEMSELVES
did. Three cells are therefore reported for every split:

  `sigma/m`     the reco split, reproducing the cards exactly (same selection,
                same `eta_lead` = |eta| of the LEADING-pT leg, same thresholds);
  predicted     the split on a GEN-ONLY PREDICTOR of `sigma/m`: the mean
                `sigma/m` in a 40x40 quantile grid of (gen |eta| of the leading
                leg, gen pT of the subleading leg). It is a function of gen
                variables alone, so it cannot see the residual; with ~2300
                candidates per cell the residual leaks into the cell mean at
                `corr/sqrt(2300)` < 1e-4 of a sigma.
  gen pT_sub    the single cleanest gen variable, since `sigma_m/m` is
                dominated by the worse-measured leg.

`sigma_bar = sigma (1 - a z)` is NOT used as a split variable: it is built from
`z`, so it is ANTI-correlated with the residual by construction
(`corr(sigma_bar/m, z) = -0.090` measured against `corr(sigma/m, z) = -0.006`),
i.e. it is a worse conditioning variable than the thing it was meant to fix.
Regression to the mean; the gen predictor is the right instrument.

and the DIRECT mass shift `<(m_reco - m_gen)/m_gen>` is carried next to `A` in
every cell, because if the mass pattern is the legs then `-A` must equal it.

    python3 legscale.py [--trim 0.05]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MZ = 91.153509740726733

# certified fitted m_Z, MeV from the generator (sec. 0f.44 / the RESUME table)
FITTED = {"barrel |eta|<0.9, sigma/m LOW": -4.56, "barrel, sigma/m HIGH": -40.90,
          "endcap 1.6-3.0, sigma/m HIGH": +65.87, "barrel, both": -21.08,
          "endcap, both": +34.22, "endcap 1.6-3.0, sigma/m LOW": +28.38}


def trimmed(v, w, trim):
    """Trimmed weighted mean; the residual has Landau-like tails."""
    if len(v) < 20:
        return np.nan
    lo, hi = np.percentile(v, [100 * trim, 100 * (1 - trim)])
    k = (v >= lo) & (v <= hi)
    return float(np.average(v[k], weights=w[k]))


def AM(dp, dm, w, trim):
    """Charge-even A and charge-odd M from the two legs of the same candidates."""
    a_p, a_m = trimmed(dp, w, trim), trimmed(dm, w, trim)
    return 0.5 * (a_p + a_m), 0.5 * (a_p - a_m)


def boot(dp, dm, w, trim, nb, rng):
    n = len(w)
    if n < 200:
        return np.nan, np.nan
    A, M = [], []
    for _ in range(nb):
        k = rng.integers(0, n, n)          # bootstrap CANDIDATES: the two legs
        a, m = AM(dp[k], dm[k], w[k], trim)  # of one candidate are correlated
        A.append(a)
        M.append(m)
    return float(np.std(A)), float(np.std(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", type=float, default=0.05)
    ap.add_argument("--nboot", type=int, default=120)
    ap.add_argument("--acoef", type=float, default=1.2110)
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)

    P = np.load(os.path.join(HERE, "runs/zpairs_dyv2_full.npz"))
    S = np.load(os.path.join(HERE, "runs/auxseed_dyv2.npz"))
    G = np.load(os.path.join(HERE, "runs/auxgen_dyv2.npz"))
    z = P["z"].astype(np.float64)
    assert np.array_equal(S["z"].astype(np.float64), z), "auxseed misaligned"
    assert np.array_equal(G["z"].astype(np.float64), z), "auxgen misaligned"
    sig = P["sigma"].astype(np.float64)
    mgen = P["eta"].astype(np.float64)          # this cache stores m_gen as `eta`
    mreco = z * sig + mgen
    w = P["w"].astype(np.float64)
    srel = sig / np.maximum(np.abs(mreco), 1e-9)
    sbar_rel = srel * (1.0 - a.acoef * srel * z)   # diagnostic only, NOT a split
    dmm = (mreco - mgen) / mgen                 # the DIRECT mass shift

    dp = (S["qopref_p"] / S["qopgen_p"] - 1.0).astype(np.float64)
    dmn = (S["qopref_m"] / S["qopgen_m"] - 1.0).astype(np.float64)

    # `eta_lead` EXACTLY as make_card.py defines it (l.365-375): the |eta| of
    # the leg with the larger RECO pT -- NOT max(|eta_p|, |eta_m|), which is a
    # different variable and gives a barrel population 2.2x smaller.
    ptp, ptm = P["ptp"].astype(np.float64), P["ptm"].astype(np.float64)
    plus_leads = ptp >= ptm
    etal = np.abs(np.where(plus_leads, P["etap"], P["etam"])).astype(np.float64)
    # and the GEN counterpart of the same definition (sec. 0f.57: they agree)
    gplus_leads = G["gpt_p"].astype(np.float64) >= G["gpt_m"].astype(np.float64)
    getal = np.abs(np.where(gplus_leads, G["geta_p"], G["geta_m"])).astype(np.float64)
    gpt_sub = np.minimum(G["gpt_p"], G["gpt_m"]).astype(np.float64)

    # the card's own selection, so the cells are the cells that were FITTED
    c2 = (P["chisqval"].astype(np.float64)
          / np.maximum(P["ndof"].astype(np.float64), 1.0))
    ok = (np.isfinite(dp) & np.isfinite(dmn) & np.isfinite(z) & (sig > 0)
          & np.isfinite(w) & (srel < 0.10) & (mreco >= 60.0) & (mreco <= 120.0)
          & (c2 < 3.0))

    # GEN-ONLY predictor of sigma/m: the mean sigma/m in a quantile grid of
    # (gen |eta| lead, gen pT sub). A function of gen variables alone.
    NB = 40
    qe = np.unique(np.percentile(getal[ok], np.linspace(0, 100, NB + 1)))
    qp = np.unique(np.percentile(gpt_sub[ok], np.linspace(0, 100, NB + 1)))
    ie = np.clip(np.digitize(getal, qe[1:-1]), 0, len(qe) - 2)
    ip = np.clip(np.digitize(gpt_sub, qp[1:-1]), 0, len(qp) - 2)
    cell = ie * (len(qp) - 1) + ip
    ncell = (len(qe) - 1) * (len(qp) - 1)
    num = np.bincount(cell[ok], weights=srel[ok], minlength=ncell)
    den = np.bincount(cell[ok], minlength=ncell)
    pred = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    srel_pred = pred[cell]
    print(f"gen predictor: {ncell} cells, median occupancy "
          f"{np.median(den[den>0]):.0f}, corr(pred, sigma/m) = "
          f"{np.corrcoef(srel_pred[ok], srel[ok])[0,1]:+.4f}")
    npix = None
    if "npix_p" in S.files:
        npix = np.minimum(S["npix_p"], S["npix_m"]).astype(np.float64)

    print(f"{int(ok.sum())} candidates ({2*int(ok.sum())} legs), "
          f"trim {a.trim:.0%} per tail, bootstrap over CANDIDATES.\n")
    print("corr diagnostics on the selected sample:")
    for nm, v in (("sigma/m", srel), ("sigma_bar/m NOT USED", sbar_rel),
                  ("GEN predicted sigma/m", srel_pred),
                  ("gen pT sub", gpt_sub), ("|eta| lead reco", etal),
                  ("|eta| lead gen", getal)):
        print(f"    corr({nm:16s}, z) = {np.corrcoef(v[ok], z[ok])[0,1]:+.4f}")
    print()

    hdr = (f"{'cell':34s} {'n':>9s} {'A [1e-4]':>17s} {'M [1e-4]':>17s} "
           f"{'-A [1e-4]':>10s} {'<dm/m> [1e-4]':>14s} {'fitted m_Z':>11s}")
    rows = {}

    def row(lab, sel):
        m_ = ok & sel
        n = int(m_.sum())
        if n < 500:
            print(f"{lab:34s} {n:9d}   (too few)")
            return
        A, M = AM(dp[m_], dmn[m_], w[m_], a.trim)
        eA, eM = boot(dp[m_], dmn[m_], w[m_], a.trim, a.nboot, rng)
        d = trimmed(dmm[m_], w[m_], a.trim)
        f = FITTED.get(lab)
        rows[lab] = (A, eA, d, n)
        print(f"{lab:34s} {n:9d} {1e4*A:+9.3f} +-{1e4*eA:5.3f} "
              f"{1e4*M:+9.3f} +-{1e4*eM:5.3f} {-1e4*A:+10.3f} {1e4*d:+14.3f} "
              f"{('%+8.2f' % f) if f is not None else '':>11s}")

    print("=== 1. per GEN |eta| of the LEADING-pT leg (the card's own variable)")
    print(hdr)
    print("-" * len(hdr))
    row("inclusive", np.ones(len(z), bool))
    for lo, hi in ((0, 0.9), (0.9, 1.6), (1.6, 3.0)):
        row(f"gen |eta|lead {lo}-{hi}", (getal >= lo) & (getal < hi))

    print("\n=== 2. THE MASS SPLIT REPRODUCED: sigma/m halves at FIXED eta")
    print("    (the RECO split, exactly the cards' own thresholds)")
    print(hdr)
    print("-" * len(hdr))
    bar = (etal >= 0) & (etal < 0.9)      # RECO, exactly as the cards cut
    end = (etal >= 1.6) & (etal < 3.0)
    row("barrel |eta|<0.9, sigma/m LOW", bar & (srel < 0.01032))
    row("barrel, sigma/m HIGH", bar & (srel >= 0.01032))
    row("barrel, both", bar)
    row("endcap 1.6-3.0, sigma/m LOW", end & (srel < 0.01658))
    row("endcap 1.6-3.0, sigma/m HIGH", end & (srel >= 0.01658))
    row("endcap, both", end)

    print("\n=== 3. THE SAME SPLIT ON A GEN-ONLY PREDICTOR OF sigma/m")
    print("    no selection on the residual is possible; a real leg bias SURVIVES")
    print(hdr)
    print("-" * len(hdr))
    tb = np.median(srel_pred[ok & bar])
    te = np.median(srel_pred[ok & end])
    print(f"    (predictor medians: barrel {tb:.5f}, endcap {te:.5f})")
    row("barrel, PRED sigma/m LOW", bar & (srel_pred < tb))
    row("barrel, PRED sigma/m HIGH", bar & (srel_pred >= tb))
    row("endcap, PRED sigma/m LOW", end & (srel_pred < te))
    row("endcap, PRED sigma/m HIGH", end & (srel_pred >= te))
    print("    and on the single cleanest gen variable, the SOFTER leg's gen pT:")
    for tag, b in (("barrel", bar), ("endcap", end)):
        t = np.median(gpt_sub[ok & b])
        row(f"{tag}, gen pT_sub HIGH (better)", b & (gpt_sub >= t))
        row(f"{tag}, gen pT_sub LOW (worse)", b & (gpt_sub < t))

    if npix is not None:
        print("\n=== 4. per PIXEL-HIT COUNT (min over the two legs)")
        print(hdr)
        print("-" * len(hdr))
        for lo, hi, lab in ((0, 3, "npix <= 2"), (3, 4, "npix = 3"),
                            (4, 5, "npix = 4"), (5, 99, "npix >= 5")):
            row(f"{lab}", (npix >= lo) & (npix < hi))
        for tag, b in (("barrel", bar), ("endcap", end)):
            for lo, hi, lab in ((0, 4, "npix <= 3"), (4, 99, "npix >= 4")):
                row(f"{tag}, {lab}", b & (npix >= lo) & (npix < hi))

    print("\n=== WHAT THE MASS SPLIT NEEDS, against what the legs deliver")
    print(f"{'comparison':44s} {'needed dA':>11s} {'measured dA':>18s} "
          f"{'measured d<dm/m>':>17s}")
    print("-" * 94)
    for lab, hi, lo in (
            ("barrel HIGH - LOW  (-36.34 MeV)", "barrel, sigma/m HIGH",
             "barrel |eta|<0.9, sigma/m LOW"),
            ("endcap HIGH - barrel LOW (+70.43)", "endcap 1.6-3.0, sigma/m HIGH",
             "barrel |eta|<0.9, sigma/m LOW"),
            ("barrel HIGH - LOW, GEN PREDICTOR", "barrel, PRED sigma/m HIGH",
             "barrel, PRED sigma/m LOW"),
            ("endcap HIGH - LOW, GEN PREDICTOR", "endcap, PRED sigma/m HIGH",
             "endcap, PRED sigma/m LOW")):
        if hi not in rows or lo not in rows:
            continue
        need = -(FITTED.get(hi, np.nan) - FITTED.get(lo, np.nan)) / (1e3 * MZ)
        dA = rows[hi][0] - rows[lo][0]
        edA = np.hypot(rows[hi][1], rows[lo][1])
        dd = rows[hi][2] - rows[lo][2]
        print(f"{lab:44s} {1e4*need:+11.3f} {1e4*dA:+11.3f} +-{1e4*edA:4.3f} "
              f"{1e4*dd:+17.3f}")
    print("\nAll columns 1e-4. `-A` is what the legs predict for `<dm/m>`; if the\n"
          "two agree the mass shift IS the leg momenta. `needed dA` is what the\n"
          "certified fitted `m_Z` split would require of the legs.")


if __name__ == "__main__":
    main()

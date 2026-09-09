#!/usr/bin/env python3
"""Measure `a_i` directly, in truth cells, and compare it with the two forms.

`a` is DEFINED as the fractional response of the REPORTED width to the
standardized residual, `a = d ln sigma_fit / dx` with `x = (m_reco - m_gen)/sigma`
-- that is what makes `sigma_bar = sigma_fit (1 - a x)` the truth-referenced
width. So it can be MEASURED, with no fit and no model: bin candidates in cells
of quantities that do not themselves fluctuate with the residual, and regress
`ln sigma_fit` on `z` inside each cell. The slope is `a`.

Two predictions to compare it with:

  spec       a/(sigma/m) = 1 + vgf
  per leg    a/(sigma/m) = 1 + 2 (1-f_ang)^2 sum_l s_l^2 e_l

with `s_l = r_l^2 / (r_1^2 + r_2^2)` the leg's share of the momentum variance
(`sigrelp`, `sigrelm` in the pairs cache) and `e_l = d ln r_l / d ln p_l`. For
symmetric legs (`s_l = 1/2`) the second reduces to the first; for asymmetric
legs it is up to twice as far from 1. Writing `asym = 2(s_1^2 + s_2^2) - 1`
(0 symmetric, 1 fully asymmetric) and taking `e_l ~ e ~ vgf` as the stand-in
until the per-leg exponents are tabulated,

  per leg    a/(sigma/m) = 1 + vgf (1 + asym) (1-f_ang)^2

so the two differ by `vgf * asym * (sigma/m)`, which is the column to watch.

A THIRD form, `--aux`: the closed form of `oddmoment/aux_gen.py`,

  closed     a/(sigma/m) = 1 + f_hit - f_ioni

with `f_hit` and `f_ioni` the per-family shares of the mass variance taken from
the fit's own **Q matrix** (`resinfvarv` grouped by the runtree `parmtype`:
8/9 hit, 10 multiple scattering, 11 ionisation; parmtype 15 is a RE-PARTITION
of 10+11 into material groups and must NOT be added -- see
`ResidualGlobalCorrectionMakerTwoTrackG4e.cc:4758`). These are the well-defined
quantities that the CF exponent's `-S''(0)` is NOT (STATE sec. 0f.47b: the
Moliere and Landau exponents have no finite second moment, so a
finite-difference second derivative is cut-dependent).

    python3 measure_a.py [--pairs runs/zpairs_dyv2_full.npz] \
        [--aux runs/auxgen_dyv2.npz]
"""
import argparse

import numpy as np


def wls_slope(x, y, w):
    """Weighted least-squares slope of y on x, and its error."""
    W = w.sum()
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    sxx = (w * (x - mx) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan
    b = (w * (x - mx) * (y - my)).sum() / sxx
    r = y - my - b * (x - mx)
    # effective dof with weights
    neff = W ** 2 / (w ** 2).sum()
    s2 = (w * r ** 2).sum() / W * neff / max(neff - 2, 1)
    return b, np.sqrt(s2 * (w.sum() / W) / sxx * W / W) if sxx > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="runs/zpairs_dyv2_full.npz")
    ap.add_argument("--zmax", type=float, default=2.0,
                    help="|z| range of the regression: `a` is the FIRST-order "
                         "response, so it must be measured where the "
                         "linearisation holds")
    ap.add_argument("--nsig", type=int, default=3)
    ap.add_argument("--nasym", type=int, default=3)
    ap.add_argument("--gen-cells", action="store_true",
                    help="ALSO measure `a` in cells of GENERATOR quantities "
                         "(requires --aux). `asym` and `sigma/m` are both built "
                         "from the REPORTED per-leg widths, which are "
                         "sigma_bar(1 + a x): binning on them is a cut on the "
                         "residual and attenuates the very slope being "
                         "measured. Gen leg kinematics cannot be.")
    ap.add_argument("--aux", default=None,
                    help="auxgen npz aligned ROW BY ROW to --pairs; adds the "
                         "closed form 1 + f_hit - f_ioni from the Q matrix")
    a = ap.parse_args()

    d = np.load(a.pairs)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mgen = d["eta"].astype(np.float64)
    m = z * sig + mgen
    vgf = d["vgf"].astype(np.float64)
    fang = d["fang"].astype(np.float64)
    sp, sm = d["sigrelp"].astype(np.float64), d["sigrelm"].astype(np.float64)
    etal = np.maximum(np.abs(d["etap"]), np.abs(d["etam"]))
    w = d["w"].astype(np.float64) if "w" in d.files else np.ones(len(z))
    if a.aux:
        g = np.load(a.aux)
        if len(g["z"]) != len(z):
            raise SystemExit("aux has %d rows, pairs %d -- not the aligned file"
                             % (len(g["z"]), len(z)))
        if not np.array_equal(g["z"].astype(np.float64), z):
            raise SystemExit("aux `z` is not bit-identical to the pairs `z`")
        fhit = g["fhit"].astype(np.float64)
        fioni = g["fioni"].astype(np.float64)
        gpt = np.stack([g["gpt_p"].astype(np.float64),
                        g["gpt_m"].astype(np.float64)])
        geta = np.stack([g["geta_p"].astype(np.float64),
                         g["geta_m"].astype(np.float64)])
        fms = g["fms"].astype(np.float64)
        clo = np.abs(fhit + fms + fioni - 1.0)
        print("aux aligned: <f_hit> = %.5f  <f_ms> = %.5f  <f_ioni> = %.3e ; "
              "Q-matrix closure |f_hit+f_ms+f_ioni-1| mean %.2e max %.2e"
              % (fhit.mean(), fms.mean(), fioni.mean(), clo.mean(), clo.max()))
    else:
        fhit = fioni = gpt = geta = None

    ok = (np.isfinite(z) & np.isfinite(sig) & (sig > 0) & np.isfinite(m) & (m > 0)
          & np.isfinite(sp) & np.isfinite(sm) & (sp > 0) & (sm > 0)
          & np.isfinite(vgf) & np.isfinite(w) & (np.abs(z) < a.zmax)
          & (sig / m < 0.10))
    z, sig, m, vgf, fang, sp, sm, etal, w = (v[ok] for v in
                                             (z, sig, m, vgf, fang, sp, sm, etal, w))
    if fhit is not None:
        fhit, fioni = fhit[ok], fioni[ok]
        gpt, geta = gpt[:, ok], geta[:, ok]
    srel = sig / m
    s1 = sp ** 2 / (sp ** 2 + sm ** 2)
    asym = 2.0 * (s1 ** 2 + (1 - s1) ** 2) - 1.0
    lns = np.log(sig)

    print(f"{len(z)} candidates, |z| < {a.zmax}, |m_reco-m_gen| implicit\n")
    print("a = d ln sigma_fit / dz, MEASURED in the cell, against the two forms.")
    print("All three columns are a/(sigma/m), so 1 + vgf is the spec.\n")
    hdr = (f"{'cell':26s} {'n':>9s} {'sigma/m':>8s} {'vgf':>6s} {'asym':>6s} "
           f"{'MEASURED':>16s} {'spec':>7s} {'per-leg':>8s} {'meas-spec':>10s}")
    if fhit is not None:
        hdr += f" {'f_ioni':>9s} {'closed':>7s} {'meas-clo':>9s}"
    print(hdr)
    print("-" * len(hdr))

    def row(lab, m_):
        if m_.sum() < 2000:
            print(f"{lab:26s} {int(m_.sum()):9d}   (too few)")
            return
        b, be = wls_slope(z[m_], lns[m_], w[m_])
        sr = np.average(srel[m_], weights=w[m_])
        v = np.average(vgf[m_], weights=w[m_])
        A = np.average(asym[m_], weights=w[m_])
        fa = np.average(fang[m_], weights=w[m_])
        meas = b / sr
        spec = 1.0 + v
        perleg = 1.0 + v * (1.0 + A) * (1.0 - fa) ** 2
        extra = ""
        if fhit is not None:
            fi = np.average(fioni[m_], weights=w[m_])
            fh = np.average(fhit[m_], weights=w[m_])
            closed = 1.0 + fh - fi
            extra = f" {fi:9.2e} {closed:7.4f} {meas - closed:+9.4f}"
        print(f"{lab:26s} {int(m_.sum()):9d} {sr:8.5f} {v:6.3f} {A:6.3f} "
              f"{meas:9.4f} +-{be/sr:5.4f} {spec:7.4f} {perleg:8.4f} "
              f"{meas - spec:+10.4f}" + extra)

    row("inclusive", np.ones(len(z), bool))
    print()
    for lo, hi, lab in ((0, 0.9, "|eta_lead| < 0.9"), (0.9, 1.6, "0.9 - 1.6"),
                        (1.6, 3.0, "1.6 - 3.0")):
        row(lab, (etal >= lo) & (etal < hi))
    print()
    # `asym` ALONE. Binning in sigma/m conditions on sigma, and sigma is
    # sigma_bar (1 + a x) -- so a sigma/m bin is a cut on x, which attenuates
    # the very slope being measured. `asym` is built from the two REPORTED
    # per-leg resolutions and is to first order a property of the kinematics,
    # not of the residual, so binning in it is safe. This is the clean test of
    # whether the leg-asymmetry term is the missing piece.
    qa2 = np.percentile(asym, np.linspace(0, 100, 6))
    for j in range(5):
        m_ = (asym >= qa2[j]) & (asym < qa2[j + 1])
        row(f"asym quintile {j+1}", m_)
    print()
    qs = np.percentile(srel, np.linspace(0, 100, a.nsig + 1))
    qa = np.percentile(asym, np.linspace(0, 100, a.nasym + 1))
    for i in range(a.nsig):
        for j in range(a.nasym):
            m_ = ((srel >= qs[i]) & (srel < qs[i + 1])
                  & (asym >= qa[j]) & (asym < qa[j + 1]))
            row(f"sigma/m t{i+1}, asym t{j+1}", m_)
    if a.gen_cells and gpt is not None:
        print("\n--- GENERATOR cells: nothing here is built from the reported "
              "widths ---")
        # `max |eta|` of the two legs, the SAME definition the reco rows use
        # (`etal = max(|etap|, |etam|)`), so the two are comparable; binning on
        # the eta of the leading-pT leg instead is a different variable and
        # gives visibly different numbers.
        getal = np.max(np.abs(geta), axis=0)
        gratio = gpt.min(axis=0) / np.maximum(gpt.max(axis=0), 1e-9)
        gdeta = np.abs(geta[0] - geta[1])
        for nm, v, nq in (("gen |eta| lead", getal, 0),
                          ("gen pT ratio", gratio, 5),
                          ("gen |d eta| legs", gdeta, 5)):
            if nq == 0:
                for lo, hi, lab in ((0, 0.9, "gen max|eta| < 0.9"),
                                    (0.9, 1.6, "gen 0.9 - 1.6"),
                                    (1.6, 3.0, "gen 1.6 - 3.0")):
                    row(lab, (v >= lo) & (v < hi))
                continue
            q = np.percentile(v, np.linspace(0, 100, nq + 1))
            for j in range(nq):
                row(f"{nm} q{j+1}", (v >= q[j]) & (v < q[j + 1]))
            print()

    print("\n`meas - spec` is the coefficient error the spec's form carries, in "
          "units of sigma/m.\nIf the leg-asymmetry term is the cause it must "
          "track `asym`, and `per-leg` must\nsit on `MEASURED` where `spec` "
          "does not.")


if __name__ == "__main__":
    main()

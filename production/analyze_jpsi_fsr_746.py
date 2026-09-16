#!/usr/bin/env python3
"""The TRUE FSR kernel of a J/psi MC sample, separated from its lineshape.

The generator draws the resonance mass from a Breit-Wigner truncated to
[mMin, mMax] and then lets PHOTOS++ radiate.  The post-FSR gen mass
``Jpsigen_mass`` therefore mixes the two.  ``Jpsigenpre_masslep`` -- the
invariant mass of the two status-746 (PHOTOS history) muons -- is the pre-FSR
resonance mass of the RADIATING candidates, so

    u = -ln(m_post / m_pre)

is the FSR kernel variable with the lineshape divided out exactly, candidate by
candidate.

Inputs: the cache written by ``extract_jpsi_gen_fsr.py``.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/zchannel")
import fsr_analytic as fa                                         # noqa: E402

MJPSI = 3.0969
GAMMA = 9.26e-5
MMIN, MMAX = 3.0960, 3.0978
U0 = np.array([1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.0566, 0.113])
# analytic exp2nll / mass-exact / pair=(e,mu) kernel at m = 3.0969, from
# fsr_analytic.FSRKernel(...).tail(U0)  -- PHOTONIC only
ANA_TAIL = np.array([0.27851408, 0.23271521, 0.18401463, 0.15971224,
                     0.13228274, 0.10657712, 0.07790755, 0.05200609,
                     0.03781909, 0.02391913])
ANA_MEAN_U_GAM = 0.010558535191164068
ANA_MEAN_U_PAIR = 0.000283953
ANA_MEAN_U = 0.01084248815897609


def bw_cdf(m, m0=MJPSI, g=GAMMA, lo=MMIN, hi=MMAX):
    """CDF of the non-relativistic BW truncated to [lo, hi]."""
    f = lambda x: np.arctan(2.0 * (x - m0) / g)                    # noqa: E731
    return (f(m) - f(lo)) / (f(hi) - f(lo))


def bw_quantile(q, m0=MJPSI, g=GAMMA, lo=MMIN, hi=MMAX):
    a, b = math.atan(2 * (lo - m0) / g), math.atan(2 * (hi - m0) / g)
    return m0 + 0.5 * g * np.tan(a + np.asarray(q, float) * (b - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--cache", required=True)
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    ap.add_argument("-o", "--output",
                    default="/work/submit/david_w/ZMass/calibration_studies/"
                            "zchannel/data/jpsi_fsr_from_746.npz")
    args = ap.parse_args()

    d = np.load(args.cache)
    m_post = d["m_post"].astype(np.float64)
    masslep = d["masslep"].astype(np.float64)
    c2 = d["chisqval"].astype(np.float64)
    nd = d["ndof"].astype(np.float64)
    n_raw = len(m_post)

    ok = np.isfinite(m_post) & (m_post > 0) & np.isfinite(masslep)
    ok &= np.isfinite(c2) & (nd > 0)
    n_fin = int(ok.sum())
    ok &= (c2 / np.maximum(nd, 1.0)) < args.max_chi2_ndof
    n_sel = int(ok.sum())
    print(f"candidates: raw {n_raw}  finite {n_fin}  "
          f"chi2/ndof < {args.max_chi2_ndof:g}: {n_sel}")

    m_post = m_post[ok]
    masslep = masslep[ok]
    rad = masslep > 0.0
    m_pre = np.where(rad, masslep, m_post)
    u = -np.log(m_post / m_pre)
    n = len(u)

    f_rad = rad.mean()
    zero = (u == 0.0)
    f_zero = zero.mean()
    mean_u = float(u.mean())
    print()
    print("=== 1. the FSR kernel of the sample ===")
    print(f"n                 = {n}")
    print(f"P(masslep filled) = {f_rad:.6f}   (PHOTOS emitted >= 1 photon)")
    print(f"P(u == 0 exact)   = {f_zero:.6f}  (float32 floor, see report)")
    print(f"P(u < 0)          = {(u < 0).mean():.3e}   n = {(u < 0).sum()}")
    print(f"<u>               = {mean_u:.8e}")
    print(f"<u | u != 0>      = {u[~zero].mean():.8e}")
    print(f"<u^2>             = {float((u * u).mean()):.8e}")
    print(f"<1 - m_post/m_pre>= {float((1.0 - m_post / m_pre).mean()):.8e}")
    print(f"m_post/m_pre ulp  = {np.spacing(np.float32(MJPSI)) / MJPSI:.4e}")

    tail_p = np.array([(u > x).mean() for x in U0])
    tail_e = np.sqrt(tail_p * (1 - tail_p) / n)
    print()
    print(f"{'u0':>10} {'P(u>u0) meas':>14} {'+-':>9} {'analytic':>10} "
          f"{'meas/ana':>9}")
    for x, p, e, a in zip(U0, tail_p, tail_e, ANA_TAIL):
        print(f"{x:10.3e} {p:14.6f} {e:9.2e} {a:10.6f} {p / a:9.4f}")

    ue = np.geomspace(1e-6, 0.2, 61)
    cnt, _ = np.histogram(u, bins=ue)
    dens = cnt / n / np.diff(np.log(ue))
    uc = np.sqrt(ue[:-1] * ue[1:])
    print()
    print("  u dP/dlnu  (60 log bins, 1e-6 .. 0.2)")
    derr = dens / np.sqrt(np.maximum(cnt, 1.0))
    for a, b, c, v, e in zip(ue[:-1], ue[1:], uc, dens, derr):
        print(f"  [{a:9.3e},{b:9.3e}]  uc={c:9.3e}  dens={v:10.6f} +-{e:9.6f}")

    # ---- 2. the pre-FSR mass ------------------------------------------
    print()
    print("=== 2. pre-FSR mass vs the truncated Breit-Wigner ===")
    for name, s in (("masslep only (radiating)", masslep[rad]),
                    ("m_pre (masslep, else m_post)", m_pre)):
        out = (s < MMIN) | (s > MMAX)
        print(f"  {name}:  n={len(s)}  min={s.min():.7f}  max={s.max():.7f}"
              f"  <m>-m0={(s.mean() - MJPSI) * 1e3:+.6f} MeV"
              f"  frac outside window={out.mean():.3e}")
    s = masslep[rad]
    sin = s[(s >= MMIN) & (s <= MMAX)]
    qs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"  quantiles of masslep (inside window, n={len(sin)}):")
    print(f"  {'q':>6} {'measured':>12} {'trunc BW':>12} {'diff [keV]':>12}")
    for q in qs:
        me, th = np.quantile(sin, q), bw_quantile(q)
        print(f"  {q:6.2f} {me:12.7f} {th:12.7f} {(me - th) * 1e6:12.2f}")
    nb = 60
    edg = np.linspace(MMIN, MMAX, nb + 1)
    print()
    print("  the SELECTED lineshape vs the generator's truncated BW")
    print(f"  {'sample':>34} {'n':>10} {'<m>-m0 [keV]':>13} {'+-':>7} "
          f"{'chi2/ndf':>10} {'tilt c':>9} {'+-':>7}")
    for nm, ss in (("masslep (radiating)", masslep[rad]),
                   ("m_post, non-radiating (m_post==m_pre)", m_post[~rad]),
                   ("m_pre = masslep else m_post", m_pre)):
        si = ss[(ss >= MMIN) & (ss <= MMAX)]
        ch, _ = np.histogram(si, bins=edg)
        ex = len(si) * np.diff(bw_cdf(edg))
        chi2 = float(((ch - ex) ** 2 / np.maximum(ex, 1e-9)).sum())
        # p(m) ~ BW(m) (1 + c t),  t = (m - m0)/0.0009 in [-1, 1]
        t = (0.5 * (edg[1:] + edg[:-1]) - MJPSI) / 0.0009
        w = ex * t
        A = float((w * w / np.maximum(ex, 1e-9)).sum())
        b = float((w * (ch - ex) / np.maximum(ex, 1e-9)).sum())
        c, ec = b / A, 1.0 / math.sqrt(A)
        chi2t = chi2 - b * b / A
        print(f"  {nm:>34} {len(si):10d} "
              f"{(si.mean() - MJPSI) * 1e6:13.3f} "
              f"{si.std() / math.sqrt(len(si)) * 1e6:7.3f} "
              f"{chi2 / (nb - 1):10.2f} {c:9.5f} {ec:7.5f}"
              f"   (chi2/ndf after tilt {chi2t / (nb - 2):.2f})")

    # ---- 3. consistency ----------------------------------------------
    print()
    print("=== 3. <m_post> decomposition (MeV) ===")
    dpost = (m_post.mean() - MJPSI) * 1e3
    dpre = (m_pre.mean() - MJPSI) * 1e3
    dfsr = (m_post - m_pre).mean() * 1e3
    print(f"  (i)  <m_post> - m0                 = {dpost:+.6f} MeV")
    print(f"  (ii) <m_pre> - m0                  = {dpre:+.6f} MeV")
    print(f"       <m_post - m_pre>              = {dfsr:+.6f} MeV")
    print(f"       sum                           = {dpre + dfsr:+.6f} MeV")
    print(f"       (i) - sum                     = {dpost - dpre - dfsr:+.2e} MeV")
    print(f"  -m0 <u>                            = {-MJPSI * mean_u * 1e3:+.6f} MeV")

    # ---- 4. the masslep loss ------------------------------------------
    print()
    print("=== 4. the lost-masslep candidates ===")
    for cut in (3.0955, 3.0950, 3.0940, 3.0900, 3.0800):
        hard = m_post < cut
        if hard.sum() == 0:
            continue
        fl = (~rad[hard]).mean()
        print(f"  m_post < {cut:.4f}: n={int(hard.sum()):8d}  "
              f"P(masslep==-99 | hard) = {fl:.5f} "
              f"+- {math.sqrt(fl * (1 - fl) / hard.sum()):.5f}")
    hard = m_post < 3.0955
    f_loss = float((~rad[hard]).mean())
    lost = hard & ~rad
    u_fix = u.copy()
    u_fix[lost] = -np.log(m_post[lost] / MJPSI)
    print(f"  <u> with the identifiable lost ones given m_pre = m0: "
          f"{u_fix.mean():.8e}  (shift {(u_fix.mean() - mean_u):+.3e}, "
          f"{(u_fix.mean() / mean_u - 1) * 100:+.3f} %)")
    corr = mean_u / (1.0 - f_loss)
    print(f"  <u> scaling the radiating population by 1/(1-f_loss): "
          f"{corr:.8e}  ({(corr / mean_u - 1) * 100:+.3f} %)")
    tail_c = tail_p / (1.0 - f_loss)
    print(f"  P(u>1e-6) so corrected = {tail_c[0]:.6f} "
          f"(ana {ANA_TAIL[0]:.6f}, ratio {tail_c[0] / ANA_TAIL[0]:.4f})")

    # ---- 5. vs analytic ------------------------------------------------
    print()
    print("=== 5. vs the analytic QED kernel ===")
    print(f"  <u> measured                 = {mean_u:.6e}")
    print(f"  <u> analytic photonic        = {ANA_MEAN_U_GAM:.6e}  "
          f"ratio {mean_u / ANA_MEAN_U_GAM:.4f}")
    print(f"  <u> analytic photonic + pair = {ANA_MEAN_U:.6e}  "
          f"ratio {mean_u / ANA_MEAN_U:.4f}")

    k = fa.FSRKernel(MJPSI, variant="exp2nll", pair=(), mass_exact=True)
    ue2 = np.geomspace(1e-6, 0.2, 61)
    uc2 = np.sqrt(ue2[:-1] * ue2[1:])
    ana_dens = uc2 * k.pdf_u(uc2)
    print()
    print("  LOCAL density u dP/dlnu, meas / analytic (acceptance-free)")
    print(f"  {'u':>10} {'meas':>10} {'+-':>9} {'ana':>10} {'ratio':>8}")
    for c, v, e, a in zip(uc2, dens, derr, ana_dens):
        print(f"  {c:10.3e} {v:10.6f} {e:9.6f} {a:10.6f} {v / a:8.4f}")

    print()
    print("  kernel CONDITIONED on u < u_cut (removes the ALCARECO mass-window"
          " truncation)")
    cut_grid, cut_meas, cut_ana = [], [], []
    print(f"  {'u_cut':>8} {'n':>10} {'<u|<cut> meas':>15} {'ana':>12} "
          f"{'ratio':>8}")
    for ucut in (0.003, 0.01, 0.03, 0.0566, 0.113):
        s2 = (u >= 0) & (u < ucut)
        mb = k.moments_below(ucut)
        me = float(u[s2].sum() / s2.sum())
        cut_grid.append(ucut); cut_meas.append(me); cut_ana.append(mb["u"])
        print(f"  {ucut:8.4f} {int(s2.sum()):10d} {me:15.6e} "
              f"{mb['u']:12.6e} {me / mb['u']:8.4f}")
    print()
    print("  P(u > u0 | u < 0.0566): meas vs analytic")
    ucut = 0.0566
    s2 = (u >= 0) & (u < ucut)
    pa_cut = float(k.tail(np.array([ucut]))[0])
    cond_u0, cond_m, cond_a = [], [], []
    for x in U0[U0 < ucut]:
        pm = (u[s2] > x).mean()
        pa = (float(k.tail(np.array([x]))[0]) - pa_cut) / (1.0 - pa_cut)
        cond_u0.append(x); cond_m.append(pm); cond_a.append(pa)
        print(f"  u0={x:9.3e}  meas {pm:9.6f}  ana {pa:9.6f}  "
              f"ratio {pm / pa:7.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, u_edges=ue, dens=dens, dens_err=derr, u_centres=uc, counts=cnt,
             tail_u0=U0, tail_p=tail_p, tail_err=tail_e,
             tail_analytic=ANA_TAIL,
             n=np.int64(n), mean_u=np.float64(mean_u),
             frac_unrad=np.float64(f_zero),
             frac_no_masslep=np.float64(1.0 - f_rad),
             mean_u2=np.float64(float((u * u).mean())),
             f_loss=np.float64(f_loss),
             dens_analytic=ana_dens,
             cut_grid=np.array(cut_grid), cut_mean_u=np.array(cut_meas),
             cut_mean_u_analytic=np.array(cut_ana),
             cond_ucut=np.float64(ucut), cond_u0=np.array(cond_u0),
             cond_tail_p=np.array(cond_m),
             cond_tail_analytic=np.array(cond_a),
             mean_u_analytic_photon=np.float64(ANA_MEAN_U_GAM),
             mean_u_analytic_total=np.float64(ANA_MEAN_U),
             frac_neg_u=np.float64(float((u < 0).mean())),
             max_chi2_ndof=np.float64(args.max_chi2_ndof))
    print()
    print("wrote", args.output)


if __name__ == "__main__":
    main()

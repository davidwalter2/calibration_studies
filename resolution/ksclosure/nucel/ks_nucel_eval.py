#!/usr/bin/env python3
"""Does the nuclear-elastic family describe the K_S mass tail?  Candidate-level
diagnostics of the per-candidate CF density WITH and WITHOUT the family, on the
same candidates, at each variant's own fitted scale.

For every candidate the mass-residual pull z = (m_reco - m_gen)/sigma_m is set
against its own CF density,

    phi_i(t) = exp( -vgf t^2/2 + S_ms + S_ioni + S_rad [+ S_nucel] ),

(the exponents cubic-spline upsampled x4 on the maker's 64-point grid, as
`MassCFTerm` does), shifted by the fitted m_ref alpha 1e-3 / sigma_i.  Reported:

  * the tail fraction P(|z| >= 3) and >= 5 in the data against the model's
    expectation sum_i P_i(|z| >= c) / n -- inclusive and per decay-radius /
    momentum bin (the radius dependence of the tail);
  * the PIT pull  z_PIT = Phi^-1( F_i(z_i) ), F_i by Gil-Pelaez inversion:
    N(0, 1) exactly if the density is right, whatever its shape -- the
    "pull of the new density": width, robust width, P(|z_PIT| >= 3) against
    0.27 %;
  * a binned Pearson chi2/ndof of the pull histogram (+-10, 80 bins, bins with
    expectation >= 5) against the summed densities;
  * figures (PDF + PNG, ratio panels) in ~/public_html/ZMass/cvh/<date>_ks_nucel/.

The corrections of the fit (a_res, Jensen) move the density by ~0.01 sigma and
are not applied here; alpha is.

usage:
  python3 ks_nucel_eval.py --base kspairs_base_all.npz --nuc kspairs_nuc_all.npz \
      --alpha-base <json> --alpha-nuc <json> --out eval.json
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtri

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import mplhep as hep                   # noqa: E402
from wums import logging               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RES)
import pubhtml                         # noqa: E402
import ratiopanel                      # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3)
M_REF = 0.497611
UPS = 4


def load(fn, with_nucel):
    d = np.load(fn, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['_nucel'] = with_nucel and ('Snuc_re' in d.files)
    if with_nucel and not out['_nucel']:
        raise SystemExit(f'{fn} has no Snuc_re')
    return out


def fine_grid(tg):
    return np.linspace(tg[0], tg[-1], (len(tg) - 1) * UPS + 1)


def exponent(d, sl, tg, tf):
    """Upsampled complex exponent (n, ntf) for the slice sl."""
    S = (-0.5 * d['vgf'][sl].astype(np.float64)[:, None] * tg[None, :] ** 2
         + d['Sms'][sl].astype(np.float64)
         + d['Sio_re'][sl].astype(np.float64) + 1j * d['Sio_im'][sl].astype(np.float64))
    if 'Srad_re' in d:
        S = S + d['Srad_re'][sl].astype(np.float64) + 1j * d['Srad_im'][sl].astype(np.float64)
    if d['_nucel']:
        S = S + d['Snuc_re'][sl].astype(np.float64)
        if 'Snuc_im' in d:
            S = S + 1j * d['Snuc_im'][sl].astype(np.float64)
    if UPS == 1:
        return S
    return (CubicSpline(tg, S.real, axis=1)(tf) + 1j * CubicSpline(tg, S.imag, axis=1)(tf))


def trap_w(t):
    w = np.gradient(t)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def evaluate(d, alpha, zgrid, chunk=512):
    """Per-candidate CDF at the observed z and the summed density on zgrid."""
    tg = np.asarray(d['tgrid'], dtype=np.float64)
    tf = fine_grid(tg)
    w = trap_w(tf)
    n = len(d['z'])
    z = d['z'].astype(np.float64)
    sig = d['sigma'].astype(np.float64)
    shift = M_REF * alpha * 1e-3 / sig
    x = z - shift
    F = np.empty(n)
    P3 = np.empty(n)
    P5 = np.empty(n)
    dens = np.zeros(len(zgrid))
    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        phi = np.exp(exponent(d, sl, tg, tf))                # (c, t)
        # Gil-Pelaez: F(x) = 1/2 - (1/pi) int_0^T Im(e^{-itx} phi(t))/t dt
        mu = phi[:, 1].imag / tf[1]                          # Im phi'(0)
        def cdf(xx):
            g = np.empty(phi.shape)
            e = np.exp(-1j * tf[None, 1:] * xx[:, None]) * phi[:, 1:]
            g[:, 1:] = e.imag / tf[None, 1:]
            g[:, 0] = mu - xx
            return 0.5 - (g * w[None, :]).sum(1) / np.pi
        xs = x[sl]
        F[sl] = cdf(xs)
        sh = shift[sl]
        for c, P in ((3.0, P3), (5.0, P5)):
            P[sl] = 1.0 - (cdf(c - sh) - cdf(-c - sh))
        arg = tf[None, None, :] * (zgrid[None, :, None] - sh[:, None, None])
        dens += np.einsum('ct,cnt->n', phi.real * w, np.cos(arg)) / np.pi
        dens += np.einsum('ct,cnt->n', phi.imag * w, np.sin(arg)) / np.pi
    return F, P3, P5, dens / n


def summarize(tag, d, F, P3, P5, edges, dmid, dlo, dhi):
    z = d['z'].astype(np.float64)
    zp = ndtri(np.clip(F, 1e-12, 1 - 1e-12))
    h, _ = np.histogram(z, bins=edges)
    mavg = ratiopanel.bin_average(dlo, dmid, dhi)
    expc = mavg * len(z) * np.diff(edges)
    use = expc >= 5
    chi2 = float(((h - expc) ** 2 / expc)[use].sum())
    ndof = int(use.sum()) - 1
    res = dict(tag=tag, n=len(z),
               data_tail3=float(np.mean(np.abs(z) >= 3)), model_tail3=float(P3.mean()),
               data_tail5=float(np.mean(np.abs(z) >= 5)), model_tail5=float(P5.mean()),
               pit_std=float(zp.std()),
               pit_rob=float(0.5 * np.diff(np.percentile(zp, [15.865, 84.135]))[0]),
               pit_mean=float(zp.mean()),
               pit_tail3=float(np.mean(np.abs(zp) >= 3)),
               pit_hi3=float(np.mean(zp >= 3)), pit_lo3=float(np.mean(zp <= -3)),
               data_hi3=float(np.mean(z >= 3)), data_lo3=float(np.mean(z <= -3)),
               chi2=chi2, ndof=ndof)
    logger.info(f"{tag}: |z|>=3 data {100*res['data_tail3']:.3f} % model {100*res['model_tail3']:.3f} %; "
                f"|z|>=5 data {100*res['data_tail5']:.3f} % model {100*res['model_tail5']:.3f} %; "
                f"PIT std {res['pit_std']:.4f} rob {res['pit_rob']:.4f} |z_PIT|>=3 "
                f"{100*res['pit_tail3']:.3f} % (hi {100*res['pit_hi3']:.3f} lo {100*res['pit_lo3']:.3f}); "
                f"chi2/ndof {chi2:.1f}/{ndof}")
    return res, zp, h, mavg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--nuc', required=True)
    ap.add_argument('--third', default='',
                    help='a third arm (a diagnostic variant of the family)')
    ap.add_argument('--third-label', default='third arm')
    ap.add_argument('--alpha-base', required=True)
    ap.add_argument('--alpha-nuc', required=True)
    ap.add_argument('--alpha-third', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--outdir', default=os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today().strftime('%y%m%d')}_ks_nucel"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pubhtml.ensure_index(args.outdir)

    def alpha_of(js):
        p = json.load(open(js))
        return float(p['fitted'][p['params'].index('alpha')])

    edges = np.linspace(-10., 10., 81)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    variants = [('without nuclear elastic', args.base, False, args.alpha_base, 'off'),
                ('with nuclear elastic', args.nuc, True, args.alpha_nuc, 'on')]
    if args.third:
        variants.append((args.third_label, args.third, True, args.alpha_third, 'third'))
    out = {}
    store = {}
    for label, fn, nuc, js, tg in variants:
        d = load(fn, nuc)
        al = alpha_of(js)
        grid = np.concatenate([edges[:-1], ctr, edges[1:]])
        F, P3, P5, dens = evaluate(d, al, grid)
        nb = len(ctr)
        dlo, dmid, dhi = dens[:nb], dens[nb:2 * nb], dens[2 * nb:]
        res, zp, h, mavg = summarize(tg, d, F, P3, P5, edges, dmid, dlo, dhi)
        res['alpha'] = al
        # the tail per bin of decay radius / softer-pion momentum / origin
        bins = {}
        rd = d['ks_rdec']
        pm = np.minimum(d['ks_pgenp'], d['ks_pgenm'])
        for bl, m in (('r0_1', rd < 1), ('r1_2', (rd >= 1) & (rd < 2)),
                      ('r2_3', (rd >= 2) & (rd < 3)), ('r3_4', (rd >= 3) & (rd < 4)),
                      ('r4_10', (rd >= 4) & (rd < 10)), ('r10_60', rd >= 10),
                      ('plo', pm < 0.8), ('pmid', (pm >= 0.8) & (pm < 1.5)), ('phi', pm >= 1.5),
                      ('fromb', d['ks_fromb'] > 0), ('prompt', d['ks_fromb'] == 0)):
            z = d['z'][m]
            bins[bl] = dict(n=int(m.sum()), data3=float(np.mean(np.abs(z) >= 3)),
                            model3=float(P3[m].mean()), data5=float(np.mean(np.abs(z) >= 5)),
                            model5=float(P5[m].mean()),
                            pit3=float(np.mean(np.abs(zp[m]) >= 3)),
                            pitstd=float(zp[m].std()),
                            nucN=float(d['nuc_N'][m].mean()) if 'nuc_N' in d else np.nan)
        res['bins'] = bins
        out[tg] = res
        store[tg] = dict(label=label, h=h, mavg=mavg, zp=zp, P3=P3, d=d)
    json.dump(out, open(args.out, 'w'), indent=1)
    logger.info(f"wrote {args.out}")

    # ---- figures -------------------------------------------------------------
    z = store['off']['d']['z']
    n = len(z)
    wbin = edges[1] - edges[0]
    cols = {'off': 'crimson', 'on': 'royalblue', 'third': 'darkgreen'}
    for suffix, logy in (('', False), ('_log', True)):
        fig, ax, rax = ratiopanel.make_ratio_fig()
        h = store['off']['h']
        ax.errorbar(ctr, h, yerr=np.sqrt(np.maximum(h, 1)), fmt='o', ms=3, color='black',
                    label=r'$K^0_S$ candidates (truth-matched)')
        for tg in [t for t in ('off', 'on', 'third') if t in store]:
            ax.plot(ctr, store[tg]['mavg'] * n * wbin, color=cols[tg], lw=1.5,
                    label=f"CF model, {store[tg]['label']}")
        ax.set_ylabel('candidates / bin')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(0.5, None)
        ax.legend(fontsize='small')
        ratiopanel.draw_ratio(rax, edges, h, store['on']['mavg'], n,
                              ylabel='data / model', clamp=(0.0, 3.0),
                              xlabel=r'$(m_{\mathrm{reco}}-m_{\mathrm{gen}})/\sigma_m$',
                              color=cols['on'])
        for tg, mk in (('off', 's'), ('third', '^')):
            if tg not in store:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                rr = (h / (n * wbin)) / store[tg]['mavg']
                ee = rr / np.sqrt(np.where(h > 0, h, np.nan))
            rax.errorbar(ctr, rr, ee, fmt=mk, ms=3, color=cols[tg], elinewidth=0.8)
        rax.set_ylim(0.0, 3.0)
        pubhtml.savefig(fig, os.path.join(args.outdir, f'ks_nucel_pull{suffix}.pdf'))
        plt.close(fig)

    # the PIT pull
    pe = np.linspace(-6, 6, 61)
    pc = 0.5 * (pe[1:] + pe[:-1])
    from scipy.stats import norm
    gauss = (norm.cdf(pe[1:]) - norm.cdf(pe[:-1])) / np.diff(pe)
    fig, ax, rax = ratiopanel.make_ratio_fig()
    for tg in ('off', 'on'):
        hp, _ = np.histogram(store[tg]['zp'], bins=pe)
        ax.errorbar(pc, hp, yerr=np.sqrt(np.maximum(hp, 1)), fmt='o', ms=3, color=cols[tg],
                    label=f"PIT pull, {store[tg]['label']}")
        if tg == 'on':
            ratiopanel.draw_ratio(rax, pe, hp, gauss, n, ylabel='data / N(0,1)',
                                  clamp=(0.0, 4.0), color=cols['on'],
                                  xlabel=r'$\Phi^{-1}(F_i(z_i))$')
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                rax.plot(pc, (hp / (n * np.diff(pe))) / gauss, 's', ms=3, color=cols['off'])
    ax.plot(pc, gauss * n * np.diff(pe), color='gray', lw=1.2, label='N(0,1)')
    ax.set_yscale('log')
    ax.set_ylim(0.5, None)
    ax.set_ylabel('candidates / bin')
    ax.legend(fontsize='small')
    pubhtml.savefig(fig, os.path.join(args.outdir, 'ks_nucel_pit.pdf'))
    plt.close(fig)

    # the tail per decay-radius bin: data vs model
    rb = ['r0_1', 'r1_2', 'r2_3', 'r3_4', 'r4_10', 'r10_60']
    rx = np.array([0.5, 1.5, 2.5, 3.5, 7.0, 20.0])
    fig, ax, rax = ratiopanel.make_ratio_fig()
    b0 = out['off']['bins']
    d3 = np.array([b0[k]['data3'] for k in rb])
    nn = np.array([b0[k]['n'] for k in rb])
    e3 = np.sqrt(d3 * (1 - d3) / nn)
    ax.errorbar(rx, 100 * d3, 100 * e3, fmt='o', color='black', label='data')
    for tg in [t for t in ('off', 'on', 'third') if t in store]:
        m3 = np.array([out[tg]['bins'][k]['model3'] for k in rb])
        ax.plot(rx, 100 * m3, 's-', color=cols[tg], label=f"model, {store[tg]['label']}")
        rax.errorbar(rx, d3 / m3, e3 / m3, fmt='o', color=cols[tg])
    ax.axhline(0.27, color='gray', ls=':', lw=1, label='Gaussian')
    ax.set_xscale('log')
    rax.set_xscale('log')
    for rr, lab in ((2.2, 'beam pipe'), (2.9, 'BPix L1')):
        ax.axvline(rr, color='0.6', ls='--', lw=0.8)
    ax.set_ylabel(r'$P(|z| \geq 3)$ [%]')
    ax.legend(fontsize='small')
    rax.axhline(1.0, color='gray')
    rax.set_ylabel('data / model', fontsize='small')
    rax.set_xlabel(r'true $K^0_S$ decay radius [cm]')
    pubhtml.savefig(fig, os.path.join(args.outdir, 'ks_nucel_tail_vs_radius.pdf'))
    plt.close(fig)
    logger.info(f"figures in {args.outdir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Figures for the K_S -> pi pi displaced closure.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 ks_plots.py --cache runs/kspairs_all.npz --tag ksclosure
"""
import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pubhtml      # noqa: E402
import ratiopanel   # noqa: E402

hep.style.use(hep.style.ROOT)

M_KS = 0.497611


def hist(ax, x, bins, label=None, **kw):
    h, e = np.histogram(x, bins=bins)
    c = 0.5 * (e[1:] + e[:-1])
    ax.step(e, np.r_[h[0], h], where='pre', label=label, **kw)
    return h, e, c


def model_pull_density(d, zgrid, alpha=0.0, chunk=8192):
    """Sum of the per-candidate CF model densities on a pull grid.

    Each candidate's log-CF is S_i(tau) = -vgf_i tau^2/2 + S_ms + S_ioni
    + S_rad, in the variable x = (m_reco - m_true)/sigma_i, so the density of
    the PULL is the cosine/sine transform of exp(S_i) -- the same construction
    the likelihood integrates, with every k family at 1.  `alpha` shifts the
    reference by m_ref*alpha*1e-3 in each candidate's own sigma units.
    """
    TG = np.asarray(d['tgrid'], dtype=np.float64)
    n = len(d['z'])
    sig = d['sigma'].astype(np.float64)
    vgf = d['vgf'].astype(np.float64)
    shift = (M_KS * alpha * 1e-3) / sig
    w = np.gradient(TG)
    w[0] *= 0.5
    w[-1] *= 0.5
    out = np.zeros(len(zgrid))
    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        S = (-0.5 * vgf[sl][:, None] * TG[None, :] ** 2
             + d['Sms'][sl].astype(np.float64)
             + d['Sio_re'][sl].astype(np.float64)
             + 1j * d['Sio_im'][sl].astype(np.float64))
        if 'Srad_re' in d.files:
            S = S + (d['Srad_re'][sl].astype(np.float64)
                     + 1j * d['Srad_im'][sl].astype(np.float64))
        phi = np.exp(S)                                   # (nc, nt)
        arg = TG[None, None, :] * (zgrid[None, :, None] - shift[sl][:, None, None])
        out += np.einsum('ct,cnt->n', phi.real * w, np.cos(arg)) / np.pi
        out += np.einsum('ct,cnt->n', phi.imag * w, np.sin(arg)) / np.pi
    return out / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tag', default='ksclosure')
    ap.add_argument('--outpath', default=None)
    ap.add_argument('--jpsi-cache', default='/work/submit/david_w/ZMass/'
                    'calibration_studies/resolution/runs/'
                    'cf_masspairs_btojpsix_v3_260904f_m0.npz',
                    help='J/psi -> mu mu cache of the SAME MC, for the '
                         'channel comparison; "" to skip')
    args = ap.parse_args()
    out = args.outpath or pubhtml.figdir(args.tag)
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)

    d = np.load(args.cache, allow_pickle=True)
    z = d['z']
    sig = d['sigma']
    mgen = d['eta']
    m = mgen + sig * z
    res = m - mgen
    r = d['ks_rdec'] if 'ks_rdec' in d.files else None
    fromb = d['ks_fromb'] if 'ks_fromb' in d.files else None
    print(f'{len(z)} candidates')

    # 1. refit K_S mass
    fig, ax = plt.subplots(figsize=(8, 6))
    hist(ax, m, np.linspace(0.46, 0.535, 76), color='black')
    ax.axvline(M_KS, color='crimson', ls='--', lw=1.2, label=r'$M(\mathrm{K^0_S})$ PDG')
    ax.set_xlabel(r'$m(\pi^+\pi^-)$ after the CVH refit [GeV]')
    ax.set_ylabel('candidates / bin')
    ax.legend()
    pubhtml.savefig(fig, os.path.join(out, 'ks_mass.pdf'))
    plt.close(fig)

    # 2. residual and pull
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    hist(axs[0], 1e3 * res, np.linspace(-40, 40, 81), color='black')
    axs[0].set_xlabel(r'$m_{\mathrm{reco}} - m_{\mathrm{gen}}$ [MeV]')
    axs[0].set_ylabel('candidates / bin')
    axs[0].text(0.03, 0.95, f'median {1e3*np.median(res):+.3f} MeV\n'
                            f'RMS {1e3*res.std():.2f} MeV',
                transform=axs[0].transAxes, va='top', fontsize=13)
    hist(axs[1], z, np.linspace(-6, 6, 61), color='black')
    axs[1].set_xlabel(r'$(m_{\mathrm{reco}} - m_{\mathrm{gen}})/\sigma_m$')
    axs[1].set_ylabel('candidates / bin')
    rob = 0.7413 * (np.quantile(z, .75) - np.quantile(z, .25))
    axs[1].text(0.03, 0.95, f'median {np.median(z):+.3f}\nrobust width {rob:.3f}',
                transform=axs[1].transAxes, va='top', fontsize=13)
    pubhtml.savefig(fig, os.path.join(out, 'ks_residual_pull.pdf'))
    plt.close(fig)

    # 3. decay radius and sigma_m/m
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    if r is not None:
        hist(axs[0], np.clip(r, 0, 60), np.linspace(0, 60, 61), color='black')
        axs[0].set_xlabel(r'$K^0_S$ decay radius [cm]')
        axs[0].set_ylabel('candidates / bin')
        axs[0].set_yscale('log')
    hist(axs[1], sig / m, np.linspace(0, 0.05, 51), color='black')
    axs[1].set_xlabel(r'$\sigma_m/m$')
    axs[1].set_ylabel('candidates / bin')
    pubhtml.savefig(fig, os.path.join(out, 'ks_radius_sigma.pdf'))
    plt.close(fig)

    # 4. residual vs decay radius (the displaced test, model-free)
    if r is not None:
        edges = np.array([0., 1., 2., 3., 4., 6., 10., 20., 60.])
        ctr, med, err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (r >= lo) & (r < hi)
            if s.sum() < 20:
                continue
            ctr.append(0.5 * (lo + hi))
            med.append(1e3 * np.median(res[s]))
            err.append(1e3 * 1.2533 * res[s].std() / np.sqrt(s.sum()))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(ctr, med, yerr=err, fmt='o', color='black')
        ax.axhline(0, color='crimson', ls='--', lw=1.0)
        ax.set_xlabel(r'$K^0_S$ decay radius [cm]')
        ax.set_ylabel(r'median $(m_{\mathrm{reco}}-m_{\mathrm{gen}})$ [MeV]')
        pubhtml.savefig(fig, os.path.join(out, 'ks_resid_vs_radius.pdf'))
        plt.close(fig)

    # 4b. pull density against the CF model, with a ratio panel.
    # Two ranges: the core, and a log-scale +-10 sigma view where an
    # UNMODELLED tail would show. The in-maker CF has families hit / MS /
    # ionisation / radiative and NO nuclear-elastic term, while a pion crossing
    # the tracker takes on average 0.05 elastic nuclear collisions of 25-35
    # mrad -- so 5-10 % of K_S candidates carry one unmodelled angular kick on
    # one leg. That is a tail, not a width, and this is where it would appear.
    for suffix, edges, logy in (('', np.linspace(-5., 5., 51), False),
                                ('_tails', np.linspace(-10., 10., 81), True)):
      try:
        ctr = 0.5 * (edges[1:] + edges[:-1])
        h, _ = np.histogram(z, bins=edges)
        mlo = model_pull_density(d, edges[:-1])
        mhi = model_pull_density(d, edges[1:])
        mmid = model_pull_density(d, ctr)
        mavg = ratiopanel.bin_average(mlo, mmid, mhi)
        fig, ax, rax = ratiopanel.make_ratio_fig()
        wbin = edges[1] - edges[0]
        ax.errorbar(ctr, h, yerr=np.sqrt(np.maximum(h, 1)), fmt='o', ms=3,
                    color='black', label='candidates')
        ax.plot(ctr, mavg * len(z) * wbin, color='crimson', lw=1.5,
                label='per-candidate CF model')
        ax.set_ylabel('candidates / bin')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(0.5, None)
        ax.legend()
        ratiopanel.draw_ratio(rax, edges, h, mavg, len(z),
                              ylabel='data / model', clamp=(0.0, 3.0),
                              xlabel=r'$(m_{\mathrm{reco}}-m_{\mathrm{gen}})/\sigma_m$')
        pubhtml.savefig(fig, os.path.join(out, f'ks_pull_model{suffix}.pdf'))
        plt.close(fig)
        core = np.abs(z) < 2.
        tail = np.abs(z) >= 3.
        print(f'pull{suffix}: |z|<2 {100*core.mean():.2f} %, |z|>=3 '
              f'{100*tail.mean():.3f} % of candidates')
      except Exception as e:
        print('pull-vs-model plot skipped:', type(e).__name__, e)

    # 4c. chi2/ndof, and the channel comparison against the J/psi of the same MC
    if 'chisqval' in d.files and 'ndof' in d.files:
        c2 = d['chisqval'] / np.maximum(d['ndof'], 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        hist(ax, np.clip(c2, 0, 4), np.linspace(0, 4, 61), color='black',
             label=r'$K^0_S \to \pi^+\pi^-$')
        ax.axvline(3.0, color='crimson', ls='--', lw=1.0,
                   label=r'selection $\chi^2/\mathrm{ndof} < 3$')
        ax.set_xlabel(r'two-track fit $\chi^2/\mathrm{ndof}$')
        ax.set_ylabel('candidates / bin')
        ax.legend()
        pubhtml.savefig(fig, os.path.join(out, 'ks_chi2ndof.pdf'))
        plt.close(fig)
        print(f'chi2/ndof: median {np.median(c2):.3f}, '
              f'>3 {100*(c2 > 3).mean():.2f} %; ndof median {np.median(d["ndof"]):.0f}')

    jp = None
    if args.jpsi_cache and os.path.exists(args.jpsi_cache):
        jp = np.load(args.jpsi_cache, allow_pickle=True)
        zj = jp['z']
        mj = jp['eta'] + jp['sigma'] * zj
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        for a_, x, lab in ((axs[0], sig / m, r'$K^0_S \to \pi\pi$'),
                           (axs[0], jp['sigma'] / mj, r'$J/\psi \to \mu\mu$')):
            hist(a_, x, np.linspace(0, 0.03, 61), label=lab,
                 color=('black' if 'K' in lab else 'crimson'))
        axs[0].set_xlabel(r'$\sigma_m/m$')
        axs[0].set_ylabel('candidates / bin (each normalised below)')
        axs[0].set_yscale('log')
        axs[0].legend()
        for a_, x, lab in ((axs[1], z, r'$K^0_S \to \pi\pi$'),
                           (axs[1], zj, r'$J/\psi \to \mu\mu$')):
            h, e = np.histogram(x, bins=np.linspace(-6, 6, 61))
            a_.step(e, np.r_[h[0], h] / max(h.sum(), 1), where='pre', label=lab,
                    color=('black' if 'K' in lab else 'crimson'))
        axs[1].set_xlabel(r'$(m_{\mathrm{reco}}-m_{\mathrm{gen}})/\sigma_m$')
        axs[1].set_ylabel('fraction of candidates / bin')
        axs[1].set_yscale('log')
        axs[1].legend()
        pubhtml.savefig(fig, os.path.join(out, 'ks_vs_jpsi.pdf'))
        plt.close(fig)
        rob = lambda x: 0.7413 * (np.quantile(x, .75) - np.quantile(x, .25))
        print(f'J/psi (same MC): n {len(zj)}, sigma/m med '
              f'{np.median(jp["sigma"]/mj):.5f}, pull med {np.median(zj):+.4f} '
              f'std {zj.std():.4f} robust {rob(zj):.4f}, vgf med '
              f'{np.median(jp["vgf"]):.4f}')
        print(f'K_S:            n {len(z)}, sigma/m med {np.median(sig/m):.5f}, '
              f'pull med {np.median(z):+.4f} std {z.std():.4f} '
              f'robust {rob(z):.4f}, vgf med {np.median(d["vgf"]):.4f}')

    # 5. resolution family shares at tau -> the CF exponents
    tg = d['tgrid']
    it = min(len(tg) - 1, int(np.searchsorted(tg, 2.0)))
    fams = [('hit', -0.5 * d['vgf'] * tg[it] ** 2),
            ('MS', d['Sms'][:, it]),
            ('ionisation', d['Sio_re'][:, it]),
            ('radiative', d['Srad_re'][:, it])]
    tot = sum(np.abs(v) for _, v in fams)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels, vals = [], []
    for nm, v in fams:
        labels.append(nm)
        vals.append(np.median(np.abs(v) / np.maximum(tot, 1e-30)))
    ax.bar(labels, vals, color=['#4878d0', '#ee854a', '#6acc64', '#d65f5f'])
    ax.set_ylabel(fr'median share of $|S_f(\tau={tg[it]:.2f})|$')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{100*v:.1f}%', ha='center', va='bottom')
    pubhtml.savefig(fig, os.path.join(out, 'ks_families.pdf'))
    plt.close(fig)
    print('family shares at tau=%.2f: ' % tg[it]
          + ', '.join(f'{n} {100*v:.1f}%' for n, v in zip(labels, vals)))
    if jp is not None and 'tgrid' in jp.files:
        tgj = jp['tgrid']
        itj = min(len(tgj) - 1, int(np.searchsorted(tgj, tg[it])))
        fj = [('hit', -0.5 * jp['vgf'] * tgj[itj] ** 2), ('MS', jp['Sms'][:, itj]),
              ('ionisation', jp['Sio_re'][:, itj]), ('radiative', jp['Srad_re'][:, itj])]
        totj = sum(np.abs(v) for _, v in fj)
        print('J/psi family shares at tau=%.2f: ' % tgj[itj]
              + ', '.join(f'{n} {100*np.median(np.abs(v)/np.maximum(totj,1e-30)):.1f}%'
                          for n, v in fj))
    print('figures ->', out)


if __name__ == '__main__':
    main()

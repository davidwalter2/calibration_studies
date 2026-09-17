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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tag', default='ksclosure')
    ap.add_argument('--outpath', default=None)
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
    print('figures ->', out)


if __name__ == '__main__':
    main()

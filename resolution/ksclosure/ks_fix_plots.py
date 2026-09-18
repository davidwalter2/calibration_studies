#!/usr/bin/env python3
"""Old code beside fixed code for the K_S -> pi pi displaced closure.

The 260917 production was made before the MS within-step correlation-sign fix
(`res(1,4) = +S3`); the 260918 one after it, from the same inputs with the same
flags.  Every figure here puts the two side by side on the SAME candidates --
the two pairs caches are matched on (run, lumi, event) and the ordinal within
the event -- so nothing but the CVH code differs.

  usage:
    source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    python3 ks_fix_plots.py --old runs/kspairs_all.npz \
        --new runs/fixed/kspairs_all.npz --tag ksclosure_fixed
"""
import argparse
import json
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
C_OLD, C_NEW = 'tab:blue', 'crimson'
L_OLD, L_NEW = 'before the fix (260917)', 'after the fix (260918)'


def load(fn):
    d = dict(np.load(fn, allow_pickle=True))
    d['m'] = d['eta'] + d['sigma'] * d['z']
    return d


def ordkey(d):
    """(run, lumi, event, ordinal-within-event) -- the same witness
    `resolution/msksign/compare_arms.py` matches on."""
    k = np.stack([d['run'].astype(np.int64), d['lumi'].astype(np.int64),
                  d['event'].astype(np.int64)], axis=1)
    seen, o = {}, np.zeros(len(k), dtype=np.int64)
    for i, t in enumerate(map(tuple, k)):
        o[i] = seen.get(t, 0)
        seen[t] = o[i] + 1
    return [(*t, int(x)) for t, x in zip(map(tuple, k), o)]


def match(a, b):
    ia = {t: i for i, t in enumerate(ordkey(a))}
    pr = [(ia[t], j) for j, t in enumerate(ordkey(b)) if t in ia]
    return np.array([p[0] for p in pr]), np.array([p[1] for p in pr])


def stepped(ax, x, edges, **kw):
    h, e = np.histogram(x, bins=edges)
    ax.step(e, np.r_[h[0], h], where='pre', **kw)
    return h


def diff_panel(rax, x, dy, ey, ylabel):
    rax.errorbar(x, dy, yerr=ey, fmt='o', ms=4, color='black')
    rax.axhline(0.0, color='grey', ls='--', lw=1.0)
    rax.set_ylabel(ylabel, fontsize=13)


def med_vs(x, y, edges, minn=20):
    """median of y with its 1.2533 sigma/sqrt(n) error, in bins of x"""
    c, m, e = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < minn:
            continue
        c.append(0.5 * (lo + hi))
        m.append(np.median(y[s]))
        e.append(1.2533 * y[s].std() / np.sqrt(s.sum()))
    return np.array(c), np.array(m), np.array(e)


def eps_bins(runs, tags, labels):
    """(label, eps, err) per fit tag from a runs directory, NaN when absent"""
    out = []
    for t, lab in zip(tags, labels):
        fn = os.path.join(runs, 'results', f'{t}.json')
        cache = os.path.join(runs, f'kspairs_{t}.npz')
        if not (os.path.exists(fn) and os.path.exists(cache)):
            out.append((lab, float('nan'), float('nan')))
            continue
        r = json.load(open(fn))
        i = r['params'].index('alpha')
        with np.load(cache, allow_pickle=True) as d:
            w = 1.0 / np.asarray(d['sigma'], dtype=np.float64) ** 2
            f = np.asarray(d['ks_fmom'], dtype=np.float64)
            fw = float((w * f).sum() / w.sum())
        out.append((lab, r['fitted'][i] / fw, r['err'][i] / fw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--old-runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--new-runs', default=os.path.join(HERE, 'runs', 'fixed'))
    ap.add_argument('--tag', default='ksclosure_fixed')
    ap.add_argument('--outpath', default=None)
    args = ap.parse_args()

    out = args.outpath or pubhtml.figdir(args.tag)
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)

    A, B = load(args.old), load(args.new)
    print(f'old {len(A["z"])} candidates, new {len(B["z"])}')
    si, sj = match(A, B)
    print(f'matched {len(si)}')

    # ---------------------------------------------------- 1. what the fix moved
    ma, mb = A['m'][si], B['m'][sj]
    sa, sb = A['sigma'][si], B['sigma'][sj]
    dm = (mb - ma) / ma
    ds = (sb - sa) / sa
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    stepped(axs[0], np.clip(dm, -2e-3, 2e-3), np.linspace(-2e-3, 2e-3, 81),
            color='black')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'$(m_{\rm fixed}-m_{\rm old})/m$')
    axs[0].set_ylabel('candidates / bin')
    axs[0].text(0.03, 0.95, f'median $|\\Delta m/m|$ {np.median(np.abs(dm)):.2e}\n'
                f'{100*(np.abs(dm) > 1e-4).mean():.1f} % above $10^{{-4}}$',
                transform=axs[0].transAxes, va='top', fontsize=13)
    stepped(axs[1], np.clip(ds, -0.01, 0.03), np.linspace(-0.01, 0.03, 81),
            color='black')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$(\sigma_{m,\rm fixed}-\sigma_{m,\rm old})/\sigma_m$')
    axs[1].set_ylabel('candidates / bin')
    axs[1].text(0.60, 0.95, f'mean {ds.mean():+.2e}\nmedian {np.median(ds):+.2e}',
                transform=axs[1].transAxes, va='top', fontsize=13)
    pubhtml.savefig(fig, os.path.join(out, 'ksfix_shift.pdf'))
    plt.close(fig)
    print(f'dm/m: median|.| {np.median(np.abs(dm)):.3e}, '
          f'p95|.| {np.percentile(np.abs(dm), 95):.3e}, '
          f'>1e-4 {100*(np.abs(dm) > 1e-4).mean():.2f} %')
    print(f'dsigma/sigma: mean {ds.mean():+.3e} +- {ds.std(ddof=1)/np.sqrt(len(ds)):.1e}, '
          f'median {np.median(ds):+.3e}')

    # ------------------------------------------------- 2. the pull, and its tail
    for suffix, edges, logy in (('', np.linspace(-5, 5, 51), False),
                                ('_tails', np.linspace(-10, 10, 81), True)):
        za = np.clip(A['z'], edges[0], edges[-1])
        zb = np.clip(B['z'], edges[0], edges[-1])
        ha, _ = np.histogram(za, bins=edges)
        hb, _ = np.histogram(zb, bins=edges)
        ha = ha * (len(zb) / len(za))       # same normalisation, different n
        ctr = 0.5 * (edges[1:] + edges[:-1])
        fig, ax, rax = ratiopanel.make_ratio_fig()
        ax.step(edges, np.r_[ha[0], ha], where='pre', color=C_OLD, label=L_OLD)
        ax.errorbar(ctr, hb, yerr=np.sqrt(np.maximum(hb, 1)), fmt='o', ms=3,
                    color=C_NEW, label=L_NEW)
        ax.set_ylabel('candidates / bin')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(0.5, None)
        ax.legend()
        good = ha > 0
        r = np.where(good, hb / np.maximum(ha, 1e-9), np.nan)
        er = np.where(good, np.sqrt(np.maximum(hb, 1)) / np.maximum(ha, 1e-9), np.nan)
        rax.errorbar(ctr, r, yerr=er, fmt='o', ms=3, color='black')
        rax.axhline(1.0, color='grey', ls='--', lw=1.0)
        rax.set_ylim(0.7, 1.3)
        rax.set_ylabel('fixed / old', fontsize=13)
        rax.set_xlabel(r'$(m_{\rm reco}-m_{\rm gen})/\sigma_m$')
        pubhtml.savefig(fig, os.path.join(out, f'ksfix_pull{suffix}.pdf'))
        plt.close(fig)
    for nm, d in (('old', A), ('new', B)):
        z = d['z']
        print(f'{nm}: pull mean {z.mean():+.4f} median {np.median(z):+.4f} '
              f'std {z.std():.4f} robust {0.7413*(np.percentile(z, 75)-np.percentile(z, 25)):.4f} '
              f'|z|>=3 {100*(np.abs(z) >= 3).mean():.3f} %')

    # ------------------------------ 3. model-free median residual vs decay radius
    edges = np.array([0., 1., 2., 3., 4., 6., 10., 20., 60.])
    ca, mda, ea = med_vs(A['ks_rdec'], 1e3 * (A['m'] - A['eta']), edges)
    cb, mdb, eb = med_vs(B['ks_rdec'], 1e3 * (B['m'] - B['eta']), edges)
    fig, ax, rax = ratiopanel.make_ratio_fig()
    ax.errorbar(ca, mda, yerr=ea, fmt='s', ms=5, color=C_OLD, label=L_OLD)
    ax.errorbar(cb, mdb, yerr=eb, fmt='o', ms=5, color=C_NEW, label=L_NEW)
    ax.axhline(0, color='grey', ls='--', lw=1.0)
    ax.set_ylabel(r'median $(m_{\rm reco}-m_{\rm gen})$ [MeV]')
    ax.legend()
    n = min(len(ca), len(cb))
    diff_panel(rax, cb[:n], (mdb - mda)[:n], np.hypot(ea, eb)[:n], 'fixed $-$ old')
    rax.set_xlabel(r'$K^0_S$ decay radius [cm]')
    pubhtml.savefig(fig, os.path.join(out, 'ksfix_resid_vs_radius.pdf'))
    plt.close(fig)
    for lo, hi, x, y, e in zip(edges[:-1], edges[1:], cb, mdb, eb):
        print(f'  r {lo:5.1f}-{hi:5.1f} cm: median residual {y:+.3f} +- {e:.3f} MeV')

    # ------------------------------------------- 4. the closure in bins, old/new
    tags = ['all', 'fromb', 'prompt', 'r0_2', 'r2_4', 'r4_10', 'r10_60',
            'plo', 'pmid', 'phi']
    labs = ['all', 'from B', 'prompt', r'$r<2$', r'$2-4$', r'$4-10$',
            r'$r>10$', r'$p<0.8$', r'$0.8-1.5$', r'$p>1.5$']
    ea_ = eps_bins(args.old_runs, tags, labs)
    eb_ = eps_bins(args.new_runs, tags, labs)
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(x - 0.12, [r[1] for r in ea_], yerr=[r[2] for r in ea_],
                fmt='s', ms=6, color=C_OLD, label=L_OLD)
    ax.errorbar(x + 0.12, [r[1] for r in eb_], yerr=[r[2] for r in eb_],
                fmt='o', ms=6, color=C_NEW, label=L_NEW)
    ax.axhline(0, color='grey', ls='--', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha='right')
    ax.set_ylabel(r'momentum scale $\varepsilon$  [$10^{-3}$]')
    ax.legend()
    pubhtml.savefig(fig, os.path.join(out, 'ksfix_eps_bins.pdf'))
    plt.close(fig)
    print(f'{"bin":<12} {"eps old":>18} {"eps fixed":>18}')
    for (l, a1, a2), (_, b1, b2) in zip(ea_, eb_):
        print(f'{l:<12} {a1:+9.4f} +- {a2:6.4f} {b1:+9.4f} +- {b2:6.4f}')
    print('figures ->', out)


if __name__ == '__main__':
    main()

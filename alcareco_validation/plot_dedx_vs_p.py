"""
Plot dE/dx vs. track momentum p for V0 daughters from the CVH refit output.

Inputs (CVH ntuplizer per-channel ROOT trees with the dE/dx branches
projected from the ALCAReco step via DeDxValueMapProjector):
  - KS:     globalcor_ks_*.root       both daughters are pi
  - Lambda: globalcor_lambda_*.root   Muplus = proton, Muminus = pi

Branches used:
  Mupluskin_pt, Mupluskin_eta, Muplus_dedxHarmonic2, Muplus_dedxPixelHarmonic2
  Muminuskin_pt, Muminuskin_eta, Muminus_dedxHarmonic2, Muminus_dedxPixelHarmonic2

Per channel x estimator (strip Harmonic2, pixel Harmonic2) we produce a 2D
hist of (p, dE/dx). For Lambda the two daughters are kept on separate panels
so the species bands are visible. For KS both daughters are pi+/pi- so they
are combined.

The plus/minus daughter naming is fixed by V0Producer's positive-charge-first
convention preserved through VertexCompositeCandidateRemapper: slot 0
(Muplus_*) = positive daughter (pi+ for KS, p/pbar for Lambda), slot 1
(Muminus_*) = negative daughter.

J/psi -> mu+ mu- is intentionally excluded: the dimuon ALCAReco does not
wire DeDxValueMapProjector (muon PID is not dE/dx-driven), so those CVH
trees have no dedx* branches.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ROOT
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools


def read_tree(path, tname='tree'):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return None
    t = f.Get(tname)
    if not t:
        return None
    n = int(t.GetEntries())
    branches = [
        'Mupluskin_pt', 'Mupluskin_eta',
        'Muminuskin_pt', 'Muminuskin_eta',
        'Muplus_dedxHarmonic2', 'Muplus_dedxPixelHarmonic2', 'Muplus_dedxAllHarmonic2',
        'Muminus_dedxHarmonic2', 'Muminus_dedxPixelHarmonic2', 'Muminus_dedxAllHarmonic2',
    ]
    available = {b.GetName() for b in t.GetListOfBranches()}
    arr = {b: np.zeros(n, dtype=np.float64) for b in branches if b in available}
    for i in range(n):
        t.GetEntry(i)
        for b in arr:
            arr[b][i] = float(getattr(t, b))
    f.Close()
    return arr


def dedx_vs_p_hist(ax, p, dedx, label, color):
    """2D histogram of (p, dE/dx) plotted as scatter for low N, hist2d above."""
    mask = np.isfinite(p) & np.isfinite(dedx) & (dedx > 0) & (p > 0)
    p, d = p[mask], dedx[mask]
    if len(p) == 0:
        ax.text(0.5, 0.5, '(no entries)', transform=ax.transAxes,
                ha='center', va='center', color=color, fontsize=12)
        return
    ax.scatter(p, d, s=12, alpha=0.6, color=color, label='%s (n=%d)' % (label, len(p)),
               edgecolor='none')


# Empirical CMS Ih parametrization for the Harmonic-2 estimator on silicon:
#   Ih(p, M) = K + C * M^2 / p^2     [MeV/cm]
# Reference: CMS HSCP analyses (e.g. EXO-12-026, arXiv:1305.0491). Constants
# are illustrative MIP-region values; absolute scale of strip vs pixel
# depends on calibration of each estimator.
IH_PARAMS = {
    'Harmonic2':      (2.30, 3.15),  # strip-only
    'PixelHarmonic2': (1.80, 2.50),  # pixel-only (slightly different MIP plateau)
    'AllHarmonic2':   (2.20, 3.00),  # joint strip+pixel (Harmonic-2 truncated, T085)
}

# Particle masses [GeV] for the reference curves.
REF_SPECIES = [
    (r'$\pi$',  0.139570, 'forestgreen', '--'),
    (r'$K$',    0.493677, 'royalblue',   '-.'),
    (r'$p$',    0.938272, 'darkorange',  ':'),
]


def overlay_bethe_bloch(ax, estimator_branch, p_grid):
    K, C = IH_PARAMS[estimator_branch]
    for label, mass, color, ls in REF_SPECIES:
        ih = K + C * mass**2 / np.maximum(p_grid, 1e-6)**2
        ax.plot(p_grid, ih, color=color, linestyle=ls, linewidth=1.6,
                label='%s ref' % label, zorder=5)


_EST_SHORT = {'Harmonic2': 'strip', 'PixelHarmonic2': 'pixel', 'AllHarmonic2': 'all'}

# Common axis ranges used by both the scatter and the 2D-histogram panels.
P_MIN, P_MAX = 0.2, 30.0
D_MIN, D_MAX = 1.0, 12.0


def _resolve_p_d(arr, p_branch, d_branch):
    """Compute (p, dE/dx) pair-wise from the per-event arrays, masking
    pathological entries (NaN, non-positive p or dE/dx)."""
    pt = arr[p_branch]
    eta = arr[p_branch.replace('_pt', '_eta')]
    d = arr[d_branch]
    p = pt * np.cosh(eta)
    mask = np.isfinite(p) & np.isfinite(d) & (d > 0) & (p > 0)
    return p[mask], d[mask]


def make_panel(channel_key, channel_label, estimator_branch, estimator_label,
               species_pairs, arr, outdir):
    """Scatter overlay: all species on a single panel. species_pairs is a
    list of (label, color, p_branch, dedx_branch) tuples. `channel_key` is
    the short ascii name (e.g. 'ks') used in the output filename."""
    fig, ax = plot_tools.figure(
        np.array([P_MIN, P_MAX]),
        xlabel=r'$p$ [GeV]',
        ylabel='dE/dx %s [MeV/cm]' % estimator_label,
        xlim=(P_MIN, P_MAX), ylim=(D_MIN, D_MAX),
        automatic_scale=False, width_scale=1.2)
    ax.set_xscale('log')

    for label, color, p_branch, d_branch in species_pairs:
        p, d = _resolve_p_d(arr, p_branch, d_branch)
        dedx_vs_p_hist(ax, p, d, label, color)

    p_grid = np.geomspace(P_MIN, P_MAX, 200)
    overlay_bethe_bloch(ax, estimator_branch, p_grid)

    ax.legend(loc='upper right', fontsize=10, frameon=True, ncol=2)
    ax.text(0.04, 0.96, channel_label, transform=ax.transAxes, va='top', ha='left',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
    base = '%s_dedx_%s_vs_p' % (channel_key, _EST_SHORT[estimator_branch])
    plot_tools.save_pdf_and_png(outdir, base, fig)
    plt.close(fig)
    print('  saved %s' % base)


def _hist2d_one_panel(channel_key, channel_label, estimator_branch, estimator_label,
                      p, d, panel_label, suffix, outdir):
    """Render one (p, dE/dx) 2D histogram into a single-panel figure and
    save as <chan>_dedx_<est>_vs_p_hist2d_<suffix>.{png,pdf}."""
    p_bins = np.geomspace(P_MIN, P_MAX, 80)
    d_bins = np.linspace(D_MIN, D_MAX, 80)
    p_grid = np.geomspace(P_MIN, P_MAX, 200)

    fig, ax = plot_tools.figure(
        np.array([P_MIN, P_MAX]),
        xlabel=r'$p$ [GeV]',
        ylabel='dE/dx %s [MeV/cm]' % estimator_label,
        xlim=(P_MIN, P_MAX), ylim=(D_MIN, D_MAX),
        automatic_scale=False, width_scale=1.2)
    ax.set_xscale('log')

    if len(p) == 0:
        ax.text(0.5, 0.5, '(no entries)', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    else:
        h, _, _ = np.histogram2d(p, d, bins=[p_bins, d_bins])
        h_safe = np.where(h > 0, h, np.nan)
        im = ax.pcolormesh(p_bins, d_bins, h_safe.T, cmap='viridis',
                           norm=LogNorm(), shading='auto')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('entries / bin')

    overlay_bethe_bloch(ax, estimator_branch, p_grid)
    ax.legend(loc='upper right', fontsize=10, frameon=True, ncol=2)
    ax.text(0.04, 0.96,
            '%s\n%s  (n=%d)' % (channel_label, panel_label, len(p)),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)

    base = '%s_dedx_%s_vs_p_hist2d_%s' % (channel_key, _EST_SHORT[estimator_branch], suffix)
    plot_tools.save_pdf_and_png(outdir, base, fig)
    plt.close(fig)
    print('  saved %s' % base)


def make_panel_hist2d(channel_key, channel_label, estimator_branch, estimator_label,
                      species_pairs, arr, outdir):
    """2D-histogram variant: produces three separate figures per (channel,
    estimator) -- one for each daughter slot individually, plus one for
    both daughters combined. Higher density resolution than the scatter
    overlay; the per-daughter panels are useful for spotting species bands
    that separate cleanly with sufficient statistics."""
    # First daughter (positive-charge slot, Muplus_*)
    p1, d1 = _resolve_p_d(arr, *species_pairs[0][2:])
    label1 = species_pairs[0][0]
    _hist2d_one_panel(channel_key, channel_label, estimator_branch, estimator_label,
                      p1, d1, label1, 'd1', outdir)

    # Second daughter (negative-charge slot, Muminus_*)
    if len(species_pairs) >= 2:
        p2, d2 = _resolve_p_d(arr, *species_pairs[1][2:])
        label2 = species_pairs[1][0]
        _hist2d_one_panel(channel_key, channel_label, estimator_branch, estimator_label,
                          p2, d2, label2, 'd2', outdir)

        # Combined: both daughters in one histogram (concat of arrays)
        p_both = np.concatenate([p1, p2])
        d_both = np.concatenate([d1, d2])
        _hist2d_one_panel(channel_key, channel_label, estimator_branch, estimator_label,
                          p_both, d_both, '%s + %s' % (label1, label2), 'both', outdir)
    else:
        # Single-daughter channel: 'both' is just that one daughter again,
        # but emit it under the 'both' suffix for naming consistency.
        _hist2d_one_panel(channel_key, channel_label, estimator_branch, estimator_label,
                          p1, d1, label1, 'both', outdir)


def main():
    # Per-channel CVH outputs from run_cvh_variants.sh, _v_p variant (vertex
    # + pointing constraint). dE/dx values come from upstream-derived
    # ValueMaps so the refit constraints do not affect them; _v_p is just
    # the default kinematic snapshot.
    indir_tmpl = '/ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants/{chan}_v_p'
    # Output dir is prefixed with today's date (YYMMDD) so successive runs
    # accumulate side by side rather than overwriting each other; _v2
    # postfix avoids clobbering the prior d0-channel run from earlier today.
    import datetime
    date_tag = datetime.date.today().strftime('%y%m%d')
    outdir = f'/home/submit/david_w/public_html/ZMass/{date_tag}_dedx_vs_p_v2'
    os.makedirs(outdir, exist_ok=True)

    # copy index.php so the dir is browsable
    src_php = '/home/submit/david_w/public_html/index.php'
    dst_php = os.path.join(outdir, 'index.php')
    if os.path.exists(src_php) and not os.path.exists(dst_php):
        import shutil
        shutil.copy(src_php, dst_php)

    channels = [
        ('ks',     r'$K_S^0 \to \pi^+ \pi^-$',
         [('$\\pi$ (Mu+)',  'forestgreen', 'Mupluskin_pt',  'Muplus_dedx%s'),
          ('$\\pi$ (Mu-)',  'darkgreen',   'Muminuskin_pt', 'Muminus_dedx%s')]),
        ('lambda', r'$\Lambda^0 \to p \pi^-$',
         [('$p$/$\\bar{p}$',  'darkorange',  'Mupluskin_pt',  'Muplus_dedx%s'),
          ('$\\pi$',          'forestgreen', 'Muminuskin_pt', 'Muminus_dedx%s')]),
    ]

    estimators = [
        ('Harmonic2',      'Harmonic2 (strip)',                ''),
        ('PixelHarmonic2', 'Harmonic2 (pixel)',                ''),
        ('AllHarmonic2',   'Harmonic2-truncated (strip+pixel)', ''),
    ]

    file_map = {
        'ks':     sorted(glob.glob(os.path.join(indir_tmpl.format(chan='ks'),     'globalcor_ks_*.root'))),
        'lambda': sorted(glob.glob(os.path.join(indir_tmpl.format(chan='lambda'), 'globalcor_lambda_*.root'))),
    }

    for chan, chan_label, species in channels:
        files = file_map[chan]
        if not files:
            print('--- %s: no input files in %s, skipping' % (chan, indir_tmpl.format(chan=chan)))
            continue
        print('--- %s: %s' % (chan, files[0]))
        arr = read_tree(files[0])
        if arr is None or len(arr['Mupluskin_pt']) == 0:
            print('  no entries')
            continue
        print('  entries: %d' % len(arr['Mupluskin_pt']))
        for est_branch, est_label, _unit in estimators:
            sp = [(label, color, p_b, d_b % est_branch) for (label, color, p_b, d_b) in species]
            make_panel(chan, chan_label, est_branch, est_label, sp, arr, outdir)
            make_panel_hist2d(chan, chan_label, est_branch, est_label, sp, arr, outdir)

    print('all plots in', outdir)


if __name__ == '__main__':
    main()

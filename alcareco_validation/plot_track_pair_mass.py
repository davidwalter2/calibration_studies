"""
Plot the raw track-pair invariant mass for KS and Lambda,
showing the full TwoBodyDecaySelector window including sidebands.
"""
import numpy as np
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

ks_in   = '/tmp/ks_pair_mass.txt'
lam_in  = '/tmp/lam_pair_mass.txt'
outdir  = '/home/submit/david_w/public_html/ZMass/260424'

ks  = np.loadtxt(ks_in)
lam = np.loadtxt(lam_in)

# ─── KS ──────────────────────────────────────────────────────────────────────
# TwoBodyDecaySelector window: [0.40, 0.60] GeV. V0Producer cut: 497.6 +/- 70 MeV.
bins = np.linspace(400, 600, 80)
counts, _ = np.histogram(ks, bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
width = bins[1] - bins[0]

fig, ax = plot_tools.figure(
    centers, xlabel=r'Raw $m(\pi^+\pi^-)$ [MeV]',
    ylabel='Pairs / %.1f MeV' % width,
    xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1)
ax.bar(centers, counts, width=width, align='center',
       color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5)
ax.axvline(497.611, color='red', linestyle='--', linewidth=1.2, label='PDG: 497.6 MeV')
ax.axvspan(427.6, 567.6, color='red', alpha=0.08, label=r'V0Producer: $\pm 70$ MeV')
ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=12)
ax.text(0.03, 0.97, 'N pairs = %d' % len(ks),
        transform=ax.transAxes, va='top', ha='left', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'ks_pair_mass', fig)
print('Saved ks_pair_mass')

# ─── Lambda ──────────────────────────────────────────────────────────────────
# TwoBodyDecaySelector window: [1.08, 1.15] GeV. V0Producer cut: 1115.7 +/- 50 MeV.
bins = np.linspace(1060, 1170, 80)
counts, _ = np.histogram(lam, bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
width = bins[1] - bins[0]

fig, ax = plot_tools.figure(
    centers, xlabel=r'Raw $m(p\pi^-)$ [MeV]  (best of $p\pi$ / $\pi p$)',
    ylabel='Pairs / %.2f MeV' % width,
    xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1)
ax.bar(centers, counts, width=width, align='center',
       color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5)
ax.axvline(1115.683, color='red', linestyle='--', linewidth=1.2, label='PDG: 1115.7 MeV')
ax.axvspan(1065.7, 1165.7, color='red', alpha=0.08, label=r'V0Producer: $\pm 50$ MeV')
ax.axvspan(1080,   1150,   color='blue', alpha=0.08, label='ALCARECO: [1.08, 1.15] GeV')
ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=12)
ax.text(0.03, 0.97, 'N pairs = %d' % len(lam),
        transform=ax.transAxes, va='top', ha='left', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_pair_mass', fig)
print('Saved lam_pair_mass')

print('Done.')

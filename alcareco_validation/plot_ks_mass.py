import numpy as np
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

ks = np.loadtxt("/tmp/ks_kinematics.txt")  # mass(MeV) pt eta phi vx vy vz Lxy L3d ...

bins = np.linspace(400, 600, 60)
counts, _ = np.histogram(ks[:,0], bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
width = bins[1] - bins[0]

fig, ax = plot_tools.figure(
    centers,
    xlabel=r'$m(\pi^+\pi^-)$ [MeV]',
    ylabel='Candidates / %.1f MeV' % width,
    xlim=(bins[0], bins[-1]),
    automatic_scale=False,
    width_scale=1,
)

ax.bar(centers, counts, width=width, align='center',
       color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5)
ax.axvline(497.611, color='red', linestyle='--', linewidth=1.2, label='PDG: 497.6 MeV')

n    = len(ks[:,0])
mean = np.mean(ks[:,0])
std  = np.std(ks[:,0])
ax.text(0.97, 0.97,
        'N = %d\nmean = %.1f MeV\nRMS = %.1f MeV' % (n, mean, std),
        transform=ax.transAxes, va='top', ha='right', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(loc='upper left', fontsize=14)

plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)

outdir = '/home/submit/david_w/public_html/ZMass/260424'
plot_tools.save_pdf_and_png(outdir, 'ks_mass', fig)

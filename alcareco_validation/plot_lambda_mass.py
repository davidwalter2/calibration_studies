import numpy as np
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

lam = np.loadtxt("/tmp/lam_kinematics.txt")  # mass(MeV) pt eta phi vx vy vz Lxy L3d ...

bins = np.linspace(1080, 1160, 60)
counts, _ = np.histogram(lam[:,0], bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
width = bins[1] - bins[0]

fig, ax = plot_tools.figure(
    centers,
    xlabel=r'$m(p\pi^-)$ [MeV]',
    ylabel='Candidates / %.1f MeV' % width,
    xlim=(bins[0], bins[-1]),
    automatic_scale=False,
    width_scale=1,
)

ax.bar(centers, counts, width=width, align='center',
       color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5)
ax.axvline(1115.683, color='red', linestyle='--', linewidth=1.2, label='PDG: 1115.7 MeV')

n    = len(lam[:,0])
mean = np.mean(lam[:,0])
std  = np.std(lam[:,0])
ax.text(0.97, 0.55,
        'N = %d\nmean = %.1f MeV\nRMS = %.1f MeV' % (n, mean, std),
        transform=ax.transAxes, va='top', ha='right', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(loc='upper right', fontsize=14)

plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)

outdir = '/home/submit/david_w/public_html/ZMass/260424'
plot_tools.save_pdf_and_png(outdir, 'lambda_mass', fig)

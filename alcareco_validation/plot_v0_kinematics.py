import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

outdir = '/home/submit/david_w/public_html/ZMass/260424'

# columns: mass pt eta phi vx vy vz Lxy L3d  d0_pt d0_eta d0_phi d0_q  d1_pt d1_eta d1_phi d1_q
ks  = np.loadtxt('/tmp/ks_kinematics.txt')
lam = np.loadtxt('/tmp/lam_kinematics.txt')
ks_nev  = np.loadtxt('/tmp/ks_per_event.txt')
lam_nev = np.loadtxt('/tmp/lam_per_event.txt')

# for KS: daughters are pi+ and pi- (same mass) — label by charge
# for Lambda: daughter 0 = proton (higher pt generally), daughter 1 = pion
# We'll split by charge: d0_q>0 is pi+ (KS) or proton (Lambda), d0_q<0 is pi- or anti-proton

def split_by_charge(data, d0_col=9, d0_q_col=12, d1_col=13, d1_q_col=16):
    """Return (pos_pt, pos_eta, pos_phi, neg_pt, neg_eta, neg_phi) based on daughter charges."""
    q0 = data[:, d0_q_col]
    pos_mask = q0 > 0
    pos_pt  = np.where(pos_mask, data[:,d0_col],   data[:,d1_col])
    pos_eta = np.where(pos_mask, data[:,d0_col+1], data[:,d1_col+1])
    pos_phi = np.where(pos_mask, data[:,d0_col+2], data[:,d1_col+2])
    neg_pt  = np.where(pos_mask, data[:,d1_col],   data[:,d0_col])
    neg_eta = np.where(pos_mask, data[:,d1_col+1], data[:,d0_col+1])
    neg_phi = np.where(pos_mask, data[:,d1_col+2], data[:,d0_col+2])
    return pos_pt, pos_eta, pos_phi, neg_pt, neg_eta, neg_phi

def make_plot(data_x, bins, xlabel, ylabel, color, name, vline=None, vline_label=None,
              xscale='linear', logx=False, stats=True):
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(data_x, bins=bins)
    width = bins[1] - bins[0] if not logx else None
    fig, ax = plot_tools.figure(
        centers, xlabel=xlabel, ylabel=ylabel,
        xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1,
    )
    if logx:
        ax.set_xscale('log')
        ax.bar(centers, counts, width=np.diff(bins), align='center',
               color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
    else:
        ax.bar(centers, counts, width=width, align='center',
               color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
    if vline is not None:
        ax.axvline(vline, color='red', linestyle='--', linewidth=1.2, label=vline_label)
        ax.legend(loc='upper right', fontsize=13)
    if stats:
        n, mean, rms = len(data_x), np.mean(data_x), np.std(data_x)
        ax.text(0.97, 0.97, 'N = %d\nmean = %.2f\nRMS = %.2f' % (n, mean, rms),
                transform=ax.transAxes, va='top', ha='right', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
    plot_tools.save_pdf_and_png(outdir, name, fig)
    plt.close(fig)
    print('Saved', name)

# ─── KS: candidates per event ───────────────────────────────────────────────
make_plot(ks_nev[ks_nev > 0], np.arange(0.5, 10.5, 1.0),
          r'$K_S$ candidates per event', 'Events',
          'steelblue', 'ks_ncands_per_event', stats=True)

# ─── Lambda: candidates per event ────────────────────────────────────────────
make_plot(lam_nev[lam_nev > 0], np.arange(0.5, 6.5, 1.0),
          r'$\Lambda^0$ candidates per event', 'Events',
          'darkorange', 'lam_ncands_per_event', stats=True)

# ─── KS V0 pT ────────────────────────────────────────────────────────────────
make_plot(ks[:,1], np.linspace(0, 15, 40),
          r'$K_S$ $p_T$ [GeV]', 'Candidates / 0.38 GeV',
          'steelblue', 'ks_pt')

# ─── KS V0 eta ───────────────────────────────────────────────────────────────
make_plot(ks[:,2], np.linspace(-3, 3, 40),
          r'$K_S$ $\eta$', 'Candidates / 0.15',
          'steelblue', 'ks_eta')

# ─── KS V0 phi ───────────────────────────────────────────────────────────────
make_plot(ks[:,3], np.linspace(-np.pi, np.pi, 40),
          r'$K_S$ $\phi$ [rad]', 'Candidates / 0.16 rad',
          'steelblue', 'ks_phi')

# ─── KS daughter (pion) pT ───────────────────────────────────────────────────
ks_pos_pt, ks_pos_eta, ks_pos_phi, ks_neg_pt, ks_neg_eta, ks_neg_phi = split_by_charge(ks)

fig, ax = plot_tools.figure(
    np.linspace(0, 8, 40), xlabel=r'Pion $p_T$ [GeV]',
    ylabel='Tracks / 0.2 GeV', xlim=(0, 8), automatic_scale=False, width_scale=1)
bins_pt = np.linspace(0, 8, 40)
w = bins_pt[1] - bins_pt[0]
c_pos, _ = np.histogram(ks_pos_pt, bins=bins_pt)
c_neg, _ = np.histogram(ks_neg_pt, bins=bins_pt)
centers_pt = 0.5 * (bins_pt[:-1] + bins_pt[1:])
ax.bar(centers_pt, c_pos, width=w, align='center', color='steelblue', alpha=0.7,
       edgecolor='steelblue', linewidth=0.5, label=r'$\pi^+$')
ax.bar(centers_pt, c_neg, width=w, align='center', color='tomato', alpha=0.7,
       edgecolor='tomato', linewidth=0.5, bottom=0, label=r'$\pi^-$')
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'ks_pion_pt', fig)
plt.close(fig)
print('Saved ks_pion_pt')

# ─── KS daughter eta ─────────────────────────────────────────────────────────
fig, ax = plot_tools.figure(
    np.linspace(-3, 3, 40), xlabel=r'Pion $\eta$',
    ylabel='Tracks / 0.15', xlim=(-3, 3), automatic_scale=False, width_scale=1)
bins_eta = np.linspace(-3, 3, 40)
w = bins_eta[1] - bins_eta[0]
c_pos, _ = np.histogram(ks_pos_eta, bins=bins_eta)
c_neg, _ = np.histogram(ks_neg_eta, bins=bins_eta)
centers_eta = 0.5 * (bins_eta[:-1] + bins_eta[1:])
ax.bar(centers_eta, c_pos, width=w, color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5, label=r'$\pi^+$')
ax.bar(centers_eta, c_neg, width=w, color='tomato', alpha=0.7, edgecolor='tomato', linewidth=0.5, label=r'$\pi^-$')
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'ks_pion_eta', fig)
plt.close(fig)
print('Saved ks_pion_eta')

# ─── KS flight lengths ────────────────────────────────────────────────────────
make_plot(ks[:,7], np.linspace(0, 25, 50),
          r'$K_S$ transverse flight length $L_{xy}$ [cm]', 'Candidates / 0.5 cm',
          'steelblue', 'ks_Lxy')
make_plot(ks[:,8], np.linspace(0, 40, 50),
          r'$K_S$ 3D flight length $L_{3D}$ [cm]', 'Candidates / 0.8 cm',
          'steelblue', 'ks_L3d')

# ─── Lambda V0 pT ─────────────────────────────────────────────────────────────
make_plot(lam[:,1], np.linspace(0, 15, 40),
          r'$\Lambda^0$ $p_T$ [GeV]', 'Candidates / 0.38 GeV',
          'darkorange', 'lam_pt')

# ─── Lambda V0 eta ────────────────────────────────────────────────────────────
make_plot(lam[:,2], np.linspace(-3, 3, 40),
          r'$\Lambda^0$ $\eta$', 'Candidates / 0.15',
          'darkorange', 'lam_eta')

# ─── Lambda V0 phi ────────────────────────────────────────────────────────────
make_plot(lam[:,3], np.linspace(-np.pi, np.pi, 40),
          r'$\Lambda^0$ $\phi$ [rad]', 'Candidates / 0.16 rad',
          'darkorange', 'lam_phi')

# ─── Lambda daughter pT (proton vs pion) ─────────────────────────────────────
# For Lambda: positive daughter = proton, negative = pion
lam_p_pt,  lam_p_eta,  lam_p_phi,  lam_pi_pt,  lam_pi_eta,  lam_pi_phi  = split_by_charge(lam)

fig, ax = plot_tools.figure(
    np.linspace(0, 10, 40), xlabel=r'Daughter $p_T$ [GeV]',
    ylabel='Tracks / 0.25 GeV', xlim=(0, 10), automatic_scale=False, width_scale=1)
bins_pt = np.linspace(0, 10, 40)
w = bins_pt[1] - bins_pt[0]
c_p,  _ = np.histogram(lam_p_pt,  bins=bins_pt)
c_pi, _ = np.histogram(lam_pi_pt, bins=bins_pt)
centers_pt = 0.5 * (bins_pt[:-1] + bins_pt[1:])
ax.bar(centers_pt, c_p,  width=w, color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5, label='proton')
ax.bar(centers_pt, c_pi, width=w, color='forestgreen', alpha=0.7, edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^-$')
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_daughter_pt', fig)
plt.close(fig)
print('Saved lam_daughter_pt')

# ─── Lambda daughter eta ──────────────────────────────────────────────────────
fig, ax = plot_tools.figure(
    np.linspace(-3, 3, 40), xlabel=r'Daughter $\eta$',
    ylabel='Tracks / 0.15', xlim=(-3, 3), automatic_scale=False, width_scale=1)
bins_eta = np.linspace(-3, 3, 40)
w = bins_eta[1] - bins_eta[0]
c_p,  _ = np.histogram(lam_p_eta,  bins=bins_eta)
c_pi, _ = np.histogram(lam_pi_eta, bins=bins_eta)
centers_eta = 0.5 * (bins_eta[:-1] + bins_eta[1:])
ax.bar(centers_eta, c_p,  width=w, color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5, label='proton')
ax.bar(centers_eta, c_pi, width=w, color='forestgreen', alpha=0.7, edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^-$')
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_daughter_eta', fig)
plt.close(fig)
print('Saved lam_daughter_eta')

# ─── Lambda flight lengths ────────────────────────────────────────────────────
# clip outlier with very large flight (secondary decay in material?)
lam_Lxy = lam[:,7]
lam_L3d = lam[:,8]
make_plot(lam_Lxy[lam_Lxy < 50], np.linspace(0, 50, 50),
          r'$\Lambda^0$ transverse flight length $L_{xy}$ [cm]', 'Candidates / 1 cm',
          'darkorange', 'lam_Lxy')
make_plot(lam_L3d[lam_L3d < 80], np.linspace(0, 80, 50),
          r'$\Lambda^0$ 3D flight length $L_{3D}$ [cm]', 'Candidates / 1.6 cm',
          'darkorange', 'lam_L3d')

print("All plots saved to", outdir)

"""
Replot Lambda daughter kinematic distributions with correct labels.
V0Producer convention: daughter(0) = baryon (p for Lambda0, pbar for anti-Lambda0),
                       daughter(1) = pion  (pi- for Lambda0, pi+ for anti-Lambda0).
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

outdir = '/home/submit/david_w/public_html/ZMass/260424'

# columns: mass pt eta phi vx vy vz Lxy L3d cosTheta_XY  d0_pt d0_eta d0_phi d0_q  d1_pt d1_eta d1_phi d1_q
lam = np.loadtxt('/tmp/lam_kinematics.txt')

# daughter(0) = p/pbar (baryon), daughter(1) = pi-/pi+ (pion)
bar_pt  = lam[:,10]
bar_eta = lam[:,11]
bar_phi = lam[:,12]
bar_q   = lam[:,13].astype(int)  # +1 for Lambda0 (proton), -1 for anti-Lambda0 (antiproton)
pi_pt   = lam[:,14]
pi_eta  = lam[:,15]
pi_phi  = lam[:,16]

n_lam     = np.sum(bar_q > 0)
n_antilam = np.sum(bar_q < 0)

# ─── daughter pT: p/pbar vs pi ────────────────────────────────────────────────
bins_pt = np.linspace(0, 10, 101)   # 0.1 GeV bins; edge at 0.35 GeV aligns with ptMin cut
w = bins_pt[1] - bins_pt[0]
centers_pt = 0.5 * (bins_pt[:-1] + bins_pt[1:])
c_bar, _ = np.histogram(bar_pt, bins=bins_pt)
c_pi,  _ = np.histogram(pi_pt,  bins=bins_pt)

fig, ax = plot_tools.figure(
    centers_pt, xlabel=r'Daughter $p_T$ [GeV]',
    ylabel='Tracks / %.1f GeV' % w, xlim=(0, 10), automatic_scale=False, width_scale=1)
ax.bar(centers_pt, c_bar, width=w, color='darkorange', alpha=0.7,
       edgecolor='darkorange', linewidth=0.5, label=r'$p$ / $\bar{p}$')
ax.bar(centers_pt, c_pi,  width=w, color='forestgreen', alpha=0.7,
       edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^\mp$')
ax.legend(fontsize=13)
ax.text(0.97, 0.97,
        r'$\Lambda^0$: %d,  $\bar{\Lambda}^0$: %d' % (n_lam, n_antilam),
        transform=ax.transAxes, va='top', ha='right', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_daughter_pt', fig)
plt.close(fig)
print('Saved lam_daughter_pt')

# ─── daughter eta: p/pbar vs pi ───────────────────────────────────────────────
bins_eta = np.linspace(-3, 3, 40)
w = bins_eta[1] - bins_eta[0]
centers_eta = 0.5 * (bins_eta[:-1] + bins_eta[1:])
c_bar, _ = np.histogram(bar_eta, bins=bins_eta)
c_pi,  _ = np.histogram(pi_eta,  bins=bins_eta)

fig, ax = plot_tools.figure(
    centers_eta, xlabel=r'Daughter $\eta$',
    ylabel='Tracks / %.2f' % w, xlim=(-3, 3), automatic_scale=False, width_scale=1)
ax.bar(centers_eta, c_bar, width=w, color='darkorange', alpha=0.7,
       edgecolor='darkorange', linewidth=0.5, label=r'$p$ / $\bar{p}$')
ax.bar(centers_eta, c_pi,  width=w, color='forestgreen', alpha=0.7,
       edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^\mp$')
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_daughter_eta', fig)
plt.close(fig)
print('Saved lam_daughter_eta')

# ─── Armenteros-Podolanski plot ───────────────────────────────────────────────
# alpha = (pL+ - pL-) / (pL+ + pL-),  qT = transverse momentum of daughter w.r.t. V0
import math

alpha_list = []
qT_list    = []

for i in range(len(lam)):
    # V0 momentum direction
    v0_pt  = lam[i,1]
    v0_eta = lam[i,2]
    v0_phi = lam[i,3]
    v0_px  = v0_pt * math.cos(v0_phi)
    v0_py  = v0_pt * math.sin(v0_phi)
    v0_pz  = v0_pt * math.sinh(v0_eta)
    v0_p   = math.sqrt(v0_px**2 + v0_py**2 + v0_pz**2)
    ux, uy, uz = v0_px/v0_p, v0_py/v0_p, v0_pz/v0_p

    # daughter 0 (baryon)
    d0_pt  = lam[i,10]
    d0_eta = lam[i,11]
    d0_phi = lam[i,12]
    d0_q   = lam[i,13]
    d0_px  = d0_pt * math.cos(d0_phi)
    d0_py  = d0_pt * math.sin(d0_phi)
    d0_pz  = d0_pt * math.sinh(d0_eta)

    # daughter 1 (pion)
    d1_pt  = lam[i,14]
    d1_eta = lam[i,15]
    d1_phi = lam[i,16]
    d1_px  = d1_pt * math.cos(d1_phi)
    d1_py  = d1_pt * math.sin(d1_phi)
    d1_pz  = d1_pt * math.sinh(d1_eta)

    # longitudinal components along V0
    pL0 = d0_px*ux + d0_py*uy + d0_pz*uz
    pL1 = d1_px*ux + d1_py*uy + d1_pz*uz

    # sign convention: alpha defined w.r.t. positive daughter
    if d0_q > 0:
        pL_pos, pL_neg = pL0, pL1
        # qT of pion (lighter daughter)
        d_px, d_py, d_pz = d1_px, d1_py, d1_pz
    else:
        pL_pos, pL_neg = pL1, pL0
        d_px, d_py, d_pz = d0_px, d0_py, d0_pz

    alpha = (pL_pos - pL_neg) / (pL_pos + pL_neg) if (pL_pos + pL_neg) != 0 else 0

    # qT: transverse component of pion w.r.t. V0 axis
    dot = d_px*ux + d_py*uy + d_pz*uz
    qT = math.sqrt(max(0, d_px**2 + d_py**2 + d_pz**2 - dot**2))

    alpha_list.append(alpha)
    qT_list.append(qT)

alpha_arr = np.array(alpha_list)
qT_arr    = np.array(qT_list)

fig, ax = plot_tools.figure(
    np.linspace(-1, 1, 2),
    xlabel=r'Armenteros $\alpha$',
    ylabel=r'$q_T$ [GeV]',
    xlim=(-1.1, 1.1), automatic_scale=False, width_scale=1)
ax.scatter(alpha_arr, qT_arr, s=4, color='darkorange', alpha=0.6)
ax.set_ylim(0, 0.35)
ax.text(0.03, 0.97,
        r'$\Lambda^0$: %d ($\alpha>0$),  $\bar{\Lambda}^0$: %d ($\alpha<0$)' % (n_lam, n_antilam),
        transform=ax.transAxes, va='top', ha='left', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_armenteros', fig)
plt.close(fig)
print('Saved lam_armenteros')

print("Done.")

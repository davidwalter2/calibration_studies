import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/work/submit/david_w/WRemnants')
sys.path.insert(0, '/work/submit/david_w/WRemnants/wums')
import wums.plot_tools as plot_tools

outdir = '/home/submit/david_w/public_html/ZMass/260424'

# Total events processed by cmsRun (from the -n config parameter).
# Update this when re-running on a different sample.
TOTAL_EVENTS = 1000

# columns: mass pt eta phi vx vy vz Lxy L3d cosTheta_XY  d0_pt d0_eta d0_phi d0_q  d1_pt d1_eta d1_phi d1_q
ks  = np.loadtxt('/tmp/ks_kinematics.txt')
lam = np.loadtxt('/tmp/lam_kinematics.txt')
ks_nev  = np.loadtxt('/tmp/ks_per_event.txt')
lam_nev = np.loadtxt('/tmp/lam_per_event.txt')

# for KS: daughters are pi+ and pi- (same mass) — label by charge
# for Lambda: daughter 0 = proton (higher pt generally), daughter 1 = pion
# We'll split by charge: d0_q>0 is pi+ (KS) or proton (Lambda), d0_q<0 is pi- or anti-proton

def split_by_charge(data, d0_col=10, d0_q_col=13, d1_col=14, d1_q_col=17):
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

def make_ncands_plot(nev, total_events, max_cands, xlabel, color, name):
    """Histogram of candidates per event including the zero bin for rejected events."""
    n_zero = total_events - len(nev)
    bins = np.arange(-0.5, max_cands + 1.5, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(nev, bins=bins)
    counts[0] = n_zero  # fill zero bin with events that had no candidate

    fig, ax = plot_tools.figure(
        centers, xlabel=xlabel, ylabel='Events',
        xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1,
    )
    ax.bar(centers, counts, width=1.0, align='center',
           color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
    ax.text(0.97, 0.97,
            'Total: %d\nAccepted: %d (%.0f%%)\nMean cands (accepted): %.2f' % (
                total_events, len(nev), 100.*len(nev)/total_events, np.mean(nev)),
            transform=ax.transAxes, va='top', ha='right', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
    plot_tools.save_pdf_and_png(outdir, name, fig)
    plt.close(fig)
    print('Saved', name)

# ─── KS: candidates per event (including zero bin) ───────────────────────────
make_ncands_plot(ks_nev, TOTAL_EVENTS, 10,
                 r'$K_S$ candidates per event', 'steelblue', 'ks_ncands_per_event')

# ─── Lambda: candidates per event (including zero bin) ───────────────────────
make_ncands_plot(lam_nev, TOTAL_EVENTS, 6,
                 r'$\Lambda^0$ candidates per event', 'darkorange', 'lam_ncands_per_event')

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

bins_pt = np.linspace(0, 8, 81)   # 0.1 GeV bins; edge at 0.35 GeV aligns with ptMin cut
w = bins_pt[1] - bins_pt[0]
centers_pt = 0.5 * (bins_pt[:-1] + bins_pt[1:])
fig, ax = plot_tools.figure(
    centers_pt, xlabel=r'Pion $p_T$ [GeV]',
    ylabel='Tracks / %.1f GeV' % w, xlim=(0, 8), automatic_scale=False, width_scale=1)
c_pos, _ = np.histogram(ks_pos_pt, bins=bins_pt)
c_neg, _ = np.histogram(ks_neg_pt, bins=bins_pt)
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

# ─── Lambda daughter pT (proton/antiproton vs pion) ──────────────────────────
# V0Producer convention: daughter(0) = baryon (p or pbar), daughter(1) = pion.
# Use d0/d1 directly — do NOT use split_by_charge, which would swap them for anti-Lambda0.
lam_bar_pt  = lam[:,10]
lam_bar_eta = lam[:,11]
lam_pi_pt   = lam[:,14]
lam_pi_eta  = lam[:,15]
n_lam     = np.sum(lam[:,13] > 0)
n_antilam = np.sum(lam[:,13] < 0)

bins_pt = np.linspace(0, 10, 101)   # 0.1 GeV bins; edge at 0.35 GeV aligns with ptMin cut
w = bins_pt[1] - bins_pt[0]
centers_pt = 0.5 * (bins_pt[:-1] + bins_pt[1:])
fig, ax = plot_tools.figure(
    centers_pt, xlabel=r'Daughter $p_T$ [GeV]',
    ylabel='Tracks / %.1f GeV' % w, xlim=(0, 10), automatic_scale=False, width_scale=1)
c_bar, _ = np.histogram(lam_bar_pt, bins=bins_pt)
c_pi,  _ = np.histogram(lam_pi_pt,  bins=bins_pt)
ax.bar(centers_pt, c_bar, width=w, color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5, label=r'$p$ / $\bar{p}$')
ax.bar(centers_pt, c_pi,  width=w, color='forestgreen', alpha=0.7, edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^\mp$')
ax.legend(fontsize=13)
ax.text(0.97, 0.97,
        r'$\Lambda^0$: %d,  $\bar{\Lambda}^0$: %d' % (n_lam, n_antilam),
        transform=ax.transAxes, va='top', ha='right', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
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
c_bar, _ = np.histogram(lam_bar_eta, bins=bins_eta)
c_pi,  _ = np.histogram(lam_pi_eta,  bins=bins_eta)
centers_eta = 0.5 * (bins_eta[:-1] + bins_eta[1:])
ax.bar(centers_eta, c_bar, width=w, color='darkorange', alpha=0.7, edgecolor='darkorange', linewidth=0.5, label=r'$p$ / $\bar{p}$')
ax.bar(centers_eta, c_pi,  width=w, color='forestgreen', alpha=0.7, edgecolor='forestgreen', linewidth=0.5, label=r'$\pi^\mp$')
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

# ─── Pointing angle: 1 - cos(theta_XY) ───────────────────────────────────────
# cosTheta_XY computed in extraction using actual beamspot position (col 9),
# consistent with V0Producer's cut. V0Producer requires cos > 0.998 (1-cos < 0.002).

ks_1mcos  = 1.0 - ks[:,9]
lam_1mcos = 1.0 - lam[:,9]

for label, vals, color, name in [
    (r'$K_S$',     ks_1mcos,  'steelblue',  'ks_pointing_angle'),
    (r'$\Lambda^0$', lam_1mcos, 'darkorange', 'lam_pointing_angle'),
]:
    bins = np.logspace(-6, 0, 60)
    centers = np.sqrt(bins[:-1] * bins[1:])
    counts, _ = np.histogram(vals, bins=bins)
    fig, ax = plot_tools.figure(
        centers,
        xlabel=r'$1 - \cos\,\theta_{XY}$',
        ylabel='Candidates',
        xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1,
    )
    ax.set_xscale('log')
    ax.bar(centers, counts, width=np.diff(bins), align='center',
           color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
    ax.axvline(1 - 0.998, color='red', linestyle='--', linewidth=1.2,
               label=r'V0Producer cut: $\cos\theta > 0.998$')
    ax.legend(fontsize=13)
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            va='top', ha='left', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
    plot_tools.save_pdf_and_png(outdir, name, fig)
    plt.close(fig)
    print('Saved', name)

# ─── Flight significance Lxy/sigma ───────────────────────────────────────────
# col 18; V0Producer cut is sig_Lxy > 15
for label, data, color, name in [
    (r'$K_S$',       ks,  'steelblue',  'ks_flight_sig'),
    (r'$\Lambda^0$', lam, 'darkorange', 'lam_flight_sig'),
]:
    sig = data[:,18]
    sig = sig[sig > 0]
    bins = np.linspace(15, 200, 60)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(sig, bins=bins)
    fig, ax = plot_tools.figure(
        centers, xlabel=r'$L_{xy}/\sigma(L_{xy})$', ylabel='Candidates',
        xlim=(bins[0], bins[-1]), automatic_scale=False, width_scale=1,
    )
    ax.bar(centers, counts, width=bins[1]-bins[0], align='center',
           color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
    ax.axvline(15, color='red', linestyle='--', linewidth=1.2,
               label='V0Producer cut: sig > 15')
    ax.legend(fontsize=13)
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            va='top', ha='right', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
    plot_tools.save_pdf_and_png(outdir, name, fig)
    plt.close(fig)
    print('Saved', name)

# ─── ΔR between daughters ────────────────────────────────────────────────────
# cols: d0_eta=11, d0_phi=12, d1_eta=15, d1_phi=16
def delta_r(data):
    deta = data[:,11] - data[:,15]
    dphi = data[:,12] - data[:,16]
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
    return np.sqrt(deta**2 + dphi**2)

ks_dr  = delta_r(ks)
lam_dr = delta_r(lam)

make_plot(ks_dr,  np.linspace(0, 4, 50),
          r'$\Delta R(\pi^+,\pi^-)$', 'Candidates / 0.08',
          'steelblue', 'ks_dR')
make_plot(lam_dr, np.linspace(0, 4, 50),
          r'$\Delta R(p/\bar{p},\,\pi^\mp)$', 'Candidates / 0.08',
          'darkorange', 'lam_dR')

# ─── cos theta* (helicity angle) ──────────────────────────────────────────────
# Angle of positive daughter in V0 rest frame w.r.t. V0 boost direction.
# KS (scalar): flat in cos theta*
# Lambda (1/2+, parity-violating decay): |cos theta*|^2 distribution
# cols: pt=1, eta=2, phi=3, mass=0; d0_pt=10, d0_eta=11, d0_phi=12, d0_q=13

def cos_hel(data, m_v0, m_d0, m_d1, use_d0=False):
    """Compute cos(theta*) for each candidate.

    use_d0=False (KS): always pick the positive-charge daughter.
    use_d0=True (Lambda): always pick daughter(0) = baryon (p/pbar),
      regardless of charge — V0Producer convention.
    """
    import math as _math
    result = []
    for row in data:
        # V0 4-momentum
        pt_v0, eta_v0, phi_v0 = row[1], row[2], row[3]
        px_v0 = pt_v0 * _math.cos(phi_v0)
        py_v0 = pt_v0 * _math.sin(phi_v0)
        pz_v0 = pt_v0 * _math.sinh(eta_v0)
        E_v0  = _math.sqrt(px_v0**2 + py_v0**2 + pz_v0**2 + m_v0**2)

        # daughter 4-momentum selection
        if use_d0:
            # Lambda: always use d0 = baryon (proton or antiproton)
            pt_d, eta_d, phi_d, m_d = row[10], row[11], row[12], m_d0
        elif row[13] > 0:
            pt_d, eta_d, phi_d, m_d = row[10], row[11], row[12], m_d0
        else:
            pt_d, eta_d, phi_d, m_d = row[14], row[15], row[16], m_d1
        px_d = pt_d * _math.cos(phi_d)
        py_d = pt_d * _math.sin(phi_d)
        pz_d = pt_d * _math.sinh(eta_d)
        E_d  = _math.sqrt(px_d**2 + py_d**2 + pz_d**2 + m_d**2)

        # boost to V0 rest frame
        beta_x = px_v0 / E_v0
        beta_y = py_v0 / E_v0
        beta_z = pz_v0 / E_v0
        beta2  = beta_x**2 + beta_y**2 + beta_z**2
        gamma  = 1.0 / _math.sqrt(1.0 - beta2)
        bdotd  = beta_x*px_d + beta_y*py_d + beta_z*pz_d
        gfac   = (gamma - 1.0) / beta2 if beta2 > 0 else 0.0
        px_d_r = px_d + gfac*bdotd*beta_x - gamma*beta_x*E_d
        py_d_r = py_d + gfac*bdotd*beta_y - gamma*beta_y*E_d
        pz_d_r = pz_d + gfac*bdotd*beta_z - gamma*beta_z*E_d

        # V0 direction in lab frame (unit vector)
        p_v0 = _math.sqrt(px_v0**2 + py_v0**2 + pz_v0**2)
        ux, uy, uz = px_v0/p_v0, py_v0/p_v0, pz_v0/p_v0

        p_d_r = _math.sqrt(px_d_r**2 + py_d_r**2 + pz_d_r**2)
        cos_t = (px_d_r*ux + py_d_r*uy + pz_d_r*uz) / p_d_r if p_d_r > 0 else 0.0
        result.append(cos_t)
    return np.array(result)

ks_cos  = cos_hel(ks,  0.49761, 0.13957, 0.13957)              # KS: both pions, pick positive
lam_cos = cos_hel(lam, 1.11568, 0.93827, 0.13957, use_d0=True) # Lambda: always d0=baryon

make_plot(ks_cos,  np.linspace(-1, 1, 40),
          r'$\cos\,\theta^*$ (helicity, $\pi^+$ vs $K_S$ boost)', 'Candidates / 0.05',
          'steelblue', 'ks_costheta_star')
make_plot(lam_cos, np.linspace(-1, 1, 40),
          r'$\cos\,\theta^*$ (helicity, $p/\bar{p}$ in $\Lambda^0/\bar{\Lambda}^0$ rest frame)', 'Candidates / 0.05',
          'darkorange', 'lam_costheta_star')

# ─── cos theta* split by Lambda0 vs anti-Lambda0 ─────────────────────────────
# Both should be flat and identical for unpolarized production at LHC.
# Any asymmetry between the two signals an acceptance artifact or analysis bug.
lam0_mask    = lam[:,13] > 0   # d0 = proton  → Lambda0
antilam_mask = lam[:,13] < 0   # d0 = pbar    → anti-Lambda0
lam_cos_arr  = np.array(lam_cos)
bins_hel = np.linspace(-1, 1, 20)
centers_hel = 0.5 * (bins_hel[:-1] + bins_hel[1:])
w_hel = bins_hel[1] - bins_hel[0]
c_lam0,   _ = np.histogram(lam_cos_arr[lam0_mask],    bins=bins_hel)
c_antilam,_ = np.histogram(lam_cos_arr[antilam_mask], bins=bins_hel)

fig, ax = plot_tools.figure(
    centers_hel,
    xlabel=r'$\cos\,\theta^*$ (helicity, baryon in V0 rest frame)',
    ylabel='Candidates / 0.10',
    xlim=(-1, 1), automatic_scale=False, width_scale=1)
ax.step(np.append(bins_hel[:-1], bins_hel[-1]),
        np.append(c_lam0, c_lam0[-1]),
        where='post', color='darkorange', linewidth=1.5, label=r'$\Lambda^0$ (proton, N=%d)' % n_lam)
ax.step(np.append(bins_hel[:-1], bins_hel[-1]),
        np.append(c_antilam, c_antilam[-1]),
        where='post', color='royalblue', linewidth=1.5, linestyle='--',
        label=r'$\bar{\Lambda}^0$ (antiproton, N=%d)' % n_antilam)
ax.legend(fontsize=13)
plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True)
plot_tools.save_pdf_and_png(outdir, 'lam_costheta_star_split', fig)
plt.close(fig)
print('Saved lam_costheta_star_split')

print("All plots saved to", outdir)

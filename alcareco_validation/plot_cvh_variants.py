"""
Compare the ditrack candidate mass and related kinematics across the four
configurations of the step-2 chain for KS, Lambda and J/psi:

    (a) raw track-pair invariant mass with daughter mass hypothesis
        (= Jpsitrk_mass; identical in all CVH outputs)
    (b) CVH refit, NO common-vertex constraint, NO pointing constraint
    (c) CVH refit, common-vertex constraint, NO pointing constraint
    (d) CVH refit, common-vertex constraint, AND pointing constraint
        (sigma = 1 mrad)

Inputs are produced by:
    run_cvh_variants.sh
    -> /ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants/<channel>_<variant>/<basename>_0.root
       channels: ks, lambda, jpsi
       variants: nv_np, v_np, v_p

Run inside the wmassdev singularity:
    APPTAINER_BIND="/tmp,/home/submit,/work/submit,/scratch/submit,/ceph/submit,/cvmfs,/etc/grid-security,/run" \
      singularity run --nv \
      /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:v48_patch0 \
      bash -c "source /work/submit/david_w/WRemnants/setup.sh > /dev/null 2>&1 && \
               python3 /work/submit/david_w/ZMass/calibration_studies/alcareco_validation/plot_cvh_variants.py"
"""

import os
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot
import wums.plot_tools as plot_tools


INBASE = '/ceph/submit/data/user/d/david_w/ZMass/cvh/260506_variants'
# Output dir is prefixed with today's date (YYMMDD) so successive runs
# accumulate side by side rather than overwriting each other.
_DATE = datetime.date.today().strftime('%y%m%d')
OUTDIR = f'/home/submit/david_w/public_html/ZMass/{_DATE}_cvh_variants_v2'
os.makedirs(OUTDIR, exist_ok=True)

# copy index.php so the directory is browsable in the public_html web view
_src_php = '/home/submit/david_w/public_html/index.php'
_dst_php = os.path.join(OUTDIR, 'index.php')
if os.path.exists(_src_php) and not os.path.exists(_dst_php):
    import shutil
    shutil.copy(_src_php, _dst_php)

# (channel-tag, output-basename, PDG mass [MeV], mass-window [MeV], display label)
CHANNELS = [
    ('ks',     'globalcor_ks',      497.611, (440,  560),  r'$K_S \to \pi^+\pi^-$'),
    ('lambda', 'globalcor_lambda', 1115.683, (1080, 1160), r'$\Lambda^0 \to p\,\pi^-$'),
    ('jpsi',   'globalcor_jpsi',   3096.900, (2700, 3400), r'$J/\psi \to \mu^+\mu^-$'),
]

# Per-channel daughter labels for the Muplus / Muminus slots in the CVH ntuplizer
# output. The candidate producer orders daughters with positive-charge first:
#   KS     -- both pions (Muplus = pi+, Muminus = pi-)
#   Lambda -- (proton, pion); the proton-candidate slot mixes p and pbar across
#             Lambda + anti-Lambda events, the pion-candidate slot mixes pi- and pi+
#   J/psi  -- (mu+, mu-) by the TwoBodyDecayCandidateProducer's positive-first
#             ordering (matches V0Producer's KS convention)
DAUGHTER_LABELS = {
    'ks':     (r'$\pi^+$',       r'$\pi^-$'),
    'lambda': (r'$p / \bar{p}$', r'$\pi^\mp$'),
    'jpsi':   (r'$\mu^+$',       r'$\mu^-$'),
}

VARIANTS = [
    ('nv_np', 'no vtx, no pt',  'tab:blue'),
    ('v_np',  'vtx, no pt',     'tab:orange'),
    ('v_p',   'vtx + pt (1 mrad)', 'tab:green'),
]

# branches we care about
BRANCHES = [
    'Jpsitrk_mass', 'Jpsitrk_pt', 'Jpsitrk_eta', 'Jpsitrk_phi',
    'Jpsi_mass', 'Jpsi_pt', 'Jpsi_eta', 'Jpsi_phi',
    'Jpsi_x', 'Jpsi_y', 'Jpsi_z',
    'Muplus_pt', 'Muplus_eta', 'Muplus_phi',
    'Muminus_pt', 'Muminus_eta', 'Muminus_phi',
    # CMSSW kinematic vertex fit results (always vertex-constrained;
    # input is the original tracks, so the value does not depend on the
    # CVH constraint variant). 'kincons' carries the same fit with an
    # additional dimuon-mass constraint -- only filled when
    # doMassConstraint=True in the channel cfi.
    'Jpsikin_mass', 'Jpsikincons_mass',
    'chisqval', 'niter',
]


def load(channel_tag, basename, variant):
    path = f'{INBASE}/{channel_tag}_{variant}/{basename}_0.root'
    if not os.path.exists(path):
        print(f'  missing: {path}')
        return None
    with uproot.open(path) as f:
        t = f['tree']
        arr = t.arrays(BRANCHES, library='np')
    return arr


def stats_box(ax, n, mean, rms, x=0.97, y=0.97, ha='right'):
    ax.text(x, y, f'N = {n}\nmean = {mean:.3f}\nRMS = {rms:.3f}',
            transform=ax.transAxes, va='top', ha=ha, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


# -----------------------------------------------------------------------------
# Per-channel comparison plots
# -----------------------------------------------------------------------------
for chan, basename, pdg_mev, mass_window, channel_label in CHANNELS:
    print(f'== {chan} ==')
    data = {v: load(chan, basename, v) for v, _, _ in VARIANTS}
    if any(d is None for d in data.values()):
        print(f'  skipping {chan}: some variant outputs missing')
        continue

    # take Jpsitrk_mass from variant nv_np for scenario (a) — identical across variants
    base_var = data['nv_np']
    raw_mass_mev = base_var['Jpsitrk_mass'] * 1000.0

    # --- (1) ditrack mass overlay (with ratio panel: each variant / v_p) -
    bins = np.linspace(mass_window[0], mass_window[1], 81)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    # Two-panel layout: main hist on top (75%), ratio below (25%).
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # histograms first (raw + 3 CVH variants + CMSSW kin fit + optional kincons)
    # Legend "N" = (in-window-count / total-tree-count) so the user can see
    # how many fall outside the [mass_window] plot range -- particularly
    # important for the raw track-pair distribution which has long tails.
    def _nlabel(values_mev, cnt):
        return f'{int(cnt.sum())} / {len(values_mev)}'

    cnt_a, _ = np.histogram(raw_mass_mev, bins=bins)
    cnts = {}
    for v, _, _ in VARIANTS:
        cnts[v], _ = np.histogram(data[v]['Jpsi_mass'] * 1000.0, bins=bins)
    # CMSSW KinematicParticleVertexFitter result -- vertex-constrained
    # by construction, no mass constraint, fed by the input tracks (so
    # variant-independent). Take from the nv_np tree.
    kin_mass_mev = base_var['Jpsikin_mass'] * 1000.0
    cnt_kin, _   = np.histogram(kin_mass_mev, bins=bins)
    # CMSSW KinematicConstrainedVertexFitter (vertex + mass constraint)
    # -- only filled when doMassConstraint=True; show only when non-zero
    kincons_mass_mev = base_var['Jpsikincons_mass'] * 1000.0
    cnt_kincons, _   = np.histogram(kincons_mass_mev, bins=bins)
    show_kincons = cnt_kincons.sum() > 0

    # main panel: each legend entry uses a 2-line label -- description on
    # the first line, N value indented underneath. Larger font for both.
    ax.step(np.append(bins[:-1], bins[-1]), np.append(cnt_a, cnt_a[-1]),
            where='post', color='black', linewidth=1.4,
            label=f'(a) raw track pair\n      N = {_nlabel(raw_mass_mev, cnt_a)}')
    for v, vlabel, vcolor in VARIANTS:
        cnt = cnts[v]
        tag = {'nv_np': 'b', 'v_np': 'c', 'v_p': 'd'}[v]
        m_v = data[v]['Jpsi_mass'] * 1000.0
        ax.step(np.append(bins[:-1], bins[-1]), np.append(cnt, cnt[-1]),
                where='post', color=vcolor, linewidth=1.4,
                label=f'({tag}) CVH {vlabel}\n      N = {_nlabel(m_v, cnt)}')
    ax.step(np.append(bins[:-1], bins[-1]), np.append(cnt_kin, cnt_kin[-1]),
            where='post', color='tab:purple', linewidth=1.4, linestyle='--',
            label=f'(e) CMSSW kin fit (vtx)\n      N = {_nlabel(kin_mass_mev, cnt_kin)}')
    if show_kincons:
        ax.step(np.append(bins[:-1], bins[-1]), np.append(cnt_kincons, cnt_kincons[-1]),
                where='post', color='tab:brown', linewidth=1.4, linestyle='--',
                label=f'(f) CMSSW kin fit (vtx + mass)\n      N = {_nlabel(kincons_mass_mev, cnt_kincons)}')
    ax.axvline(pdg_mev, color='red', linestyle='--', linewidth=1.0,
               label=f'PDG: {pdg_mev:.2f} MeV')
    ax.set_ylabel(f'Candidates / {width:.2f} MeV')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, loc='upper right',
              labelspacing=0.6, handletextpad=0.5)
    ax.text(0.03, 0.97, channel_label, transform=ax.transAxes,
            va='top', ha='left', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ratio panel: each variant (and raw) divided by v_p, bin-by-bin
    ref = cnts['v_p'].astype(float)
    safe = ref > 0
    def _ratio(num):
        r = np.full(len(num), np.nan)
        r[safe] = num[safe] / ref[safe]
        return r
    # raw
    r_raw = _ratio(cnt_a.astype(float))
    axr.step(np.append(bins[:-1], bins[-1]), np.append(r_raw, r_raw[-1]),
             where='post', color='black', linewidth=1.2)
    # variants
    for v, vlabel, vcolor in VARIANTS:
        r = _ratio(cnts[v].astype(float))
        axr.step(np.append(bins[:-1], bins[-1]), np.append(r, r[-1]),
                 where='post', color=vcolor, linewidth=1.2)
    # CMSSW kin fit (always shown) + kincons (when filled)
    r_kin = _ratio(cnt_kin.astype(float))
    axr.step(np.append(bins[:-1], bins[-1]), np.append(r_kin, r_kin[-1]),
             where='post', color='tab:purple', linewidth=1.2, linestyle='--')
    if show_kincons:
        r_kincons = _ratio(cnt_kincons.astype(float))
        axr.step(np.append(bins[:-1], bins[-1]), np.append(r_kincons, r_kincons[-1]),
                 where='post', color='tab:brown', linewidth=1.2, linestyle='--')
    axr.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    axr.axvline(pdg_mev, color='red', linestyle='--', linewidth=0.8)
    axr.set_xlabel(r'$m(\mathrm{ditrack})$ [MeV]')
    axr.set_ylabel('ratio / (d) v_p')
    axr.set_xlim(mass_window)
    axr.set_ylim(0.5, 1.6)

    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{chan}_mass_compare.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{chan}_mass_compare.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {chan}_mass_compare')

    # --- (2) mass vs Lxy_proxy, one panel per scenario ------------------
    # We don't have BS in the tree; use sqrt(x^2+y^2) as a proxy. The
    # absolute Lxy is shifted by the BS (~mm) which is small relative to
    # the typical KS/Lambda flight (~cm). For D0 the proxy is dominated
    # by the BS itself — interpret with care.
    def lxy_proxy(d):
        return np.hypot(d['Jpsi_x'], d['Jpsi_y'])

    # axis ranges per channel
    if chan == 'jpsi':
        lxy_max = 2.0    # cm; J/psi is mostly prompt with non-prompt tail
    elif chan == 'ks':
        lxy_max = 25.0
    else:
        lxy_max = 40.0
    lxy_bins = np.linspace(0, lxy_max, 41)
    mass_bins = np.linspace(mass_window[0], mass_window[1], 41)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
    # panel (a): raw track-pair mass vs lxy (using nv_np vertex as proxy)
    ax = axes[0]
    h, _, _ = np.histogram2d(lxy_proxy(base_var), raw_mass_mev,
                             bins=[lxy_bins, mass_bins])
    h_safe = np.where(h.T > 0, h.T, np.nan)
    im = ax.imshow(h_safe, origin='lower', aspect='auto',
                   extent=[lxy_bins[0], lxy_bins[-1], mass_bins[0], mass_bins[-1]],
                   cmap='viridis', interpolation='nearest')
    ax.set_title('(a) raw track pair', fontsize=11)
    ax.axhline(pdg_mev, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$L_{xy}^{\mathrm{proxy}}$ [cm]')
    ax.set_ylabel(r'$m(\mathrm{ditrack})$ [MeV]')

    for i, (v, vlabel, _) in enumerate(VARIANTS, start=1):
        ax = axes[i]
        m = data[v]['Jpsi_mass'] * 1000.0
        x = lxy_proxy(data[v])
        h, _, _ = np.histogram2d(x, m, bins=[lxy_bins, mass_bins])
        h_safe = np.where(h.T > 0, h.T, np.nan)
        ax.imshow(h_safe, origin='lower', aspect='auto',
                  extent=[lxy_bins[0], lxy_bins[-1], mass_bins[0], mass_bins[-1]],
                  cmap='viridis', interpolation='nearest')
        tag = {'nv_np': 'b', 'v_np': 'c', 'v_p': 'd'}[v]
        ax.set_title(f'({tag}) CVH {vlabel}', fontsize=11)
        ax.axhline(pdg_mev, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel(r'$L_{xy}^{\mathrm{proxy}}$ [cm]')

    fig.suptitle(channel_label + r' — $m$ vs $L_{xy}$ proxy', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{chan}_mass_vs_lxy.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{chan}_mass_vs_lxy.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {chan}_mass_vs_lxy')

    # --- (3) profile: <mass> vs Lxy proxy (with ratio panel: each / v_p) -
    def profile_mean(x, y):
        idx = np.digitize(x, lxy_bins) - 1
        means = np.full(len(lxy_bins)-1, np.nan)
        errs  = np.full(len(lxy_bins)-1, np.nan)
        for i in range(len(means)):
            sel = idx == i
            if sel.sum() >= 5:
                means[i] = np.mean(y[sel])
                errs[i]  = np.std(y[sel]) / math.sqrt(sel.sum())
        return means, errs

    centers = 0.5 * (lxy_bins[:-1] + lxy_bins[1:])
    m_a, e_a = profile_mean(lxy_proxy(base_var), raw_mass_mev)
    profiles = {}
    for v, _, _ in VARIANTS:
        x = lxy_proxy(data[v])
        m = data[v]['Jpsi_mass'] * 1000.0
        profiles[v] = profile_mean(x, m)

    # two-panel layout
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # main panel
    ax.errorbar(centers, m_a, yerr=e_a, fmt='o-', color='black',
                markersize=4, linewidth=1.0, label='(a) raw track pair')
    for v, vlabel, vcolor in VARIANTS:
        mu, er = profiles[v]
        tag = {'nv_np': 'b', 'v_np': 'c', 'v_p': 'd'}[v]
        ax.errorbar(centers, mu, yerr=er, fmt='o-', color=vcolor,
                    markersize=4, linewidth=1.0,
                    label=f'({tag}) {vlabel}')
    ax.axhline(pdg_mev, color='red', linestyle='--', linewidth=0.9, label='PDG')
    ax.set_ylabel(r'$\langle m(\mathrm{ditrack}) \rangle$ [MeV]')
    ax.set_xlim(lxy_bins[0], lxy_bins[-1])
    ax.legend(fontsize=10, loc='best')
    ax.text(0.03, 0.97, channel_label, transform=ax.transAxes,
            va='top', ha='left', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ratio panel: each variant (and raw) divided by the channel's PDG
    # mass, bin-by-bin. Reference is a constant (no error), so the ratio
    # error is just num_err / pdg_mev.
    def _profile_ratio(num, num_err):
        r  = np.full_like(num, np.nan)
        re = np.full_like(num, np.nan)
        sel = ~np.isnan(num)
        r[sel]  = num[sel]     / pdg_mev
        re[sel] = num_err[sel] / pdg_mev
        return r, re
    r_raw, e_raw = _profile_ratio(m_a, e_a)
    axr.errorbar(centers, r_raw, yerr=e_raw, fmt='o-', color='black',
                 markersize=3, linewidth=1.0)
    for v, _, vcolor in VARIANTS:
        mu, er = profiles[v]
        rr, re = _profile_ratio(mu, er)
        axr.errorbar(centers, rr, yerr=re, fmt='o-', color=vcolor,
                     markersize=3, linewidth=1.0)
    axr.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    axr.set_xlabel(r'$L_{xy}^{\mathrm{proxy}}$ [cm]')
    axr.set_ylabel('ratio / PDG')
    # tight ratio range — means typically deviate by O(few permille)
    axr.set_ylim(0.998, 1.002)

    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{chan}_meanmass_vs_lxy.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{chan}_meanmass_vs_lxy.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {chan}_meanmass_vs_lxy')

    # --- (4) V0 pT, eta, rapidity, Lxy proxy, daughter pT (using v_p variant) ----
    d = data['v_p']
    # Rapidity y = atanh(p_z / E),   p_z = pT sinh(eta),  E = sqrt(p^2 + m^2)
    pt_arr  = np.asarray(d['Jpsi_pt'])
    eta_arr = np.asarray(d['Jpsi_eta'])
    m_arr   = np.asarray(d['Jpsi_mass'])
    pz_arr  = pt_arr * np.sinh(eta_arr)
    p_arr   = pt_arr * np.cosh(eta_arr)
    E_arr   = np.sqrt(p_arr**2 + m_arr**2)
    y_arr   = 0.5 * np.log((E_arr + pz_arr) / (E_arr - pz_arr))
    d['Jpsi_rapidity'] = y_arr

    pt_max = 30.0 if chan == 'jpsi' else 15.0
    cand_label = 'mother' if chan == 'jpsi' else 'V0'
    plot_specs = [
        ('Jpsi_pt',       np.linspace(0, pt_max, 60),     fr'{cand_label} $p_T$ [GeV]', f'{chan}_v0_pt'),
        ('Jpsi_eta',      np.linspace(-3, 3, 60),         fr'{cand_label} $\eta$',      f'{chan}_v0_eta'),
        ('Jpsi_rapidity', np.linspace(-3, 3, 60),         fr'{cand_label} $y$',         f'{chan}_v0_rapidity'),
        ('Jpsi_phi',      np.linspace(-np.pi, np.pi, 60), fr'{cand_label} $\phi$ [rad]', f'{chan}_v0_phi'),
    ]
    for branch, bins_, xlabel, name in plot_specs:
        fig, ax = plot_tools.figure(
            0.5*(bins_[:-1] + bins_[1:]),
            xlabel=xlabel, ylabel='Candidates',
            xlim=(bins_[0], bins_[-1]),
            automatic_scale=False, width_scale=1)
        cnt, _ = np.histogram(d[branch], bins=bins_)
        c = 0.5 * (bins_[:-1] + bins_[1:])
        ax.bar(c, cnt, width=np.diff(bins_), align='center',
               color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5)
        stats_box(ax, len(d[branch]), float(np.mean(d[branch])),
                  float(np.std(d[branch])))
        ax.text(0.03, 0.97, channel_label + r' (CVH, vtx+pt)',
                transform=ax.transAxes, va='top', ha='left', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
        plot_tools.save_pdf_and_png(OUTDIR, name, fig)
        plt.close(fig)
        print(f'  saved {name}')

    # Lxy proxy
    bins_ = np.linspace(0, lxy_max, 60)
    fig, ax = plot_tools.figure(
        0.5*(bins_[:-1] + bins_[1:]),
        xlabel=r'$L_{xy}^{\mathrm{proxy}}$ [cm]', ylabel='Candidates',
        xlim=(bins_[0], bins_[-1]), automatic_scale=False, width_scale=1)
    x = lxy_proxy(d)
    cnt, _ = np.histogram(x, bins=bins_)
    c = 0.5 * (bins_[:-1] + bins_[1:])
    ax.bar(c, cnt, width=np.diff(bins_), align='center',
           color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5)
    stats_box(ax, len(x), float(np.mean(x)), float(np.std(x)))
    ax.text(0.03, 0.97, channel_label + r' (CVH, vtx+pt)',
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    plot_tools.save_pdf_and_png(OUTDIR, f'{chan}_v0_lxy', fig)
    plt.close(fig)
    print(f'  saved {chan}_v0_lxy')

    # Daughter pT (Mu+/Mu- branches; for KS/Lambda/D0 they are pi+/pi-, p/pi, K/pi)
    bins_ = np.linspace(0, 10, 80)
    c = 0.5 * (bins_[:-1] + bins_[1:])
    w = bins_[1] - bins_[0]
    fig, ax = plot_tools.figure(
        c, xlabel=r'Daughter $p_T$ [GeV]', ylabel=f'Tracks / {w:.2f} GeV',
        xlim=(bins_[0], bins_[-1]), automatic_scale=False, width_scale=1)
    c_pos, _ = np.histogram(d['Muplus_pt'],  bins=bins_)
    c_neg, _ = np.histogram(d['Muminus_pt'], bins=bins_)
    pos_lbl, neg_lbl = DAUGHTER_LABELS.get(chan, ('positive daughter', 'negative daughter'))
    ax.bar(c, c_pos, width=w, color='steelblue', alpha=0.7,
           edgecolor='steelblue', linewidth=0.5, label=pos_lbl)
    ax.bar(c, c_neg, width=w, color='tomato', alpha=0.7,
           edgecolor='tomato', linewidth=0.5, label=neg_lbl)
    ax.legend(fontsize=12)
    ax.text(0.03, 0.97, channel_label + r' (CVH, vtx+pt)',
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    plot_tools.save_pdf_and_png(OUTDIR, f'{chan}_daughter_pt', fig)
    plt.close(fig)
    print(f'  saved {chan}_daughter_pt')

    # Daughter eta
    bins_ = np.linspace(-3, 3, 60)
    c = 0.5 * (bins_[:-1] + bins_[1:])
    w = bins_[1] - bins_[0]
    fig, ax = plot_tools.figure(
        c, xlabel=r'Daughter $\eta$', ylabel=f'Tracks / {w:.2f}',
        xlim=(bins_[0], bins_[-1]), automatic_scale=False, width_scale=1)
    c_pos, _ = np.histogram(d['Muplus_eta'],  bins=bins_)
    c_neg, _ = np.histogram(d['Muminus_eta'], bins=bins_)
    ax.bar(c, c_pos, width=w, color='steelblue', alpha=0.7,
           edgecolor='steelblue', linewidth=0.5, label=pos_lbl)
    ax.bar(c, c_neg, width=w, color='tomato', alpha=0.7,
           edgecolor='tomato', linewidth=0.5, label=neg_lbl)
    ax.legend(fontsize=12)
    ax.text(0.03, 0.97, channel_label + r' (CVH, vtx+pt)',
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    plot_tools.save_pdf_and_png(OUTDIR, f'{chan}_daughter_eta', fig)
    plt.close(fig)
    print(f'  saved {chan}_daughter_eta')

    # ΔR between daughters
    deta = d['Muplus_eta'] - d['Muminus_eta']
    dphi = d['Muplus_phi'] - d['Muminus_phi']
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    dr   = np.sqrt(deta**2 + dphi**2)
    bins_ = np.linspace(0, 4, 60)
    c = 0.5 * (bins_[:-1] + bins_[1:])
    fig, ax = plot_tools.figure(
        c, xlabel=r'$\Delta R$(daughter, daughter)', ylabel='Candidates',
        xlim=(bins_[0], bins_[-1]), automatic_scale=False, width_scale=1)
    cnt, _ = np.histogram(dr, bins=bins_)
    ax.bar(c, cnt, width=np.diff(bins_), align='center',
           color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0.5)
    stats_box(ax, len(dr), float(np.mean(dr)), float(np.std(dr)))
    ax.text(0.03, 0.97, channel_label + r' (CVH, vtx+pt)',
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plot_tools.add_cms_decor(ax, label='Preliminary', lumi=None, data=True, loc=0)
    plot_tools.save_pdf_and_png(OUTDIR, f'{chan}_deltaR', fig)
    plt.close(fig)
    print(f'  saved {chan}_deltaR')

    # --- (5) fit-quality comparison: chisqval and niter across variants -
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    cbins = np.linspace(0, 200, 80)
    ibins = np.arange(0, 21)
    for v, vlabel, vcolor in VARIANTS:
        c_chi, _ = np.histogram(data[v]['chisqval'], bins=cbins)
        c_it,  _ = np.histogram(data[v]['niter'],   bins=ibins)
        cc = 0.5*(cbins[:-1] + cbins[1:])
        ic = 0.5*(ibins[:-1] + ibins[1:])
        tag = {'nv_np': 'b', 'v_np': 'c', 'v_p': 'd'}[v]
        axes[0].step(np.append(cbins[:-1], cbins[-1]),
                     np.append(c_chi, c_chi[-1]),
                     where='post', color=vcolor, linewidth=1.4,
                     label=f'({tag}) {vlabel}')
        axes[1].step(np.append(ibins[:-1], ibins[-1]),
                     np.append(c_it, c_it[-1]),
                     where='post', color=vcolor, linewidth=1.4,
                     label=f'({tag}) {vlabel}')
    axes[0].set_xlabel(r'$\chi^2$')
    axes[0].set_ylabel('Candidates')
    axes[0].legend(fontsize=10)
    axes[0].set_title(f'{channel_label}: fit chisq')
    axes[1].set_xlabel('niter')
    axes[1].set_ylabel('Candidates')
    axes[1].legend(fontsize=10)
    axes[1].set_title(f'{channel_label}: iterations to convergence')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{chan}_fit_quality.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{chan}_fit_quality.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {chan}_fit_quality')

print(f'\nAll plots written to {OUTDIR}')

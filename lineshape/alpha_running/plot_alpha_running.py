"""
Plot the running of the QED fine-structure constant alpha_EM(M) and its
hadronic, leptonic, W-boson, and top-quark components with uncertainty bands.

Run with the mfs venv:
  source ../../../mfs/.venv/bin/activate
  python plot_alpha_running.py

Output: ../../../../Documents/AN-EN-XXX/plots/alpha_running.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import os

hep.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 16})

OUTPUT_DIR = "/work/submit/david_w/Documents/AN-EN-XXX/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_0  = 1.0 / 137.036   # fine-structure constant at q²=0
MZ       = 91.1876          # Z-pole mass (GeV)
M_TOP    = 172.76           # top quark MS-bar mass (GeV)
M_W      = 80.377           # W boson mass (GeV)

# PDG anchor for 5-flavour hadronic VP at M_Z
# (Burkhardt & Pietrzyk 2011; Davier et al. 2019)
DALPHA_HAD_MZ     = 0.02764
DALPHA_HAD_MZ_ERR = 0.00010   # total uncertainty (conservative)

# Lepton masses (GeV)
LEPTONS = [('e',   5.110e-4),
           ('mu',  0.10566),
           ('tau', 1.7769)]

# 5 light quarks: MS-bar masses (GeV)
QUARKS = [(2/3, 0.070),   # u
          (1/3, 0.070),   # d
          (1/3, 0.150),   # s
          (2/3, 1.28),    # c
          (1/3, 4.18)]    # b


# ---------------------------------------------------------------------------
# VP functions — leading-log one-loop, valid for M >> m_f
# ---------------------------------------------------------------------------

def _ll(M, charge, mass, Nc):
    """Leading-log VP from one Dirac fermion."""
    if M <= mass:
        return 0.0
    return Nc * charge**2 * (ALPHA_0 / (3.0 * np.pi)) * (np.log(M**2 / mass**2) - 5.0 / 3.0)


def delta_alpha_lep(M):
    return sum(_ll(M, 1.0, ml, 1) for _, ml in LEPTONS)


def _delta_alpha_had_pQCD(M):
    return sum(_ll(M, Qq, mq, 3) for Qq, mq in QUARKS)


_HAD_OFFSET = DALPHA_HAD_MZ - _delta_alpha_had_pQCD(MZ)


def delta_alpha_had(M):
    return _delta_alpha_had_pQCD(M) + _HAD_OFFSET


def delta_alpha_top(M):
    """Top-quark VP (Dirac fermion, Nc=3, Q=2/3).
    Non-negligible only for M >> M_TOP; effectively zero below ~400 GeV."""
    return _ll(M, 2.0/3.0, M_TOP, 3)


def delta_alpha_W(M):
    """W-boson VP contribution in the leading-log limit.
    Bosonic gauge loop enters with the OPPOSITE sign to fermion loops
    (EW analogue of asymptotic freedom), slowing the running of alpha
    above M_W. Coefficient 7/(12π) from the spin-1 photon-WW vertex."""
    if M <= M_W:
        return 0.0
    return -(7.0 * ALPHA_0 / (12.0 * np.pi)) * np.log(M**2 / M_W**2)


def delta_alpha_total(M):
    return delta_alpha_lep(M) + delta_alpha_had(M) + delta_alpha_top(M) + delta_alpha_W(M)


def inv_alpha(M):
    return (1.0 - delta_alpha_total(M)) / ALPHA_0


# ---------------------------------------------------------------------------
# Build grids
# ---------------------------------------------------------------------------
M_grid = np.logspace(np.log10(1.0), np.log10(1000.0), 800)

dal_lep = np.array([delta_alpha_lep(m)   for m in M_grid])
dal_had = np.array([delta_alpha_had(m)   for m in M_grid])
dal_top = np.array([delta_alpha_top(m)   for m in M_grid])
dal_W   = np.array([delta_alpha_W(m)     for m in M_grid])
dal_tot = dal_lep + dal_had + dal_top + dal_W
ialpha  = np.array([inv_alpha(m)         for m in M_grid])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6, 8),
    gridspec_kw={"height_ratios": [2, 3], "hspace": 0.05},
)

# --- top panel: 1/alpha(M) ---
ax_top.plot(M_grid, ialpha, color='black', lw=2.0)
ax_top.axvline(MZ, color='grey', lw=0.9, ls='--')
ax_top.set_xscale('log')
ax_top.set_xlim(M_grid[0], M_grid[-1])
ax_top.set_ylim(124, 138)
ax_top.set_ylabel(r'$1/\alpha(M)$')
ax_top.tick_params(labelbottom=False)
ax_top.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))
# Annotate M_Z after axis is set
ax_top.text(MZ * 1.06, ax_top.get_ylim()[0] + 0.5, r'$M_Z$',
            fontsize=12, color='grey', va='bottom')

# --- bottom panel: Delta alpha components ---
ax_bot.plot(M_grid, dal_lep, color='C0', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{lep}$')
ax_bot.plot(M_grid, dal_had, color='C3', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{had}^{(5)}$')
ax_bot.fill_between(M_grid,
                    dal_had - DALPHA_HAD_MZ_ERR,
                    dal_had + DALPHA_HAD_MZ_ERR,
                    color='C3', alpha=0.25)
ax_bot.plot(M_grid, dal_W,   color='C2', lw=2.0,
            label=r'$\Delta\alpha_{W}$')
ax_bot.plot(M_grid, dal_top, color='C4', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{top}$')
ax_bot.plot(M_grid, dal_tot, color='black', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{total}$')
ax_bot.fill_between(M_grid,
                    dal_tot - DALPHA_HAD_MZ_ERR,
                    dal_tot + DALPHA_HAD_MZ_ERR,
                    color='grey', alpha=0.30)
ax_bot.axvline(MZ, color='grey', lw=0.9, ls='--')
ax_bot.set_xscale('log')
ax_bot.set_xlim(M_grid[0], M_grid[-1])
ax_bot.set_ylim(-0.005, 0.10)
ax_bot.axhline(0, color='grey', lw=0.7, ls=':')
ax_bot.set_xlabel(r'$M$ (GeV)')
ax_bot.set_ylabel(r'$\Delta\alpha(M)$')
ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax_bot.legend(loc='upper left', fontsize=12, framealpha=0.9, ncol=2)

for ext in ('pdf', 'png'):
    path = os.path.join(OUTPUT_DIR, f'alpha_running.{ext}')
    fig.savefig(path, dpi=150 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

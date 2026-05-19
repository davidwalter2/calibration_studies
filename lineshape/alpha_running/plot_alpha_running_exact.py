"""
plot_alpha_running_exact.py

One-loop running of alpha_EM(M) with explicit threshold structure
(square-root cusps at s = 4 m_f^2 and s = 4 M_W^2), instead of the
leading-log step approximation used in plot_alpha_running.py.

Contributions:

- Charged fermions: literal one-loop dispersion form
      Delta alpha_f(s) = (2 alpha_0 / pi) Q^2 N_c
                         * Re int_0^1 dx x(1-x) ln[1 - x(1-x) s/m^2 - i eps]
  Reduces to (alpha_0 / 3 pi) Q^2 N_c [ln(s/m^2) - 5/3] at s >> m^2
  (the canonical leading-log with the on-shell -5/3 constant) and
  contains the smooth one-loop turn-on at s = 4 m^2.

- W boson: B_0-based one-loop ansatz matched to the canonical LL slope
      Delta alpha_W(s) = (7 alpha_0 / 12 pi) * Re Bbar_0(s, M_W, M_W)
  where Bbar_0(s, m, m) = B_0(s, m, m) - B_0(0, m, m) is the once-
  subtracted (UV-finite) Passarino-Veltman scalar two-point function.
  Reduces to -(7 alpha_0 / 12 pi) ln(s/M_W^2) at s >> M_W^2 and
  exhibits the cusp at s = 4 M_W^2 from the W+W- threshold opening
  in Im Pi_gamma_gamma. This is *not* the literal full one-loop
  bosonic VP (which in 't Hooft-Feynman gauge requires summing the
  W loop with charged-Goldstone, ghost, and seagull diagrams); it
  reproduces the dominant threshold and asymptotic structure. The
  difference from the full bosonic VP is an O(alpha) finite
  constant, invisible at the scale of this plot.

- Hadronic: pQCD via the same fermion one-loop form, with effective
  light-quark masses anchored to Delta alpha_had^(5)(M_Z) = 0.02764.
  This is wrong below ~2 GeV where pQCD does not apply; for an
  AN-quality hadronic curve use Jegerlehner's alphaQED data-driven
  parametrisation.

Run with the mfs venv:
    source ../../../mfs/.venv/bin/activate
    python plot_alpha_running_exact.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad
import mplhep as hep

hep.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 16})

OUTPUT_DIR = "/work/submit/david_w/Documents/AN-EN-XXX/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_0 = 1.0 / 137.036
MZ      = 91.1876
M_TOP   = 172.76
M_W     = 80.377

DALPHA_HAD_MZ     = 0.02764
DALPHA_HAD_MZ_ERR = 0.00010

LEPTONS = [('e',   5.110e-4),
           ('mu',  0.10566),
           ('tau', 1.7769)]

# 5 light quarks (effective MS-bar masses; not physical below ~2 GeV)
QUARKS = [(2/3, 0.070),
          (1/3, 0.070),
          (1/3, 0.150),
          (2/3, 1.28),
          (1/3, 4.18)]

# ---------------------------------------------------------------------------
# Subtracted Passarino-Veltman B_0 with equal masses (real part)
# ---------------------------------------------------------------------------

def B0_bar_real(s, m):
    """Re[B_0(s, m, m) - B_0(0, m, m)].

    Vanishes at s=0; finite cusp at s = 4 m^2; ~ 2 - ln(s/m^2) at
    s >> m^2. Square-root threshold singularity in the slope at
    s = 4 m^2."""
    if s == 0.0:
        return 0.0
    z = s / (m * m)
    if z < 0:
        # spacelike: real, no branch cut
        beta = np.sqrt(1.0 - 4.0/z)            # > 1
        return 2.0 - beta * np.log((beta + 1.0) / (beta - 1.0))
    elif z < 4.0:
        beta_p = np.sqrt(4.0/z - 1.0)
        return 2.0 - 2.0 * beta_p * np.arctan(1.0 / beta_p)
    elif z == 4.0:
        return 2.0
    else:  # z > 4: above threshold
        beta = np.sqrt(1.0 - 4.0/z)            # 0 < beta < 1
        return 2.0 - beta * np.log((1.0 + beta) / (1.0 - beta))


# ---------------------------------------------------------------------------
# Exact one-loop fermion VP (real part)
# ---------------------------------------------------------------------------

def delta_alpha_fermion_exact(s, m, charge, Nc):
    """Re[(2 alpha_0 / pi) Q^2 N_c int_0^1 dx x(1-x) ln(1 - x(1-x) s/m^2 - i eps)]."""
    if s <= 0.0 or m <= 0.0:
        return 0.0
    z = s / (m * m)

    def integrand(x):
        arg = 1.0 - x * (1.0 - x) * z
        if arg == 0.0:
            return 0.0
        return x * (1.0 - x) * np.log(abs(arg))

    integral, _ = quad(integrand, 0.0, 1.0, limit=300, epsabs=1e-14, epsrel=1e-10)
    return (2.0 * ALPHA_0 / np.pi) * charge ** 2 * Nc * integral


# ---------------------------------------------------------------------------
# Component sums
# ---------------------------------------------------------------------------

def delta_alpha_lep(M):
    s = M * M
    return sum(delta_alpha_fermion_exact(s, ml, 1.0, 1) for _, ml in LEPTONS)


def _delta_alpha_had_pQCD(M):
    s = M * M
    return sum(delta_alpha_fermion_exact(s, mq, Qq, 3) for Qq, mq in QUARKS)


_HAD_OFFSET = DALPHA_HAD_MZ - _delta_alpha_had_pQCD(MZ)


def delta_alpha_had(M):
    return _delta_alpha_had_pQCD(M) + _HAD_OFFSET


def delta_alpha_top(M):
    return delta_alpha_fermion_exact(M * M, M_TOP, 2.0/3.0, 3)


def delta_alpha_W(M):
    """W contribution via subtracted B_0 (see module docstring)."""
    return (7.0 * ALPHA_0 / (12.0 * np.pi)) * B0_bar_real(M * M, M_W)


def delta_alpha_total(M):
    return (delta_alpha_lep(M) + delta_alpha_had(M)
            + delta_alpha_top(M) + delta_alpha_W(M))


# ---------------------------------------------------------------------------
# Build grids — denser near 2 M_W and 2 m_t to resolve cusps
# ---------------------------------------------------------------------------
M_grid_log = np.logspace(np.log10(1.0), np.log10(1000.0), 600)
# extra resolution near W and top pair-production thresholds
M_grid_W   = np.linspace(2*M_W - 5, 2*M_W + 30, 200)
M_grid_t   = np.linspace(2*M_TOP - 10, 2*M_TOP + 60, 200)
M_grid     = np.unique(np.concatenate([M_grid_log, M_grid_W, M_grid_t]))
M_grid     = M_grid[M_grid > 0]

print(f"Computing {len(M_grid)} grid points...")
dal_lep = np.array([delta_alpha_lep(m) for m in M_grid])
dal_had = np.array([delta_alpha_had(m) for m in M_grid])
dal_top = np.array([delta_alpha_top(m) for m in M_grid])
dal_W   = np.array([delta_alpha_W(m)   for m in M_grid])
dal_tot = dal_lep + dal_had + dal_top + dal_W
ialpha  = (1.0 - dal_tot) / ALPHA_0
print("Done.")

# Print a few sanity values
def report(M):
    print(f"  M = {M:7.2f} GeV: 1/alpha = {(1-delta_alpha_total(M))/ALPHA_0:.3f}, "
          f"Da_W = {delta_alpha_W(M):+.5f}, Da_top = {delta_alpha_top(M):+.5f}")
print("Sanity checks:")
for M in (10.0, MZ, M_W, 2*M_W, 200.0, M_TOP, 2*M_TOP, 1000.0):
    report(M)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(6, 8),
    gridspec_kw={"height_ratios": [2, 3], "hspace": 0.05},
)

# top panel: 1/alpha(M)
ax_top.plot(M_grid, ialpha, color='black', lw=2.0)
ax_top.axvline(MZ,        color='grey', lw=0.9, ls='--')
ax_top.axvline(2*M_W,     color='C2',   lw=0.7, ls=':', alpha=0.7)
ax_top.axvline(2*M_TOP,   color='C4',   lw=0.7, ls=':', alpha=0.7)
ax_top.set_xscale('log')
ax_top.set_xlim(M_grid[0], M_grid[-1])
ax_top.set_ylim(124, 138)
ax_top.set_ylabel(r'$1/\alpha(M)$')
ax_top.tick_params(labelbottom=False)
ax_top.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))

ymin, ymax = ax_top.get_ylim()
ax_top.text(MZ * 1.06,    ymin + 0.5, r'$M_Z$',  fontsize=12, color='grey', va='bottom')
ax_top.text(2*M_W * 1.04, ymin + 0.5, r'$2M_W$', fontsize=11, color='C2',   va='bottom')
ax_top.text(2*M_TOP*1.04, ymin + 0.5, r'$2m_t$', fontsize=11, color='C4',   va='bottom')

# bottom panel: components
ax_bot.plot(M_grid, dal_lep, color='C0', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{lep}$')
ax_bot.plot(M_grid, dal_had, color='C3', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{had}^{(5)}$')
ax_bot.fill_between(M_grid, dal_had - DALPHA_HAD_MZ_ERR, dal_had + DALPHA_HAD_MZ_ERR,
                    color='C3', alpha=0.25)
ax_bot.plot(M_grid, dal_W,   color='C2', lw=2.0,
            label=r'$\Delta\alpha_W$')
ax_bot.plot(M_grid, dal_top, color='C4', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{top}$')
ax_bot.plot(M_grid, dal_tot, color='black', lw=2.0,
            label=r'$\Delta\alpha_\mathrm{total}$')
ax_bot.fill_between(M_grid, dal_tot - DALPHA_HAD_MZ_ERR, dal_tot + DALPHA_HAD_MZ_ERR,
                    color='grey', alpha=0.30)
ax_bot.axvline(MZ,      color='grey', lw=0.9, ls='--')
ax_bot.axvline(2*M_W,   color='C2',   lw=0.7, ls=':', alpha=0.7)
ax_bot.axvline(2*M_TOP, color='C4',   lw=0.7, ls=':', alpha=0.7)
ax_bot.axhline(0,       color='grey', lw=0.7, ls=':')
ax_bot.set_xscale('log')
ax_bot.set_xlim(M_grid[0], M_grid[-1])
ax_bot.set_ylim(-0.005, 0.10)
ax_bot.set_xlabel(r'$M$ (GeV)')
ax_bot.set_ylabel(r'$\Delta\alpha(M)$')
ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax_bot.legend(loc='upper left', fontsize=12, framealpha=0.9, ncol=2)

for ext in ('pdf', 'png'):
    path = os.path.join(OUTPUT_DIR, f'alpha_running_exact.{ext}')
    fig.savefig(path, dpi=150 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

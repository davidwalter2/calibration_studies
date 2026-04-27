"""
Plot the dimuon lineshape for J/psi, Upsilon(1S), and Z/gamma* as three separate
figures, each with a top (lineshape) and bottom (ratio) panel, showing pre-FSR
(NR BW / DY LO) and post-FSR (x QED radiation kernel).

Run with the mfs venv:
  source ../../mfs/.venv/bin/activate
  python plot_lineshape_kernel.py

Output: ../../../../Documents/AN-EN-XXX/plots/lineshape_kernel_{jpsi,upsilon,z}.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import functions
import constants

hep.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 16})

OUTPUT_DIR = "/work/submit/david_w/Documents/AN-EN-XXX/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: pre-FSR shape and its kernel-folded version for a given mass grid
# ---------------------------------------------------------------------------
def nrbw(m, mass, width):
    return functions.non_relativistic_breit_wigner(m, mass, width)

def dy_shape(m):
    """Parton-level γ+Z DY shape: term_1+term_2+term_3 summed over u/d (L+R).
    All terms have units GeV^-4, giving correct relative γ vs Z normalization."""
    Q2 = m**2
    mz, gz, s2 = constants.mass_z, constants.width_z, constants.sin2theta_w
    quarks = [
        ( 2/3,  1/2 - 2/3*s2, -2/3*s2),   # u: (e_f, gL, gR)
        (-1/3, -1/2 + 1/3*s2,  1/3*s2),   # d
    ]
    D = (Q2 - mz**2)**2 + mz**2 * gz**2   # fixed-width denominator
    def _t1(e_f):    return e_f**2 / (2*Q2**2)
    def _t2(e_f, g): return ((1 - mz**2/Q2) / D) * (1 - 4*s2)/(4*s2*(1-s2)) * e_f * g
    def _t3(g):      return g**2 * (1 + (1-4*s2)**2) / (32*s2**2*(1-s2)**2 * D)
    total = 0.0
    for e_f, gL, gR in quarks:
        for g in (gL, gR):
            total += _t1(e_f) + _t2(e_f, g) + _t3(g)
    return total

def make_figure(m_grid, prefsr_fn, label_peak, xlim, ylim_top, xlabel_unit="GeV",
                beta_label=None, tag=""):
    """Produce one standalone figure (top: lineshape, bottom: ratio)."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    prefsr  = prefsr_fn(m_grid)
    postfsr = functions.convolution_fft(prefsr_fn, m_grid)

    norm = prefsr.max()
    prefsr  = prefsr  / norm
    postfsr = postfsr / norm

    ax_top.plot(m_grid, prefsr,  color='C0', lw=2.0, ls='-',  label='Pre-FSR')
    ax_top.plot(m_grid, postfsr, color='C1', lw=2.0, ls='--',
                label=r'Post-FSR ($\otimes\,K_\mathrm{QED}$)')

    ax_top.set_xlim(xlim)
    ax_top.set_ylim(ylim_top)
    ax_top.set_ylabel(r"d$\sigma$/d$M$ (norm.)")
    ax_top.tick_params(labelbottom=False)
    ax_top.text(0.97, 0.95, label_peak, transform=ax_top.transAxes,
                ha='right', va='top', fontsize=15)
    if beta_label:
        ax_top.text(0.97, 0.72, beta_label, transform=ax_top.transAxes,
                    ha='right', va='top', fontsize=13, color='gray')
    ax_top.legend(loc='upper left', fontsize=13, framealpha=0.9)

    ratio = np.where(prefsr > 1e-6, postfsr / prefsr, 1.0)
    ax_bot.plot(m_grid, ratio, color='C1', lw=2.0)
    ax_bot.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax_bot.set_xlim(xlim)
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlabel(r"$M_{\mu\mu}$ (%s)" % xlabel_unit)
    ax_bot.set_ylabel("Post/Pre")
    ax_bot.xaxis.set_major_locator(ticker.MaxNLocator(4, prune='both'))

    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR, f"lineshape_kernel_{tag}.{ext}")
        fig.savefig(path, dpi=150 if ext == 'png' else None, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)

# --- J/psi ---
m_jpsi = np.linspace(constants.mass_j - 0.0008, constants.mass_j + 0.0008, 600)
make_figure(
    m_jpsi,
    lambda m: nrbw(m, constants.mass_j, constants.width_j),
    label_peak=r"$\mathrm{J}/\psi\ (3.097\ \mathrm{GeV})$",
    xlim=(m_jpsi[0], m_jpsi[-1]),
    ylim_top=(-0.05, 1.25),
    beta_label=r"$\beta_t \approx 0.029$",
    tag="jpsi",
)

# --- Upsilon(1S) ---
m_upsi = np.linspace(constants.mass_u1 - 0.0004, constants.mass_u1 + 0.0004, 600)
make_figure(
    m_upsi,
    lambda m: nrbw(m, constants.mass_u1, constants.width_u1),
    label_peak=r"$\Upsilon(1\mathrm{S})\ (9.460\ \mathrm{GeV})$",
    xlim=(m_upsi[0], m_upsi[-1]),
    ylim_top=(-0.05, 1.25),
    beta_label=r"$\beta_t \approx 0.041$",
    tag="upsilon",
)

# --- Z/gamma* ---
m_z = np.linspace(60., 120., 1200)
make_figure(
    m_z,
    dy_shape,
    label_peak=r"$\mathrm{Z}/\gamma^*\ (91.2\ \mathrm{GeV})$",
    xlim=(60., 120.),
    ylim_top=(-0.05, 1.25),
    beta_label=r"$\beta_t \approx 0.062$",
    tag="z",
)

# --- QED radiation kernel K(z; M) for J/psi, Upsilon(1S), Z ---
z_grid = np.linspace(0.0, 0.98, 500)

KERNEL_MASSES = [
    (constants.mass_j,  'C0', r'$m_{\mathrm{J}/\psi}$'),
    (constants.mass_u1, 'C2', r'$m_{\Upsilon(1\mathrm{S})}$'),
    (constants.mass_z,  'C4', r'$m_{\mathrm{Z}}$'),
]

fig_k, (ax_kt, ax_kb) = plt.subplots(
    2, 1, figsize=(6, 6),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

kernels = {}
for mass, color, label in KERNEL_MASSES:
    K = functions.radiator_kernel(z_grid, mass)
    norm = np.trapz(K, z_grid)
    K /= norm
    kernels[label] = K
    ax_kt.plot(z_grid, K, color=color, lw=2.0, label=label)

K_ref = kernels[r'$m_{\mathrm{J}/\psi}$']
for mass, color, label in KERNEL_MASSES:
    ratio_k = kernels[label] / K_ref
    ax_kb.plot(z_grid, ratio_k, color=color, lw=2.0)

ax_kt.set_xlim(0, 0.98)
ax_kt.set_ylim(bottom=0)
ax_kt.set_ylabel(r"$K_\mathrm{QED}(z;\,M)$ (norm.)")
ax_kt.tick_params(labelbottom=False)
ax_kt.legend(loc='upper left', fontsize=13, framealpha=0.9)

ax_kb.axhline(1.0, color='gray', lw=0.8, ls='--')
ax_kb.set_xlim(0, 0.98)
ax_kb.set_ylim(0.9, 1.1)
ax_kb.set_xlabel(r"$z = m_{\mu\mu}^{\prime\,2}/M^2$")
ax_kb.set_ylabel(r"Ratio to J/$\psi$")
ax_kb.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='both'))

for ext in ('pdf', 'png'):
    path = os.path.join(OUTPUT_DIR, f"lineshape_kernel_kernel.{ext}")
    fig_k.savefig(path, dpi=150 if ext == 'png' else None, bbox_inches='tight')
    print(f"Saved: {path}")
plt.close(fig_k)

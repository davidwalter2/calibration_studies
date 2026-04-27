"""
Plot sin²θW vs Q: SM prediction (V-shape), existing measurements, and future projections.

Run with the mfs venv:
  source ../../mfs/.venv/bin/activate
  python plot_sin2thetaW_running.py

Output: ../../Documents/AN-EN-XXX/plots/sin2thetaW_running.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import os

hep.style.use(hep.style.ROOT)
plt.rcParams.update({'font.size': 19})

OUTPUT_DIR = "/work/submit/david_w/Documents/AN-EN-XXX/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SM prediction curve (MS-bar scheme, Erler-Langacker PDG 2020 Fig 10.1
# + Czarnecki-Marciano 2000, Table I)
# ---------------------------------------------------------------------------
MZ = 91.1876

Q_lo = np.array([0.001, 0.002, 0.005, 0.010, 0.020, 0.050,
                 0.100, 0.158, 0.35, 0.5, 1.0, 2.0, 3.0,
                 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, MZ])
s2_lo = np.array([0.23882, 0.23881, 0.23880, 0.23879, 0.23877, 0.23873,
                  0.23867, 0.23862, 0.23849, 0.23843, 0.23824, 0.23800, 0.23782,
                  0.23751, 0.23714, 0.23694, 0.23654, 0.23622, 0.23564, 0.23481, 0.23416, 0.23122])

Q_hi = np.array([MZ, 100., 120., 150., 200., 250., 300.,
                 400., 500., 700., 1000., 1500., 2000., 3000., 3250.])
s2_hi = np.array([0.23122, 0.23147, 0.23173, 0.23215, 0.23284, 0.23343, 0.23395,
                  0.23479, 0.23548, 0.23663, 0.23797, 0.23959, 0.24075, 0.24260, 0.24290])

def sm_interp(Q):
    if Q <= MZ:
        return float(np.interp(Q, Q_lo, s2_lo))
    else:
        return float(np.interp(Q, Q_hi, s2_hi))

# ---------------------------------------------------------------------------
# Existing measurements
# ---------------------------------------------------------------------------
msbar_data = [
    (0.0024,  0.2356, 0.0011, "Cs APV"),
    (0.1585,  0.2397, 0.0013, "E158"),
]

nutev = (20.0, 0.2277, 0.0016, "NuTeV")

# ---------------------------------------------------------------------------
# Future projections — centred on SM prediction at that Q
# ---------------------------------------------------------------------------
proj_jlab = [
    (0.086, sm_interp(0.086), 0.00028, "MOLLER"),
    (0.35,  sm_interp(0.35),  0.00030, "SoLID"),
]

amoroso_Q = [130., 173., 245., 387., 707., 3250.]
amoroso_s2 = [sm_interp(q) for q in amoroso_Q]
amoroso_run3_err  = [s * 0.010 for s in amoroso_s2]
amoroso_hllhc_err = [s * 0.003 for s in amoroso_s2]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))

# SM curve
ax.plot(Q_lo, s2_lo, 'k-', lw=1.8, zorder=3,
        label=r"SM prediction ($\overline{\mathrm{MS}}$)")
ax.plot(Q_hi, s2_hi, 'k-', lw=1.8, zorder=3)
ax.axvline(MZ, color='gray', lw=0.8, ls='--', zorder=1)

# MS-bar fixed-target measurements (filled)
for (Q, val, err, lbl) in msbar_data:
    ax.errorbar(Q, val, yerr=err,
                fmt='o', ms=6, color='C0', elinewidth=1.5,
                capsize=3.5, zorder=6, label=lbl)

# NuTeV (on-shell, filled)
ax.errorbar(nutev[0], nutev[1], yerr=nutev[2],
            fmt='s', ms=6, color='gray', elinewidth=1.2,
            capsize=3, zorder=5,
            label=r"NuTeV (on-shell, disputed)")

# JLab future projections (open)
for i, (Q, val, err, lbl) in enumerate(proj_jlab):
    ax.errorbar(Q, val, yerr=err,
                fmt='s', ms=7, color='C2', elinewidth=1.5,
                capsize=3.5, zorder=7, mfc='none', mew=1.8,
                label=f"JLab {lbl} (proj.)")

# LHC Run 3 projections — slight rightward offset (open)
run3_x = [q * 1.04 for q in amoroso_Q]
for i, (Q, s2, err) in enumerate(zip(run3_x, amoroso_s2, amoroso_run3_err)):
    ax.errorbar(Q, s2, yerr=err,
                fmt='^', ms=6, color='C1', elinewidth=1.3,
                capsize=3, zorder=7, mfc='none', mew=1.5,
                label=r"LHC Run 3, 300 fb$^{-1}$ [Amoroso et al.]" if i == 0 else None)

# HL-LHC projections — slight leftward offset (open)
hllhc_x = [q * 0.96 for q in amoroso_Q]
for i, (Q, s2, err) in enumerate(zip(hllhc_x, amoroso_s2, amoroso_hllhc_err)):
    ax.errorbar(Q, s2, yerr=err,
                fmt='v', ms=6, color='C4', elinewidth=1.3,
                capsize=3, zorder=7, mfc='none', mew=1.5,
                label=r"HL-LHC, 3 ab$^{-1}$ [Amoroso et al.]" if i == 0 else None)

# Scheme note and mZ label — lower left, same baseline, no box
ax.text(0.02, 0.03,
        r"$\overline{\mathrm{MS}}$ scheme (except NuTeV: on-shell)",
        transform=ax.transAxes, va='bottom')
# mZ annotation at the same vertical level, just right of the dashed line
ax.text(MZ * 1.04, 0.2207, r"$m_Z$", color='gray', va='bottom')

# Axis
ax.set_xscale('log')
ax.set_xlim(0.0015, 5000.)
ax.set_ylim(0.220, 0.254)
ax.set_xlabel(r"Energy scale $Q$ (GeV)")
ax.set_ylabel(r"$\sin^2\!\theta_W$")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}" if x >= 1 else f"{x:.3g}"))

ax.legend(loc='upper left', ncols=2, framealpha=0.9, borderpad=0.8)

plt.tight_layout()

for ext in ('pdf', 'png'):
    path = os.path.join(OUTPUT_DIR, f"sin2thetaW_running.{ext}")
    plt.savefig(path, dpi=150 if ext == 'png' else None, bbox_inches='tight')
    print(f"Saved: {path}")

plt.close()

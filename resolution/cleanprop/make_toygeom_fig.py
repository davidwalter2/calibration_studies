#!/usr/bin/env python3
"""Schematic of the LAYERED TOY geometry used by the clean-propagation closure.

The numbers are not retyped: the shell radii, the layer thickness / density
relation and the scoring radii are read from the two generators that build the
geometry and the sim driver together,

    Analysis/HitAnalyzer/data/gen_toy_layers.py   material shells (15)
    Analysis/HitAnalyzer/test/gen_toy_config.py   scoring radii (14)

so a change there shows up here instead of the slide going stale.  RMIN/RMAX/DZ
and the T -> rho rule come from the same file.

usage (from calibration_studies/resolution):
    python cleanprop/make_toygeom_fig.py
"""
import ast
import datetime
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.patches import Circle

hep.style.use(hep.style.ROOT)

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides", "assets")
DATED = os.path.expanduser(
    f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_toygeom/")

RED, BLUE, GREY = "#A31F34", "#1F4E79", "#8A8B8C"

T = 0.10                      # cm, the NSUB=1 / T=0.10 configuration
PT, BFIELD, ETA = 3.0, 3.8, 0.30


def _listlit(path, name):
    """The literal value of a module-level `name = [...]` assignment."""
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _scalars(path, names):
    """Module-level scalar assignments, read as literals (RMIN, RMAX, DZ)."""
    src, out = open(path).read(), {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        t0 = node.targets[0]
        elts = t0.elts if isinstance(t0, ast.Tuple) else [t0]
        tg = [getattr(e, "id", None) for e in elts]
        if tg == list(names):
            out = dict(zip(names, ast.literal_eval(node.value)))
    if not out:
        raise KeyError(f"{names} not found in {path}")
    return out


shells = _listlit(f"{CMSSW}/data/gen_toy_layers.py", "radii")
score = _listlit(f"{CMSSW}/test/gen_toy_config.py", "RADII")
geo = _scalars(f"{CMSSW}/data/gen_toy_layers.py", ["RMIN", "RMAX", "DZ"])
rho = 9.0 * 0.10 / T                                   # the file's own rule

print(f"{len(shells)} material shells: {shells}")
print(f"{len(score)} scoring radii : {score}")
print(f"T = {T} cm, rho = {rho:.4f} g/cm3, "
      f"radial budget = {len(shells) * T * rho:.2f} g/cm2")
print(f"RMIN {geo['RMIN']}  RMAX {geo['RMAX']}  DZ {geo['DZ']} cm")

# the reference muon: a circle of radius R = pT/(0.3 B) through the origin,
# tangent to +x at it, and a constant pitch cot(theta) = sinh(eta)
R = PT / (0.3 * BFIELD) * 100.0                        # cm
cot = np.sinh(ETA)


def arc_xy(rmax):
    """(x, y) of the transverse projection out to radius rmax."""
    psi = np.linspace(0, 2 * np.arcsin(min(rmax, 2 * R) / (2 * R)), 400)
    return R * np.sin(psi), R * (1 - np.cos(psi))


fig, (ax, az) = plt.subplots(1, 2, figsize=(13.2, 5.6),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))

# ---------------------------------------------------------------- transverse
for r in shells:
    ax.add_patch(Circle((0, 0), r, fill=False, lw=1.1,
                        ec=RED if r in score else GREY,
                        ls="-" if r in score else "--"))
x, y = arc_xy(max(shells) + 6)
ax.plot(x, y, color=BLUE, lw=2.0, zorder=5)
# the crossing points, at each scoring radius
rr = np.hypot(x, y)
for r in score:
    i = int(np.argmin(np.abs(rr - r)))
    ax.plot(x[i], y[i], "o", ms=4.5, color=BLUE, zorder=6)
ax.set_xlim(-8, 120)
ax.set_ylim(-8, 120)
ax.set_aspect("equal")
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$y$ [cm]")
ax.set_title(f"transverse — $\\mu$, $p_T={PT:.0f}$ GeV, $R={R/100:.2f}$ m",
             fontsize=17, pad=10)

# --------------------------------------------------------------- longitudinal
for r in shells:
    az.plot([-geo["DZ"], geo["DZ"]], [r, r], lw=1.1,
            color=RED if r in score else GREY,
            ls="-" if r in score else "--")
s = np.linspace(0, 1.02 * max(shells) * np.sqrt(1 + cot ** 2), 400)
rl = s / np.sqrt(1 + cot ** 2)
az.plot(rl * cot, rl, color=BLUE, lw=2.0, zorder=5,
        label=rf"reference, $\eta={ETA:.2f}$")
az.set_xlim(-geo["DZ"] - 5, geo["DZ"] + 5)
az.set_ylim(0, geo["RMAX"] * 1.02)
az.set_xlabel("$z$ [cm]")
az.set_ylabel("$r$ [cm]")
az.set_title(f"longitudinal — barrel only, $|z| < {geo['DZ']:.0f}$ cm",
             fontsize=17, pad=10)
az.legend(loc="upper left", fontsize=15, frameon=False)

fig.text(0.5, 0.995,
         f"{len(shells)} shells of {T*10:.0f} mm at the real barrel-layer radii, "
         rf"$\rho = {rho:.0f}$ g/cm$^3$, $Z=8$, $A=16$ — vacuum in between",
         ha="center", va="top", fontsize=17)
fig.text(0.5, 0.938,
         f"solid red = the {len(score)} scoring planes (r = {min(score)}–"
         f"{max(score)} cm)   ·   dashed grey = the double-sided partner at "
         f"{[r for r in shells if r not in score][0]} cm, not scored",
         ha="center", va="top", fontsize=14, color=GREY)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.11, wspace=0.24)

os.makedirs(OUT, exist_ok=True)
os.makedirs(DATED, exist_ok=True)
for d in (OUT, DATED):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"toygeom.{ext}"), dpi=160,
                    bbox_inches="tight")
print(f"wrote toygeom.png/pdf -> {OUT} and {DATED}")

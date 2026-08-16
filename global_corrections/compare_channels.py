"""Compare how the different CVH calibration channels (J/psi, Ks, cosmics)
constrain the global parameter block (field modes parmtype 14 + material
groups parmtype 15), and how combining them breaks correlations.

Consumes the per-channel information npz written by
    fit_global_grads.py --save-info <chan>.npz
(each holds grad, hess, parmtype, names, priors on the identical floated
parameter set -- the productions share one 81824-param catalog).

For each channel and the combination it reports, on the floated params:
  - per-parameter constraint sigma (sqrt of diag of the covariance)
  - which channel best constrains each parameter (the complementarity)
  - the Hessian eigenspectrum (how many well-measured directions each
    channel supplies, and how the combination adds rank/information)
  - max off-diagonal correlation per channel vs combined (correlation
    breaking)
Plots the per-parameter sigma bar chart and the field/material correlation
matrices (per channel + combined).

Usage:
    python compare_channels.py --jpsi jpsi.npz --ks ks.npz --cosmics cos.npz \
        --nmr <nmr.npz> -o <outdir>
"""
import argparse
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from wums import output_tools, plot_tools

hep.style.use(hep.style.ROOT)


def load_channel(path):
    d = np.load(path, allow_pickle=True)
    return dict(grad=d["grad"], hess=d["hess"], parmtype=d["parmtype"],
               names=[str(n) for n in d["names"]], priors=d["priors"],
               ncand=int(d["ncand"]))


def cov_of(hess, priors, nmr=None, m14=None):
    """Covariance = 2 * (H + priors + NMR)^-1, matching fit_global_grads."""
    H = hess + np.diag(2. * priors)
    if nmr is not None and m14 is not None:
        gm, res, sig = nmr
        for p in range(len(res)):
            w = 1. / sig[p] ** 2
            H[np.ix_(m14, m14)] += 2. * w * np.outer(gm[p], gm[p])
    return 2. * np.linalg.inv(H), H


def corr(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


parser = argparse.ArgumentParser()
parser.add_argument("--jpsi")
parser.add_argument("--ks")
parser.add_argument("--cosmics")
parser.add_argument("--nmr", default=None, help="NMR constraint npz (field block)")
parser.add_argument("-o", "--outpath",
                    default=f"/home/submit/david_w/public_html/ZMass/MagneticField/{datetime.date.today().strftime('%y%m%d')}_channel_comparison")
args = parser.parse_args()

chans = {}
for tag, p in [("Jpsi", args.jpsi), ("Ks", args.ks), ("cosmics", args.cosmics)]:
    if p and os.path.exists(p):
        chans[tag] = load_channel(p)
assert chans, "no channel npz provided"

# reference metadata from the first channel (all share the catalog)
ref = next(iter(chans.values()))
parmtype = ref["parmtype"]
names = ref["names"]
nfit = len(parmtype)
m14 = np.where(parmtype == 14)[0]
m15 = np.where(parmtype == 15)[0]
priors = ref["priors"]

nmr = None
if args.nmr:
    n = np.load(args.nmr)
    nmr = (n["g"], n["resid"], n["sigma"])

# per-channel + combined covariance
combos = dict(chans)
combos["combined"] = None  # filled below
hess_sum = sum(c["hess"] for c in chans.values())
grad_sum = sum(c["grad"] for c in chans.values())
combos["combined"] = dict(hess=hess_sum, grad=grad_sum, priors=priors,
                          parmtype=parmtype, names=names,
                          ncand=sum(c["ncand"] for c in chans.values()))

print(f"floated params: {nfit}  (field {len(m14)}, material {len(m15)})\n")
covs = {}
print(f"{'channel':10s} {'candidates':>12s} {'well-meas dirs':>14s} {'max|corr|':>9s} "
      f"{'field sig[mode0]':>16s}")
for tag, c in combos.items():
    cov, H = cov_of(c["hess"], c["priors"], nmr, m14)
    covs[tag] = cov
    # well-measured directions: eigenvalues of the prior-subtracted info
    # (data information only), counted above a relative threshold
    info = c["hess"]
    ev = np.linalg.eigvalsh(info)
    wellmeas = int((ev > 1e-6 * ev.max()).sum()) if ev.max() > 0 else 0
    cc = corr(cov)
    maxcorr = np.abs(cc - np.eye(nfit)).max()
    sig0 = np.sqrt(cov[m14[0], m14[0]])
    print(f"{tag:10s} {c['ncand']:>12,d} {wellmeas:>14d} {maxcorr:>9.3f} {sig0:>16.4f}")

# --- per-parameter sigma: which channel constrains what ---
outdir = output_tools.make_plot_dir(args.outpath)
fig, ax = plt.subplots(figsize=(15, 7))
x = np.arange(nfit)
colors = {"Jpsi": "#A31F34", "Ks": "#1f77b4", "cosmics": "#2ca02c", "combined": "#222222"}
for tag in ["Jpsi", "Ks", "cosmics", "combined"]:
    if tag not in covs:
        continue
    sig = np.sqrt(np.clip(np.diag(covs[tag]), 0, None))
    ax.plot(x, sig, ".-", color=colors[tag], label=tag, lw=1.4, ms=5,
            alpha=0.9 if tag == "combined" else 0.7)
ax.axvline(len(m14) - 0.5, color="k", ls=":", lw=1)
ax.text(len(m14) / 2, ax.get_ylim()[1] * 0.9, "field modes (14)", ha="center", fontsize=15)
ax.text(len(m14) + len(m15) / 2, ax.get_ylim()[1] * 0.9, "material (15)", ha="center", fontsize=15)
ax.set_yscale("log")
ax.set_xlabel("floated parameter index")
ax.set_ylabel(r"constraint $\sigma$ (sqrt diag covariance)")
ax.legend(fontsize=16, ncol=4)
plot_tools.add_decor(ax, "CMS", "Work in progress", data=True, lumi=None, no_energy=True)
plot_tools.save_pdf_and_png(outdir, "per_param_sigma")
plt.close(fig)

# --- correlation matrices ---
for tag in covs:
    cc = corr(covs[tag])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    for b in [len(m14) - 0.5]:
        ax.axhline(b, color="k", lw=0.8); ax.axvline(b, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title(f"{tag}: parameter correlations (field | material)", fontsize=15)
    plot_tools.save_pdf_and_png(outdir, f"corr_{tag}")
    plt.close(fig)

output_tools.write_index_and_log(outdir, "channel_comparison", args=args)
print(f"\nplots in {outdir}")

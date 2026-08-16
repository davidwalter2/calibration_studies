"""Show the effect of the NMR probe constraints on the combined
multi-channel fit: parameter correlations and field-scale precision
WITHOUT vs WITH the 4 NMR |B| rows. Uses the saved per-channel info npz."""
import argparse, datetime, os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from wums import output_tools, plot_tools
hep.style.use(hep.style.ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/ceph/submit/data/user/d/david_w/ZMass/cvh/channel_info_260721")
ap.add_argument("--nmr", default="/work/submit/david_w/ZMass/MuonMomentumScaleCalibration/Analysis/data/nmr_constraints_2016_custom50.npz")
ap.add_argument("-o", "--outpath",
                default=f"/home/submit/david_w/public_html/ZMass/MagneticField/{datetime.date.today().strftime('%y%m%d')}_nmr_effect")
args = ap.parse_args()

ch = {t: np.load(f"{args.dir}/{t}.npz", allow_pickle=True) for t in ["Jpsi", "Ks", "cosmics"]}
ref = ch["Jpsi"]
pt = ref["parmtype"]; names = [str(n) for n in ref["names"]]; pri = ref["priors"]
m14 = np.where(pt == 14)[0]; m15 = np.where(pt == 15)[0]
n = np.load(args.nmr); gm, res, sig = n["g"], n["resid"], n["sigma"]

Hbase = sum(c["hess"] for c in ch.values()) + np.diag(2 * pri)


def solve(withnmr):
    H = Hbase.copy()
    if withnmr:
        for p in range(len(res)):
            w = 1. / sig[p] ** 2
            H[np.ix_(m14, m14)] += 2. * w * np.outer(gm[p], gm[p])
    cov = 2. * np.linalg.inv(H)
    d = np.sqrt(np.diag(cov))
    return cov, cov / np.outer(d, d), d


results = {}
for tag, wn in [("no NMR", False), ("with NMR", True)]:
    cov, corr, d = solve(wn)
    cc = np.abs(corr - np.eye(len(pt)))
    i, j = np.unravel_index(cc.argmax(), cc.shape)
    fieldfield = cc[np.ix_(m14, m14)]
    results[tag] = dict(corr=corr, d=d, maxc=cc[i, j], maxpair=(names[i], names[j]),
                        sig0=d[m14[0]], nhi=(fieldfield > 0.8).sum() // 2)

print(f"{'':10s} {'sig(mode0)':>11s} {'max|corr|':>10s} {'max-corr pair':>26s} "
      f"{'field pairs >0.8':>16s}")
for tag in ["no NMR", "with NMR"]:
    r = results[tag]
    print(f"{tag:10s} {r['sig0']:>11.4f} {r['maxc']:>10.3f} "
          f"{r['maxpair'][0]+'/'+r['maxpair'][1]:>26s} {r['nhi']:>16d}")
# improvement in the low-order field modes from NMR
imp = results["no NMR"]["d"][m14[:6]] / results["with NMR"]["d"][m14[:6]]
print(f"\nlow-l field mode sigma improvement (no NMR / with NMR): "
      f"{np.round(imp, 2)}  (modes 0-5)")

# --- side-by-side field-block correlation matrices ---
outdir = output_tools.make_plot_dir(args.outpath)
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, tag in zip(axes, ["no NMR", "with NMR"]):
    cf = results[tag]["corr"][np.ix_(m14, m14)]
    im = ax.imshow(cf, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.xaxis.set_minor_locator(plt.NullLocator()); ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title(f"field-mode correlations, combined ({tag})\n"
                 f"max|corr|={results[tag]['maxc']:.3f}, sigma(mode0)={results[tag]['sig0']:.4f}",
                 fontsize=14)
fig.colorbar(im, ax=axes, label="correlation", fraction=0.025)
plot_tools.save_pdf_and_png(outdir, "field_corr_nmr_vs_nonmr")
plt.close(fig)
output_tools.write_index_and_log(outdir, "nmr_effect", args=args)
print(f"\nplots in {outdir}")

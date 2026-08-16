"""2D (z,r) maps of Delta Bz + uncertainty for J/psi only, Ks only, and
J/psi+Ks combined, each panel with its OWN colour scale (the channels differ
by orders of magnitude: Ks alone floats the scale to ~+300 mT, so a common
scale would wash out J/psi and combined)."""
import datetime, sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from wums import output_tools, plot_tools
sys.path.insert(0, "/work/submit/david_w/ZMass/mfs")
import harmonic_basis
hep.style.use(hep.style.ROOT)

W = "/ceph/submit/data/user/d/david_w/ZMass/cvh/channel_info_260721"
COEF = "/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt"
meta, params = {}, []
for line in open(COEF):
    if line.startswith("#"):
        p = line[1:].split()
        if len(p) == 2: meta[p[0]] = p[1]
        continue
    _, l, m, cs, _ = line.split(); params.append((int(l), int(m), cs))
r_scale = float(meta["r_scale"]); z0 = float(meta["z0"]); cmssw_norm = bool(int(meta["cmssw_norm"]))
ch = {t: np.load(f"{W}/{t}.npz", allow_pickle=True) for t in ["Jpsi", "Ks", "cosmics"]}
pt = ch["Jpsi"]["parmtype"]; pri = ch["Jpsi"]["priors"]; m14 = np.where(pt == 14)[0]

rr = np.linspace(1.0, 115.0, 45); zz = np.linspace(-280.0, 280.0, 113)
pp = np.linspace(-np.pi, np.pi, 16, endpoint=False)
R, Z, P = np.meshgrid(rr, zz, pp, indexing="ij"); shape = R.shape
rf, zf, pf = R.ravel(), Z.ravel(), P.ravel()
M = np.array([harmonic_basis.bz_basis(l, m, cs, rf, pf, zf, r_scale=r_scale, z0=z0,
                                      cmssw_norm=cmssw_norm) for (l, m, cs) in params]).T


def solve(names):
    H = sum(ch[t]["hess"] for t in names) + np.diag(2. * pri)
    g = sum(ch[t]["grad"] for t in names); Hinv = np.linalg.inv(H)
    dc = (-Hinv @ g)[m14]; covf = (2 * Hinv)[np.ix_(m14, m14)]
    dbz = (M @ dc).reshape(shape).mean(axis=2) * 1e3
    var = np.einsum("pi,ij,pj->p", M, covf, M)
    sbz = np.sqrt(np.clip(var, 0, None)).reshape(shape).mean(axis=2) * 1e3
    return dbz, sbz


cols = [("J/$\\psi$ only", ["Jpsi"]), ("Ks only", ["Ks"]), ("J/$\\psi$ + Ks", ["Jpsi", "Ks"])]
outdir = output_tools.make_plot_dir(f"/home/submit/david_w/public_html/ZMass/MagneticField/{datetime.date.today().strftime('%y%m%d')}_bz_jpsi_ks")
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True, sharey=True)
for k, (lab, names) in enumerate(cols):
    dbz, sbz = solve(names)
    lim = np.abs(dbz).max()
    im0 = axes[0, k].pcolormesh(zz, rr, dbz, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0, k].set_title(lab, fontsize=16)
    fig.colorbar(im0, ax=axes[0, k], fraction=0.046, pad=0.02, label="mT" if k == 2 else "")
    im1 = axes[1, k].pcolormesh(zz, rr, sbz, cmap="viridis", vmin=0, vmax=sbz.max())
    fig.colorbar(im1, ax=axes[1, k], fraction=0.046, pad=0.02, label="mT" if k == 2 else "")
    for a in (axes[0, k], axes[1, k]):
        a.xaxis.set_minor_locator(plt.NullLocator()); a.yaxis.set_minor_locator(plt.NullLocator())
    axes[1, k].set_xlabel("z [cm]")
    print(f"{lab:14s} dBz mean {dbz.mean():+8.2f} rms {dbz.std():6.2f} mT | sigma median {np.median(sbz):.3f} mT")
axes[0, 0].set_ylabel(r"$\langle\Delta B_z\rangle_\phi$" + "\nr [cm]", fontsize=13)
axes[1, 0].set_ylabel(r"$\langle\sigma(B_z)\rangle_\phi$" + "\nr [cm]", fontsize=13)
fig.suptitle("per-panel colour scale (channels differ by orders of magnitude)", fontsize=13, y=0.98)
plot_tools.save_pdf_and_png(outdir, "bz_jpsi_ks_combined")
plt.close(fig)
output_tools.write_index_and_log(outdir, "bz_jpsi_ks")
print(f"plots in {outdir}")

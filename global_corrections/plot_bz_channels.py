"""2D (z, r) maps of the fitted Delta Bz correction and the propagated Bz
uncertainty for cumulative channel combinations:
  J/psi | +Ks | +cosmics | +NMR
from the saved per-channel info npz (grad, hess on the floated params).

For each combo:  H = sum(channel hess) + 2*priors [+ NMR rows];
  delta_c = -H^-1 grad ;  Cov = 2 H^-1
Delta Bz(z,r)   = phi-avg of  sum_i delta_c_i * bz_basis_i(r,phi,z)
sigma Bz(z,r)   = phi-avg of  sqrt( b(r,phi,z)^T Cov_field b(r,phi,z) )
(b = the Bz basis vector over the 50 field modes; Cov_field = the
parmtype-14 block of Cov). phi-averaged pointwise, so m!=0 modes enter
the uncertainty.
"""
import argparse, datetime, os, sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from wums import output_tools, plot_tools

sys.path.insert(0, "/work/submit/david_w/ZMass/mfs")
import harmonic_basis
hep.style.use(hep.style.ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/ceph/submit/data/user/d/david_w/ZMass/cvh/channel_info_260721")
ap.add_argument("--coeffs", default="/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt")
ap.add_argument("--nmr", default="/work/submit/david_w/ZMass/MuonMomentumScaleCalibration/Analysis/data/nmr_constraints_2016_custom50.npz")
ap.add_argument("-o", "--outpath",
                default=f"/home/submit/david_w/public_html/ZMass/MagneticField/{datetime.date.today().strftime('%y%m%d')}_bz_channels")
args = ap.parse_args()

# basis definition
meta, params = {}, []
for line in open(args.coeffs):
    if line.startswith("#"):
        p = line[1:].split()
        if len(p) == 2:
            meta[p[0]] = p[1]
        continue
    _, l, m, cs, _ = line.split()
    params.append((int(l), int(m), cs))
r_scale = float(meta["r_scale"]); z0 = float(meta["z0"]); cmssw_norm = bool(int(meta["cmssw_norm"]))

ch = {t: np.load(f"{args.dir}/{t}.npz", allow_pickle=True) for t in ["Jpsi", "Ks", "cosmics"]}
ref = ch["Jpsi"]
pt = ref["parmtype"]; pri = ref["priors"]
m14 = np.where(pt == 14)[0]
assert len(m14) == len(params), (len(m14), len(params))
n = np.load(args.nmr); gm, res, sig = n["g"], n["resid"], n["sigma"]

# (r,z) grid + phi samples for phi-averaging
rr = np.linspace(1.0, 115.0, 45)
zz = np.linspace(-280.0, 280.0, 113)
pp = np.linspace(-np.pi, np.pi, 16, endpoint=False)
R, Z, P = np.meshgrid(rr, zz, pp, indexing="ij")   # (nr, nz, nphi)
shape = R.shape
rf, zf, pf = R.ravel(), Z.ravel(), P.ravel()

# Bz design matrix over the 50 field modes at every (r,phi,z) point
print("building Bz design matrix ...")
M = np.zeros((rf.size, len(params)))
for i, (l, m, cs) in enumerate(params):
    M[:, i] = harmonic_basis.bz_basis(l, m, cs, rf, pf, zf,
                                       r_scale=r_scale, z0=z0, cmssw_norm=cmssw_norm)


def combo_solve(names, withnmr):
    H = sum(ch[t]["hess"] for t in names) + np.diag(2. * pri)
    g = sum(ch[t]["grad"] for t in names)
    if withnmr:
        for p in range(len(res)):
            w = 1. / sig[p] ** 2
            H[np.ix_(m14, m14)] += 2. * w * np.outer(gm[p], gm[p])
    Hinv = np.linalg.inv(H)
    dc = -Hinv @ g
    cov = 2. * Hinv
    covf = cov[np.ix_(m14, m14)]            # 50x50 field-mode covariance
    dbz = (M @ dc[m14]).reshape(shape).mean(axis=2) * 1e3               # mT, phi-avg
    var = np.einsum("pi,ij,pj->p", M, covf, M)                         # per-point Bz variance
    sbz = np.sqrt(np.clip(var, 0, None)).reshape(shape).mean(axis=2) * 1e3   # mT, phi-avg
    return dbz, sbz


combos = [("J/$\\psi$", ["Jpsi"], False),
          ("J/$\\psi$+Ks", ["Jpsi", "Ks"], False),
          ("J/$\\psi$+Ks+cosmics", ["Jpsi", "Ks", "cosmics"], False),
          ("J/$\\psi$+Ks+cosmics+NMR", ["Jpsi", "Ks", "cosmics"], True)]

dbzs, sbzs, labels = [], [], []
for lab, names, wn in combos:
    dbz, sbz = combo_solve(names, wn)
    dbzs.append(dbz); sbzs.append(sbz); labels.append(lab)
    print(f"{lab:26s} dBz[mT] mean {dbz.mean():+.2f} rms {dbz.std():.2f} | "
          f"sigmaBz[mT] median {np.median(sbz):.3f} max {sbz.max():.2f}")

outdir = output_tools.make_plot_dir(args.outpath)
# common scales
dlim = np.percentile(np.abs(np.array(dbzs)), 99)
slim = np.percentile(np.array(sbzs), 99)

fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True, sharey=True)
for k in range(4):
    a0 = axes[0, k]
    im0 = a0.pcolormesh(zz, rr, dbzs[k], cmap="RdBu_r", vmin=-dlim, vmax=dlim)
    a0.set_title(labels[k], fontsize=15)
    a1 = axes[1, k]
    im1 = a1.pcolormesh(zz, rr, sbzs[k], cmap="viridis", vmin=0, vmax=slim)
    for a in (a0, a1):
        a.xaxis.set_minor_locator(plt.NullLocator()); a.yaxis.set_minor_locator(plt.NullLocator())
    if k == 0:
        a0.set_ylabel(r"$\Delta B_z$" + "\n\nr [cm]", fontsize=14)
        a1.set_ylabel(r"$\sigma(B_z)$" + "\n\nr [cm]", fontsize=14)
    a1.set_xlabel("z [cm]")
fig.colorbar(im0, ax=axes[0, :], label=r"$\langle\Delta B_z\rangle_\phi$ [mT]", fraction=0.02)
fig.colorbar(im1, ax=axes[1, :], label=r"$\langle\sigma(B_z)\rangle_\phi$ [mT]", fraction=0.02)
plot_tools.save_pdf_and_png(outdir, "bz_delta_and_sigma_channels")
plt.close(fig)
output_tools.write_index_and_log(outdir, "bz_channels", args=args)
print(f"\nplots in {outdir}")

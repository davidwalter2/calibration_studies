"""Delta|B| vs radius (at the probe z) from the combined J/psi+Ks+cosmics
track fit, with its uncertainty band, against the NMR probe measurements --
showing that the track fit extrapolates catastrophically beyond the
track-sampled region (r < 1.15 m) and that the NMR probes anchor the
boundary the tracks cannot reach."""
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
NMR = "/work/submit/david_w/ZMass/MuonMomentumScaleCalibration/Analysis/data/nmr_constraints_2016_custom50.npz"

meta, params = {}, []
for line in open(COEF):
    if line.startswith("#"):
        p = line[1:].split()
        if len(p) == 2:
            meta[p[0]] = p[1]
        continue
    _, l, m, cs, _ = line.split(); params.append((int(l), int(m), cs))
r_scale = float(meta["r_scale"]); z0 = float(meta["z0"]); cmssw_norm = bool(int(meta["cmssw_norm"]))

ch = {t: np.load(f"{W}/{t}.npz", allow_pickle=True) for t in ["Jpsi", "Ks", "cosmics"]}
pt = ch["Jpsi"]["parmtype"]; pri = ch["Jpsi"]["priors"]; m14 = np.where(pt == 14)[0]
H = sum(ch[t]["hess"] for t in ["Jpsi", "Ks", "cosmics"]) + np.diag(2 * pri)
g = sum(ch[t]["grad"] for t in ["Jpsi", "Ks", "cosmics"])
Hinv = np.linalg.inv(H); dc = (-Hinv @ g)[m14]; covf = (2 * Hinv)[np.ix_(m14, m14)]

n = np.load(NMR); res, sig = n["res"] if "res" in n.files else n["resid"], n["sigma"]
# probe layout: A/E midplane r=291.5 z~0 ; C/D r=65.1 z=+-283
probes = {"A": (291.5, -0.6), "E": (291.5, 0.6), "C": (65.1, -283.5), "D": (65.1, 283.1)}
pres = {"A": res[0], "E": res[1], "C": res[2], "D": res[3]}
psig = {"A": sig[0], "E": sig[1], "C": sig[2], "D": sig[3]}


def dBz_along_r(rcm, zcm, phi=0.0):
    M = np.array([harmonic_basis.bz_basis(l, m, cs, rcm, np.full_like(rcm, phi),
                                          np.full_like(rcm, zcm), r_scale=r_scale, z0=z0,
                                          cmssw_norm=cmssw_norm) for (l, m, cs) in params]).T
    val = M @ dc
    var = np.einsum("pi,ij,pj->p", M, covf, M)
    return val * 1e3, np.sqrt(np.clip(var, 0, None)) * 1e3


outdir = output_tools.make_plot_dir(f"/home/submit/david_w/public_html/ZMass/MagneticField/{datetime.date.today().strftime('%y%m%d')}_nmr_boundary")
fig, ax = plt.subplots(figsize=(13, 6.2))
rr = np.linspace(1, 300, 300)
# midplane profile (relevant for A/E)
val, err = dBz_along_r(rr, 0.0)
ax.axvspan(0, 115, color="#c8e6c9", alpha=0.5, label="tracker (tracks live here)")
ax.plot(rr, val, color="#A31F34", lw=2, label="track fit $\\Delta B_z$ (J/$\\psi$+Ks+cosmics), z=0")
ax.fill_between(rr, val - err, val + err, color="#A31F34", alpha=0.25)
for lbl in ["A", "E"]:
    ax.errorbar([probes[lbl][0]], [pres[lbl] * 1e3], yerr=[psig[lbl] * 1e3], fmt="o",
                ms=10, color="#222222", capsize=6, label="NMR measured (A,E)" if lbl == "A" else None)
    ax.annotate(lbl, (probes[lbl][0] + 4, pres[lbl] * 1e3), fontsize=15)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("r [cm]"); ax.set_ylabel(r"$\Delta B_z$ [mT]")
ax.set_ylim(-300, 60)
ax.legend(fontsize=15, loc="lower left")
ax.text(120, 25, "track fit diverges beyond\nthe sampled region", fontsize=14, color="#A31F34")
plot_tools.add_decor(ax, "CMS", "Work in progress", data=True, lumi=None, no_energy=True)
plot_tools.save_pdf_and_png(outdir, "dBz_vs_r_boundary")
plt.close(fig)
output_tools.write_index_and_log(outdir, "nmr_boundary")
print("A/E track prediction vs NMR: A", round(dBz_along_r(np.array([291.5]), 0.)[0][0], 1),
      "vs", round(pres["A"] * 1e3, 1), "mT")
print(f"plots in {outdir}")

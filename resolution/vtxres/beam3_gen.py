#!/usr/bin/env python3
"""THE SIMULATED LUMINOUS REGION -- the closure reference for `--beam3`.

The beam parameters the fit returns are corrections to the `offlineBeamSpot`
RECORD.  What they must reproduce is the luminous region the MC was actually
smeared with, and on this sample that is known in CLOSED FORM, not only
measurable: `BetafuncEvtVtxGenerator` with
`Realistic25ns13TeV2016CollisionVtxSmearingParameters`,

    X = Gauss(0, sigma(Z)/sqrt(2)) + X0        <- the `+ Z*fdxdz` term is
    Y = Gauss(0, sigma(Z)/sqrt(2)) + Y0           COMMENTED OUT in the source
    Z = Gauss(0, SigmaZ) + Z0
    sigma(z) = sqrt(emittance (betastar + (z - Z0)^2/betastar))

with `Phi = Alpha = 0` (they are the Lorentz boost of the event, not a
rotation of the vertex distribution).  So

  * dxdz = dydz = 0 EXACTLY -- the simulation has NO tilt;
  * rho_xy = 0 EXACTLY -- X and Y are independent draws;
  * sigma_x = sigma_y identically, z-dependent, with marginal rms
    sqrt(emittance (betastar + SigmaZ^2/betastar) / 2) = 9.947 um;
  * X0, Y0, Z0 are the generator's, to the digit.

The record, by contrast, has sigma_x != sigma_y, non-zero tilts, and no rho at
all.  Every one of those differences is a NUMBER THE FIT MUST RETURN, which is
what makes this a closure and not a consistency check.

This script measures the same quantities FROM THE GEN VERTICES of the very
candidates the fit uses -- so that the reference carries the sample's own
statistical fluctuation and its selection -- and prints both.

THE ESTIMATORS ARE ROBUST AND BIAS-CORRECTED.  The gen production vertex has
heavy tails (rms 32 / 91 um against a MAD of 10 um: ~0.05 % of candidates
carry a genuinely displaced gen vertex, a leg matched to a heavy-flavour
decay), so a plain rms is meaningless.  A TRIMMED standard deviation is biased
LOW and the bias is computed exactly here rather than ignored, and the tilt is
fitted by iteratively-reweighted least squares rather than by `polyfit`, whose
answer on this sample is dominated by the same tail.  Errors are BOOTSTRAP.

usage:
  python3 beam3_gen.py --npz <dy_bsx.npz> [--nboot 200]
  python3 beam3_gen.py --files '<prod>/task_*/globalcor_*.root'
"""
import argparse
import glob
import sys

import numpy as np
from scipy.stats import norm as _norm

# Realistic25ns13TeV2016CollisionVtxSmearingParameters (CMSSW
# IOMC/EventVertexGenerators/python/VtxSmearedParameters_cfi.py)
GEN = dict(X0=0.09163, Y0=0.16955, Z0=0.9315, SigmaZ=3.65,
           BetaStar=40.0, Emittance=4.906e-8, Phi=0.0, Alpha=0.0)


def gen_sigma_t():
    """The marginal transverse width of the generator, cm.

    `sigma^2(z) = emittance (betastar + (z-Z0)^2/betastar) / 2` and z is
    Gaussian with width SigmaZ, so `<sigma^2> = emittance (betastar +
    SigmaZ^2/betastar)/2`.
    """
    e, b, sz = GEN["Emittance"], GEN["BetaStar"], GEN["SigmaZ"]
    return np.sqrt(e * (b + sz * sz / b) / 2.0)


def trim_bias(frac):
    """E[s_trimmed]/sigma for a Gaussian trimmed at the `frac/2` tails.

    Trimming at the quantiles `+-q` keeps a truncated Gaussian whose variance
    is `1 - 2 q phi(q)/(1-frac)`; the retained standard deviation is that,
    and dividing by it is what turns a trimmed spread into an estimate of
    sigma.  (For frac = 0.01 the factor is 0.96164.)
    """
    q = _norm.ppf(1.0 - frac / 2.0)
    return np.sqrt(1.0 - 2.0 * q * _norm.pdf(q) / (1.0 - frac))


def rob_width(v, frac=0.01):
    lo, hi = np.quantile(v, [frac / 2, 1 - frac / 2])
    m = (v >= lo) & (v <= hi)
    return v[m].std() / trim_bias(frac)


def rob_centre(v, frac=0.01):
    lo, hi = np.quantile(v, [frac / 2, 1 - frac / 2])
    m = (v >= lo) & (v <= hi)
    return v[m].mean()


def irls_slope(z, v, niter=8, c=3.0):
    """Slope of `v` against `z` by iteratively-reweighted least squares.

    Plain least squares on this sample is a measurement of the 0.05 % of
    candidates with a displaced gen vertex, not of the beam tilt: the raw rms
    of `x` is 3x its MAD and 90x for `y`.  Each iteration re-fits with the
    outliers beyond `c` robust sigmas dropped, which converges in two or three
    passes.
    """
    w = np.ones(len(z), bool)
    s = b = 0.0
    for _ in range(niter):
        s, b = np.polyfit(z[w], v[w], 1)
        r = v - (s * z + b)
        sd = 1.4826 * np.median(np.abs(r - np.median(r)))
        w2 = np.abs(r - np.median(r)) < c * sd
        if w2.sum() < 10 or np.array_equal(w2, w):
            w = w2
            break
        w = w2
    return s, b, w


def rob_corr(x, y, c=3.0):
    """Correlation after ONE robust mask that keeps the pair, not each
    coordinate separately -- masking them in turn would pair different rows."""
    keep = np.ones(len(x), bool)
    for v in (x, y):
        sd = 1.4826 * np.median(np.abs(v - np.median(v)))
        keep &= np.abs(v - np.median(v)) < c * sd
    return float(np.corrcoef(x[keep], y[keep])[0, 1]), int(keep.sum())


def measure(gx, gy, gz):
    sx_, bx, _ = irls_slope(gz, gx)
    sy_, by, _ = irls_slope(gz, gy)
    rx = gx - sx_ * gz
    ry = gy - sy_ * gz
    rho, _n = rob_corr(rx - np.median(rx), ry - np.median(ry))
    return dict(
        x0=rob_centre(gx), y0=rob_centre(gy), z0=rob_centre(gz),
        dxdz=sx_, dydz=sy_,
        sigx=rob_width(rx), sigy=rob_width(ry), sigz=rob_width(gz),
        rho=rho,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None,
                    help="an extract_vtx.py npz (it carries genvtx_* and the "
                         "record, and it carries the SELECTED candidates)")
    ap.add_argument("--files", default=None, help="globalcor_*.root glob")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args()

    if a.npz:
        d = np.load(a.npz, allow_pickle=False)
        gx, gy, gz = (np.asarray(d[f"genvtx_{k}"], np.float64)
                      for k in "xyz")
        rec = dict(spot=np.asarray(d["bsspot"], np.float64),
                   width=np.asarray(d["bswidth"], np.float64),
                   slope=np.asarray(d["bsslope"], np.float64))
    elif a.files:
        import uproot
        gx, gy, gz, sp, wd, sl = [], [], [], [], [], []
        for p in sorted(glob.glob(a.files)):
            try:
                t = uproot.open(p)["tree"]
            except Exception as e:
                print(f"  [skip] {p}: {e}")
                continue
            arr = t.arrays(["Jpsigen_x", "Jpsigen_y", "Jpsigen_z",
                            "Jpsi_bsspot", "Jpsi_bswidth", "Jpsi_bsslope"],
                           library="np")
            gx.append(arr["Jpsigen_x"])
            gy.append(arr["Jpsigen_y"])
            gz.append(arr["Jpsigen_z"])
            sp.append(np.asarray(arr["Jpsi_bsspot"]).reshape(-1, 3))
            wd.append(np.asarray(arr["Jpsi_bswidth"]).reshape(-1, 3))
            sl.append(np.asarray(arr["Jpsi_bsslope"]).reshape(-1, 2))
        gx, gy, gz = (np.concatenate(v) for v in (gx, gy, gz))
        rec = dict(spot=np.concatenate(sp), width=np.concatenate(wd),
                   slope=np.concatenate(sl))
    else:
        sys.exit("give --npz or --files")

    # -99 is the NO-GEN-MATCH sentinel, written on all three at once
    ok = (np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gz)
          & (np.abs(gx) < 5.0) & (np.abs(gy) < 5.0) & (np.abs(gz) < 60.0))
    print(f"{ok.sum()} gen-matched of {ok.size} ({100*ok.mean():.2f} %)")
    gx, gy, gz = gx[ok], gy[ok], gz[ok]
    R = {k: np.median(v, axis=0) for k, v in rec.items()}
    for k, v in rec.items():
        u = np.unique(np.round(v[ok], 12), axis=0)
        if len(u) > 1:
            print(f"  NOTE: the record's `{k}` is not unique "
                  f"({len(u)} rows); the median is used")

    m = measure(gx, gy, gz)
    rng = np.random.default_rng(a.seed)
    boot = {k: [] for k in m}
    n = len(gx)
    for _ in range(a.nboot):
        i = rng.integers(0, n, n)
        b = measure(gx[i], gy[i], gz[i])
        for k in b:
            boot[k].append(b[k])
    err = {k: float(np.std(v, ddof=1)) for k, v in boot.items()}

    st = gen_sigma_t()
    print("\nTHE GENERATOR (Realistic25ns13TeV2016Collision, closed form)")
    print(f"  X0 {GEN['X0']:.6f} cm   Y0 {GEN['Y0']:.6f} cm   "
          f"Z0 {GEN['Z0']:.6f} cm")
    print(f"  sigma_t (marginal) {st*1e4:.4f} um   SigmaZ {GEN['SigmaZ']:.4f} cm")
    print(f"  dxdz 0 EXACTLY   dydz 0 EXACTLY   rho_xy 0 EXACTLY "
          f"(Phi = Alpha = 0, and the tilt term is commented out)")

    print(f"\nMEASURED FROM {n} GEN VERTICES (robust, bootstrap x{a.nboot})")
    print(f"  x0    {m['x0']*1e4:+10.3f} +- {err['x0']*1e4:.3f} um")
    print(f"  y0    {m['y0']*1e4:+10.3f} +- {err['y0']*1e4:.3f} um")
    print(f"  z0    {m['z0']:+10.5f} +- {err['z0']:.5f} cm")
    print(f"  sigx  {m['sigx']*1e4:10.4f} +- {err['sigx']*1e4:.4f} um")
    print(f"  sigy  {m['sigy']*1e4:10.4f} +- {err['sigy']*1e4:.4f} um")
    print(f"  sigz  {m['sigz']:10.5f} +- {err['sigz']:.5f} cm")
    print(f"  dxdz  {m['dxdz']:+10.3e} +- {err['dxdz']:.1e}")
    print(f"  dydz  {m['dydz']:+10.3e} +- {err['dydz']:.1e}")
    print(f"  rho   {m['rho']:+10.4f} +- {err['rho']:.4f}")

    rx0, ry0, rz0 = R["spot"]
    rsx, rsy, rsz = R["width"]
    rdx, rdy = R["slope"]
    print("\nTHE RECORD THE MAKER USED")
    print(f"  x0 {rx0*1e4:+.3f} um  y0 {ry0*1e4:+.3f} um  z0 {rz0:+.5f} cm")
    print(f"  sigx {rsx*1e4:.4f} um  sigy {rsy*1e4:.4f} um  sigz {rsz:.4f} cm")
    print(f"  dxdz {rdx:+.4e}  dydz {rdy:+.4e}  rho 0 (NOT STORED)")

    print("\nTHE CLOSURE TARGETS -- what the fit must return, in CARD UNITS")
    print(f"  {'parameter':16s} {'target':>12s} {'from gen':>12s} "
          f"{'gen err':>10s}")
    tg = {
        "beamwidth_x": ((st / rsx) ** 2 - 1.0, (m['sigx'] / rsx) ** 2 - 1.0,
                        2 * m['sigx'] * err['sigx'] / rsx ** 2),
        "beamwidth_y": ((st / rsy) ** 2 - 1.0, (m['sigy'] / rsy) ** 2 - 1.0,
                        2 * m['sigy'] * err['sigy'] / rsy ** 2),
        "beamcorr_xy": (0.0, np.arctanh(m['rho']), err['rho']),
        "beamtilt_x": ((0.0 - rdx) / 1e-5, (m['dxdz'] - rdx) / 1e-5,
                       err['dxdz'] / 1e-5),
        "beamtilt_y": ((0.0 - rdy) / 1e-5, (m['dydz'] - rdy) / 1e-5,
                       err['dydz'] / 1e-5),
        "beamcentre_x": ((GEN["X0"] - rx0) / 1e-4, (m['x0'] - rx0) / 1e-4,
                         err['x0'] / 1e-4),
        "beamcentre_y": ((GEN["Y0"] - ry0) / 1e-4, (m['y0'] - ry0) / 1e-4,
                         err['y0'] / 1e-4),
    }
    for k, (t_, g_, e_) in tg.items():
        print(f"  {k:16s} {t_:+12.4f} {g_:+12.4f} {e_:10.4f}")
    print("\n(`target` is the GENERATOR's closed form, `from gen` the same "
          "quantity\n measured on this sample's own gen vertices.  The card "
          "units are:\n eps for the two widths, eta = atanh(rho), 1e-5 for "
          "the tilts, 1 um for\n the centres.)")
    np.savez(a.npz.replace(".npz", "_genref.npz") if a.npz else "genref.npz",
             **{f"meas_{k}": v for k, v in m.items()},
             **{f"err_{k}": v for k, v in err.items()},
             **{f"target_{k}": v[0] for k, v in tg.items()},
             **{f"genmeas_{k}": v[1] for k, v in tg.items()},
             **{f"generr_{k}": v[2] for k, v in tg.items()},
             record=np.concatenate([R["spot"], R["width"], R["slope"]]),
             n=n)


if __name__ == "__main__":
    main()

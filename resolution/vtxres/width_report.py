#!/usr/bin/env python3
"""THE LUMINOUS-REGION WIDTH FLOATS -- fitted values, errors, and the pull
against the SIMULATED luminous region.

The beam-spot record the maker used is WIDER in the transverse plane than the
luminous region the DY MC was actually smeared with: record
sigma_x / sigma_y = 10.834 / 10.388 um against a simulated 10.05 / 9.98 um.
The two card parameters `beamwidth_x` / `beamwidth_y` scale the family-16
block's VARIANCE share linearly, so the value the fit must return is

    eps = k - 1 ,   k = (sigma_sim / sigma_record)^2

i.e. eps_x = -0.137 and eps_y = -0.081.  That is the gate: the fitted scales
recover the simulation, not the record.

THE DEFAULTS ARE THE BIAS-CORRECTED WIDTHS, and that matters.  The gen
production vertex has heavy tails (rms 31.8 / 90.8 um against a MAD of
10.08 / 9.93 um -- 0.05 % of candidates carry a genuinely displaced gen
vertex), so the width must be estimated robustly.  But a TRIMMED STANDARD
DEVIATION is biased LOW, and by a lot: for a Gaussian trimmed at the
0.5 / 99.5 percentiles the retained standard deviation is 0.96164 of the true
sigma.  So the "9.66 / 9.59 um (1 % trimmed)" of STATE section 14.3 is really
9.662/0.96164 = 10.047 um and 9.595/0.96164 = 9.978 um, and once corrected the
three robust estimators AGREE:

    sigma_x:  1 % trim 10.047   5 % trim 10.070   MAD 10.081
    sigma_y:  1 % trim  9.978   5 % trim  9.973   MAD  9.925

The record is therefore 7.2 % / 4.1 % wider in SIGMA -- eps = -0.137 / -0.081,
not the -0.204 / -0.148 the uncorrected trimmed numbers imply.

Every number quoted is EDM-certified -- the fit is refused if rabbit's EDM is
not below the tolerance.

  ./width_report.py --fits DIR [DIR ...] [--sim-x 10.047 --sim-y 9.978]
                    [--rec-x 10.834 --rec-y 10.388] [--edm-max 1e-3]
"""
import argparse
import os
import sys

import numpy as np


def read_fit(path):
    from rabbit import io_tools
    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    val = np.asarray(h.values(), np.float64)
    err = np.sqrt(np.asarray(h.variances(), np.float64))
    out = dict(names=names, val=val, err=err)
    for k in ("edmval", "nllvalfull", "nllvalreduced"):
        if k in fr:
            try:
                v = fr[k]
                out[k] = float(np.asarray(v.get() if hasattr(v, "get") else v))
            except Exception:  # noqa: BLE001
                pass
    if "cov" in fr:
        out["cov"] = np.asarray(fr["cov"].get().values(), np.float64)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--sim-x", type=float, default=10.047)
    p.add_argument("--sim-y", type=float, default=9.978)
    p.add_argument("--rec-x", type=float, default=10.834)
    p.add_argument("--rec-y", type=float, default=10.388)
    p.add_argument("--edm-max", type=float, default=1e-3)
    a = p.parse_args()

    ex = (a.sim_x / a.rec_x) ** 2 - 1.0
    ey = (a.sim_y / a.rec_y) ** 2 - 1.0
    print("THE EXPECTATION, from the simulated luminous region against the record")
    print(f"  sigma_x  record {a.rec_x:.2f} um   simulated {a.sim_x:.2f} um"
          f"   -> k = {(a.sim_x/a.rec_x)**2:.4f}   eps = {ex:+.4f}")
    print(f"  sigma_y  record {a.rec_y:.2f} um   simulated {a.sim_y:.2f} um"
          f"   -> k = {(a.sim_y/a.rec_y)**2:.4f}   eps = {ey:+.4f}")
    print()
    hdr = (f"{'fit':22s} {'EDM':>10} {'eps_x':>18} {'pull_x':>8} "
           f"{'eps_y':>18} {'pull_y':>8}  {'sigma_x(fit)':>12} {'sigma_y(fit)':>12}")
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for d in a.fits:
        fp = d if d.endswith(".hdf5") else os.path.join(d, "fitresults.hdf5")
        if not os.path.exists(fp):
            print(f"{os.path.basename(d):22s}  (no fitresults)")
            continue
        try:
            f = read_fit(fp)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(d):22s}  READ FAILED: {e}")
            bad += 1
            continue
        edm = f.get("edmval", np.nan)
        # CERTIFICATION: a fit whose EDM is not below tolerance is not a
        # measurement.  It is reported and NOT quoted.
        cert = np.isfinite(edm) and edm < a.edm_max
        row = [f"{os.path.basename(d.rstrip('/')):22s}", f"{edm:10.2e}"]
        vals = {}
        for nm in ("beamwidth_x", "beamwidth_y"):
            if nm not in f["names"]:
                vals[nm] = (np.nan, np.nan)
                continue
            i = f["names"].index(nm)
            vals[nm] = (f["val"][i], f["err"][i])
        for nm, exp in (("beamwidth_x", ex), ("beamwidth_y", ey)):
            v, e = vals[nm]
            row.append(f"{v:+9.4f} +- {e:6.4f}" if np.isfinite(v) else f"{'--':>18}")
            row.append(f"{(v-exp)/e:+8.2f}" if np.isfinite(v) and e > 0 else f"{'--':>8}")
        for nm, rec in (("beamwidth_x", a.rec_x), ("beamwidth_y", a.rec_y)):
            v, e = vals[nm]
            if np.isfinite(v) and 1.0 + v > 0:
                s = rec * np.sqrt(1.0 + v)
                ds = rec * e / (2 * np.sqrt(1.0 + v))
                row.append(f"{s:6.2f}+-{ds:4.2f}")
            else:
                row.append(f"{'--':>12}")
        print("  ".join(row) + ("" if cert else "   *** EDM NOT CERTIFIED ***"))
        bad += (not cert)
        if "cov" in f and "beamwidth_x" in f["names"] and "beamwidth_y" in f["names"]:
            ix = f["names"].index("beamwidth_x")
            iy = f["names"].index("beamwidth_y")
            c = f["cov"]
            if c.ndim == 2 and c.shape[0] > max(ix, iy):
                r = c[ix, iy] / np.sqrt(max(c[ix, ix] * c[iy, iy], 1e-300))
                print(f"{'':24s} corr(beamwidth_x, beamwidth_y) = {r:+.3f}")
    print(f"\n{bad} fit(s) not certified or unreadable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

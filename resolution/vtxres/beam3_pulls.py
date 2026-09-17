#!/usr/bin/env python3
"""WHAT FLOATING THE LUMINOUS REGION DOES TO THE TWO BEAM PULLS.

The maker builds the whitened pair `z = L^-1 r_bs` against the beam-spot
RECORD: its centre, its two tilts, its two widths and `rho = 0`.  If the
record is not the luminous region the events were drawn from, those pulls are
neither centred nor unit-variance, and both defects are PREDICTED by the
fitted beam parameters:

    the MEAN       z -> z + sum_k (d z/d p_k) p_k        (`Jpsi_bsmean*`, and
                                                          the tilt's lever arm
                                                          `z_v - z0`)
    the VARIANCE   1 -> 1 + Delta v_bs(p) + sum_c eps_c v_c + Delta v_material

so the question "do the pulls become unit-variance once rho and the tilts
float" has a number on both sides: the measured variance of the corrected
pull, and the model's own prediction for it.  They are reported together;
agreement is the closure, not the corrected variance being 1 by itself (the
fit's own vertex covariance is ~18 % low, which is what the hit-class scales
are for and what no beam parameter can fix).

usage:
  ./run_tf.sh python3 beam3_pulls.py --npz-dir DIR --fit DIR [--base DIR]
      --groups <materialGroups50.txt> [--maxn 8000]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_vtx_card as MC  # noqa: E402
import selection  # noqa: E402


def read_fit(path):
    from rabbit import io_tools
    fp = path if path.endswith(".hdf5") else os.path.join(path, "fitresults.hdf5")
    fr = io_tools.get_fitresult(fp)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    val = np.asarray(h.values(), np.float64)
    err = np.sqrt(np.asarray(h.variances(), np.float64))
    edm = np.nan
    if "edmval" in fr:
        v = fr["edmval"]
        edm = float(np.asarray(v.get() if hasattr(v, "get") else v))
    return dict(zip(names, val)), dict(zip(names, err)), edm


def robust_var(z, frac=0.01):
    """Trimmed variance, bias-corrected for a Gaussian (see beam3_gen.py)."""
    from beam3_gen import trim_bias
    lo, hi = np.quantile(z, [frac / 2, 1 - frac / 2])
    m = (z >= lo) & (z <= hi)
    return float(z[m].var() / trim_bias(frac) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--prefix", default="dy")
    ap.add_argument("--fit", required=True, help="the --beam3 fit")
    ap.add_argument("--base", default=None, help="the widths-only fit")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--maxn", type=int, default=8000)
    ap.add_argument("--max-abs-z", type=float, default=40.0)
    ap.add_argument("--edm-max", type=float, default=1e-3)
    selection.add_args(ap)
    a = ap.parse_args()
    a.max_chi2_ndof = getattr(a, "max_chi2_ndof", 3.0)

    # the SAME candidate set the card used: the intersection over the four
    # channels, truncated to --maxn, exactly as `make_vtx_card.common_index`
    a.vtx_npz = os.path.join(a.npz_dir, f"{a.prefix}_vtx.npz")
    a.bsx_npz = os.path.join(a.npz_dir, f"{a.prefix}_bsx.npz")
    a.bsy_npz = os.path.join(a.npz_dir, f"{a.prefix}_bsy.npz")
    a.mass_npz = os.path.join(a.npz_dir, f"{a.prefix}_mass.npz")
    a.keep_mask = None
    a.m_ref = 91.1876
    MC.MREF[0], MC.MREF[1] = 91.1876, 30.0
    chans = [(nm, p) for nm, p in MC.CHANNELS(a) if p and os.path.exists(p)]
    idx = MC.common_index(chans, a)
    print(f"{len(idx)} candidates (the card's set)")

    pars, errs, edm = read_fit(a.fit)
    print(f"beam3 fit EDM {edm:.2e} "
          f"{'(certified)' if edm < a.edm_max else '*** NOT CERTIFIED ***'}")
    if not (edm < a.edm_max):
        sys.exit("refusing to quote an uncertified fit")

    import groups as G
    gmap, _ = G.read_groups(a.groups)
    ngroups = (max(gmap) + 1) if gmap else 0
    gparams, _gp = G.group_param_names(ngroups, a.groups)
    gunits = G.card_group_units(ngroups, a.groups, whiten=True)
    import hitres_classes
    hparams = [f"hitres_{c}" for c in hitres_classes.CLASSES]

    # BOTH estimators are printed.  The model's variance is the FULL second
    # moment of a non-Gaussian CF density, so the raw sample variance is its
    # comparator; the 1 %-trimmed one (bias-corrected for a Gaussian) is what
    # the tail-insensitive statement needs, and the two differ by ~10 % here
    # because the beam residual has a 5 sigma tail (section 14.19 item 1).
    print(f"\n{'channel':8s} {'N':>6s} {'Var raw':>9s} {'Var raw+m':>10s} "
          f"{'Var trim':>9s} {'Vtrim+m':>9s} {'model Var':>10s} "
          f"{'raw/model':>9s} {'mean(z)':>10s} {'mean(z+m)':>10s}")
    for nm in ("bsx", "bsy"):
        d = np.load(os.path.join(a.npz_dir, f"{a.prefix}_{nm}.npz"),
                    allow_pickle=False)
        z = np.asarray(d["m0"], np.float64)[idx]
        bm = np.asarray(d["bsmean"], np.float64)[idx]
        vtx = np.asarray(d["bsvtx"], np.float64)[idx]
        spot = np.asarray(d["bsspot"], np.float64)[idx]
        lever = vtx[:, 2] - spot[:, 2]
        # the MEAN response, with the card's own units
        dz = (bm[:, 0] * MC.BEAM3_UNITS["beamcentre_x"] * pars.get("beamcentre_x", 0.0)
              + bm[:, 1] * MC.BEAM3_UNITS["beamcentre_y"] * pars.get("beamcentre_y", 0.0)
              + bm[:, 0] * lever * MC.BEAM3_UNITS["beamtilt_x"]
              * pars.get("beamtilt_x", 0.0)
              + bm[:, 1] * lever * MC.BEAM3_UNITS["beamtilt_y"]
              * pars.get("beamtilt_y", 0.0))
        zc = z + dz

        # the model's VARIANCE at the fitted point, in units of the exported
        # sigma^2 (which is 1 for the whitened pulls, by construction)
        sigma = np.asarray(d["sigma"], np.float64)[idx]
        q, ref, v0, _dv, _mean, _dm = MC.beam3_block(
            d, idx, sigma, {}, log=lambda *_: None)
        cov = MC.beam3_cov(
            ref[:, 0] * np.sqrt(1.0 + pars.get("beamwidth_x", 0.0)),
            ref[:, 1] * np.sqrt(1.0 + pars.get("beamwidth_y", 0.0)),
            ref[:, 2],
            ref[:, 3] + pars.get("beamtilt_x", 0.0) * MC.BEAM3_UNITS["beamtilt_x"],
            ref[:, 4] + pars.get("beamtilt_y", 0.0) * MC.BEAM3_UNITS["beamtilt_y"],
            rho=np.tanh(pars.get("beamcorr_xy", 0.0)))
        cov0 = MC.beam3_cov(*[ref[:, i] for i in range(5)])
        dvbs = ((cov - cov0) * q * MC.BEAM3_MULT).sum(-1)

        # the hit classes
        hptr = np.asarray(d["hit_ptr"], np.int64)
        rows = np.concatenate([np.arange(hptr[i], hptr[i + 1]) for i in idx])
        hcls = np.asarray(d["hit_cls"], np.int64)[rows]
        hv = np.asarray(d["hit_v"], np.float64)[rows]
        hseg = np.repeat(np.arange(len(idx)), np.diff(hptr)[idx])
        eps = np.array([pars.get(p, 0.0) for p in hparams])
        dvhit = np.zeros(len(idx))
        np.add.at(dvhit, hseg, eps[hcls] * hv)

        # the material, through the fit's own per-group Q variance
        dvmat = np.zeros(len(idx))
        if "Gvqms" in d.files:
            gptr = np.asarray(d["grp_ptr"], np.int64)
            grows = np.concatenate([np.arange(gptr[i], gptr[i + 1]) for i in idx])
            gid = np.asarray(d["grp_id"], np.int64)[grows]
            gq = (np.asarray(d["Gvqms"], np.float64)[grows]
                  + np.asarray(d["Gvqio"], np.float64)[grows])
            k = np.array([pars.get(p, 0.0) for p in gparams]) * gunits
            gseg = np.repeat(np.arange(len(idx)), np.diff(gptr)[idx])
            np.add.at(dvmat, gseg, (np.exp(k[gid]) - 1.0) * gq)

        vmod = 1.0 + dvbs + dvhit + dvmat
        n = len(idx)
        print(f"{nm:8s} {n:6d} {z.var():9.4f} {zc.var():10.4f} "
              f"{robust_var(z):9.4f} {robust_var(zc):9.4f} "
              f"{np.mean(vmod):10.4f} {z.var()/np.mean(vmod):9.4f} "
              f"{z.mean():+10.4f} {zc.mean():+10.4f}")
        print(f"{'':8s} shares: d v_bs {np.mean(dvbs):+.5f}, "
              f"d v_hit {np.mean(dvhit):+.5f}, d v_mat {np.mean(dvmat):+.5f}; "
              f"mean shift rms {dz.std():.4f}")


if __name__ == "__main__":
    main()

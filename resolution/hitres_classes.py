#!/usr/bin/env python3
"""Hit CLASSES and their measured characteristic functions.

One definition, shared by hitres_cf_conditioning.py, hitres_clt_forecast.py
and cf_track_resolution.py, so the class boundaries cannot drift between the
identifiability study, the forecast and the CF itself.

The splits are the ones the pull study found to move (NOTES_HITRES sections 3
and 10): strips on the cluster width N and on uProj -- the CPE's own
independent variable, across which the core runs +12 % to -13 % -- and pixels
on the template charge bin, across which the core runs 0.52 to 1.17.

`build_cf_bank` returns, per class, an interpolator for log phi_c(s), the log
characteristic function of the RAW per-hit pull (rec - sim)/sigma_CPE. Raw,
not standardised, so that phi_c carries BOTH the variance ratio theta_c and
the shape -- which is what the CF consumes:

    log phi_hit(t) = sum_b log phi_c(b)( t sqrt(v_b) / sigma )

with v_b = w_b^T V_b w_b the block's exported variance contribution.
"""
import glob
import os

import numpy as np

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
# argument grid for log phi_c(s). s = t sqrt(v_b)/sigma; t runs to 14 on the
# CF's TG grid and sqrt(v_b)/sigma is at most ~0.4, so 8 is generous.
SGRID = np.linspace(0.0, 8.0, 1601)


# CANONICAL ordering. The index must not depend on the order classes happen
# to be encountered, because extract_parallel.py shards the extraction and
# concatenates the per-shard arrays: a shard-local index would silently
# relabel every hit of every shard but the first.
CLASSES = ([f"pix_{a}_q{q}" for a in ("x", "y") for q in range(4)]
           + [f"str_N{n}_{u}" for n in range(1, 6) for u in ("lo", "hi")])
_CLSIDX = {c: i for i, c in enumerate(CLASSES)}


def class_index(name):
    return _CLSIDX[name]


def class_of(subdet, N, uproj, qbin, isy):
    """Label for one hit. subdet is DetId::subdetId() (1,2 = pixel)."""
    if subdet <= 2:
        return f"pix_{'y' if isy else 'x'}_q{min(max(int(qbin), 0), 3)}"
    n = min(max(int(N), 1), 5)
    return f"str_N{n}_{'lo' if uproj < 0.25 else 'hi'}"


def build_cf_bank(subdir="hitres3", tag="mugun_lowpt", nfiles=0,
                  trim_core=10.0, minhits=500, logger=None):
    """Measured log phi_c(s) per class, from a gen-anchored production.

    trim_core removes |pull| > trim_core * core. That population is a
    WRONG-HIT rate (NOTES_HITRES section 6: 0.0005 % for muons rising to
    0.12 % for protons), not a resolution: it is leverage on the fit rather
    than a width, it is quantified separately, and leaving it in would put a
    component in the CF that the closure has no power to test.
    """
    import uproot
    br = ["dxrecsim", "dxerr", "dyrecsim", "dyerr", "hitDetId", "hitUProj",
          "clusterSizeX", "clusterChargeBin"]
    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/{subdir}_{tag}/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if nfiles:
        fs = fs[:nfiles]
    if not fs:
        raise SystemExit(f"no complete files in {subdir}_{tag}")
    cols = {b: [] for b in br}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(br, library="np")
        for b in br:
            cols[b].append(np.concatenate(a[b]))
    d = {b: np.concatenate(v) for b, v in cols.items()}
    sd = (d["hitDetId"].astype(np.uint32) >> 25) & 0x7

    pools = {}
    for isy, (rk, ek) in enumerate((("dxrecsim", "dxerr"), ("dyrecsim", "dyerr"))):
        m = (d[rk] > -98) & (d[ek] > 0)
        if isy:
            m &= sd <= 2                       # local y is a pixel-only block
        idx = np.where(m)[0]
        pull = d[rk][idx] / d[ek][idx]
        for k, i in enumerate(idx):
            c = class_of(sd[i], d["clusterSizeX"][i], d["hitUProj"][i],
                         d["clusterChargeBin"][i], bool(isy))
            pools.setdefault(c, []).append(pull[k])

    bank, meta = {}, {}
    for c, v in sorted(pools.items()):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size < minhits:
            continue
        med = np.median(v)
        q16, q84 = np.percentile(v, [15.865, 84.135])
        core = 0.5 * (q84 - q16)
        keep = np.abs(v - med) <= trim_core * core
        ntrim = int((~keep).sum())
        x = v[keep] - med                      # centred: the CF's mean is a bias,
        # and a per-hit bias belongs to alignment, not to the resolution model
        # phi_c(s) = E[exp(i s x)] on SGRID
        ph = np.exp(1j * SGRID[:, None] * x[None, :]).mean(axis=1)
        bank[c] = np.log(np.where(np.abs(ph) < 1e-12, 1e-12 + 0j, ph))
        meta[c] = dict(n=int(x.size), var=float(x.var()), core=float(core),
                       trimmed=ntrim / max(v.size, 1))
        if logger is not None:
            logger.info(f"  {c:<14} n={x.size:7d}  var {x.var():7.4f}  "
                        f"core {core:6.4f}  trimmed {100*ntrim/max(v.size,1):.4f}%")
    return bank, meta


def logphi(bank_c, s):
    """log phi_c at arbitrary s >= 0, by linear interpolation on SGRID.

    Beyond the grid the CF has long decayed; clamping to the last value is
    harmless there and keeps the exponent finite.
    """
    s = np.clip(np.abs(s), SGRID[0], SGRID[-1])
    re = np.interp(s, SGRID, bank_c.real)
    im = np.interp(s, SGRID, bank_c.imag)
    return re + 1j * im

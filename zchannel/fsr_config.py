#!/usr/bin/env python3
r"""The two FSR kernel configurations of the Z channel.

``mc``
    the kernel of the **generator sample**, at unlimited statistics: the
    Photos++ 3.61 configuration the UL16 DY sample was produced with, run
    standalone (`photos_standalone/`) over the sample's own ``m_pre``
    distribution.  Use it to close the likelihood against that MC.  It is
    *not* the best description of nature: it carries whatever Photos leaves
    out of its own configuration.

``data``
    the best available QED: the analytic radiator of `fsr_analytic.py` -
    exponentiated exact O(alpha) through the O(alpha^2) leading log and its
    NLL term (`DATA_VARIANT`) - with real lepton- and hadron-pair emission
    added for every species the muon line can produce (`DATA_PAIRS`).  First-principles throughout, no
    fitted constant, and its dependence on the fitted mass is exact.

Both are written as ``(r, w, m_lo, m_hi)`` atom files for the
``rabbit.lineshapes.zgamma`` provider, banded in ``m_pre``.

    python3 fsr_config.py --config mc   -o data/kern_cfg_mc.npz
    python3 fsr_config.py --config data -o data/kern_cfg_data.npz
"""
import argparse
import json
import os

import numpy as np

import fsr_analytic as FA

#: `fsr_analytic` settings of the ``data`` configuration.  One constant, so
#: that adding a variant to `fsr_analytic` is a one-line change here.
DATA_VARIANT = "exp2nll"
#: every species whose pair emission the muon line can produce.  The photonic
#: radiator is 1.43 % (e) / 2.26 % (all) larger with these included.
DATA_PAIRS = ("e", "mu", "tau", "had")

#: the standalone Photos run that reproduces the sample.  The UL16 DY production
#: ran Photos++ 3.61 with exponentiation, ``XPHCUT = 1e-7``, the exact Z
#: matrix-element correction ON and real lepton-pair emission ON, but that
#: correction only *fires* on the 38.9 % of events whose Z has two opposite-sign
#: fermion mothers; the rest are gluon-initiated.  The run is therefore the
#: measured mixture of the two, built by ``photos_standalone/mix.py`` from
#:   run.sh mcB  12 2200000 --me=1 --pairs=1 --fint=8      (correction fires)
#:   run.sh pair 12 2200000 --me=0 --pairs=1 --fint=8      (it does not)
#: with the per-band weight of ``data/photos/f_me.json``, measured in the
#: sample's own gen record.  ``--run data/photos/gen_mcB.npz`` gives the
#: unmixed variant; it moves ``m_Z`` by 0.34 MeV.
MC_RUN = "data/photos/gen_mcMix.npz"

#: discretisation.  ``mc`` uses the ``sigma_cap`` of the empirical kernels of
#: `fit_gen.py kernel`, ``data`` the ``var_budget`` of `fsr_analytic.py kernel`,
#: so each is like for like with its own established control.
SIGMA_CAP = 3.3e-4
VAR_BUDGET = 6e-10
BAND_WIDTH = 2.0


# --------------------------------------------------------------------------
# the MC configuration: atoms from the standalone Photos histograms
# --------------------------------------------------------------------------
def atoms_from_hist(s0, s1, s2, w_norad, u_fine, sigma_cap=SIGMA_CAP,
                    var_budget=None, tail=None, u_tail0=2.0, u_tail_w=0.05):
    """`fit_gen.build_kernel`'s merge, run on pre-binned ``(s0, s1, s2)``.

    ``w_norad`` is the weight of the events Photos left untouched; they become
    a single atom at ``r = 1``.  In the standalone their ``u`` is zero to
    machine precision, so no float-noise floor is needed (the sample's kernel
    uses ``u <= 1e-5`` for that, worth < 0.05 MeV on the fold).

    ``tail`` is the optional coarse ``(t0, t1, t2)`` block above ``u_tail0``.
    """
    s0 = np.asarray(s0, float).copy()
    s1 = np.asarray(s1, float).copy()
    s2 = np.asarray(s2, float).copy()
    s0[0] = max(s0[0] - w_norad, 0.0)          # the r = 1 atom is split off
    if tail is not None:
        s0 = np.concatenate([s0, np.asarray(tail[0], float)])
        s1 = np.concatenate([s1, np.asarray(tail[1], float)])
        s2 = np.concatenate([s2, np.asarray(tail[2], float)])

    rj, wj = [1.0], [float(w_norad)]
    a0 = a1 = a2 = 0.0
    for k in range(len(s0)):
        b0, b1, b2 = a0 + s0[k], a1 + s1[k], a2 + s2[k]
        if b0 <= 0.0:
            continue
        mu = b1 / b0
        var = max(b2 / b0 - mu * mu, 0.0)
        too_wide = (b0 * var > var_budget) if var_budget else (
            np.sqrt(var) > sigma_cap)
        if too_wide and a0 > 0.0:
            rj.append(float(np.exp(-a1 / a0)))
            wj.append(float(a0))
            a0, a1, a2 = s0[k], s1[k], s2[k]
        else:
            a0, a1, a2 = b0, b1, b2
    if a0 > 0.0:
        rj.append(float(np.exp(-a1 / a0)))
        wj.append(float(a0))
    rj, wj = np.array(rj), np.array(wj)
    o = np.argsort(-rj)
    rj, wj = rj[o], wj[o]
    return rj, wj / wj.sum(), wj.sum()


def build_mc(run=MC_RUN, band_width=BAND_WIDTH, sigma_cap=SIGMA_CAP,
             var_budget=None, u_max=None):
    """Banded atoms from a standalone Photos run.

    The generated 2 GeV bands are merged into bands of ``band_width`` with the
    *sample's* band weights, so the within-band mass distribution of the
    kernel is the sample's even though the generation was flat in band index.
    """
    d = np.load(run, allow_pickle=True)
    lo, hi = d["bands_lo"], d["bands_hi"]
    nb = len(lo)
    # generated band width is 2 GeV up to 200 GeV; anything wider is kept as is
    groups = []
    i = 0
    while i < nb:
        if hi[i] - lo[i] > 2.0 + 1e-9 or lo[i] >= 200.0:
            groups.append([i])
            i += 1
            continue
        j = i
        while (j + 1 < nb and hi[j] - lo[i] < band_width - 1e-9
               and hi[j + 1] - lo[j + 1] <= 2.0 + 1e-9 and lo[j + 1] < 200.0):
            j += 1
        groups.append(list(range(i, j + 1)))
        i = j + 1

    R, W, LO, HI, info = [], [], [], [], []
    nfine = d["fine_s0"].shape[1]
    ufine = 2.0 / nfine
    for gi, g in enumerate(groups):
        # weight the generated bands by the sample's own population so a merged
        # band carries the sample's conditional mass distribution
        sw = np.array([d["n"][k] for k in g], float)
        # `n` is the generated count, flat by construction; reweight by the
        # sample fraction instead
        frac = _sample_frac()[g]
        rw = np.where(sw > 0, frac / np.maximum(sw, 1.0), 0.0)
        s0 = sum(rw[a] * d["fine_s0"][k] for a, k in enumerate(g))
        s1 = sum(rw[a] * d["fine_s1"][k] for a, k in enumerate(g))
        s2 = sum(rw[a] * d["fine_s2"][k] for a, k in enumerate(g))
        t0 = sum(rw[a] * d["tail_s0"][k] for a, k in enumerate(g))
        t1 = sum(rw[a] * d["tail_s1"][k] for a, k in enumerate(g))
        t2 = sum(rw[a] * d["tail_s2"][k] for a, k in enumerate(g))
        # "Photos emitted nothing" -- no photon and no pair; with pair
        # emission on this is not the same as "no photon"
        key = "n_noemit" if "n_noemit" in d.files else "n_nophot"
        wn = float(sum(rw[a] * d[key][k] for a, k in enumerate(g)))
        r, w, tot = atoms_from_hist(s0, s1, s2, wn, ufine, sigma_cap,
                                    var_budget, tail=(t0, t1, t2))
        blo = 0.0 if gi == 0 else float(lo[g[0]])
        bhi = np.inf if gi == len(groups) - 1 else float(hi[g[-1]])
        R.append(r); W.append(w)
        LO.append(np.full(len(r), blo)); HI.append(np.full(len(r), bhi))
        info.append(dict(lo=blo, hi=None if not np.isfinite(bhi) else bhi,
                         natoms=len(r), p_norad=float(w[r == 1.0].sum()),
                         mean_u=-float(np.sum(w * np.log(r)))))
    return (dict(r=np.concatenate(R), w=np.concatenate(W),
                 m_lo=np.concatenate(LO), m_hi=np.concatenate(HI)), info)


_FRAC = None


def _sample_frac(path="data/photos/mpre_bands.bin"):
    """The sample's weight in each generated band (written by prep_input.py)."""
    global _FRAC
    if _FRAC is None:
        import struct
        with open(path, "rb") as f:
            blob = f.read()
        nb = struct.unpack_from("<i", blob, 0)[0]
        out = np.empty(nb)
        for i in range(nb):
            out[i] = struct.unpack_from("<d", blob, 4 + i * 40 + 16)[0]
        _FRAC = out / out.sum()
    return _FRAC


# --------------------------------------------------------------------------
def build_data(band_width=BAND_WIDTH, sigma_cap=None, var_budget=VAR_BUDGET,
               lo=50.0, hi=200.0, variant=DATA_VARIANT, pair=DATA_PAIRS):
    return FA.build_banded(FA.band_edges(lo, hi, band_width), variant=variant,
                           pair=tuple(pair), sigma_cap=sigma_cap,
                           var_budget=var_budget)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=("mc", "data"))
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--run", default=MC_RUN, help="mc: the standalone run npz")
    ap.add_argument("--band-width", type=float, default=BAND_WIDTH)
    ap.add_argument("--sigma-cap", type=float, default=None)
    ap.add_argument("--var-budget", type=float, default=None)
    ap.add_argument("--variant", default=DATA_VARIANT)
    ap.add_argument("--pair", nargs="*", default=None)
    a = ap.parse_args()

    if a.config == "mc":
        sc = a.sigma_cap if (a.sigma_cap or a.var_budget) else SIGMA_CAP
        k, info = build_mc(a.run, a.band_width, sc, a.var_budget)
        meta = dict(config="mc", run=a.run, band_width=a.band_width,
                    sigma_cap=sc, var_budget=a.var_budget)
    else:
        vb = a.var_budget if (a.sigma_cap or a.var_budget) else VAR_BUDGET
        k, info = build_data(a.band_width, a.sigma_cap, vb,
                             variant=a.variant,
                             pair=DATA_PAIRS if a.pair is None else a.pair)
        meta = dict(config="data", variant=a.variant,
                    pair=list(DATA_PAIRS if a.pair is None else a.pair),
                    band_width=a.band_width, sigma_cap=a.sigma_cap,
                    var_budget=vb)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez(a.output, **k)
    print(f"{a.output}: {len(k['r'])} atoms, {len(info)} bands")
    print("  " + json.dumps(meta))
    for i in (0, len(info) // 2, len(info) - 1):
        d = info[i]
        print(f"  band {i:3d} [{d['lo']:.1f}, {d['hi']}): "
              f"{d['natoms']} atoms, P(norad) = {d['p_norad']:.5f}, "
              f"<u> = {d['mean_u']:.6e}")


if __name__ == "__main__":
    main()

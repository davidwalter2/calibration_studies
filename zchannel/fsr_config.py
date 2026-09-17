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
    NLL term (`DATA_VARIANT`) - convoluted with the exact O(alpha^2) real
    pair radiator for every species the muon line can produce (`DATA_PAIRS`).
    First-principles throughout, no fitted constant, and its dependence on the
    fitted mass is exact.

Both are written as cell-integrated kernel **tables** ``(m_nodes, u_edges, K,
u_mean, p0)`` for the ``rabbit.lineshapes.zgamma`` provider (`fsr_table`): one
row per pre-FSR mass node, interpolated at every Born grid point, and each
``u`` cell integrated rather than collapsed onto an atom.  Neither the band
width nor ``sigma_cap``/``var_budget`` enters, and the two knobs that are left
-- the node spacing `fsr_table.DM_NODE` and the cell count `fsr_table.N_CELL`
-- are convergent (README, "Fold matrix from the kernel table").

``--atoms`` writes the superseded banded ``(r, w, m_lo, m_hi)`` form instead.
It is kept because the published atom rows were measured on it; it is not what
a new kernel should be built in.

Under a lepton ``p_T`` cut the kernel has to be resolved **leg by leg**, which
is `fsr_perleg`.  ``--htable`` turns either configuration into its
selection-conditional form with the two-leg law `SHARE_MODE`: the pair variable
``z`` keeps the kernel below untouched and the two muons' energy fractions are
drawn from the exact O(alpha) recoil sharing at fixed ``z`` -- carried by EVERY
photon of the kernel's Levy measure, which makes the law exact at O(alpha) in
the angle and exactly ``D (x) D`` in the collinear limit -- so that

    K_sel(u | m) = K(u | m) Gbar(u | m) ,

with ``Gbar`` the selection weight of the boson-kinematics table.  ``data``
hands it the closed form, ``mc`` the tabulated histograms of `MC_RUN`.  The
collinear product ``D (x) D`` with ``D`` the convolution square root --
`fsr_perleg.LegRadiator` for ``data``, ``legsqrt`` for ``mc`` -- is the
superseded form; it treats the two legs as independent and costs -2.5 MeV on
``Gamma_Z``.

    python3 fsr_config.py --config mc   -o data/ktab_cfg_mc.npz
    python3 fsr_config.py --config data -o data/ktab_cfg_data.npz
    python3 fsr_config.py --config data --htable data/ht_pt25_1.0gev.npz \
            --pt-cuts 25 25 -o data/ktab_cfg_data_2525.npz \
            -a data/atab_cfg_data_2525.json
    python3 fsr_config.py --config data --atoms \
            -o data/kern_cfg_data_vb6e-10.npz          # the legacy atom form
"""
import argparse
import json
import os

import numpy as np

import fsr_analytic as FA

#: `fsr_analytic` settings of the ``data`` configuration.  One constant, so
#: that adding a variant to `fsr_analytic` is a one-line change here.
DATA_VARIANT = "exp2nll"
#: every species whose pair emission the muon line can produce.  At the Z the
#: exact pair radiator adds N_pair = 3.54e-3 emissions and 2.15 % to the mean
#: mass loss (1.10 % for e+e- alone, 1.40 % for e and mu, the species Photos
#: generates).  Requires the table `fsr_analytic.PAIR_TABLE`.
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

#: the two-leg law the selection-conditional form uses (``--htable``, and
#: ``fsr_perleg.py corr --mode``).  ``multi`` is the exponentiated exact
#: O(alpha) sharing: right at
#: O(alpha) in the recoil angle AND collapsing exactly to ``D (x) D`` in the
#: collinear limit, so it needs no matching scale.  ``single`` is the one-photon
#: sharing it supersedes; the two differ by -0.03 / +0.05 MeV at 25/25 and
#: +0.02 / +0.00 at 25/10, below the closure test's resolution, so the choice is
#: made on correctness rather than on a measured gain (README).
SHARE_MODE = "multi"

#: discretisation of the LEGACY atom form (``--atoms``).  ``mc`` uses the
#: ``sigma_cap`` of the empirical kernels of `fit_gen.py kernel`, ``data`` the
#: ``var_budget`` of `fsr_analytic.py kernel`, so each is like for like with its
#: own established control.  The table form has neither.
SIGMA_CAP = 3.3e-4
VAR_BUDGET = 6e-10
BAND_WIDTH = 2.0


# --------------------------------------------------------------------------
# the table form (the default): `fsr_table` at this module's settings
# --------------------------------------------------------------------------
def build_table(config, htable=None, pt_cuts=None, run=MC_RUN,
                variant=DATA_VARIANT, pair=DATA_PAIRS, mode=SHARE_MODE,
                rho=None, resol=None, resol_mode="shape", h4=None,
                dm_node=None, n_cell=None, u_min=None, u_max=None,
                m_lo=None, m_hi=None, free_grid=False):
    """``(table, acceptance, meta)`` of one configuration.

    Inclusive without ``htable``, selection-conditional with it: the same
    `fsr_table.build_corr` every other ``corr`` producer calls, so the two
    differ only in which cells go into it -- `fsr_table.build_mc` for ``mc``,
    `fsr_table.build_analytic` for ``data``.

    The ``u`` grid of the inclusive ``mc`` table is a COARSENING of the
    standalone run's own histogram bins, which makes the rebinning exact;
    everywhere else it is the geometric ladder `fsr_table.u_grid`.
    """
    import fsr_perleg as PL
    import fsr_table as FT

    dm_node = FT.DM_NODE if dm_node is None else dm_node
    n_cell = FT.N_CELL if n_cell is None else n_cell
    u_min = FT.U_MIN if u_min is None else u_min
    u_max = FT.U_MAX if u_max is None else u_max
    m_lo = FT.M_LO if m_lo is None else m_lo
    m_hi = FT.M_HI if m_hi is None else m_hi
    pair = tuple(pair)
    nodes = FT.m_nodes(m_lo, m_hi, dm_node)
    mc = config == "mc"
    if mc and htable is None and not free_grid:
        ue = FT.mc_u_grid(run, n_cell, u_max)
    else:
        ue = FT.u_grid(n_cell, u_min, u_max)
    meta = dict(config=config, dm_node=dm_node, n_cell=len(ue) - 1,
                u_min=u_min, u_max=u_max, m_lo=m_lo, m_hi=m_hi)
    meta.update(dict(run=run) if mc else
                dict(variant=variant, pair=list(pair), var_budget=VAR_BUDGET))

    if htable is None:
        tab = (FT.build_mc(nodes, ue, run) if mc else
               FT.build_analytic(nodes, ue, variant, pair,
                                 var_budget=VAR_BUDGET))
        return tab, None, dict(meta, kind=config)

    ht = PL.load_htable(htable)
    pr = PL.make_pass(ht, argparse.Namespace(pt_cuts=pt_cuts, resol=resol,
                                             resol_mode=resol_mode, h4=h4))
    cbb = (lambda: PL.tabulated_cells(run, ht["bands"])) if mc else None
    rt = FT.load_rho(ht, mode, rho, cells_by_band=cbb, variant=variant,
                     pair=pair, pr=pr)
    tab, acc = FT.build_corr(nodes, ue, ht, run=run if mc else None,
                             variant=variant, pair=pair,
                             var_budget=VAR_BUDGET, pr=pr, rho=rt)
    acc = dict(acc, _meta=dict(selection=PL._sel_meta(pr, ht),
                               eta_cut=ht["eta_cut"]))
    meta.update(kind="corr", htable=os.path.abspath(htable), mode=mode,
                rho=rho, selection=PL._sel_meta(pr, ht))
    return tab, acc, meta


# --------------------------------------------------------------------------
# the LEGACY atom form: atoms from the standalone Photos histograms
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


def build_mc_atoms(run=MC_RUN, band_width=BAND_WIDTH, sigma_cap=SIGMA_CAP,
                   var_budget=None, u_max=None):
    """Banded atoms from a standalone Photos run (the legacy form).

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
def build_data_atoms(band_width=BAND_WIDTH, sigma_cap=None,
                     var_budget=VAR_BUDGET, lo=50.0, hi=200.0,
                     variant=DATA_VARIANT, pair=DATA_PAIRS):
    """Banded atoms of the analytic radiator (the legacy form)."""
    return FA.build_banded(FA.band_edges(lo, hi, band_width), variant=variant,
                           pair=tuple(pair), sigma_cap=sigma_cap,
                           var_budget=var_budget)


# --------------------------------------------------------------------------
def _atoms_main(a):
    """The legacy ``(r, w, m_lo, m_hi)`` output of ``--atoms``."""
    if a.htable:
        raise SystemExit("--atoms is inclusive only; the banded atoms of a "
                         "selection are `fsr_perleg.py corr --atoms`")
    a.band_width = BAND_WIDTH if a.band_width is None else a.band_width
    if a.config == "mc":
        sc = a.sigma_cap if (a.sigma_cap or a.var_budget) else SIGMA_CAP
        k, info = build_mc_atoms(a.run, a.band_width, sc, a.var_budget)
        meta = dict(config="mc", run=a.run, band_width=a.band_width,
                    sigma_cap=sc, var_budget=a.var_budget)
    else:
        vb = a.var_budget if (a.sigma_cap or a.var_budget) else VAR_BUDGET
        k, info = build_data_atoms(a.band_width, a.sigma_cap, vb,
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
        extra = (f"P(norad) = {d['p_norad']:.5f}" if "p_norad" in d
                 else f"N_pair = {d.get('pair_rate', 0.0):.4e}")
        print(f"  band {i:3d} [{d['lo']:.1f}, {d['hi']}): "
              f"{d['natoms']} atoms, {extra}, <u> = {d['mean_u']:.6e}")


def main():
    import fsr_table as FT

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=("mc", "data"))
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--run", default=MC_RUN, help="mc: the standalone run npz")
    ap.add_argument("--variant", default=DATA_VARIANT)
    ap.add_argument("--pair", nargs="*", default=None)
    # the selection-conditional form
    ap.add_argument("--htable", default=None,
                    help="an `fsr_perleg.py htable` boson-kinematics table: "
                         "write K_sel(u|m) = K(u|m) Gbar(u|m) and its A(m) "
                         "instead of the inclusive kernel")
    ap.add_argument("-a", "--acceptance", default=None,
                    help="where A(m) goes, required with --htable")
    ap.add_argument("--pt-cuts", nargs="*", type=float, default=None,
                    help="leading and trailing pT thresholds [GeV]; one value "
                         "or none is the symmetric cut at the table's pT ref")
    ap.add_argument("--mode", default=SHARE_MODE,
                    choices=("single", "lin", "multi", "coll", "matched"),
                    help="the two-leg law (`fsr_perleg.py corr --mode`)")
    ap.add_argument("--rho", default=None,
                    help="cache file of the two-leg law's correction table")
    ap.add_argument("--h4", default=None, help="h4 table, required with --resol")
    ap.add_argument("--resol", default=None,
                    help="a `ptres.py` resolution npz: the cuts then act on "
                         "the RECONSTRUCTED pT and the pass region is smooth")
    ap.add_argument("--resol-mode", default="shape", choices=("shape", "gauss"))
    # the table's own discretisation, both convergent
    ap.add_argument("--dm-node", type=float, default=FT.DM_NODE)
    ap.add_argument("--n-cell", type=int, default=FT.N_CELL)
    ap.add_argument("--u-min", type=float, default=FT.U_MIN)
    ap.add_argument("--u-max", type=float, default=FT.U_MAX)
    ap.add_argument("--m-lo", type=float, default=FT.M_LO)
    ap.add_argument("--m-hi", type=float, default=FT.M_HI)
    ap.add_argument("--free-grid", action="store_true",
                    help="mc: the geometric u grid instead of a coarsening of "
                         "the run's own bins (then the rebinning is not exact)")
    # the legacy atom form
    ap.add_argument("--atoms", action="store_true",
                    help="LEGACY: write banded (r, w, m_lo, m_hi) atoms "
                         "instead of a cell-integrated table.  Kept so the "
                         "published atom rows can be regenerated; --band-width,"
                         " --sigma-cap and --var-budget apply only to it")
    ap.add_argument("--band-width", type=float, default=None)
    ap.add_argument("--sigma-cap", type=float, default=None)
    ap.add_argument("--var-budget", type=float, default=None)
    a = ap.parse_args()

    if a.atoms:
        _atoms_main(a)
        return
    for name in ("band_width", "sigma_cap", "var_budget"):
        if getattr(a, name) is not None:
            raise SystemExit(f"--{name.replace('_', '-')} is a discretisation "
                             "of the banded atom form and only applies to "
                             "--atoms")
    if a.htable and not a.acceptance:
        raise SystemExit("--htable needs -a/--acceptance")
    tab, acc, meta = build_table(
        a.config, htable=a.htable, pt_cuts=a.pt_cuts, run=a.run,
        variant=a.variant, pair=DATA_PAIRS if a.pair is None else a.pair,
        mode=a.mode, rho=a.rho, resol=a.resol, resol_mode=a.resol_mode,
        h4=a.h4, dm_node=a.dm_node, n_cell=a.n_cell, u_min=a.u_min,
        u_max=a.u_max, m_lo=a.m_lo, m_hi=a.m_hi, free_grid=a.free_grid)
    FT.save(a.output, tab, meta, acc=acc, acc_path=a.acceptance)
    FT.describe(a.output)


if __name__ == "__main__":
    main()

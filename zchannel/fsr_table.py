#!/usr/bin/env python3
r"""Cell-integrated FSR kernel tables for the `rabbit.lineshapes.zgamma` fold.

The format
----------
A table is an npz (or a mapping) with

===========  =========================  ====================================
`m_nodes`    `(n_m,)`                   pre-FSR mass nodes, GeV, increasing
`u_edges`    `(n_c + 1,)`               cell edges in `u = -ln(m_post/m_pre)`,
                                        increasing, `u_edges[0] = 0`
`K`          `(n_m, n_c)`               probability MASS in each cell
`u_mean`     `(n_m, n_c)`               its first moment `<u>`, in the cell
`p0`         `(n_m,)`                   point mass at `u = 0` (may be 0)
===========  =========================  ====================================

with `sum_c K[i, c] + p0[i] = 1` for every node `i` (the provider renormalises
rows anyway).  `u_mean` of an empty cell is ignored.

This replaces the `(r, w, m_lo, m_hi)` atom form and removes both of its
discretisations at once:

* the **bands**.  The atom form is piecewise constant in `m_pre`; the table is
  interpolated in `m` at every Born grid point, so the kernel's mass
  dependence -- `beta(m)`, the phase space, the pair table, the selection
  weight -- is continuous.  Nodes 0.5-1 GeV apart are far finer than the
  2 GeV bands and cost one row each, not one atom set each.
* the **atoms** (`sigma_cap` / `var_budget`).  An atom is a group of cells
  collapsed onto its mean, so the fold loses the group's second moment; the
  table keeps the cells and the provider deposits each one by the rule that is
  exact for its own width -- a point mass at the cell's first moment while the
  cell is narrower than the output grid spacing, the exact hat-basis projection
  of its own density once it is wider.

The producers below all write the same format:

`analytic`
    `fsr_analytic.FSRKernel` at each node.  The photonic part is integrated
    **cell by cell** in `t = (1-z)^beta`, the substitution that removes the
    `C beta (1-z)^{beta-1}` endpoint singularity exactly: the singular piece
    contributes `C [(1-z_lo)^beta - (1-z_hi)^beta]` per cell analytically (it
    is a constant integrand in `t`, so Gauss-Legendre is exact on it) and the
    regular remainder is the same quadrature the atom form uses.  The pair
    branch -- `K = K_phot (x) [(1-N) delta + R_pair]` -- is folded in by
    depositing the (coarse photonic atom) x (pair atom) products into the
    cells at their own `u`, mass and first moment exact.
`mc`
    the standalone Photos++ histograms (`photos_standalone/`): the fine `u`
    bins ARE cells, so the table is a coarsening of them (bins are merged into
    cells, which adds their masses and first moments -- exact), and the 78
    generated 2 GeV bands are interpolated onto the mass nodes.  `p0` is the
    generator's unradiated fraction, a genuine delta.
`corr`
    the selection-conditional kernel of `fsr_perleg`, `K_sel(u|m) = K(u|m)
    Gbar(u|m)`: the inclusive cells of either configuration above, multiplied
    per cell by `Gbar` at the cell's own mean.  `Gbar` is smooth and is
    computed on the `h` table's bands and interpolated in `m` onto the nodes;
    the acceptance `A(m) = int K Gbar` comes out of the same integral and is
    written as the `{"kind": "grid"}` acceptance file the provider takes.
`atoms`
    an existing `(r, w)` atom file turned into a table of point-like cells
    `[u - eps, u + eps]` -- the identity test of the two paths, which the
    provider then folds into the atom form's own fold matrix.

    python3 fsr_table.py analytic -o data/ktab_data_dm10_c2000.npz --dm-node 1.0
    python3 fsr_table.py mc       -o data/ktab_mc_dm10.npz         --dm-node 1.0
    python3 fsr_table.py corr --htable data/ht_ref10_1.0gev.npz --pt-cuts 25 25 \
            -o data/ktab_corr_data_2525.npz -a data/atab_corr_data_2525.json
    python3 fsr_table.py check -i data/ktab_data_dm10_c2000.npz
    ./run_tf_z.sh python3 -u fsr_table.py fit --gen data/genmerged_full.npz \
            --nm 8192 --shape 5 --row "label=data/ktab_data_dm10_c2000.npz"
"""
import argparse
import json
import os
import time

import numpy as np

import fsr_analytic as FA
import fsr_config as FC

#: default `u` grid: everything below `U_MIN` is one cell (the analytic kernel
#: puts 31 % of its mass there, all of it at `u < 1e-9`, i.e. at `m' = m` to
#: 1e-8 of a grid spacing), then geometric to `U_MAX` -- the 2-muon threshold
#: at the Z is `u = 6.06`, and the fold's output window cuts at `u = 0.6`.
U_MIN = 1e-9
U_MAX = 7.0
N_CELL = 2000
#: mass nodes: the Born grid of the fit runs to the luminosity table's 200 GeV
M_LO = 50.0
M_HI = 200.0
DM_NODE = 1.0


def u_grid(n=N_CELL, u_min=U_MIN, u_max=U_MAX):
    """`u_edges`: `[0, u_min]` then geometric to `u_max`."""
    return np.concatenate([[0.0], np.geomspace(u_min, u_max, n)])


def m_nodes(lo=M_LO, hi=M_HI, dm=DM_NODE):
    return lo + dm * np.arange(int(np.ceil((hi - lo) / dm)) + 1)


def _bin_moments(u, w, u_edges):
    """Deposit point masses `(u, w)` into cells: `(mass, first moment)`."""
    n = len(u_edges) - 1
    i = np.clip(np.searchsorted(u_edges, u, "right") - 1, 0, n - 1)
    return (np.bincount(i, w, n), np.bincount(i, w * u, n))


def _rebin(src_edges, s0, s1, u_edges):
    """Merge source cells into target cells by their centre.

    Exact -- masses and first moments simply add -- when `u_edges` is a
    coarsening of `src_edges`, which is what `mc_u_grid` returns.  On any
    other target grid a source cell that straddles a target edge is assigned
    whole to the side its centre falls on; its first moment goes with it, so
    the mean is still right and only the cell boundary moves.
    """
    n = len(u_edges) - 1
    cen = 0.5 * (src_edges[:-1] + src_edges[1:])
    i = np.clip(np.searchsorted(u_edges, cen, "right") - 1, 0, n - 1)
    return np.bincount(i, s0, n), np.bincount(i, s1, n)


# --------------------------------------------------------------------------
# the analytic kernel
# --------------------------------------------------------------------------
def analytic_row(m, u_edges, variant=FC.DATA_VARIANT, pair=FC.DATA_PAIRS,
                 minus_one=True, pair_table=None, ng=16, var_budget=6e-10,
                 kernel=None):
    """`(K, u_mean, p0, kernel)` of `FSRKernel(m)` on `u_edges`.

    The photonic part is the exact per-cell integral of the `t = (1-z)^beta`
    quadrature; the pair branch is deposited at its own `u`.
    """
    k = kernel or FA.FSRKernel(m, variant=variant, pair=tuple(pair),
                               minus_one=minus_one, pair_table=pair_table)
    u_max = k._u_max()
    ue = np.minimum(u_edges, u_max)
    t = k._t_of_u(ue)
    u, _, w = k._panel(t[:-1], t[1:], ng)
    m0 = w.sum(1)
    m1 = (w * u).sum(1)
    N = float(k.pair_rate)
    if N > 0.0:
        # K = K_phot (x) [(1 - N) delta(1-z) + R_pair]; the delta branch keeps
        # the cells, the pair branch multiplies a coarse photonic atom set
        # (its error carries the weight N, so the budget there is 1/N looser)
        uk, wk = k.pair_atoms(var_budget=var_budget / max(N, 1e-12))
        uc, wc = FA._merge_cells(np.stack([m0, m1, (w * u * u).sum(1)]),
                                 var_budget / max(N, 1e-12))
        uu = (uc[:, None] + uk[None, :]).ravel()
        ww = (wc[:, None] * wk[None, :]).ravel()
        b0, b1 = _bin_moments(uu, ww, u_edges)
        m0 = (1.0 - N) * m0 + b0
        m1 = (1.0 - N) * m1 + b1
    m0 = np.maximum(m0, 0.0)
    cen = 0.5 * (u_edges[:-1] + u_edges[1:])
    um = np.where(m0 > 0.0, m1 / np.where(m0 > 0.0, m0, 1.0), cen)
    um = np.clip(um, u_edges[:-1], u_edges[1:])
    return m0, um, 0.0, k


def build_analytic(nodes, u_edges, variant=FC.DATA_VARIANT, pair=FC.DATA_PAIRS,
                   minus_one=True, pair_table=None, ng=16, var_budget=6e-10,
                   verbose=True):
    K = np.empty((len(nodes), len(u_edges) - 1))
    U = np.empty_like(K)
    p0 = np.zeros(len(nodes))
    t0 = time.time()
    for i, m in enumerate(nodes):
        K[i], U[i], p0[i], _ = analytic_row(m, u_edges, variant, pair,
                                            minus_one, pair_table, ng,
                                            var_budget)
    if verbose:
        print(f"[analytic] {len(nodes)} nodes x {K.shape[1]} cells, "
              f"{time.time() - t0:.0f} s")
    return dict(m_nodes=np.asarray(nodes, float), u_edges=u_edges, K=K,
                u_mean=U, p0=p0)


# --------------------------------------------------------------------------
# the tabulated (standalone Photos) kernel
# --------------------------------------------------------------------------
def photos_bands(run=FC.MC_RUN, u_tail0=2.0, u_tail_w=0.05):
    """`(m_band, src_edges, s0, s1, w_delta)` of a standalone Photos run.

    One row per generated band, at the band centre; the histogram bins are
    cells with exact first moments, and the events the generator left
    untouched are the delta at `u = 0`.
    """
    d = np.load(run, allow_pickle=True)
    lo = np.asarray(d["bands_lo"], float)
    hi = np.asarray(d["bands_hi"], float)
    n = np.asarray(d["n"], float)
    key = "n_noemit" if "n_noemit" in d.files else "n_nophot"
    wn = np.asarray(d[key], float)
    f0, f1 = np.asarray(d["fine_s0"], float), np.asarray(d["fine_s1"], float)
    t0, t1 = np.asarray(d["tail_s0"], float), np.asarray(d["tail_s1"], float)
    nfine, ntail = f0.shape[1], t0.shape[1]
    src = np.concatenate([
        np.linspace(0.0, 2.0, nfine + 1),
        u_tail0 + u_tail_w * np.arange(1, ntail + 1),
    ])
    mb = 0.5 * (lo + hi)
    # the last band is open (its upper edge is a placeholder); put its node
    # just above its lower edge rather than at a meaningless centre
    mb = np.where(hi > 10.0 * lo, 1.05 * lo, mb)
    S0 = np.concatenate([f0, t0], axis=1) / np.maximum(n, 1.0)[:, None]
    S1 = np.concatenate([f1, t1], axis=1) / np.maximum(n, 1.0)[:, None]
    wd = wn / np.maximum(n, 1.0)
    S0[:, 0] = np.maximum(S0[:, 0] - wd, 0.0)      # split off the delta
    return mb, src, S0, S1, wd


def build_mc(nodes, u_edges, run=FC.MC_RUN, verbose=True):
    """The `mc` table: the run's bands rebinned onto `u_edges`, interpolated in m."""
    t0 = time.time()
    mb, src, S0, S1, wd = photos_bands(run)
    nb = len(mb)
    B0 = np.empty((nb, len(u_edges) - 1))
    B1 = np.empty_like(B0)
    for i in range(nb):
        B0[i], B1[i] = _rebin(src, S0[i], S1[i], u_edges)
    nodes = np.asarray(nodes, float)
    o = np.argsort(mb)
    K = np.stack([np.interp(nodes, mb[o], B0[o, c]) for c in range(B0.shape[1])],
                 axis=1)
    M1 = np.stack([np.interp(nodes, mb[o], B1[o, c]) for c in range(B1.shape[1])],
                  axis=1)
    p0 = np.interp(nodes, mb[o], wd[o])
    cen = 0.5 * (u_edges[:-1] + u_edges[1:])
    U = np.where(K > 0.0, M1 / np.where(K > 0.0, K, 1.0), cen)
    U = np.clip(U, u_edges[:-1], u_edges[1:])
    if verbose:
        print(f"[mc] {nb} generated bands -> {len(nodes)} nodes x {K.shape[1]} "
              f"cells, {time.time() - t0:.0f} s")
    return dict(m_nodes=nodes, u_edges=u_edges, K=K, u_mean=U, p0=p0)


def mc_u_grid(run=FC.MC_RUN, n=N_CELL, u_max=U_MAX):
    """A `u` grid that is a COARSENING of the run's own bins (exact rebinning).

    The generated bins are 2e-5 wide below `u = 2`; keeping them all would be
    100 k cells, so they are merged into groups whose width follows the same
    geometric ladder as `u_grid`, snapped to the source edges.
    """
    _, src, _, _, _ = photos_bands(run)
    want = u_grid(n, src[1], min(u_max, src[-1]))
    i = np.unique(np.searchsorted(src, want, "left"))
    i = i[(i >= 0) & (i < len(src))]
    e = np.unique(np.concatenate([[0.0], src[i]]))
    return e


# --------------------------------------------------------------------------
# the selection-conditional (corr) kernel
# --------------------------------------------------------------------------
def build_corr(nodes, u_edges, htable, run=None, variant=FC.DATA_VARIANT,
               pair=FC.DATA_PAIRS, minus_one=True, pair_table=None,
               var_budget=6e-10, n_gbar=2000, npanel=64, ng=8, u_min=1e-9,
               pr=None, verbose=True):
    """`(table, acceptance)` of `K_sel(u|m) = K(u|m) Gbar(u|m)`.

    `Gbar` is computed on the `h` table's own bands (it is a property of the
    boson-kinematics table, which is binned) and interpolated in `m` onto the
    nodes; the kernel it multiplies is evaluated AT the node, so the whole
    `m` dependence of the QED is continuous.  `A(m) = int K Gbar` is returned
    as the tabulated acceptance on the same nodes.
    """
    import fsr_perleg as PL

    t0 = time.time()
    bands = htable["bands"]
    b_edges = np.asarray(htable["b_edges"], float)
    H = np.asarray(htable["h"], float)
    nodes = np.asarray(nodes, float)
    un = np.concatenate([[0.0], np.geomspace(u_min, U_MAX, n_gbar)])
    mb, GB = [], []
    for k, (lo, hi) in enumerate(bands):
        mc = 0.5 * (lo + hi)
        if not np.isfinite(mc):
            mc = float(lo) * 1.05
        if k == 0:
            mc = float(bands[0][1]) * 0.95
        g = pr.grid(k) if pr is not None else PL.survival(H[k], b_edges)
        mb.append(mc)
        GB.append(PL.gbar(mc, un, g, b_edges, npanel=npanel, ng=ng))
    mb = np.asarray(mb)
    GB = np.asarray(GB)
    o = np.argsort(mb)
    if verbose:
        print(f"[corr] Gbar on {len(mb)} bands, {time.time() - t0:.0f} s")

    cells = build_mc(nodes, u_edges, run) if run else \
        build_analytic(nodes, u_edges, variant, pair, minus_one, pair_table,
                       var_budget=var_budget, verbose=verbose)
    K, U, p0 = cells["K"], cells["u_mean"], cells["p0"]
    A = np.empty(len(nodes))
    ms, Gs = mb[o], GB[o]
    for i, m in enumerate(nodes):
        j = int(np.clip(np.searchsorted(ms, m) - 1, 0, len(ms) - 2))
        f = np.clip((m - ms[j]) / (ms[j + 1] - ms[j]), 0.0, 1.0)
        gb = (1.0 - f) * Gs[j] + f * Gs[j + 1]
        lg = np.interp(np.log(np.maximum(U[i], u_min)), np.log(un[1:]), gb[1:],
                       left=gb[0], right=gb[-1])
        lg = np.where(U[i] <= u_min, gb[0], lg)
        K[i] = K[i] * lg
        p0[i] = p0[i] * gb[0]
        A[i] = K[i].sum() + p0[i]
        K[i] /= A[i]
        p0[i] /= A[i]
    if verbose:
        print(f"[corr] {len(nodes)} nodes, A = {A.min():.4f}..{A.max():.4f}, "
              f"{time.time() - t0:.0f} s")
    return cells, dict(kind="grid", m=nodes.tolist(), a=A.tolist())


# --------------------------------------------------------------------------
# atoms -> table (the identity test)
# --------------------------------------------------------------------------
def from_atoms(path, m_lo=M_LO, m_hi=M_HI, eps=1e-9):
    """A band-less atom file as a table of point-like cells: the identity test.

    Each atom gets a cell `[u - eps, u + eps]` carrying its whole weight, with
    an empty cell in the gap to the next one.  Every filled cell is then
    narrower than any output grid the provider can have, so it is deposited at
    its own `u` -- the same point mass the atom path folds -- and the empty
    ones contribute nothing.  What is left between the two paths is only the
    direction of the interpolation (see `fsr_matrix_from_table`).
    """
    with np.load(path, allow_pickle=False) as d:
        r = np.asarray(d["r"], float).ravel()
        w = np.asarray(d["w"], float).ravel()
        if "m_lo" in d and len(np.unique(d["m_lo"])) > 1:
            raise ValueError("from_atoms needs a band-less atom file")
    u = -np.log(np.minimum(r, 1.0))
    o = np.argsort(u)
    u, w = u[o], w[o] / w.sum()
    gap = np.diff(u, prepend=2.0 * u[0] - eps, append=u[-1] + 2.0 * eps)
    e = np.minimum(eps, 0.45 * np.minimum(gap[:-1], gap[1:]))
    edges = np.concatenate([np.maximum(u - e, 0.0)[:, None], (u + e)[:, None]],
                           axis=1).ravel()
    edges = np.concatenate([[0.0], edges]) if edges[0] > 0.0 else edges
    K = np.zeros(len(edges) - 1)
    U = 0.5 * (edges[:-1] + edges[1:])
    j = np.searchsorted(edges, u, "right") - 1
    K[j] = w
    U[j] = u
    return dict(m_nodes=np.array([m_lo, m_hi]),
                u_edges=edges,
                K=np.broadcast_to(K, (2, len(K))).copy(),
                u_mean=np.broadcast_to(U, (2, len(U))).copy(),
                p0=np.zeros(2))


# --------------------------------------------------------------------------
def save(path, tab, meta=None, acc=None, acc_path=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = {k: np.asarray(tab[k], float) for k in
         ("m_nodes", "u_edges", "K", "u_mean", "p0")}
    if meta:
        d["provenance"] = np.array([json.dumps(meta)])
    np.savez(path, **d)
    sz = os.path.getsize(path) / 1e6
    print(f"{path}: {d['K'].shape[0]} nodes x {d['K'].shape[1]} cells, "
          f"{sz:.1f} MB")
    if acc is not None and acc_path:
        with open(acc_path, "w") as fh:
            json.dump(acc, fh)
        print(f"{acc_path}: A(m) on {len(acc['m'])} nodes")


def describe(path):
    with np.load(path, allow_pickle=True) as d:
        K, ue, mn = d["K"], d["u_edges"], d["m_nodes"]
        p0 = d["p0"]
        U = d["u_mean"]
        tot = K.sum(1) + p0
        i = len(mn) // 2
        uu = float((K[i] * U[i]).sum() + 0.0)
        print(f"{path}")
        print(f"  {len(mn)} nodes {mn[0]:.1f}..{mn[-1]:.1f} GeV, "
              f"{K.shape[1]} cells, u {ue[0]:.0e}..{ue[-1]:.2f}")
        print(f"  row sums: {tot.min():.15f} .. {tot.max():.15f}")
        print(f"  p0: {p0.min():.5f} .. {p0.max():.5f}")
        print(f"  <u> at m = {mn[i]:.1f}: {uu:.6e}")
        print(f"  mass below u = 1e-6 / 1e-4 / 1e-2: "
              + " / ".join(f"{K[i][ue[1:] <= x].sum() + p0[i]:.4f}"
                           for x in (1e-6, 1e-4, 1e-2)))
        if "provenance" in d.files:
            print("  " + str(d["provenance"][0]))


# --------------------------------------------------------------------------
# the fit benchmark: the same rows `fit_gen.py fit` runs, atoms or tables
# --------------------------------------------------------------------------
def cmd_fit(args):
    """`fit_gen`'s closure fit on an explicit list of rows.

    `fit_gen.py fit` runs a whole suite; here each `--row` is one kernel (atom
    file or table, `none` for no fold), so an atom row and its table can be run
    side by side on the same events.  Everything else -- the events, the
    weights, the window, the `K(m)` terms -- is `fit_gen`'s own machinery.
    """
    import resource

    import tensorflow as tf

    import fit_gen as FG
    from rabbit.lineshapes import ZGammaLineshape

    g = FG.load_gen(args.gen)
    if args.nmax:
        g = {k: v[:args.nmax] for k, v in g.items() if v.ndim == 1}
    w = FG.clip_weights(g["weight"].astype(np.float64), args.wclip)
    sel = None
    if args.acc_pt is not None:
        sel = FG.fiducial(g, (args.acc_pt, args.acc_pt_trail or args.acc_pt),
                          args.acc_eta, post=True,
                          smear=FG.load_resolution(args.smear, args.smear_mode),
                          seed=FG.SMEAR_SEED)
        print(f"[fit] fiducial pT > {args.acc_pt}/"
              f"{args.acc_pt_trail or args.acc_pt}, |eta| < {args.acc_eta}: "
              f"{sel.sum()} / {len(sel)} events")
    results = {}
    for spec in args.row:
        lab, kpath, apath = FG._alt_spec(spec)
        acc = None
        if apath:
            with open(apath) as fh:
                acc = {k: v for k, v in json.load(fh).items()
                       if not k.startswith("_")}
        m = g[args.mass]
        ww = w
        if sel is not None:
            m, ww = m[sel], ww[sel]
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.time()
        z = ZGammaLineshape(
            m_ref=FG.MZ_RUNNING, window=(args.born_lo, args.born_hi),
            nm=args.nm, width_scheme="fixed", lumi="nnpdf31_nnlo_13tev",
            terms=("gamma", "int", "z"),
            fsr=None if kpath in ("none", "") else kpath, acceptance=acc,
            mz_ref=FG.MZ_FIXED, gz_ref=FG.GZ_FIXED, mz_unit=1e-3, gz_unit=1e-3,
            fsr_mmax=args.fsr_mmax, fsr_minterp=args.minterp,
            fsr_deposit=args.deposit,
            nfft=args.nm, tau_max=1.0)
        tb = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        fit = FG.GenFit(z, m, ww, tuple(args.window), shape=args.shape, tf=tf)
        res = fit.fit(verbose=False)
        d = FG.summarise(f"{lab}  [{fit.n_used} events in {tuple(args.window)}]",
                         res)
        d["_build_s"] = tb
        d["_matrix_MB"] = z.nm * z.n_born * 8 / 1e6 if z.fsr is not None else 0.0
        d["_maxrss_MB"] = (r1 - r0) / 1e3
        d["_kind"] = z.fsr_kind
        print(f"      [build {tb:.1f} s, matrix {d['_matrix_MB']:.0f} MB, "
              f"maxrss +{d['_maxrss_MB']:.0f} MB, n_born {z.n_born}, "
              f"kind {z.fsr_kind}]")
        results[lab] = d
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"[fit] -> {args.output}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-o", "--output", required=True)
        p.add_argument("--n-cell", type=int, default=N_CELL)
        p.add_argument("--u-min", type=float, default=U_MIN)
        p.add_argument("--u-max", type=float, default=U_MAX)
        p.add_argument("--dm-node", type=float, default=DM_NODE)
        p.add_argument("--m-lo", type=float, default=M_LO)
        p.add_argument("--m-hi", type=float, default=M_HI)

    a = sub.add_parser("analytic")
    common(a)
    a.add_argument("--variant", default=FC.DATA_VARIANT)
    a.add_argument("--pair", nargs="*", default=None)
    a.add_argument("--var-budget", type=float, default=6e-10)

    m = sub.add_parser("mc")
    common(m)
    m.add_argument("--run", default=FC.MC_RUN)
    m.add_argument("--free-grid", action="store_true",
                   help="use the geometric u grid instead of a coarsening of "
                        "the run's own bins (then the rebinning is not exact)")

    c = sub.add_parser("corr")
    common(c)
    c.add_argument("--htable", required=True)
    c.add_argument("--run", default=None, help="mc configuration of K")
    c.add_argument("--pair", nargs="*", default=None)
    c.add_argument("--variant", default=FC.DATA_VARIANT)
    c.add_argument("-a", "--acceptance", required=True)
    c.add_argument("--pt-cuts", type=float, nargs=2, default=None)
    c.add_argument("--resol", default=None)
    c.add_argument("--resol-mode", default="shape")
    c.add_argument("--h4", default=None)

    t = sub.add_parser("atoms")
    t.add_argument("-i", "--input", required=True)
    t.add_argument("-o", "--output", required=True)

    d = sub.add_parser("check")
    d.add_argument("-i", "--input", required=True, nargs="+")

    f = sub.add_parser("fit")
    f.add_argument("--gen", required=True)
    f.add_argument("--row", action="append", required=True,
                   help="label=kernel.npz[:acc.json]; kernel 'none' = no fold")
    f.add_argument("--mass", default="m_post")
    f.add_argument("--nm", type=int, default=8192)
    f.add_argument("--shape", type=int, default=5)
    f.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0])
    f.add_argument("--born-lo", type=float, default=50.0)
    f.add_argument("--born-hi", type=float, default=130.0)
    f.add_argument("--fsr-mmax", type=float, default=None)
    f.add_argument("--minterp", default="linear", choices=("linear", "cubic"))
    f.add_argument("--deposit", default="interp", choices=("interp", "mass"))
    f.add_argument("--acc-pt", type=float, default=None)
    f.add_argument("--acc-pt-trail", type=float, default=None)
    f.add_argument("--acc-eta", type=float, default=2.4)
    f.add_argument("--smear", default=None)
    f.add_argument("--smear-mode", default="shape")
    f.add_argument("--wclip", type=float, default=100.0)
    f.add_argument("--nmax", type=int, default=0)
    f.add_argument("-o", "--output", default=None)

    args = ap.parse_args()

    if args.cmd == "fit":
        cmd_fit(args)
        return

    if args.cmd == "check":
        for p in args.input:
            describe(p)
        return

    if args.cmd == "atoms":
        save(args.output, from_atoms(args.input),
             dict(kind="atoms", src=os.path.abspath(args.input)))
        return

    nodes = m_nodes(args.m_lo, args.m_hi, args.dm_node)
    ue = u_grid(args.n_cell, args.u_min, args.u_max)

    if args.cmd == "analytic":
        pair = FC.DATA_PAIRS if args.pair is None else tuple(args.pair)
        tab = build_analytic(nodes, ue, args.variant, pair,
                             var_budget=args.var_budget)
        save(args.output, tab,
             dict(kind="analytic", variant=args.variant, pair=list(pair),
                  var_budget=args.var_budget, dm_node=args.dm_node,
                  n_cell=args.n_cell, u_min=args.u_min, u_max=args.u_max))
    elif args.cmd == "mc":
        if not args.free_grid:
            ue = mc_u_grid(args.run, args.n_cell, args.u_max)
        tab = build_mc(nodes, ue, args.run)
        save(args.output, tab,
             dict(kind="mc", run=args.run, dm_node=args.dm_node,
                  n_cell=len(ue) - 1, free_grid=args.free_grid))
    elif args.cmd == "corr":
        import fsr_perleg as PL

        ht = PL.load_htable(args.htable)
        ns = argparse.Namespace(pt_cuts=args.pt_cuts, resol=args.resol,
                                resol_mode=args.resol_mode, h4=args.h4)
        pr = PL.make_pass(ht, ns)
        pair = FC.DATA_PAIRS if args.pair is None else tuple(args.pair)
        tab, acc = build_corr(nodes, ue, ht, run=args.run, variant=args.variant,
                              pair=pair, pr=pr)
        save(args.output, tab,
             dict(kind="corr", htable=args.htable, run=args.run,
                  variant=args.variant, pair=list(pair),
                  pt_cuts=args.pt_cuts, resol=args.resol,
                  dm_node=args.dm_node, n_cell=args.n_cell),
             acc=acc, acc_path=args.acceptance)


if __name__ == "__main__":
    main()

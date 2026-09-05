#!/usr/bin/env python3
"""Assemble the Z -> mumu channel of the unbinned CVH mass likelihood.

Inputs
------
``--pairs``   a ``cf_inmaker.py pairs`` cache built from the Z production's
              ``globalcor_*.root`` (in-maker CF exponents on the 64-point tau
              grid): keys ``z, sigma, eta, vgf, tgrid, Sms, Sio_re, Sio_im,
              Srad_re, Srad_im``.
``--kernel``  the FSR kernel npz from ``zfsr_kernel.py`` (``phik_t``,
              ``phik_re``, ``phik_im``).

Output
------
A rabbit datacard holding one ``rabbit.unbinned.MassCFTerm`` with

* physics kernel ``TabulatedLineshapeKernel(ZGammaLineshape)`` -- the Born
  Z/gamma* lineshape, POIs ``m_Z`` and ``Gamma_Z`` in MeV;
* FSR kernel ``phi_K`` from the generator record;
* resolution families ``hit`` (analytic, from ``vgf``), ``ms``, ``ioni``,
  ``rad``, each with its own scale ``k_<f>``;
* a uniform or Bernstein background on the window with a fixed or floating
  fraction;
* the **truncated** likelihood over the selection window (``norm_window``).

...and an ``.npz`` of the assembled arrays (``--dump``) so ``fit_z.py`` can
fit and project without going through the datacard.

Two windows, deliberately different
-----------------------------------
``--born-window``  the support of the *pre-FSR* lineshape the provider models.
                   The DY sample itself is generated with ``m_pre > 50`` GeV,
                   so 50 is the natural lower edge; the upper edge only has to
                   sit far enough above the selection window that FSR
                   migration into it is covered.
``--window``       the window the *observed* candidates were selected in.
                   This is what the likelihood must renormalise over.
Setting them equal would be wrong in both directions: it would forbid Born
masses that FSR moves into the window, and it would leave the observed
density unnormalised.

Usage::

    python make_z_card.py --pairs data/zpairs_smoke.npz \\
        --kernel data/zfsr_kernel_acc.npz -o data/zcard_smoke.hdf5 \\
        --dump data/zterm_smoke.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MZ_REF = 91.1876
FAMILY_ORDER = ["ms", "ioni", "rad"]
CACHE_KEYS = {"ms": ("Sms", None), "ioni": ("Sio_re", "Sio_im"),
              "rad": ("Srad_re", "Srad_im"), "del": ("Sdel", None)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pairs", required=True, help="cf_inmaker pairs cache (npz)")
    p.add_argument("--kernel", required=True, help="zfsr_kernel.py output (npz)")
    p.add_argument("-o", "--output", default=None, help="output datacard (.hdf5)")
    p.add_argument("--dump", default=None, help="also write the assembled arrays here")
    p.add_argument("--name", default="zmass", help="unbinned term name")
    p.add_argument("--channel", default="z", help="channel label")
    p.add_argument("--mref", type=float, default=MZ_REF, help="reference mass [GeV]")
    p.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0],
                   metavar=("LO", "HI"), help="observed-mass selection window")
    p.add_argument("--born-window", type=float, nargs=2, default=[50.0, 130.0],
                   metavar=("LO", "HI"), help="support of the Born lineshape")
    p.add_argument("--no-window-norm", action="store_true",
                   help="do NOT renormalise over the window (diagnostic only: "
                        "this is the biased likelihood)")
    p.add_argument("--norm-classes", type=int, default=32,
                   help="resolution classes for the truncation normalisation; "
                        "0 = one per candidate (exact, only affordable at "
                        "small n)")
    p.add_argument("--norm-nodes", type=int, default=257,
                   help="mass-grid nodes for the truncation integral")
    p.add_argument("--sigma-max", type=float, default=0.0,
                   help="drop candidates with sigma above this [GeV] (0 = keep "
                        "all); the tail of very poorly measured pairs sets the "
                        "tau range the lineshape CF must cover")
    p.add_argument("--maxn", type=int, default=0, help="use only the first N")
    p.add_argument("--del-family", action="store_true",
                   help="include the 'del' family (not part of the reference "
                        "resolution model)")
    p.add_argument("--float-krad", action="store_true",
                   help="let k_rad float freely (the reference model fixes it)")
    p.add_argument("--krad-prior", type=float, default=0.01,
                   help="Gaussian prior width pinning k_rad to 1")
    p.add_argument("--k-prior", type=float, default=0.0,
                   help="Gaussian prior width on the other resolution scales "
                        "(0 = free)")
    p.add_argument("--gz-prior", type=float, default=0.0,
                   help="Gaussian prior on Gamma_Z [MeV] (0 = free; the PDG "
                        "world average is 2.3)")
    p.add_argument("--mz-prior", type=float, default=0.0,
                   help="Gaussian prior on m_Z [MeV] (0 = free)")
    p.add_argument("--with-alpha", action="store_true",
                   help="add the momentum-scale parameter alpha. m_Z and alpha "
                        "are EXACTLY degenerate in a single-resonance fit -- "
                        "only meaningful together with another channel.")
    p.add_argument("--background", choices=["none", "uniform", "bernstein"],
                   default="uniform")
    p.add_argument("--bernstein-degree", type=int, default=2)
    p.add_argument("--fbkg", type=float, default=0.0, help="background fraction")
    p.add_argument("--float-bkg", action="store_true")
    p.add_argument("--width-scheme", choices=["fixed", "running"], default="fixed")
    p.add_argument("--nm", type=int, default=32768, help="lineshape mass grid")
    p.add_argument("--tau-max", type=float, default=40.0,
                   help="lineshape CF tabulation range [1/GeV]")
    p.add_argument("--chunk", type=int, default=32768)
    return p.parse_args(argv)


def discover_families(keys, want_del=False):
    fams = []
    for name in FAMILY_ORDER + (["del"] if want_del else []):
        re_k, im_k = CACHE_KEYS[name]
        if re_k in keys:
            fams.append((name, re_k, im_k if (im_k and im_k in keys) else None))
    return fams


def build(args, log=print):
    """Assemble everything the term needs. Returns (term, datasets, decl, info)."""
    from rabbit import unbinned
    from rabbit.lineshapes import ZGammaLineshape

    d = np.load(args.pairs, allow_pickle=True)
    k = np.load(args.kernel, allow_pickle=True)
    tgrid = np.asarray(d["tgrid"], dtype=np.float64)
    nt = len(tgrid)

    z = d["z"].astype(np.float64)
    sigma = d["sigma"].astype(np.float64)
    mgen = d["eta"].astype(np.float64)          # the gen (post-FSR) mass
    vgf = d["vgf"].astype(np.float64)
    mreco = z * sigma + mgen
    n0 = len(z)

    lo, hi = args.window
    keep = (mreco >= lo) & (mreco <= hi)
    log(f"  {n0} candidates in the pairs cache; {int(keep.sum())} with "
        f"m_obs in [{lo}, {hi}]")
    if args.sigma_max > 0:
        nb = int(keep.sum())
        keep &= sigma <= args.sigma_max
        log(f"  sigma <= {args.sigma_max} GeV drops {nb - int(keep.sum())} more")
    idx = np.flatnonzero(keep)
    if args.maxn and args.maxn < len(idx):
        idx = idx[: args.maxn]
    n = len(idx)
    sigma, mreco, vgf, mgen = sigma[idx], mreco[idx], vgf[idx], mgen[idx]
    mobs = mreco - args.mref

    fams = discover_families(set(d.files), args.del_family)
    log(f"  {n} candidates, nt = {nt}, tau grid [0, {tgrid[-1]:.4f}], "
        f"families {[f[0] for f in fams]} + hit")
    log(f"  sigma: min {sigma.min():.4f} med {np.median(sigma):.4f} "
        f"max {sigma.max():.4f} GeV -> tau needed "
        f"{tgrid[-1]/sigma.min():.2f} 1/GeV")

    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    datasets = {"sigma": sigma, "mobs": mobs, "vgf": vgf, "tgrid": tgrid}
    arrays = {}
    for name, re_k, im_k in fams:
        families.append({"name": name, "param": f"k_{name}", "kind": "tab"})
        arrays[name] = {"re": np.asarray(d[re_k])[idx]}
        datasets[f"S_re_{name}"] = arrays[name]["re"]
        if im_k:
            arrays[name]["im"] = np.asarray(d[im_k])[idx]
            datasets[f"S_im_{name}"] = arrays[name]["im"]

    phik = (np.asarray(k["phik_t"]), np.asarray(k["phik_re"]),
            np.asarray(k["phik_im"]))
    tau_needed = float(tgrid[-1] / sigma.min())
    if tau_needed > phik[0][-1]:
        raise SystemExit(
            f"the FSR kernel CF is tabulated to t = {phik[0][-1]:.2f} but the "
            f"term needs {tau_needed:.2f} 1/GeV; rebuild zfsr_kernel.py with a "
            f"larger --tmax (or use --sigma-max)"
        )
    datasets["phik_t"], datasets["phik_re"], datasets["phik_im"] = phik

    # ---- truncation normalisation classes -------------------------------
    norm = None
    if not args.no_window_norm:
        K = n if args.norm_classes <= 0 else min(args.norm_classes, n)
        if K == n:
            cls = np.arange(n)
            sig_c, vgf_c = sigma.copy(), vgf.copy()
            fam_c = {name: {c: a.astype(np.float64) for c, a in arr.items()}
                     for name, arr in arrays.items()}
        else:
            edges = np.quantile(sigma, np.linspace(0, 1, K + 1))
            cls = np.clip(np.searchsorted(edges[1:-1], sigma, "right"), 0, K - 1)
            sig_c = np.empty(K)
            vgf_c = np.empty(K)
            fam_c = {name: {c: np.empty((K, nt)) for c in arr}
                     for name, arr in arrays.items()}
            for c in range(K):
                m = cls == c
                if not m.any():          # empty class: fall back to the median
                    m = np.ones(n, bool)
                sig_c[c] = np.median(sigma[m])
                vgf_c[c] = np.mean(vgf[m])
                for name, arr in arrays.items():
                    for comp, a in arr.items():
                        fam_c[name][comp][c] = a[m].mean(axis=0)
        norm = {"sigma": sig_c, "vgf": vgf_c, "class": cls,
                "families": [dict(name=nm, **fam_c[nm]) for nm in fam_c]}
        datasets["norm_sigma"] = sig_c
        datasets["norm_vgf"] = vgf_c
        datasets["norm_class"] = cls.astype(np.int64)
        for nm, arr in fam_c.items():
            for comp, a in arr.items():
                datasets[f"S_{comp}_{nm}_norm"] = a.astype(np.float32)
        log(f"  truncation normalisation on [{lo}, {hi}] with {K} resolution "
            f"class(es), {args.norm_nodes} mass nodes")
    else:
        log("  NO window normalisation (biased likelihood; diagnostic only)")

    # ---- lineshape provider ---------------------------------------------
    t0 = time.time()
    provider = ZGammaLineshape(
        m_ref=args.mref, window=tuple(args.born_window), nm=args.nm,
        tau_max=args.tau_max, width_scheme=args.width_scheme,
    )
    need, ok = provider.check_tau_range(tgrid, sigma)
    log(f"  {provider} built in {time.time()-t0:.1f} s; tau needed {need:.2f} "
        f"1/GeV -> {'ok' if ok else 'TOO SMALL, raise --tau-max'}")
    if not ok:
        raise SystemExit("lineshape tau_max too small")

    if args.background == "none":
        background = None
    elif args.background == "uniform":
        background = unbinned.UniformBackground((lo, hi))
    else:
        background = unbinned.BernsteinBackground(
            (lo, hi), [f"bkg_c{i}" for i in range(args.bernstein_degree + 1)])

    term = unbinned.MassCFTerm(
        args.name, sigma=sigma, mobs=mobs, tgrid=tgrid,
        families=[dict(f, **arrays.get(f["name"], {})) for f in families],
        vgf=vgf, phik=phik,
        kernel=unbinned.TabulatedLineshapeKernel(provider=provider),
        background=background, m_ref=args.mref,
        scale_param="alpha" if args.with_alpha else None,
        bkg_frac_param="f_bkg" if args.float_bkg else None,
        bkg_frac=args.fbkg,
        norm_window=None if args.no_window_norm else (lo, hi),
        norm_nodes=args.norm_nodes, norm=norm,
        chunk=args.chunk, channel=args.channel,
    )

    decl = {}
    decl.update(provider.param_declarations(
        mz_prior=args.mz_prior or None, gz_prior=args.gz_prior or None))
    for f in families:
        p = f["param"]
        if p == "k_rad" and not args.float_krad:
            decl[p] = (1.0, args.krad_prior, 1.0, 0)
        else:
            decl[p] = (1.0, args.k_prior or np.nan, 1.0, 0)
    if args.with_alpha:
        decl["alpha"] = (0.0, np.nan, 0.0, 1)
    if args.float_bkg:
        decl["f_bkg"] = (args.fbkg / unbinned.FBKG_UNIT, np.nan,
                         args.fbkg / unbinned.FBKG_UNIT, 0)
    if args.background == "bernstein":
        for i in range(args.bernstein_degree + 1):
            decl[f"bkg_c{i}"] = (float(np.log(np.expm1(1.0))), np.nan,
                                 float(np.log(np.expm1(1.0))), 0)
    decl = unbinned.declare_params(term, decl)

    log(f"  parameters {list(term.param_names)}")
    log(f"    defaults {np.round(decl['param_defaults'], 4).tolist()}")
    log(f"    priors   {np.round(decl['param_prior_sigmas'], 5).tolist()}")
    log(f"    POIs     {[p for p, f in zip(term.param_names, decl['param_is_poi']) if f]}")

    info = {"n": n, "n_cache": n0, "window": [lo, hi],
            "born_window": list(args.born_window), "mreco": mreco, "mgen": mgen,
            "provider_config": provider.config()}
    return term, datasets, decl, info


def main():
    args = parse_args()
    print(f"[make_z_card] {args.pairs}")
    term, datasets, decl, info = build(args)

    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)) or ".",
                    exist_ok=True)
        np.savez_compressed(
            args.dump, config=json.dumps(term.config()),
            params=np.array(list(term.param_names)), mreco=info["mreco"],
            mgen=info["mgen"], argv=np.array(sys.argv[1:], dtype=object),
            **{k: v for k, v in datasets.items()}, **decl)
        print(f"  -> {args.dump}")

    if args.output:
        from rabbit import tensorwriter
        writer = tensorwriter.TensorWriter()
        writer.add_dummy_channel(name=f"{args.channel}_dummy")
        writer.add_unbinned_term(args.name, term.config(), term.param_names,
                                 datasets, **decl)
        out = os.path.abspath(args.output)
        name = os.path.basename(out)
        if name.endswith(".hdf5"):
            name = name[:-5]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        t0 = time.time()
        writer.write(outfolder=os.path.dirname(out) or ".", outfilename=name)
        print(f"  -> {os.path.join(os.path.dirname(out), name)}.hdf5 "
              f"in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()

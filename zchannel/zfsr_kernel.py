#!/usr/bin/env python3
"""Build the Z final-state-radiation kernel for the unbinned CVH mass likelihood.

``rabbit.lineshapes.ZGammaLineshape`` supplies the characteristic function of
the **pre-FSR (Born)** dimuon mass.  What the tracker measures is the
**post-FSR bare** pair, so the observed density is

    p(m_obs) = lineshape (x) FSR (x) resolution ,

and the object this script produces is the middle factor: the empirical
distribution of

    dm = m_postFSR - m_preFSR

and its characteristic function ``phi_K(t) = <exp(i t dm)>``, tabulated on
``[0, tmax]``.  ``rabbit.unbinned.MassCFTerm`` takes that tabulation as
``phik = (phik_t, phik_re, phik_im)`` and interpolates it onto ``tgrid/sigma_i``
per candidate -- exactly the path the J/psi channel uses, where the kernel is
the empirical distribution of ``m_gen,postFSR - M_Jpsi``.

Input: the npz written by ``dump_gen_fsr.py`` (see its docstring for the
generator-record definitions of pre- and post-FSR).

Two things about the Z kernel that do **not** hold for the J/psi and that the
``--report`` output quantifies:

* it is *not* a small perturbation -- roughly 58 % of events do not radiate at
  all (``|dm| < 1 MeV``) and the remaining 42 % have a tail reaching tens of
  GeV, so ``phi_K(t)`` does not decay to zero but plateaus at the
  no-radiation probability;
* it is *multiplicative*, not additive: ``dm/m_pre`` is nearly independent of
  ``m_pre`` while ``dm`` is not.  A single additive kernel is therefore an
  approximation; see the report and README for what it costs.

Usage::

    python zfsr_kernel.py -i data/genall.npz -o data/zfsr_kernel.npz \\
        --acc-pt 26 --acc-eta 2.4 --report
"""

import argparse
import os

import numpy as np

MZ_REF = 91.1876


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="npz file(s) from dump_gen_fsr.py")
    p.add_argument("-o", "--output", default=None, help="output kernel npz")
    p.add_argument("--acc-pt", type=float, default=0.0,
                   help="gen acceptance: min pT of BOTH post-FSR muons [GeV]")
    p.add_argument("--acc-eta", type=float, default=0.0,
                   help="gen acceptance: max |eta| of both post-FSR muons (0 = off)")
    p.add_argument("--mass-window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="restrict to m_postFSR in [LO, HI]. OFF by default: the "
                        "likelihood's own truncation normalisation covers the "
                        "window, and applying it here as well double-counts it.")
    p.add_argument("--pre-window", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="restrict to m_preFSR in [LO, HI]")
    p.add_argument("--variant", choices=["bare", "dressed"], default="bare",
                   help="post-FSR definition (the likelihood uses bare muons)")
    p.add_argument("--mode", choices=["additive", "multiplicative"],
                   default="additive",
                   help="'additive': dm samples as-is. 'multiplicative': dm "
                        "rescaled to --m-scale, i.e. (dm/m_pre) * m_scale, which "
                        "is the best single additive kernel if FSR is a pure "
                        "rescaling of the mass.")
    p.add_argument("--m-scale", type=float, default=MZ_REF,
                   help="reference mass for --mode multiplicative")
    p.add_argument("--tmax", type=float, default=10.0,
                   help="upper end of the CF tabulation [1/GeV]; must cover "
                        "max(tgrid)/min(sigma) of the term")
    p.add_argument("--npoints", type=int, default=32769,
                   help="CF tabulation points on [0, tmax]")
    p.add_argument("--report", action="store_true", help="print the diagnostics")
    p.add_argument("--max-samples", type=int, default=0,
                   help="cap the number of dm samples used for the CF")
    return p.parse_args()


def load(paths):
    d = {}
    for f in paths:
        z = np.load(f)
        for k in z.files:
            d.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in d.items()}


def selection(d, args):
    """Boolean mask implementing the gen-level acceptance / window options."""
    sel = np.ones(len(d["m_pre"]), bool)
    if args.acc_pt > 0:
        sel &= (d["pt1"] > args.acc_pt) & (d["pt2"] > args.acc_pt)
    if args.acc_eta > 0:
        sel &= (np.abs(d["eta1"]) < args.acc_eta) & (np.abs(d["eta2"]) < args.acc_eta)
    if args.mass_window:
        post = d["m_dress"] if args.variant == "dressed" else d["m_post"]
        sel &= (post >= args.mass_window[0]) & (post <= args.mass_window[1])
    if args.pre_window:
        sel &= (d["m_pre"] >= args.pre_window[0]) & (d["m_pre"] <= args.pre_window[1])
    return sel


def cf_table(dm, tmax, npoints, block=2048):
    """Empirical CF ``<exp(i t dm)>`` on a uniform grid ``[0, tmax]``.

    Blocked over the ``t`` axis only, so the mean over samples -- and hence the
    result -- is independent of the block size.
    """
    tabs = np.linspace(0.0, tmax, npoints)
    out = np.empty(npoints, np.complex128)
    for i in range(0, npoints, block):
        out[i:i + block] = np.mean(np.exp(1j * np.outer(tabs[i:i + block], dm)),
                                   axis=1)
    return tabs, out


def quantile_row(x, qs=(0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)):
    return np.percentile(x, qs)


def report(d, sel, dm, args):
    mp = d["m_pre"]
    post = d["m_dress"] if args.variant == "dressed" else d["m_post"]
    n = int(sel.sum())
    print(f"\n=== FSR kernel, {args.variant} muons, {args.mode} ===")
    print(f"  {len(mp)} gen events, {n} selected ({100*n/len(mp):.1f} %)")
    print(f"  acceptance: pT > {args.acc_pt} GeV, |eta| < {args.acc_eta or 'inf'}; "
          f"m_post window {args.mass_window or 'none'}; "
          f"m_pre window {args.pre_window or 'none'}")
    prad = float(np.mean(np.abs(dm) > 1e-3))
    print(f"\n  P(no radiation, |dm| < 1 MeV) = {1-prad:.4f}   "
          f"P(radiating) = {prad:.4f}")
    print(f"  mean   = {dm.mean()*1e3:+.1f} MeV      median = {np.median(dm)*1e3:+.3f} MeV")
    print(f"  rms    = {dm.std():.4f} GeV        mean|dm| = {np.abs(dm).mean():.4f} GeV")
    qs = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    print("  quantiles [GeV] " + "  ".join(f"{q}%:{v:+.4f}" for q, v in
                                           zip(qs, quantile_row(dm, qs))))
    print("  tail fractions:  " + "   ".join(
        f"P(|dm|>{t:g})={np.mean(np.abs(dm)>t):.4f}" for t in
        (0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)))

    print("\n  --- kernel factorisation: p(dm | m_pre) binned in m_pre ---")
    print(f"  {'m_pre bin':>13s} {'n':>7s} {'P(rad)':>7s} {'mean dm':>10s} "
          f"{'rms dm':>9s} {'q5 dm':>10s} | {'mean r':>9s} {'rms r':>8s} {'q5 r':>9s}"
          "     (r = dm/m_pre)")
    for a, b in ((60, 80), (80, 86), (86, 89), (89, 91), (91, 93), (93, 96),
                 (96, 100), (100, 110), (110, 150)):
        m = sel & (mp >= a) & (mp < b)
        if m.sum() < 100:
            continue
        x = (post - mp)[m]
        r = x / mp[m]
        print(f"  {a:5.0f}-{b:<7.0f} {m.sum():7d} "
              f"{np.mean(np.abs(x)>1e-3):7.4f} {x.mean():+10.4f} {x.std():9.4f} "
              f"{np.percentile(x,5):+10.4f} | {r.mean():+9.5f} {r.std():8.5f} "
              f"{np.percentile(r,5):+9.5f}")
    print("  -> dm varies strongly with m_pre; dm/m_pre does not. The FSR "
          "kernel is multiplicative.")

    print("\n  --- bare vs dressed (same selection) ---")
    for lab, x in (("bare   m_post-m_pre", (d["m_post"] - mp)[sel]),
                   ("dressed m_dress-m_pre", (d["m_dress"] - mp)[sel])):
        print(f"  {lab:24s} mean {x.mean()*1e3:+8.1f} MeV  rms {x.std():7.4f} GeV "
              f" P(|dm|>1 GeV) {np.mean(np.abs(x)>1):.4f}")
    print(f"  photons dressed per event: {d['nph'][sel].mean():.3f}, "
          f"mean energy {d['eph'][sel].mean():.3f} GeV")


def main():
    args = parse_args()
    d = load(args.input)
    sel = selection(d, args)
    post = d["m_dress"] if args.variant == "dressed" else d["m_post"]
    dm_raw = (post - d["m_pre"])[sel]
    if args.mode == "multiplicative":
        dm = dm_raw / d["m_pre"][sel] * args.m_scale
    else:
        dm = dm_raw
    if args.max_samples and args.max_samples < len(dm):
        dm = dm[: args.max_samples]

    if args.report:
        report(d, sel, dm_raw, args)

    tabs, phik = cf_table(dm, args.tmax, args.npoints)
    print(f"\n  CF tabulated on [0, {args.tmax}] with {args.npoints} points "
          f"(dt = {tabs[1]:.3e} 1/GeV) from {len(dm)} samples")
    print(f"  phi_K(0) = {phik[0].real:.12f}{phik[0].imag:+.2e}i   "
          f"|phi_K(tmax)| = {abs(phik[-1]):.4f}   "
          f"(plateau ~ P(no radiation) = {np.mean(np.abs(dm)<1e-3):.4f})")
    print(f"  statistical noise on phi_K at large t ~ 1/sqrt(N) = "
          f"{1/np.sqrt(len(dm)):.2e}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                    exist_ok=True)
        np.savez_compressed(
            args.output, dm=dm, phik_t=tabs, phik_re=phik.real.copy(),
            phik_im=phik.imag.copy(), m_pre=d["m_pre"][sel], m_post=post[sel],
            mode=args.mode, m_scale=args.m_scale, variant=args.variant,
            acc_pt=args.acc_pt, acc_eta=args.acc_eta,
            mass_window=np.array(args.mass_window if args.mass_window
                                 else [np.nan, np.nan]),
            nsel=len(dm))
        print(f"  -> {args.output}")


if __name__ == "__main__":
    main()

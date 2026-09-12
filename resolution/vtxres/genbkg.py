#!/usr/bin/env python3
"""Is the large-vertex-residual tail BACKGROUND?  Gen truth says so or it does
not, and this is what asks it.

Section 12.9 of `STATE.md` showed that the DY vertex-residual tail
(`P(|z_v| > 5)` = 1.86 % free / 0.165 % constrained inclusive, against 0.107 %
on the same candidate in both regimes) is carried by candidates that sit in
MULTI-CANDIDATE events.  That was a COMBINATORIAL argument -- a counting
argument over event multiplicity -- and David's question is the physics one it
does not answer: are those candidates background, by GEN TRUTH?  If they are,
a cut on the vertex residual is a background rejection tool and not only a
tail-trimming convenience.

THE GEN-MATCH CRITERION (the maker's, unchanged, `Mu*gen_dr`): the closest
status-1 gen muon of the SAME CHARGE within dR < 0.1 of the leg's fitted
momentum.  What the maker now adds (`Mu*gen_idx`, `Mu*gen_motherPdgId`,
`Jpsigen_sameDecay`) is the IDENTITY and the ANCESTRY of that particle, which
is what the classification below needs: a dR match alone cannot tell one muon
reconstructed twice from two different muons.

THE CLASSES (in the order they are tested; every candidate lands in exactly
one):

  unmatched    at least one leg has no gen muon of the right charge within
               dR < 0.1 -- a fake track, a hadron, a pileup muon or a muon
               below the MiniAOD gen-pruning threshold
  otherdecay   both legs matched, but to muons whose first non-muon ancestor
               is a DIFFERENT particle: a real muon paired with a muon from
               somewhere else in the event
  nonres       both legs matched to one decay whose mother is not in
               --resonances (kept separate so a surprise is visible rather
               than absorbed)
  dup          both legs matched to the two daughters of the resonance, but
               ANOTHER candidate of the same event matched the SAME two gen
               particles: one muon (or both) reconstructed twice, i.e. the
               combinatorial pairing.  The candidate with the smaller
               dR(+) + dR(-) keeps the `signal` label and the rest are `dup`.
               THAT TIE-BREAK USES TRUTH and is stated as such: it defines
               which of two reconstructions of one decay is "the" one, and
               nothing downstream may use it as a selection.
  signal       the rest: the two daughters of the resonance, reconstructed
               once.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 genbkg.py --npz <dy_vtx.npz> --tag dy_on --classes --cuts --mass
  python3 genbkg.py --npz <dy_vtx.npz> --tag dy_on --density
  python3 genbkg.py --raw '<prod>/task_*/globalcor_*.root' --tag dy_on --hits
"""
import argparse, datetime, glob, os, sys
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)

CLASSES = ("signal", "dup", "otherdecay", "nonres", "unmatched")
CLSCOL = {"signal": "#1f77b4", "dup": "#d62728", "otherdecay": "#ff7f0e",
          "nonres": "#9467bd", "unmatched": "#2ca02c"}
# what counts as "the resonance the two legs came from"
RESONANCES = (23, 22, 443, 100443, 553, 100553, 200553)


def binom(k, n):
    """fraction with its binomial uncertainty; (nan, nan) for an empty set."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    return p, float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def fmtpm(k, n, scale=1.0, unit=""):
    p, e = binom(k, n)
    if not np.isfinite(p):
        return "--"
    return f"{scale*p:.4f} +- {scale*e:.4f}{unit}"


def classify(d, resonances=RESONANCES):
    """-> (cls int array, names, aux dict).  See the module docstring."""
    n = len(d["vtxz"])
    need = ("genidx_plus", "genidx_minus", "gensamedecay", "genmoth_plus")
    missing = [k for k in need if k not in d]
    if missing:
        raise SystemExit(
            f"the npz has no gen provenance ({', '.join(missing)}): it was "
            f"extracted from a production written before the maker exported "
            f"it.  Re-run the production, or extract from one that has it.")
    ip = np.asarray(d["genidx_plus"], np.int64)
    im = np.asarray(d["genidx_minus"], np.int64)
    same = np.asarray(d["gensamedecay"], bool)
    moth = np.abs(np.asarray(d["genmoth_plus"], np.int64))
    drp = np.asarray(d.get("gendr_plus", np.zeros(n)), float)
    drm = np.asarray(d.get("gendr_minus", np.zeros(n)), float)
    drsum = np.where(drp > 0, drp, 0.0) + np.where(drm > 0, drm, 0.0)

    cls = np.full(n, -1, np.int8)
    idxof = {c: i for i, c in enumerate(CLASSES)}
    unmatched = (ip < 0) | (im < 0)
    cls[unmatched] = idxof["unmatched"]
    rest = cls < 0
    other = rest & ~same
    cls[other] = idxof["otherdecay"]
    rest = cls < 0
    nonres = rest & ~np.isin(moth, resonances)
    cls[nonres] = idxof["nonres"]
    rest = np.flatnonzero(cls < 0)

    # the duplicate test: same event, same PAIR of gen particles
    run = np.asarray(d["run"], np.int64)
    lumi = np.asarray(d["lumi"], np.int64)
    evt = np.asarray(d["event"], np.int64)
    best = {}
    for i in rest:
        k = (run[i], lumi[i], evt[i], ip[i], im[i])
        if k not in best or drsum[i] < drsum[best[k]]:
            best[k] = i
    keep = set(best.values())
    for i in rest:
        cls[i] = idxof["signal"] if i in keep else idxof["dup"]

    # event multiplicity of SELECTED candidates (the section-12.9 split)
    cnt = Counter(zip(run.tolist(), lumi.tolist(), evt.tolist()))
    mult = np.fromiter((cnt[(int(a), int(b), int(c))]
                        for a, b, c in zip(run, lumi, evt)), np.int64, n)
    return cls, CLASSES, {"mult": mult, "drsum": drsum, "moth": moth}


def load_npz(fn, max_chi2_ndof=0.0):
    """`vtxterm.load`, so the CF density can be evaluated on the same dict --
    it carries every 1-D per-candidate array through, the gen provenance
    included.  `max_chi2_ndof = 0` because the extraction has already applied
    whatever chi2 cut the npz was made with; passing it again here would cut
    twice and silently.""" 
    import vtxterm as VT
    return VT.load(fn, max_chi2_ndof=max_chi2_ndof)


# --------------------------------------------------------------- classes ---
def report_classes(d, cls, aux, thresholds=(3.0, 4.0, 5.0, 10.0)):
    n = len(cls)
    z = np.abs(np.asarray(d["vtxz"], float))
    logger.info(f"=== gen classes, {n} candidates")
    hdr = (f"  {'class':12s} {'N':>7s} {'frac':>18s}  " +
           "  ".join(f"P(|z|>{t:g})".rjust(20) for t in thresholds))
    print(hdr)
    for ic, c in enumerate(CLASSES):
        m = cls == ic
        if not m.any():
            print(f"  {c:12s} {0:7d}")
            continue
        row = f"  {c:12s} {int(m.sum()):7d} {fmtpm(int(m.sum()), n):>18s}  "
        row += "  ".join(fmtpm(int((z[m] > t).sum()), int(m.sum())).rjust(20)
                         for t in thresholds)
        print(row)
    sig = cls == CLASSES.index("signal")
    print(f"  {'SIGNAL':12s} {int(sig.sum()):7d} {fmtpm(int(sig.sum()), n):>18s}")
    print(f"  {'BACKGROUND':12s} {int((~sig).sum()):7d} "
          f"{fmtpm(int((~sig).sum()), n):>18s}")
    # the direct answer: composition of the tail
    for t in thresholds:
        m = z > t
        nt = int(m.sum())
        if nt == 0:
            print(f"  |z_v| > {t:g}: EMPTY")
            continue
        comp = "  ".join(f"{c} {int((cls[m] == ic).sum())}"
                         for ic, c in enumerate(CLASSES))
        print(f"  |z_v| > {t:g}: N {nt}  background fraction "
              f"{fmtpm(int((cls[m] != CLASSES.index('signal')).sum()), nt)}"
              f"   [{comp}]")
    # mother spectrum, so a surprise is visible
    moth = aux["moth"]
    print("  first non-muon ancestor |pdgId| of the plus leg (top 8): " +
          ", ".join(f"{p}:{c}" for p, c in Counter(moth.tolist()).most_common(8)))
    # is the residual signal tail the few-hit legs?
    if "nvalid_plus" in d and "nvalid_minus" in d:
        weak = np.minimum(np.asarray(d["nvalid_plus"], np.int64),
                          np.asarray(d["nvalid_minus"], np.int64))
        sig = cls == CLASSES.index("signal")
        print("  SIGNAL only, by weaker-leg valid hits: " + "   ".join(
            f"<= {k}: N {int((sig & (weak <= k)).sum())} "
            f"P(|z|>5) {fmtpm(int((sig & (weak <= k) & (z > 5)).sum()), int((sig & (weak <= k)).sum()))}"
            for k in (5, 8)) +
            f"   > 8: N {int((sig & (weak > 8)).sum())} P(|z|>5) "
            f"{fmtpm(int((sig & (weak > 8) & (z > 5)).sum()), int((sig & (weak > 8)).sum()))}")
    # the OTHER half of the match: how well the reco momentum agrees with the
    # matched gen muon.  dR alone can match a badly reconstructed track.
    if "pt_plus" in d and "genpt_plus" in d:
        print(f"  {'class':12s} {'dR(+) med/p99':>20s} {'dR(-) med/p99':>20s} "
              f"{'pT+/genpT+ med/p5/p95':>30s} {'pT-/genpT- med/p5/p95':>30s}")
        for ic, c in enumerate(CLASSES):
            m = cls == ic
            if not m.any():
                continue
            row = f"  {c:12s} "
            for leg in ("plus", "minus"):
                dr = np.asarray(d.get(f"gendr_{leg}", np.zeros(n)), float)[m]
                dr = dr[dr >= 0]
                row += (f"{np.median(dr):.5f}/{np.percentile(dr,99):.5f}".rjust(20)
                        + " " if dr.size else "--".rjust(20) + " ")
            for leg in ("plus", "minus"):
                g = np.asarray(d[f"genpt_{leg}"], float)[m]
                r = np.asarray(d[f"pt_{leg}"], float)[m]
                ok = g > 0
                if ok.sum() == 0:
                    row += "--".rjust(30) + " "
                    continue
                x = r[ok] / g[ok]
                row += (f"{np.median(x):.4f}/{np.percentile(x,5):.4f}/"
                        f"{np.percentile(x,95):.4f}").rjust(30) + " "
            print(row)


# ------------------------------------------------------------------ cuts ---
def report_cuts(d, cls, aux, thresholds=(3.0, 4.0, 5.0, 10.0)):
    z = np.abs(np.asarray(d["vtxz"], float))
    sig = cls == CLASSES.index("signal")
    mult = aux["mult"]
    for lab, sel in (("ALL events", np.ones(len(z), bool)),
                     ("1 candidate/event", mult == 1),
                     (">1 candidate/event", mult > 1)):
        ns, nb = int((sig & sel).sum()), int((~sig & sel).sum())
        logger.info(f"=== cut |z_v| < t, {lab}: {ns} signal, {nb} background")
        print(f"  {'t':>5s} {'signal eff':>22s} {'bkg rejection':>22s} "
              f"{'purity before':>16s} {'purity after':>16s}")
        p0 = fmtpm(ns, ns + nb)
        for t in thresholds:
            ks = int((sig & sel & (z < t)).sum())
            kb = int((~sig & sel & (z < t)).sum())
            rej, erej = binom(nb - kb, nb)
            print(f"  {t:5g} {fmtpm(ks, ns):>22s} "
                  f"{(f'{rej:.4f} +- {erej:.4f}' if np.isfinite(rej) else '--'):>22s} "
                  f"{p0:>16s} {fmtpm(ks, ks + kb):>16s}")


def report_dup(d, cls, aux):
    """For the candidates that are two reconstructions of ONE decay: does the
    vertex residual pick the better one?  The gen tie-break (smaller
    dR(+) + dR(-)) is truth; |z_v| and chi2/ndof are what an analysis has."""
    idup = CLASSES.index("dup")
    isig = CLASSES.index("signal")
    ip = np.asarray(d["genidx_plus"], np.int64)
    im = np.asarray(d["genidx_minus"], np.int64)
    run, lumi, evt = (np.asarray(d[k], np.int64) for k in ("run", "lumi", "event"))
    z = np.abs(np.asarray(d["vtxz"], float))
    c2 = np.asarray(d.get("chi2ndof", np.zeros(len(z))), float)
    groups = defaultdict(list)
    for i in np.flatnonzero((cls == idup) | (cls == isig)):
        groups[(run[i], lumi[i], evt[i], ip[i], im[i])].append(i)
    pairs = [v for v in groups.values() if len(v) > 1]
    logger.info(f"=== the duplicate pairings: {len(pairs)} gen decays "
                f"reconstructed more than once "
                f"({sum(len(v) for v in pairs)} candidates)")
    if not pairs:
        return
    nz, nc, nties = 0, 0, 0
    for v in pairs:
        truth = min(v, key=lambda i: (cls[i] != isig, i))
        byz = min(v, key=lambda i: z[i])
        byc = min(v, key=lambda i: c2[i])
        nz += int(byz == truth)
        nc += int(byc == truth)
        nties += 1
    print(f"  the gen-preferred candidate is also the one with the smallest "
          f"|z_v|:      {fmtpm(nz, nties)}")
    print(f"  the gen-preferred candidate is also the one with the smallest "
          f"chi2/ndof: {fmtpm(nc, nties)}")
    zt = z[cls == idup]
    zs = z[cls == isig]
    print(f"  |z_v| of the {zt.size} duplicates: median {np.median(zt):.3f}, "
          f"p90 {np.percentile(zt, 90):.3f}, "
          f"P(>5) {fmtpm(int((zt > 5).sum()), zt.size)}"
          f"   against the {zs.size} signal: median {np.median(zs):.3f}, "
          f"P(>5) {fmtpm(int((zs > 5).sum()), zs.size)}")


def report_hitcut(d, cls, aux):
    """The OTHER cut: a minimum on the weaker leg's valid hits.  Same figures
    of merit as the |z_v| cut, plus the overlap -- do the two remove the same
    candidates or different ones?"""
    if "nvalid_plus" not in d:
        logger.warning("no per-leg hit counts in the npz: skipping")
        return
    weak = np.minimum(np.asarray(d["nvalid_plus"], np.int64),
                      np.asarray(d["nvalid_minus"], np.int64))
    pair = (np.asarray(d["nvalid_plus"], np.int64)
            + np.asarray(d["nvalid_minus"], np.int64))
    z = np.abs(np.asarray(d["vtxz"], float))
    sig = cls == CLASSES.index("signal")
    ns, nb = int(sig.sum()), int((~sig).sum())
    logger.info(f"=== cut on the WEAKER leg's valid hits: {ns} signal, "
                f"{nb} background")
    print(f"  {'minLegHits':>11s} {'removed':>8s} {'signal eff':>22s} "
          f"{'bkg rejection':>22s} {'purity after':>16s} "
          f"{'bkg in removed':>16s}")
    for k in (4, 5, 6, 7, 8, 9, 10):
        keep = weak >= k
        ks, kb = int((sig & keep).sum()), int((~sig & keep).sum())
        rej, erej = binom(nb - kb, nb)
        nrem = int((~keep).sum())
        print(f"  {k:11d} {nrem:8d} {fmtpm(ks, ns):>22s} "
              f"{(f'{rej:.4f} +- {erej:.4f}'):>22s} {fmtpm(ks, ks + kb):>16s} "
              f"{fmtpm(int((~sig & ~keep).sum()), max(nrem, 1)):>16s}")
    print(f"  class of the candidates a minLegHits cut removes:")
    for k in (4, 6, 8):
        m = weak < k
        print(f"    < {k}: N {int(m.sum()):5d}  " + "  ".join(
            f"{c} {int((cls[m] == ic).sum())}" for ic, c in enumerate(CLASSES)
            if (cls[m] == ic).any()))
    # the overlap with the |z_v| cut
    logger.info("=== do the two cuts remove the SAME candidates?")
    for k in (4, 6, 8):
        a = z > 5
        b = weak < k
        print(f"  |z_v| > 5: {int(a.sum())}  weaker leg < {k}: {int(b.sum())}"
              f"  BOTH: {int((a & b).sum())}  "
              f"|z_v| only: {int((a & ~b).sum())}  hits only: {int((b & ~a).sum())}")
    print(f"  pair-hit cut (minPairHits): N(pair < 10) {int((pair < 10).sum())}, "
          f"N(pair < 11) {int((pair < 11).sum())}, "
          f"N(pair < 16) {int((pair < 16).sum())}")
    logger.info("=== the two together (drop if |z_v| >= t OR weaker leg < k)")
    print(f"  {'t':>5s} {'k':>4s} {'removed':>8s} {'signal eff':>22s} "
          f"{'bkg rejection':>22s} {'purity after':>16s}")
    for t in (3.0, 5.0):
        for k in (0, 6, 7, 8):
            keep = (z < t) & (weak >= k)
            ks, kb = int((sig & keep).sum()), int((~sig & keep).sum())
            rej, erej = binom(nb - kb, nb)
            print(f"  {t:5g} {k:4d} {int((~keep).sum()):8d} {fmtpm(ks, ns):>22s} "
                  f"{(f'{rej:.4f} +- {erej:.4f}'):>22s} "
                  f"{fmtpm(ks, ks + kb):>16s}")


# ------------------------------------------------------------------ mass ---
def report_mass(d, cls, mwin=15.0, mref=91.1876):
    if "mass_unc" not in d:
        logger.warning("no Jpsi_mass_unc in the npz (free regime): skipping")
        return
    mc = np.asarray(d["mass"], float)
    mu = np.asarray(d["mass_unc"], float)
    sm = np.asarray(d["sigmamass"], float)
    z = np.abs(np.asarray(d["vtxz"], float))
    sig = cls == CLASSES.index("signal")
    dz = np.where(sm > 0, (mc - mu) / np.maximum(sm, 1e-12), np.nan)
    logger.info("=== the mass side: what the vertex constraint does to the "
                f"mass, in units of sigma_m ({mref:g} +- {mwin:g} GeV window)")
    print(f"  {'sample':28s} {'N':>7s} {'median |dm|/sig':>18s} {'p90':>10s} "
          f"{'max':>10s} {'mean dm (MeV)':>16s}")
    for lab, m in (("signal", sig), ("background", ~sig),
                   ("background, |z_v| > 5", ~sig & (z > 5)),
                   ("signal, |z_v| > 5", sig & (z > 5)),
                   ("all, |z_v| < 3", z < 3)):
        x = dz[m & np.isfinite(dz)]
        if x.size == 0:
            print(f"  {lab:28s} {0:7d}")
            continue
        print(f"  {lab:28s} {x.size:7d} {np.median(np.abs(x)):18.4f} "
              f"{np.percentile(np.abs(x), 90):10.4f} {np.abs(x).max():10.4f} "
              f"{1e3*np.mean((mc-mu)[m & np.isfinite(dz)]):16.3f}")
    logger.info(f"=== does the constraint pull BACKGROUND into the mass window?")
    for lab, m in (("background", ~sig), ("signal", sig)):
        inu = np.abs(mu - mref) < mwin
        inc = np.abs(mc - mref) < mwin
        n_in_unc = int((m & inu).sum())
        n_in_con = int((m & inc).sum())
        moved_in = int((m & ~inu & inc).sum())
        moved_out = int((m & inu & ~inc).sum())
        print(f"  {lab:12s} N {int(m.sum()):6d}   in window UNCONSTRAINED "
              f"{n_in_unc:6d}   CONSTRAINED {n_in_con:6d}   "
              f"moved IN {moved_in:4d}   moved OUT {moved_out:4d}   "
              f"net {n_in_con - n_in_unc:+5d}")
    # inside the window, how far the constraint moves each class
    inw = np.abs(mc - mref) < mwin
    for lab, m in (("signal, in window", sig & inw),
                   ("background, in window", ~sig & inw)):
        x = (mc - mu)[m & np.isfinite(dz)]
        y = dz[m & np.isfinite(dz)]
        if x.size == 0:
            continue
        print(f"  {lab:24s} N {x.size:6d}  mean(m_c - m_unc) "
              f"{1e3*np.mean(x):+8.2f} MeV  rms {1e3*np.std(x):8.2f} MeV  "
              f"median |dm|/sigma_m {np.median(np.abs(y)):.4f}")


# --------------------------------------------------------------- density ---
def plot_density(d, cls, outdir, tag, arms=("cf",), zrange=(-6.0, 6.0),
                 nbins=60, upsample=1, minn=90):
    import matplotlib.pyplot as plt
    import mplhep as hep
    import pubhtml
    import ratiopanel
    import vtxterm as VT
    hep.style.use(hep.style.ROOT)

    zlo, zhi = zrange
    edges = np.linspace(zlo, zhi, nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fine = np.unique(np.concatenate([edges, ctr]))
    z = np.asarray(d["vtxz"], float)
    sig = cls == CLASSES.index("signal")
    groups = [("signal", sig), ("background", ~sig)]
    groups += [(c, cls == ic) for ic, c in enumerate(CLASSES)
               if ic != CLASSES.index("signal") and (cls == ic).sum() >= minn]
    for lab, m in groups:
        rows = np.flatnonzero(m)
        if rows.size < 20:
            logger.warning(f"{lab}: {rows.size} candidates, no density drawn")
            continue
        zc = np.clip(z[rows], zlo, zhi)
        h, _ = np.histogram(zc, bins=edges)
        dens = h / (rows.size * np.diff(edges))
        err = np.where(h > 0, dens / np.sqrt(np.maximum(h, 1)), np.nan)
        model = VT.mean_density(d, arms[0], rows, fine, upsample=upsample)
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.0, 7.0))
        ax.errorbar(ctr, dens, yerr=err, fmt="ko", ms=3.2, lw=1.0,
                    label=f"{lab} ({rows.size} candidates)", zorder=5)
        ax.plot(fine, model, "-", color=CLSCOL.get(lab, "#d62728"), lw=1.8,
                label="CF (full non-Gaussian)")
        ax.set_yscale("log")
        pos = dens[dens > 0]
        if pos.size:
            ax.set_ylim(max(1e-6, pos.min() * 0.3), dens.max() * 3.0)
        ax.set_ylabel("probability density")
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(f"{tag}, {lab}: the vertex-constraint residual pull "
                     r"$z_v = r_v/\sigma_v$", fontsize=12)
        mref = ratiopanel.bin_average(np.interp(edges[:-1], fine, model),
                                      np.interp(ctr, fine, model),
                                      np.interp(edges[1:], fine, model))
        r = dens / np.where(mref > 0, mref, np.nan)
        re_ = err / np.where(mref > 0, mref, np.nan)
        rax.errorbar(ctr, r, yerr=re_, fmt="ko", ms=3.0, lw=1.0)
        rax.axhline(1.0, color="k", lw=0.8)
        rax.set_yscale("log")
        rax.set_ylim(0.1, 30.0)
        rax.set_ylabel("data / CF", fontsize=10)
        rax.set_xlabel(r"$z_v = r_v/\sigma_v$")
        fn = os.path.join(outdir, f"density_{tag}_{lab.replace(' ', '')}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
        # the tail table against the model
        zg = np.linspace(-40, 40, 3201)
        dm = VT.mean_density(d, arms[0], rows, zg, upsample=upsample)
        parts = []
        for t in (3.0, 4.0, 5.0):
            fd, _ = binom(int((np.abs(z[rows]) > t).sum()), rows.size)
            lo, hi = zg <= -t, zg >= t
            fm = float(np.trapezoid(dm[lo], zg[lo]) + np.trapezoid(dm[hi], zg[hi]))
            parts.append(f"{t:g}: data {fd:.5f} CF {fm:.5f} "
                         f"ratio {fd/max(fm,1e-12):.2f}")
        logger.info(f"  {lab}: " + " | ".join(parts))


def plot_cuts(d, cls, aux, outdir, tag):
    """Two panels the class table cannot show at a glance: what the |z_v| cut
    buys (efficiency against rejection, both against gen truth) and what the
    tail is MADE of as a function of |z_v|."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    import pubhtml
    hep.style.use(hep.style.ROOT)

    z = np.abs(np.asarray(d["vtxz"], float))
    sig = cls == CLASSES.index("signal")
    mult = aux["mult"]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ts = np.concatenate([np.linspace(0.5, 10, 96), np.linspace(10, 60, 51)])
    for lab, m, col in (("all events", np.ones(len(z), bool), "#1f77b4"),
                        ("1 candidate/event", mult == 1, "#2ca02c"),
                        (">1 candidate/event", mult > 1, "#d62728")):
        ns, nb = int((sig & m).sum()), int((~sig & m).sum())
        if ns == 0 or nb == 0:
            continue
        eff = np.array([ (sig & m & (z < t)).sum() / ns for t in ts ])
        rej = np.array([ 1.0 - (~sig & m & (z < t)).sum() / nb for t in ts ])
        ax.plot(eff, rej, "-", color=col, lw=1.8,
                label=f"{lab} ({ns} S / {nb} B)")
        for t, mk in ((3.0, "o"), (4.0, "s"), (5.0, "^"), (10.0, "D")):
            e = (sig & m & (z < t)).sum() / ns
            r = 1.0 - (~sig & m & (z < t)).sum() / nb
            ax.plot([e], [r], mk, color=col, ms=6, mfc="white", mew=1.6)
            if lab == "all events":
                ax.annotate(rf"$|z_v| < {t:g}$", (e, r),
                            textcoords="offset points", xytext=(-78, 4),
                            fontsize=11, color=col)
    # the usable region only: below ~0.95 efficiency the cut is not a cut any
    # analysis would take
    ax.set_xlim(0.95, 1.002)
    ax.set_xlabel("signal efficiency (gen truth)")
    ax.set_ylabel("background rejection (gen truth)")
    ax.set_title(f"{tag}: what a cut on the vertex residual buys", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="lower left")
    fn = os.path.join(outdir, f"cut_roc_{tag}.pdf")
    pubhtml.savefig(fig, fn)
    plt.close(fig)
    logger.info(f"wrote {fn}")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # linear bins with the last one an OVERFLOW, so the (very long) tail does
    # not have to be drawn on a log axis that cannot show zero
    zmax = 20.0
    edges = np.linspace(0, zmax, 41)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    zc = np.clip(z, 0, zmax - 1e-6)
    bottom = np.zeros(len(ctr))
    for ic, c in enumerate(CLASSES):
        m = cls == ic
        if not m.any():
            continue
        h, _ = np.histogram(zc[m], bins=edges)
        ax.bar(ctr, h, width=np.diff(edges), bottom=bottom, align="center",
               color=CLSCOL[c], label=f"{c} ({int(m.sum())})", linewidth=0)
        bottom += h
    for t in (3.0, 4.0, 5.0, 10.0):
        ax.axvline(t, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(bottom.max() * 3, 10))
    ax.set_xlim(0, zmax)
    ax.set_xlabel(r"$|z_v| = |r_v|/\sigma_v$   (last bin: overflow)")
    ax.set_ylabel("candidates / bin")
    ax.set_title(f"{tag}: what the vertex-residual tail is made of", fontsize=13)
    ax.legend(fontsize=11)
    fn = os.path.join(outdir, f"cut_composition_{tag}.pdf")
    pubhtml.savefig(fig, fn)
    plt.close(fig)
    logger.info(f"wrote {fn}")


# ------------------------------------------------------------------ hits ---
RAWB = ["run", "lumi", "event", "ndof", "chisqval",
        "Jpsi_vtxz", "Jpsi_vtxsig", "Jpsi_vtxres", "Jpsi_vtxdchi2",
        "Jpsi_vtxok", "cfmass_ok", "Jpsi_vtxvchk",
        "Jpsi_mass", "Jpsi_mass_unc", "Jpsi_sigmamass",
        "Muplus_nvalid", "Muminus_nvalid",
        "Muplus_nvalidpixel", "Muminus_nvalidpixel",
        "Muplus_nhits", "Muminus_nhits",
        "Muplusgen_dr", "Muminusgen_dr"]


def _read_one(f):
    import uproot
    try:
        fh = uproot.open(f)
    except Exception as e:
        logger.warning(f"skip {f}: {e}")
        return None
    with fh:
        t = fh["tree"]
        ks = set(k.split(";")[0] for k in t.keys())
        return t.arrays([b for b in RAWB if b in ks], library="np")


def load_raw(pattern, branches=RAWB, maxfiles=0, jobs=1):
    files = sorted(glob.glob(pattern))
    if maxfiles:
        files = files[:maxfiles]
    out = {}
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            it = pool.imap(_read_one, files)
            for i, a in enumerate(it):
                if a is None:
                    continue
                for k, v in a.items():
                    out.setdefault(k, []).append(v)
                if (i + 1) % 20 == 0:
                    logger.info(f"  [{i+1}/{len(files)}] read")
    else:
        for i, f in enumerate(files):
            a = _read_one(f)
            if a is None:
                continue
            for k, v in a.items():
                out.setdefault(k, []).append(v)
            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{len(files)}] read")
    if not out:
        raise SystemExit(f"nothing read from {pattern}")
    return {k: np.concatenate(v) for k, v in out.items()}, len(files)


def report_hits(d, nfiles, tag, constraint):
    n = len(d["run"])
    ndof = np.asarray(d["ndof"], np.int64)
    nv = np.asarray(d["Muplus_nvalid"] + d["Muminus_nvalid"], np.int64)
    npx = np.asarray(d["Muplus_nvalidpixel"] + d["Muminus_nvalidpixel"], np.int64)
    nmeas = nv + npx
    off = 9 if constraint else 10
    logger.info(f"=== {tag}: {n} written candidates from {nfiles} files, "
                f"vertex constraint {'ON' if constraint else 'OFF'}")
    dev = ndof - (nmeas - off)
    print(f"  ndof == nvalid + nvalidpixel - {off} for "
          f"{int((dev == 0).sum())}/{n} candidates; deviations "
          f"{dict(sorted(Counter(dev[dev != 0].tolist()).items()))}")
    print(f"  ndof   min {ndof.min():4d}  max {ndof.max():4d}   "
          f"hist(<=12) {dict(sorted(Counter(ndof[ndof <= 12].tolist()).items()))}")
    print(f"  nvalid(pair) min {nv.min():4d}   "
          f"hist(<=14) {dict(sorted(Counter(nv[nv <= 14].tolist()).items()))}")
    print(f"  nmeas(pair)  min {nmeas.min():4d}   "
          f"hist(<=14) {dict(sorted(Counter(nmeas[nmeas <= 14].tolist()).items()))}")
    for leg in ("Muplus", "Muminus"):
        x = np.asarray(d[f"{leg}_nvalid"], np.int64)
        print(f"  nvalid({leg:8s}) min {x.min():3d}  "
              f"hist(<=6) {dict(sorted(Counter(x[x <= 6].tolist()).items()))}")
    weak = np.minimum(np.asarray(d["Muplus_nvalid"], np.int64),
                      np.asarray(d["Muminus_nvalid"], np.int64))
    print(f"  nvalid(WEAKER leg) min {weak.min():3d}  "
          f"cumulative {{k: N(weak<=k)}} "
          f"{ {k: int((weak <= k).sum()) for k in (1, 2, 3, 4, 5, 6, 8)} }")
    for b in ("Jpsi_vtxres", "Jpsi_vtxsig", "Jpsi_vtxz", "Jpsi_vtxdchi2",
              "Jpsi_mass", "Jpsi_mass_unc", "Jpsi_sigmamass"):
        if b not in d:
            continue
        x = np.asarray(d[b], float)
        nbad = int((~np.isfinite(x)).sum())
        print(f"  non-finite {b:18s} {nbad}")
    c2 = np.where(ndof > 0, np.asarray(d["chisqval"], float)
                  / np.maximum(ndof, 1), 1e9)
    keep = (np.asarray(d["Jpsi_vtxok"], bool) & np.asarray(d["cfmass_ok"], bool)
            & (np.asarray(d["Jpsi_vtxsig"], float) > 0)
            & (np.asarray(d["Jpsi_sigmamass"], float) > 0) & (c2 < 3.0)
            & (np.abs(np.asarray(d["Jpsi_vtxvchk"], float)) < 1e-4))
    print(f"  after the extraction selection: {int(keep.sum())}  "
          f"ndof min {ndof[keep].min()}  nvalid min {nv[keep].min()}  "
          f"weaker leg min {weak[keep].min()}  "
          f"N(weak<=3) {int((weak[keep] <= 3).sum())}  "
          f"N(weak<=4) {int((weak[keep] <= 4).sum())}")
    zz = np.abs(np.asarray(d["Jpsi_vtxz"], float))
    # THE FLOOR: on the gun every candidate is signal by construction, so this
    # is the irreducible tail of the resolution model -- what a |z_v| cut
    # costs in signal efficiency even with no background at all.
    print("  selected P(|z_v| > t): " + "   ".join(
        f"{t:g}: {fmtpm(int((zz[keep] > t).sum()), int(keep.sum()))}"
        for t in (3.0, 4.0, 5.0, 10.0)))
    for k in (3, 4, 5):
        m = keep & (weak <= k)
        if m.any():
            print(f"    selected with weaker leg <= {k}: N {int(m.sum())}  "
                  f"P(|z_v|>5) {fmtpm(int((zz[m] > 5).sum()), int(m.sum()))}  "
                  f"(whole selected sample "
                  f"{fmtpm(int((zz[keep] > 5).sum()), int(keep.sum()))})")
    # THE WEAKER LEG. A pair SUM cannot see a one-hit leg -- and every
    # candidate whose exported mass resolution is non-finite has one.
    sm = np.asarray(d["Jpsi_sigmamass"], float)
    nan_m = ~np.isfinite(sm)
    vok = np.asarray(d["Jpsi_vtxok"], bool)
    sv = np.asarray(d["Jpsi_vtxsig"], float)
    print(f"  non-finite sigma_m: {int(nan_m.sum())}; weaker-leg hits of those "
          f"{dict(sorted(Counter(weak[nan_m].tolist()).items()))}; "
          f"Jpsi_vtxok TRUE for {int((nan_m & vok).sum())} of them")
    for k in (1, 2, 3, 4):
        m = weak <= k
        print(f"  weaker leg <= {k}: N {int(m.sum()):6d}  "
              f"non-finite sigma_m {int((m & nan_m).sum()):4d} "
              f"({100*(m & nan_m).sum()/max(nan_m.sum(),1):.1f} % of all of them)"
              f"  sigma_v > 1000 cm {int((m & (sv > 1000)).sum()):4d}")
    print(f"  sigma_v > 1000 cm: N {int((sv > 1000).sum())} "
          f"(all Jpsi_vtxok {bool(np.all(vok[sv > 1000])) if (sv>1000).any() else '--'}), "
          f"median weaker leg "
          f"{np.median(weak[sv > 1000]) if (sv > 1000).any() else float('nan'):g}")

    # what a minimum-size cut would remove, written and selected
    minhits = 10 if constraint else 11
    for lab, m in (("written", np.ones(n, bool)), ("selected", keep)):
        nn = int(m.sum())
        cut_ndof = m & (ndof < 1)
        cut_hits = m & (nv < minhits)
        z = np.abs(np.asarray(d["Jpsi_vtxz"], float))
        print(f"  [{lab}] minNdof=1 removes {int(cut_ndof.sum())} "
              f"({100*cut_ndof.sum()/max(nn,1):.4f} %); "
              f"minPairHits={minhits} removes {int(cut_hits.sum())} "
              f"({100*cut_hits.sum()/max(nn,1):.4f} %); union "
              f"{int((cut_ndof | cut_hits).sum())}")
        u = m & (cut_ndof | cut_hits)
        if u.any():
            print(f"           their |z_v|: {np.round(z[u], 3).tolist()[:20]}")
            print(f"           |z_v| > 5 among them: {int((z[u] > 5).sum())} "
                  f"of {int(u.sum())}; in the whole {lab} sample "
                  f"{int((z[m] > 5).sum())} of {nn}")


# ------------------------------------------------------------------ gate ---
GATEB = ["run", "lumi", "event", "ndof", "chisqval",
         "Jpsi_vtxres", "Jpsi_vtxsig", "Jpsi_vtxz", "Jpsi_vtxdchi2",
         "Jpsi_vtxvchk", "Jpsi_vtxb6", "Jpsi_vtxbfree", "Jpsi_vtxok",
         "Jpsi_mass", "Jpsi_mass_unc", "Jpsi_sigmamass", "Jpsi_covmassvtx",
         "Jpsi_pt", "Jpsi_eta", "cfmass_ok",
         "Muplus_pt", "Muplus_eta", "Muplus_phi",
         "Muminus_pt", "Muminus_eta", "Muminus_phi",
         "Muplus_nvalid", "Muminus_nvalid", "Muplus_nhits", "Muminus_nhits",
         "Muplus_nvalidpixel", "Muminus_nvalidpixel"]


def _gate_load(pattern):
    import uproot
    files = sorted(glob.glob(pattern))
    out = {}
    for f in files:
        with uproot.open(f) as fh:
            t = fh["tree"]
            ks = set(k.split(";")[0] for k in t.keys())
            a = t.arrays([b for b in GATEB if b in ks], library="np")
        for k, v in a.items():
            out.setdefault(k, []).append(v)
    if not out:
        raise SystemExit(f"nothing read from {pattern}")
    return {k: np.concatenate(v) for k, v in out.items()}, len(files)


def report_gate(newpat, refpat, constraint, minleghits=0):
    """Every candidate the NEW build writes must be present in the REFERENCE
    and BIT-IDENTICAL in it; the reference may have extra candidates, and
    those extras must be exactly the ones the new minimum-size cut removes."""
    dn, _ = _gate_load(newpat)
    dr, _ = _gate_load(refpat)
    nn, nr = len(dn["run"]), len(dr["run"])
    logger.info(f"=== gate: new {nn} candidates, reference {nr}")

    def keys(d):
        z = zip(d["run"].tolist(), d["lumi"].tolist(), d["event"].tolist(),
                d["Muplus_nhits"].tolist(), d["Muminus_nhits"].tolist(),
                d["Muplus_nvalid"].tolist(), d["Muminus_nvalid"].tolist())
        seen, out = defaultdict(int), []
        for k in z:
            out.append(k + (seen[k],))
            seen[k] += 1
        return out

    kn, kr = keys(dn), keys(dr)
    rmap = {k: i for i, k in enumerate(kr)}
    # the reference run may be shorter (the new run's event window is the same
    # nEvents, so both cover the same events)
    evn = set(zip(dn["run"].tolist(), dn["lumi"].tolist(), dn["event"].tolist()))
    evr = set(zip(dr["run"].tolist(), dr["lumi"].tolist(), dr["event"].tolist()))
    common = evn & evr
    inwin = np.array([(int(a), int(b), int(c)) in common
                      for a, b, c in zip(dn["run"], dn["lumi"], dn["event"])])
    logger.info(f"  events: new {len(evn)}, reference {len(evr)}, common "
                f"{len(common)}; new candidates in common events "
                f"{int(inwin.sum())}")
    miss, bad, nsame = [], defaultdict(int), 0
    fields = [b for b in GATEB if b in dn and b in dr
              and b not in ("run", "lumi", "event")]
    for i in np.flatnonzero(inwin):
        j = rmap.get(kn[i])
        if j is None:
            miss.append(i)
            continue
        nsame += 1
        for b in fields:
            x, y = dn[b][i], dr[b][j]
            if x != y and not (isinstance(x, float) and np.isnan(x)
                               and np.isnan(y)):
                bad[b] += 1
    print(f"  matched {nsame} candidates; NOT FOUND in the reference "
          f"{len(miss)}")
    if bad:
        print("  DIFFERENCES: " + ", ".join(f"{k} {v}" for k, v in bad.items()))
    else:
        print(f"  BIT-IDENTICAL on all {len(fields)} compared branches")
    # the reference candidates the new run does not have -- must be exactly
    # those the cut removes
    nmap = set(kn)
    extra = [j for j, k in enumerate(kr)
             if k not in nmap
             and (int(dr["run"][j]), int(dr["lumi"][j]), int(dr["event"][j])) in common]
    minhits = 10 if constraint else 11
    minleg = int(minleghits)
    nlr = np.minimum(dr["Muplus_nvalid"], dr["Muminus_nvalid"])
    nvr = dr["Muplus_nvalid"] + dr["Muminus_nvalid"]
    nmr = nvr + dr["Muplus_nvalidpixel"] + dr["Muminus_nvalidpixel"]
    ndr = nmr - (9 if constraint else 10)
    print(f"  reference-only candidates in common events: {len(extra)}")
    for j in extra[:20]:
        print(f"    ndof {int(ndr[j]):3d}  nvalid {int(nvr[j]):3d} "
              f"({int(dr['Muplus_nvalid'][j])},{int(dr['Muminus_nvalid'][j])})"
              f"  |z_v| {abs(float(dr['Jpsi_vtxz'][j])):.4f}"
              f"  weakleg {int(nlr[j]):2d}"
              f"  -> cut by "
              f"{'minNdof ' if ndr[j] < 1 else ''}"
              f"{'minPairHits ' if nvr[j] < minhits else ''}"
              f"{'minLegHits ' if (minleg > 0 and nlr[j] < minleg) else ''}"
              f"{'NEITHER (INVESTIGATE)' if (ndr[j] >= 1 and nvr[j] >= minhits and not (minleg > 0 and nlr[j] < minleg)) else ''}")
    nleg = int(sum(1 for j in extra if minleg > 0 and nlr[j] < minleg))
    print(f"  of those, explained by minLegHits >= {minleg}: {nleg}")
    unexplained = [j for j in extra
                   if ndr[j] >= 1 and nvr[j] >= minhits
                   and not (minleg > 0 and nlr[j] < minleg)]
    print(f"  reference-only candidates NOT explained by the cut: "
          f"{len(unexplained)}")
    # every surviving candidate must have ndof >= 1 and finite exports
    ndn = np.asarray(dn["ndof"], np.int64)
    print(f"  new build: ndof min {ndn.min()}  "
          f"nvalid(pair) min {int((dn['Muplus_nvalid']+dn['Muminus_nvalid']).min())}")
    for b in ("Jpsi_vtxres", "Jpsi_vtxsig", "Jpsi_vtxz", "Jpsi_mass",
              "Jpsi_mass_unc", "Jpsi_sigmamass"):
        if b in dn:
            x = np.asarray(dn[b], float)
            print(f"    non-finite {b:18s} {int((~np.isfinite(x)).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="extract_vtx.py output (vtx functional)")
    ap.add_argument("--raw", help="glob of globalcor_*.root, for --hits")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--constraint", choices=["on", "off"], default="on")
    ap.add_argument("--classes", action="store_true")
    ap.add_argument("--cuts", action="store_true")
    ap.add_argument("--mass", action="store_true")
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--hits", action="store_true")
    ap.add_argument("--gate-min-leg-hits", type=int, default=0,
                    help="the NEW build's minLegHits, so the reference-only "
                         "candidates it removes are counted as explained")
    ap.add_argument("--gate", nargs=2, metavar=("NEW", "REF"),
                    help="two globs: the new build's output and the reference")
    ap.add_argument("--resonances", type=int, nargs="*", default=list(RESONANCES))
    ap.add_argument("--mref", type=float, default=91.1876)
    ap.add_argument("--max-chi2-ndof", type=float, default=0.0,
                    help="extra chi2/ndof cut at load; 0 = none (the npz "
                         "already carries the extraction's own)")
    ap.add_argument("--mwin", type=float, default=15.0)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--outtag", default="vtxbkg")
    ap.add_argument("--minclass", type=int, default=90,
                    help="smallest class that gets its own density panel")
    ap.add_argument("--dump", default=None, help="npz of the class labels")
    a = ap.parse_args()
    logging.setup_logger(__file__, 3, False)

    if a.gate:
        report_gate(a.gate[0], a.gate[1], a.constraint == "on",
                    a.gate_min_leg_hits)

    if a.hits:
        d, nf = load_raw(a.raw, maxfiles=a.max_files, jobs=a.jobs)
        report_hits(d, nf, a.tag, a.constraint == "on")

    if not (a.classes or a.cuts or a.mass or a.density or a.dump):
        return
    if not a.npz:
        raise SystemExit("--npz is required for everything but --hits")
    d = load_npz(a.npz, max_chi2_ndof=a.max_chi2_ndof)
    cls, names, aux = classify(d, resonances=tuple(a.resonances))
    if a.classes:
        report_classes(d, cls, aux)
    if a.cuts:
        report_cuts(d, cls, aux)
    if a.cuts:
        report_dup(d, cls, aux)
        report_hitcut(d, cls, aux)
    if a.mass:
        report_mass(d, cls, mwin=a.mwin, mref=a.mref)
    if a.density:
        outdir = a.outdir or os.path.join(
            os.path.expanduser("~/public_html/ZMass/cvh"),
            datetime.date.today().strftime("%y%m%d") + "_" + a.outtag)
        os.makedirs(outdir, exist_ok=True)
        import pubhtml
        pubhtml.ensure_index(outdir)
        plot_density(d, cls, outdir, a.tag, minn=a.minclass)
        plot_cuts(d, cls, aux, outdir, a.tag)
    if a.dump:
        np.savez_compressed(a.dump, cls=cls, classes=np.array(CLASSES),
                            mult=aux["mult"], vtxz=d["vtxz"],
                            run=d["run"], lumi=d["lumi"], event=d["event"])
        logger.info(f"wrote {a.dump}")


if __name__ == "__main__":
    main()

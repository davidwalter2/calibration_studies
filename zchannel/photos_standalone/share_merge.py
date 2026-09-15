#!/usr/bin/env python3
"""Merge `photos_share` band records, and build the sample-like ME mixture.

Two modes, chained:

    share_merge.py "<glob>" -o run.npz --meta "..."         # sum replicas
    share_merge.py --combine A=a.npz B=b.npz \
                   --mix mix=A,B,0.3886 -o share_peak.npz   # one file, prefixed
    share_merge.py --report share_peak.npz                  # the tables

Every stored quantity is an unweighted event count, so a ratio of two of them
carries an exact binomial error.  The mixture is the per-event average
``f q_A/n_A + (1-f) q_B/n_B`` -- each run normalised inside the band first, as
`mix.py` does -- and is therefore in units of "per event"; its error has to be
propagated from the ``A_``/``B_`` counts, which is why both are kept.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

MAGIC = b"PHSHR02"
#: (name, trailing shape spec) of every block of a band record, in file order
SCAL = ("n", "n_1g", "n_1g_rad", "n_rad", "n_rad6", "n_viol", "n_bad",
        "mpre_s1", "sum_u", "sum_up", "sum_um")
PERCL = ("cn", "cup", "cum")
PERCLT = ("tp", "tm", "tj")
PERCLS = ("sn", "smid", "su", "su2", "sz", "sm")
ARRS = PERCL + PERCLT + PERCLS + ("hj", "hf")


def read_file(fn):
    with open(fn, "rb") as f:
        blob = f.read()
    if blob[:7] != MAGIC:
        raise ValueError(f"{fn}: bad magic {blob[:8]!r}")
    o = 8
    nb, ncl, nub, nsl, nfb, nt, nfe = struct.unpack_from("<7q", blob, o)
    o += 7 * 8
    ujlo, ujhi = struct.unpack_from("<dd", blob, o); o += 2 * 8
    usl = np.frombuffer(blob, "<f8", nsl + 1, o); o += (nsl + 1) * 8
    tcut = np.frombuffer(blob, "<f8", nt, o); o += nt * 8
    fedge = np.frombuffer(blob, "<f8", nfe, o); o += nfe * 8
    shp = dict(ncl=ncl, nub=nub, nsl=nsl, nfb=nfb, nt=nt,
               ujlo=ujlo, ujhi=ujhi, usl=usl, tcut=tcut, fedge=fedge)
    sizes = dict(cn=(ncl,), cup=(ncl,), cum=(ncl,),
                 tp=(ncl, nt), tm=(ncl, nt), tj=(ncl, nt),
                 sn=(ncl, nsl), smid=(ncl, nsl), su=(ncl, nsl),
                 su2=(ncl, nsl), sz=(ncl, nsl), sm=(ncl, nsl),
                 hj=(ncl, nub, nub), hf=(ncl, nsl, nfb))
    out = []
    for _ in range(nb):
        idx = struct.unpack_from("<q", blob, o)[0]; o += 8
        lo, hi = struct.unpack_from("<dd", blob, o); o += 2 * 8
        rec = {"lo": lo, "hi": hi}
        s = np.frombuffer(blob, "<f8", len(SCAL), o); o += len(SCAL) * 8
        for k, v in zip(SCAL, s):
            rec[k] = float(v)
        for k in ARRS:
            n_ = int(np.prod(sizes[k]))
            rec[k] = np.frombuffer(blob, "<f8", n_, o).reshape(sizes[k])
            o += n_ * 8
        out.append((idx, rec))
    if o != len(blob):
        raise ValueError(f"{fn}: {len(blob) - o} trailing bytes")
    return out, shp


def sum_runs(files):
    _, shp = read_file(files[0])
    acc, lohi = {}, {}
    for fn in files:
        recs, _ = read_file(fn)
        for idx, r in recs:
            lohi[idx] = (r["lo"], r["hi"])
            d = acc.setdefault(idx, {})
            for k in SCAL:
                d[k] = d.get(k, 0.0) + r[k]
            for k in ARRS:
                d[k] = d.get(k, 0.0) + r[k]
    bands = sorted(acc)
    out = {"bands": np.array(bands, np.int64),
           "bands_lo": np.array([lohi[b][0] for b in bands]),
           "bands_hi": np.array([lohi[b][1] for b in bands])}
    for k in SCAL:
        out[k] = np.array([acc[b][k] for b in bands])
    for k in ARRS:
        out[k] = np.stack([acc[b][k] for b in bands])
    for k in ("usl", "tcut", "fedge"):
        out[k] = np.asarray(shp[k])
    out["ubin_lo"] = np.array(shp["ujlo"])
    out["ubin_hi"] = np.array(shp["ujhi"])
    #: the u_leg bin edges the `hj` axes use: bin 0 is u_leg < ubin_lo
    out["uedge"] = np.concatenate(
        [[0.0], np.geomspace(shp["ujlo"], shp["ujhi"], shp["nub"])])
    return out


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------
# Every statistic below is a smooth function of multinomial cell proportions,
# so one delta-method helper covers all of them.  A mixture is a linear
# combination of independent runs, ``p = sum_X w_X p_X``, and its variance is
# ``sum_X w_X^2 Cov_X`` with each ``Cov_X`` the multinomial covariance of that
# run at its own proportions.


def _dvar(grad, runs):
    """Variance of a statistic with gradient ``grad`` in the mixed proportions.

    ``runs`` is ``[(w, p, N), ...]``: weight, cell proportions and generated
    count of each independent run entering the mixture.
    """
    v = 0.0
    for w, p, n in runs:
        p = np.asarray(p, float)
        gp = float(np.dot(grad, p))
        v += w * w * (float(np.dot(grad * grad, p)) - gp * gp) / n
    return max(v, 0.0)


class Setting:
    """One generated (or mixed) configuration, in per-event cell proportions."""

    def __init__(self, d, name, ib=0):
        self.name = name
        self.runs = []                     # the independent runs behind it
        g = lambda k: np.asarray(d[f"{name}_{k}"], float)
        self.n = float(g("n")[ib])
        for k in SCAL[1:] + ARRS:
            setattr(self, k, g(k)[ib] / self.n)
        self.usl, self.tcut, self.fedge = g("usl"), g("tcut"), g("fedge")
        self.uedge = g("uedge")

    @staticmethod
    def build(d, name, ib=0):
        s = Setting(d, name, ib)
        src = f"{name}_src"
        if src in d.files:                 # a mixture: keep its ingredients
            n1, n2 = str(d[src]).split(",")
            fr = float(d[f"{name}_frac"])
            for nm, w in ((n1, fr), (n2, 1.0 - fr)):
                s.runs.append((w, Setting(d, nm, ib)))
        else:
            s.runs.append((1.0, s))
        return s

    def cells(self, *arrs):
        """``[(w, p, N)]`` of the ingredient runs for the given cell getter."""
        return [(w, np.concatenate([[f(r) for f in arrs],
                                    [1.0 - sum(f(r) for f in arrs)]]), r.n)
                for w, r in self.runs]


def _pmid(s, c, k):
    """``P(0.01 < f < 0.99 | class c, slice k)`` and its error."""
    a, b = s.smid[c, k], s.sn[c, k] - s.smid[c, k]
    p = a / (a + b)
    g = np.array([b, -a, 0.0]) / (a + b) ** 2
    e = np.sqrt(_dvar(g, s.cells(lambda r: r.smid[c, k],
                                 lambda r: r.sn[c, k] - r.smid[c, k])))
    return p, e


def _ptail(s, c, k, which):
    """``P(u_leg > t_k | class c)`` for ``which`` in ('tp', 'tm')."""
    a = getattr(s, which)[c, k]
    d = s.cn[c] - a
    g = np.array([d, -a, 0.0]) / (a + d) ** 2
    e = np.sqrt(_dvar(g, s.cells(lambda r: getattr(r, which)[c, k],
                                 lambda r: r.cn[c] - getattr(r, which)[c, k])))
    return a / (a + d), e


def _jratio(s, c, k):
    """``P(u_+>t, u_->t) / P(u_+>t) P(u_->t)`` on the 2x2 leg table."""
    def cell(r):
        j, tp, tm, n = r.tj[c, k], r.tp[c, k], r.tm[c, k], r.cn[c]
        return j, tp - j, tm - j, n - tp - tm + j
    a, b, cc, dd = cell(s)
    T = a + b + cc + dd
    R = a * T / ((a + b) * (a + cc))
    # d ln R / d p over the four cells of the (u_+>t) x (u_->t) table; anything
    # outside class c has gradient zero and is the implicit remainder
    g = np.array([1.0 / a + 1.0 / T - 1.0 / (a + b) - 1.0 / (a + cc),
                  1.0 / T - 1.0 / (a + b),
                  1.0 / T - 1.0 / (a + cc),
                  1.0 / T])
    runs = [(w, np.array(cell(r)), r.n) for w, r in s.runs]
    return R, R * np.sqrt(_dvar(g, runs))


def _umean(s, c, which):
    """``<u_leg | class c>`` and an error from the binned second moment."""
    m = getattr(s, "cup" if which == "tp" else "cum")[c] / s.cn[c]
    ue = s.uedge
    cen = np.concatenate([[0.5 * ue[1]], np.sqrt(ue[1:-1] * ue[2:])])
    h = s.hj[c].sum(1 if which == "tp" else 0)
    w = h / h.sum()
    v = max(float(np.dot(w, cen ** 2) - np.dot(w, cen) ** 2), 0.0)
    neff = sum(ww * ww / r.n for ww, r in s.runs) ** -1 if s.runs else s.n
    return m, np.sqrt(v / neff)


def _exact(m, z, fedge):
    """The exact O(alpha) sharing: ``P(0.01<f<0.99)`` and the binned density."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    import fsr_analytic as FA
    # npanel=1024 rather than `cmp_perleg`'s 256: with 256 the Gauss-Legendre
    # panels are only ~10 per f bin and the binned reference carries a +-0.7 %
    # jitter that the per-bin ratio would otherwise inherit
    f, w = FA.share_nodes(m, np.array([z]), npanel=1024, ng=8)
    # `w` sums to 1/2 on the stored f <= 1/2 branch; the density is symmetric
    pmid = 2.0 * float(w[0][f[0] > 0.01].sum())
    fa = np.concatenate([f[0], 1.0 - f[0]])
    wa = np.concatenate([w[0], w[0]])
    k = np.clip(np.searchsorted(fedge, fa, "right") - 1, 0, len(fedge) - 2)
    return pmid, np.bincount(k, wa, len(fedge) - 1)


def compare(path, a, b, ib=0):
    """Two settings side by side: the boost-invariance check."""
    d = np.load(path, allow_pickle=True)
    S = {n: Setting.build(d, n, ib) for n in (a, b)}
    usl, tcut = S[a].usl, S[a].tcut
    lab = ["exactly 1 photon", "all events", "k^2 ~ 0"]
    print(f"(6) {b} / {a}, the same statistic in both "
          f"(N = {S[a].n:.3e} / {S[b].n:.3e})")
    for c in range(3):
        print(f"    class = {lab[c]}")
        for k in range(len(usl) - 1):
            p1, e1 = _pmid(S[a], c, k)
            p2, e2 = _pmid(S[b], c, k)
            e = np.hypot(e1, e2)
            print(f"      P(0.01<f<0.99), u in [{usl[k]:g}, {usl[k+1]:g})"
                  f"".ljust(46)
                  + f" {p2/p1:.5f} +- {e/p1:.5f}  ({(p2-p1)/e:+.1f} sigma)")
        for k in range(len(tcut)):
            p1, e1 = _ptail(S[a], c, k, "tp")
            p2, e2 = _ptail(S[b], c, k, "tp")
            e = np.hypot(e1, e2)
            print(f"      P(u_+ > {tcut[k]:.0e})".ljust(46)
                  + f" {p2/p1:.5f} +- {e/p1:.5f}  ({(p2-p1)/e:+.1f} sigma)")
        print()


def report(path, names, ib=0):
    d = np.load(path, allow_pickle=True)
    S = {n: Setting.build(d, n, ib) for n in names}
    ref = S[names[0]]
    usl, tcut, fedge = ref.usl, ref.tcut, ref.fedge
    nsl, nt = len(usl) - 1, len(tcut)
    lab = ["exactly 1 photon", "all events", "k^2 ~ 0 (the sample's tag)"]

    print(f"# {path}")
    print(f"# band {int(d[names[0] + '_bands'][ib])} = "
          f"[{d[names[0] + '_bands_lo'][ib]:g}, "
          f"{d[names[0] + '_bands_hi'][ib]:g}) GeV")
    for n in names:
        s = S[n]
        print(f"#   {n:8s} N_gen = {sum(r.n for _, r in s.runs):.4e}"
              f"   <m_pre> = {s.mpre_s1:.5f}   <u> = {s.sum_u:.6e}")
    print()

    # ---- (1) P(0.01 < f < 0.99) -----------------------------------------
    print("(1) P(0.01 < f < 0.99) per u slice, against the exact O(alpha)")
    for c in range(len(lab)):
        print(f"    class = {lab[c]}")
        head = f"      {'u slice':>16s} | {'z_bar':>7s} | {'exact':>8s} | "
        head += " | ".join(f"{n:>21s}" for n in names)
        print(head)
        for k in range(nsl):
            zb = float(ref.sz[c, k] / ref.sn[c, k])
            mb = float(ref.sm[c, k] / ref.sn[c, k])
            ex, _ = _exact(mb, zb, fedge)
            row = (f"      [{usl[k]:g}, {usl[k+1]:g})".ljust(24)
                   + f"| {zb:7.4f} | {ex:8.4f} | ")
            cells = []
            for n in names:
                p, e = _pmid(S[n], c, k)
                cells.append(f"{p:.4f}+-{e:.4f} ({p/ex:.4f})")
            print(row + " | ".join(f"{x:>21s}" for x in cells))
        print()

    # ---- (2) the f shape -------------------------------------------------
    cen = np.sqrt(fedge[:-1] * fedge[1:])
    for CSH in (0, 2):
        print(f"(2) the f shape, Photos / exact per bin, class = {lab[CSH]}; "
              "the first and last bins are the collinear overflow")
        for k in range(nsl):
            zb = float(ref.sz[CSH, k] / ref.sn[CSH, k])
            mb = float(ref.sm[CSH, k] / ref.sn[CSH, k])
            _, hm = _exact(mb, zb, fedge)
            print(f"    u in [{usl[k]:g}, {usl[k+1]:g}),  z_bar = {zb:.4f}")
            print(f"      {'f (bin centre)':>16s} | {'exact':>10s} | "
                  + " | ".join(f"{n:>17s}" for n in names))
            for j in range(len(cen)):
                if hm[j] <= 0:
                    continue
                cells = []
                for n in names:
                    h = S[n].hf[CSH, k]
                    v = h[j] / h.sum()
                    # the MC bin content is a proportion of the slice population
                    a = S[n].hf[CSH, k, j]
                    b = S[n].sn[CSH, k] - a
                    g = np.array([b, -a, 0.0]) / (a + b) ** 2
                    e = np.sqrt(_dvar(g, S[n].cells(
                        lambda r: r.hf[CSH, k, j],
                        lambda r: r.sn[CSH, k] - r.hf[CSH, k, j])))
                    cells.append(f"{v/hm[j]:7.4f}+-{e/hm[j]:6.4f}")
                print(f"      {cen[j]:16.4e} | {hm[j]:10.3e} | "
                      + " | ".join(f"{x:>17s}" for x in cells))
            print()

    # ---- (3) leg-leg correlation ----------------------------------------
    print("(3) P(u_+>t, u_->t) / P(u_+>t) P(u_->t)")
    for c in range(len(lab)):
        print(f"    class = {lab[c]}")
        print(f"      {'t':>10s} | " + " | ".join(f"{n:>17s}" for n in names))
        for k in range(nt):
            cells = []
            for n in names:
                R, e = _jratio(S[n], c, k)
                cells.append(f"{R:8.4f}+-{e:6.4f}")
            print(f"      {tcut[k]:10.0e} | " + " | ".join(f"{x:>17s}" for x in cells))
        print()

    # ---- (4) the per-leg marginal ---------------------------------------
    print("(4) <u_leg> and P(u_leg > t)")
    for c in range(len(lab)):
        print(f"    class = {lab[c]}")
        cells = []
        for n in names:
            mp, ep = _umean(S[n], c, "tp")
            mm, _ = _umean(S[n], c, "tm")
            cells.append(f"{0.5*(mp+mm):.6e}+-{ep/np.sqrt(2):.1e}")
        print(f"      {'<u_leg>':>10s} | " + " | ".join(f"{x:>22s}" for x in cells))
        for k in range(nt):
            cells = []
            for n in names:
                pp, ep = _ptail(S[n], c, k, "tp")
                pm, _ = _ptail(S[n], c, k, "tm")
                cells.append(f"{0.5*(pp+pm):.6f}+-{ep/np.sqrt(2):.1e}")
            print(f"      P(>{tcut[k]:8.0e}) | "
                  + " | ".join(f"{x:>22s}" for x in cells))
        print()

    # ---- (5) emission composition ---------------------------------------
    print("(5) emission composition (per event, and per radiating event "
          "u > 1e-5)")
    print(f"      {'quantity':>34s} | " + " | ".join(f"{n:>17s}" for n in names))
    rows = (("P(radiating, u > 1e-5)", lambda s: s.n_rad),
            ("P(radiating, u > 1e-6)", lambda s: s.n_rad6),
            ("P(exactly 1 photon, no pair)", lambda s: s.n_1g),
            ("1 photon / radiating(1e-5)", lambda s: s.n_1g_rad / s.n_rad),
            ("k^2 ~ 0 & rad / radiating(1e-5)", lambda s: s.cn[2] / s.n_rad),
            ("k^2 ~ 0 & rad / radiating(1e-6)", lambda s: s.cn[2] / s.n_rad6),
            ("line violation (1e-9) / radiating", lambda s: s.n_viol / s.n_rad),
            ("line violation (1e-9) / all", lambda s: s.n_viol))
    for name, fn in rows:
        print(f"      {name:>34s} | "
              + " | ".join(f"{fn(S[n]):17.6f}" for n in names))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="glob(s) of photos_share .bin files")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--meta", default="")
    ap.add_argument("--combine", nargs="+", default=None,
                    help="name=path.npz ...: copy each under the `name_` prefix")
    ap.add_argument("--mix", action="append", default=[],
                    help="name=first,second,frac: per-event mixture of two "
                         "combined runs, weight `frac` on the first")
    ap.add_argument("--report", default=None,
                    help="print the tables of a combined npz instead of merging")
    ap.add_argument("--names", default="A,B,mix",
                    help="--report: the prefixes to tabulate, in order")
    ap.add_argument("--compare", default=None,
                    help="--report: `a,b`, print b/a for the same statistics")
    a = ap.parse_args()

    if a.report:
        report(a.report, a.names.split(","))
        if a.compare:
            compare(a.report, *a.compare.split(","))
        return

    if a.combine:
        out = {}
        srcs = {}
        for spec in a.combine:
            name, _, path = spec.partition("=")
            d = np.load(path, allow_pickle=True)
            srcs[name] = d
            for k in d.files:
                out[f"{name}_{k}"] = d[k]
        for spec in a.mix:
            name, _, rest = spec.partition("=")
            n1, n2, fr = rest.split(",")
            fr = float(fr)
            d1, d2 = srcs[n1], srcs[n2]
            if not np.array_equal(d1["bands"], d2["bands"]):
                raise SystemExit("mix: the two runs cover different bands")
            w1 = fr / d1["n"]
            w2 = (1.0 - fr) / d2["n"]
            for k in SCAL + ARRS:
                x1, x2 = np.asarray(d1[k], float), np.asarray(d2[k], float)
                sh = (-1,) + (1,) * (x1.ndim - 1)
                out[f"{name}_{k}"] = w1.reshape(sh) * x1 + w2.reshape(sh) * x2
            for k in ("bands", "bands_lo", "bands_hi", "usl", "tcut", "fedge",
                      "uedge", "ubin_lo", "ubin_hi"):
                out[f"{name}_{k}"] = d1[k]
            out[f"{name}_frac"] = np.array(fr)
            out[f"{name}_ngen"] = np.asarray(d1["n"]) + np.asarray(d2["n"])
            out[f"{name}_src"] = np.array(f"{n1},{n2}")
        out["meta"] = np.array(a.meta)
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        np.savez_compressed(a.output, **out)
        if not a.output:
            raise SystemExit("-o is required")
        print(f"{a.output}: combined {list(srcs)}"
              + (f", mixed {[s.split('=')[0] for s in a.mix]}" if a.mix else ""))
        return

    files = sorted(sum([glob.glob(p) for p in a.inputs], []))
    if not files:
        raise SystemExit("no input files")
    out = sum_runs(files)
    out["meta"] = np.array(a.meta)
    out["nfiles"] = np.array(len(files))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez_compressed(a.output, **out)
    n = out["n"].sum()
    print(f"{a.output}: {len(files)} files, {len(out['bands'])} bands, "
          f"N = {n:.4e}, bad = {out['n_bad'].sum():.0f}")
    print(f"  <u> = {out['sum_u'].sum()/n:.6e}, "
          f"1-photon / radiating = {out['n_1g_rad'].sum()/out['n_rad'].sum():.6f}, "
          f"line violations / radiating = "
          f"{out['n_viol'].sum()/out['n_rad'].sum():.6f}")


if __name__ == "__main__":
    main()

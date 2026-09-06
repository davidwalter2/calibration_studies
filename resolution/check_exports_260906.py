#!/usr/bin/env python3
"""Validation of the new CVH C++ exports (2026-09-06 smoke campaign).

Five independent gates on the smoke outputs under
`/work/submit/david_w/ZMass/scratch_smoke_260906`:

  1. SIGMA_G CLOSURE   sum_g S_{f,g}(tau) == S_f(tau) for the per-material-group
                       CF arrays (`cf*_grp_*`) against the flat families.
  2. HIT-CLASS BLOCKS  the parmtype-8/9 resolution blocks: their multiplicity,
                       the INVARIANCE of `cf*_vgf` / `resinfcov` with respect to
                       registering them, and the per-class variance shares.
  3. PER-LEG COV       `Jpsi_covrefmom` / `Jpsi_jacrefmom` reproduce
                       `Jpsi_sigmamass`, and the derived `sigmarel`/`rhomom`/
                       `fang` summaries are what they claim to be.
  4. PRE-FSR MASS      occupancy and offsets of `Jpsigenpre_*` /
                       `Jpsigen_massdressed`.
  5. BYTES/CANDIDATE   compressed cost of every new branch group, per config,
                       against the corresponding baseline.

Text report only -- no plots. Missing files/branches are reported, never fatal.

    python3 check_exports_260906.py [--only 1,3] [--quick] [--wait 720]
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

try:  # repo convention; falls back to prints outside a wums-enabled venv
    from wums import logging as _wumslog
    logger = _wumslog.setup_logger(__file__, 3)
except Exception:
    logger = None

import uproot
import prodfiles

# --------------------------------------------------------------------------
# where things are
# --------------------------------------------------------------------------
ROOT = "/work/submit/david_w/ZMass/scratch_smoke_260906"
OUTDIR = "/work/submit/david_w/ZMass/calibration_studies/resolution/runs/exports260906"
OUTFILE = os.path.join(OUTDIR, "check_exports.txt")

# smoke -> (output stem, tree kind).  A stem, not a file name: the smokes are
# `numberOfThreads=1` so a directory holds one stream, and
# prodfiles.single_file finds it without naming stream 0 (and warns if the
# directory turns out to hold several, which would make this check read 1/N).
SMOKES = {
    "gun_tt": ("globalcor", "tt"),
    "gun_st": ("globalcor_resclosure", "st"),
    "data_tt": ("globalcor", "tt"),
}
# configuration -> directory under ROOT.  `base_prod2` only ever holds data_tt.
CONFIGS = ["base_prod", "base_prod2", "v_noblocks", "v_default", "v_grp"]
BASE_OF = {"gun_tt": "base_prod", "gun_st": "base_prod", "data_tt": "base_prod2"}
# base_prod/data_tt is a KNOWN-BAD empty tree (0 entries); base_prod2/data_tt is
# the data baseline.  Never wait for, or read, the known-bad one.
SKIP_INPUTS = {("base_prod", "data_tt")}

# the canonical 18 hit classes, in index order (hitres_classes.py:CLASSES)
CLASSES = ([f"pix_{a}_q{q}" for a in ("x", "y") for q in range(4)]
           + [f"str_N{n}_{u}" for n in range(1, 6) for u in ("lo", "hi")])
# the offline low-pT muon-gun (single-track) reference shares, in percent
REF_SHARE = {"str_N3_lo": 24.7, "str_N2_lo": 18.3, "str_N1_lo": 10.8,
             "str_N3_hi": 9.7, "pix_x_q1": 6.8}
REF_PIXY_TOTAL = 2.3  # all four pix_y_* together

# gen-measured reference ranges on this same gun sample
REF_RHO = (-0.009, -0.004)
REF_FANG = (0.10, 0.14)

# float32 storage floor: every cf array is a vector<float>, so a sum of n of
# them can only agree with the separately-rounded total to ~eps32*sqrt(n).
EPS32 = np.finfo(np.float32).eps
F32_FLOOR = 1e-6

CF = {"tt": "cfmass", "st": "cfqop"}
NTAU = 64


def path_of(cfg, smoke):
    return prodfiles.single_file(os.path.join(ROOT, cfg, smoke), SMOKES[smoke][0])


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
class Report:
    def __init__(self, fn):
        self.fh = open(fn, "w")
        self.gates = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.fh.write(s + "\n")
        self.fh.flush()

    def head(self, s):
        self("")
        self("=" * 100)
        self(s)
        self("=" * 100)

    def sub(self, s):
        self("")
        self("-- " + s + " " + "-" * max(0, 96 - len(s)))

    def gate(self, name, ok, detail=""):
        """ok: True / False / None (= not evaluated)."""
        self.gates.append((name, ok, detail))
        tag = "PASS" if ok is True else ("FAIL" if ok is False else "SKIP")
        self(f"  [{tag}] {name}   {detail}")

    def close(self):
        self.head("PASS / FAIL SUMMARY")
        for name, ok, detail in self.gates:
            tag = "PASS" if ok is True else ("FAIL" if ok is False else "SKIP")
            self(f"  {tag:4s}  {name:<54s} {detail}")
        npass = sum(1 for _, o, _ in self.gates if o is True)
        nfail = sum(1 for _, o, _ in self.gates if o is False)
        nskip = sum(1 for _, o, _ in self.gates if o is None)
        self("")
        self(f"  TOTAL: {npass} pass, {nfail} fail, {nskip} skipped")
        self.fh.close()


# --------------------------------------------------------------------------
# file access
# --------------------------------------------------------------------------
_OPEN = {}
_MISSING = []


def ready(fn):
    """True once the file exists AND carries a readable, non-empty `tree`."""
    if not os.path.exists(fn):
        return False
    try:
        f = uproot.open(fn)
        if "tree" not in [k.split(";")[0] for k in f.keys()]:
            f.close()
            return False
        n = f["tree"].num_entries
        f.close()
        return n > 0
    except Exception:
        return False


def get(fn, rep=None):
    """Open (and cache) a file; returns the tree or None."""
    if fn in _OPEN:
        return _OPEN[fn]
    t = None
    try:
        if os.path.exists(fn):
            f = uproot.open(fn)
            if "tree" in [k.split(";")[0] for k in f.keys()]:
                t = f["tree"]
            else:
                if rep:
                    rep(f"  !! {fn}: no `tree` key (still being written?)")
        else:
            if rep:
                rep(f"  !! {fn}: MISSING")
    except Exception as e:
        if rep:
            rep(f"  !! {fn}: {type(e).__name__}: {e}")
    _OPEN[fn] = t
    if t is None and fn not in _MISSING:
        _MISSING.append(fn)
    return t


def arrs(tree, names, quick, rep=None, tag=""):
    """Read the branches that exist; report (and skip) the ones that do not."""
    have = [b for b in names if b in tree]
    miss = [b for b in names if b not in tree]
    if miss and rep:
        rep(f"  !! {tag}: missing branches {miss}")
    if not have:
        return {}, miss
    stop = 20 if quick else None
    try:
        d = tree.arrays(have, library="np", entry_stop=stop)
    except Exception as e:
        if rep:
            rep(f"  !! {tag}: read failed {type(e).__name__}: {e}")
        return {}, names
    return d, miss


def pct(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def fmtq(x, fmt="{:+.4f}"):
    """median (16,84) summary string."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n/a"
    return (fmt.format(np.median(x)) + "  (" + fmt.format(np.percentile(x, 16))
            + ", " + fmt.format(np.percentile(x, 84)) + ")")


# ==========================================================================
# (1) SIGMA_G CLOSURE
# ==========================================================================
def section1(rep, quick):
    rep.head("(1) SIGMA_G CLOSURE  --  sum_g S_{f,g}(tau) vs the flat S_f(tau)")
    rep("")
    rep("  The `cf*_grp_*` arrays are ngrp x 64 row-major in (group, tau); the flat")
    rep("  `cf*_<family>` are the same per-step sums associated differently.  BOTH are")
    rep("  stored as vector<float>, so a recomputation from the FILE can only close to")
    rep(f"  the float32 floor (~{EPS32:.1e} x sqrt(ngrp) ~ 1e-7), NOT to 1e-13.  The")
    rep("  1e-13-level statement is the maker's own `cf*_grp_closure`, which is formed")
    rep("  in float64 before the cast.  Both are reported; the recomputed gate is")
    rep(f"  therefore evaluated at the float32-appropriate {F32_FLOOR:.0e} as well.")

    fams_tt = ["ms", "ioni_re", "ioni_im", "rad_re", "rad_im"]
    any_run = False
    worst_rel_all = 0.0
    ok_all = True
    ok_all_1e13 = True
    for smoke in ("gun_tt", "gun_st", "data_tt"):
        fn = path_of("v_grp", smoke)
        t = get(fn, rep)
        if t is None:
            rep.gate(f"(1) sigma_g closure [v_grp/{smoke}]", None, "input not available")
            continue
        pre = CF[SMOKES[smoke][1]]
        fams = list(fams_tt)
        if f"{pre}_grp_del" in t:
            fams.append("del")
        rep.sub(f"v_grp / {smoke}   ({pre}_*, {t.num_entries} entries)")
        names = [f"{pre}_grp", f"{pre}_grp_closure"] + \
                [f"{pre}_grp_{f}" for f in fams] + [f"{pre}_{f}" for f in fams]
        d, miss = arrs(t, names, quick, rep, f"v_grp/{smoke}")
        if f"{pre}_grp" not in d:
            rep.gate(f"(1) sigma_g closure [v_grp/{smoke}]", None,
                     "no group arrays in this file (exportCfGroupExponents off?)")
            continue
        any_run = True
        grp = d[f"{pre}_grp"]
        n = len(grp)
        ngrp = np.array([len(g) for g in grp])
        allids = sorted(set(int(x) for g in grp for x in g))
        rep(f"  groups/candidate: mean {ngrp.mean():.2f}  median {np.median(ngrp):.1f}"
            f"  min {ngrp.min()}  max {ngrp.max()}")
        rep(f"  group ids seen ({len(allids)}): {allids}")
        # ascending-order sanity
        nonasc = sum(1 for g in grp if len(g) > 1 and np.any(np.diff(np.asarray(g)) <= 0))
        rep(f"  candidates whose group ids are NOT strictly ascending: {nonasc}")
        rep("")
        rep(f"  {'family':<10s} {'max |d|':>12s} {'max rel':>12s} {'n(rel>1e-13)':>13s}"
            f" {'n(rel>1e-6)':>12s} {'max|S|':>12s}")
        for fam in fams:
            gb, fb = f"{pre}_grp_{fam}", f"{pre}_{fam}"
            if gb not in d or fb not in d:
                rep(f"  {fam:<10s} {'--- branch missing ---':>40s}")
                ok_all = False
                continue
            dmax = relmax = smax = 0.0
            nbad13 = nbad6 = 0
            for i in range(n):
                ng = len(grp[i])
                g = np.asarray(d[gb][i], dtype=np.float64)
                f = np.asarray(d[fb][i], dtype=np.float64)
                if f.size != NTAU or g.size != ng * NTAU:
                    nbad13 += 1
                    nbad6 += 1
                    continue
                s = g.reshape(ng, NTAU).sum(axis=0)
                dd = float(np.max(np.abs(s - f)))
                sm = float(np.max(np.abs(f)))
                rel = dd / sm if sm > 0 else 0.0
                dmax = max(dmax, dd)
                relmax = max(relmax, rel)
                smax = max(smax, sm)
                if rel > 1e-13:
                    nbad13 += 1
                if rel > F32_FLOOR:
                    nbad6 += 1
            rep(f"  {fam:<10s} {dmax:12.4e} {relmax:12.4e} {nbad13:13d} {nbad6:12d} {smax:12.4e}")
            worst_rel_all = max(worst_rel_all, relmax)
            ok_all &= (nbad6 == 0)
            ok_all_1e13 &= (nbad13 == 0)
        cl = np.abs(np.asarray(d.get(f"{pre}_grp_closure", np.zeros(n)), dtype=np.float64))
        rep("")
        rep(f"  maker's own |{pre}_grp_closure|:  max {cl.max():.4e}   median {np.median(cl):.4e}")
        rep(f"  (its normalisation is the LARGEST family's max|S| over all families, so it")
        rep(f"   is <= the per-family rel above by construction)")
    if any_run:
        rep.gate("(1) sigma_g closure, per-family rel <= 1e-13 (as specified)",
                 ok_all_1e13,
                 f"max rel = {worst_rel_all:.3e}  -- unreachable from float32 storage")
        rep.gate("(1) sigma_g closure, per-family rel <= 1e-6 (float32 floor)",
                 ok_all, f"max rel = {worst_rel_all:.3e}")
    return


# ==========================================================================
# (2) HIT-CLASS BLOCKS
# ==========================================================================
def hitcls_table(rep, t, pre, quick, tag, vgf_exact_gate=False):
    """Per-class occupancy and variance share, from cf*_hitcls / cf*_hitv."""
    names = [f"{pre}_hitcls", f"{pre}_hitv", f"{pre}_vgf"]
    d, miss = arrs(t, names, quick, rep, tag)
    if f"{pre}_hitcls" not in d or f"{pre}_hitv" not in d:
        rep(f"  !! {tag}: no per-class arrays")
        return None
    cls, v = d[f"{pre}_hitcls"], d[f"{pre}_hitv"]
    n = len(cls)
    occ = np.zeros(18)
    tot = np.zeros(18)
    fracsum = np.zeros(18)
    nwith = 0
    sums = []
    for i in range(n):
        c = np.asarray(cls[i], dtype=int)
        vv = np.asarray(v[i], dtype=np.float64)
        if c.size == 0:
            continue
        nwith += 1
        s = vv.sum()
        sums.append(s)
        for cc, x in zip(c, vv):
            if 0 <= cc < 18:
                occ[cc] += 1
                tot[cc] += x
                if s > 0:
                    fracsum[cc] += x / s
    sums = np.asarray(sums)
    if nwith == 0:
        rep(f"  !! {tag}: every candidate has an empty class list")
        return None
    grand = tot.sum()
    rep("")
    rep(f"  {tag}: {nwith}/{n} candidates carry hit classes")
    rep(f"  {'idx':>3s} {'class':<12s} {'occupancy':>10s} {'share_pooled':>13s}"
        f" {'share_mean':>11s} {'ref (st gun)':>13s}")
    for c in range(18):
        ref = REF_SHARE.get(CLASSES[c])
        refs = f"{ref:.1f} %" if ref is not None else ""
        rep(f"  {c:3d} {CLASSES[c]:<12s} {occ[c]/nwith*100:9.1f}%"
            f" {tot[c]/grand*100 if grand > 0 else 0:12.2f}%"
            f" {fracsum[c]/nwith*100:10.2f}% {refs:>13s}")
    pixy = sum(tot[c] for c in range(4, 8)) / grand * 100 if grand > 0 else 0
    rep(f"  all four pix_y together: {pixy:.2f}%   (ref {REF_PIXY_TOTAL} %)")
    if f"{pre}_vgf" in d:
        vgf = np.asarray(d[f"{pre}_vgf"], dtype=np.float64)[:len(cls)]
        m = np.array([len(cls[i]) > 0 for i in range(len(cls))])
        if m.sum() > 0:
            s = np.array([np.asarray(v[i], dtype=np.float64).sum() for i in range(len(cls))])
            num = np.abs(s[m] - vgf[m])
            den = np.maximum(np.abs(vgf[m]), 1e-300)
            rel = num / den
            rep(f"  sum_c {pre}_hitv vs {pre}_vgf:  max |diff| {num.max():.4e}"
                f"   max rel {rel.max():.4e}   n(rel>1e-6) {(rel > 1e-6).sum()}")
            if vgf_exact_gate:
                rep.gate(f"(2) sum_c {pre}_hitv == {pre}_vgf [{tag}]",
                         bool(rel.max() <= 1e-6),
                         f"max rel {rel.max():.3e} (float32 exactness)")
    return tot / grand if grand > 0 else None


def section2(rep, quick):
    rep.head("(2) HIT-CLASS BLOCKS  --  multiplicity, VGF invariance, per-class shares")

    tdef = get(path_of("v_default", "gun_tt"), rep)
    tnob = get(path_of("v_noblocks", "gun_tt"), rep)

    # ---- block multiplicity -------------------------------------------
    rep.sub("parmtype-8/9 block multiplicity (from `reshitcls`)")
    counts = {}
    for lbl, t in (("v_default/gun_tt", tdef), ("v_noblocks/gun_tt", tnob)):
        if t is None:
            rep(f"  {lbl}: input not available")
            continue
        d, _ = arrs(t, ["reshitcls"], quick, rep, lbl)
        if "reshitcls" not in d:
            rep(f"  {lbl}: no `reshitcls` branch")
            continue
        rc = d["reshitcls"]
        nh = np.array([int(np.sum(np.asarray(x, dtype=int) >= 0)) for x in rc])
        nm = np.array([int(np.sum(np.asarray(x, dtype=int) < 0)) for x in rc])
        counts[lbl] = nh
        rep(f"  {lbl}: hit blocks/cand mean {nh.mean():7.2f} median {np.median(nh):6.1f}"
            f" min {nh.min():4d} max {nh.max():4d}   |   material blocks/cand"
            f" mean {nm.mean():7.2f} median {np.median(nm):6.1f}")
    if "v_default/gun_tt" in counts and "v_noblocks/gun_tt" in counts:
        rep.gate("(2) hit blocks present in v_default, absent in v_noblocks",
                 bool(counts["v_default/gun_tt"].mean() > 0
                      and counts["v_noblocks/gun_tt"].max() == 0),
                 f"default mean {counts['v_default/gun_tt'].mean():.2f}, "
                 f"noblocks max {counts['v_noblocks/gun_tt'].max()}")
    elif "v_default/gun_tt" in counts:
        rep.gate("(2) hit blocks present in v_default, absent in v_noblocks", None,
                 "v_noblocks not available")

    # ---- VGF INVARIANCE (the load-bearing check) -----------------------
    rep.sub("VGF INVARIANCE  v_default vs v_noblocks  (a change here is a silent physics change)")
    if tdef is None or tnob is None:
        rep("  one of the two files is not available")
        rep.gate("(2) VGF invariance cfmass_vgf / resinfcov", None, "inputs not available")
    else:
        key = ["run", "lumi", "event"]
        cmp_br = ["cfmass_vgf", "resinfcov", "Jpsi_sigmamass", "cfmass_ok"]
        da, _ = arrs(tdef, key + cmp_br, quick, rep, "v_default")
        db, _ = arrs(tnob, key + cmp_br, quick, rep, "v_noblocks")
        if all(k in da for k in key) and all(k in db for k in key):
            ka = list(zip(da["run"], da["lumi"], da["event"]))
            kb = list(zip(db["run"], db["lumi"], db["event"]))
            if ka == kb:
                ia = np.arange(len(ka))
                ib = np.arange(len(kb))
                rep(f"  candidate ordering identical: {len(ka)} entries matched 1:1")
            else:
                from collections import defaultdict
                pos = defaultdict(list)
                for j, k in enumerate(kb):
                    pos[k].append(j)
                ia, ib = [], []
                used = defaultdict(int)
                for i, k in enumerate(ka):
                    u = used[k]
                    if k in pos and u < len(pos[k]):
                        ia.append(i)
                        ib.append(pos[k][u])
                        used[k] += 1
                ia, ib = np.asarray(ia), np.asarray(ib)
                rep(f"  ordering differs; matched {len(ia)} of {len(ka)}/{len(kb)} on (run,lumi,event)")
            worst = 0.0
            for b in ["cfmass_vgf", "resinfcov"]:
                if b not in da or b not in db:
                    rep(f"  {b}: missing in one of the two files")
                    continue
                A = np.asarray(da[b], dtype=np.float64)[ia]
                B = np.asarray(db[b], dtype=np.float64)[ib]
                nbit = int(np.sum(np.asarray(da[b])[ia] != np.asarray(db[b])[ib]))
                den = np.maximum(np.abs(A), np.abs(B))
                den = np.where(den > 0, den, 1.0)
                rel = np.abs(A - B) / den
                worst = max(worst, float(rel.max()))
                rep(f"  {b:<14s} max |rel diff| {rel.max():.4e}   max |abs diff|"
                    f" {np.abs(A - B).max():.4e}   bitwise-different entries {nbit}/{len(ia)}"
                    f"   mean(default) {A.mean():.6e}")
            rep.gate("(2) VGF invariance cfmass_vgf / resinfcov (rel <= 1e-13)",
                     bool(worst <= 1e-13), f"max rel diff {worst:.3e}")
        else:
            rep.gate("(2) VGF invariance cfmass_vgf / resinfcov", None, "key branches missing")

    # ---- vgf split -----------------------------------------------------
    rep.sub("`resinfcovhit` and the VGF SPLIT   hit / (hits + beamspot + pointing)")
    for lbl, cfg, smoke in (("v_default/gun_tt", "v_default", "gun_tt"),
                            ("v_grp/gun_tt", "v_grp", "gun_tt"),
                            ("v_default/data_tt", "v_default", "data_tt"),
                            ("v_noblocks/gun_tt", "v_noblocks", "gun_tt")):
        t = get(path_of(cfg, smoke), rep)
        if t is None:
            rep(f"  {lbl}: input not available")
            continue
        d, _ = arrs(t, ["resinfcovhit", "cfmass_vgf", "Jpsi_sigmamass", "resinfcov"],
                    quick, rep, lbl)
        if "resinfcovhit" not in d:
            rep(f"  {lbl}: no `resinfcovhit`")
            continue
        ch = np.asarray(d["resinfcovhit"], dtype=np.float64)
        rep(f"  {lbl}: mean resinfcovhit {ch.mean():.6e}")
        if "cfmass_vgf" in d and "Jpsi_sigmamass" in d:
            vgf = np.asarray(d["cfmass_vgf"], dtype=np.float64)
            sm = np.asarray(d["Jpsi_sigmamass"], dtype=np.float64)
            den = sm ** 2 * vgf
            m = (den > 0) & np.isfinite(den) & np.isfinite(ch)
            if m.sum():
                r = ch[m] / den[m]
                rep(f"      vgf split = resinfcovhit/(sigma_m^2 cfmass_vgf) :"
                    f" median {np.median(r):.8f}   (16,84) ({np.percentile(r, 16):.8f},"
                    f" {np.percentile(r, 84):.8f})   min {r.min():.8f} max {r.max():.8f}"
                    f"   [n {int(m.sum())}]")
                med = float(np.median(r))
                rest = "nothing, to float32" if abs(1. - med) < 1e-4 else "the remainder"
                rep(f"      -> the HIT blocks carry {med * 100:.5f} % of the total"
                    f" Gaussian share; beamspot + pointing carry"
                    f" {(1. - med) * 100:+.4g} % ({rest})")
            rc = np.asarray(d["resinfcov"], dtype=np.float64) if "resinfcov" in d else None
            if rc is not None and m.sum():
                rep(f"      material share resinfcov/sigma_m^2 median"
                    f" {np.median(rc[m] / sm[m] ** 2):.6f}   |   cfmass_vgf median"
                    f" {np.median(vgf[m]):.6f}")
                idr = np.abs(sm[m] ** 2 - rc[m] - ch[m]) / sm[m] ** 2
                rep(f"      identity |sigma_m^2 - resinfcov - resinfcovhit|/sigma_m^2:"
                    f" median {np.median(idr):.3e}  max {idr.max():.3e}")

    # ---- per-class tables ----------------------------------------------
    rep.sub("PER-CLASS variance shares  --  TWO-TRACK (cfmass_hitcls / cfmass_hitv)")
    rep("  (the `ref` column is the offline low-pT muon-gun SINGLE-TRACK study; it is")
    rep("   printed here only for orientation -- the like-for-like comparison is the")
    rep("   single-track table below.  For the two-track file the `sum_c hitv vs vgf`")
    rep("   line IS the vgf split: it equals 1 only if beamspot + pointing carry no")
    rep("   variance.)")
    for lbl, cfg, smoke in (("v_default/gun_tt", "v_default", "gun_tt"),
                            ("v_grp/gun_tt", "v_grp", "gun_tt"),
                            ("v_default/data_tt", "v_default", "data_tt")):
        t = get(path_of(cfg, smoke), rep)
        if t is None:
            rep(f"  {lbl}: input not available")
            continue
        hitcls_table(rep, t, "cfmass", quick, lbl, vgf_exact_gate=False)

    rep.sub("PER-CLASS variance shares  --  SINGLE-TRACK (cfqop_hitcls / cfqop_hitv)")
    done_st = False
    for lbl, cfg in (("v_default/gun_st", "v_default"), ("v_grp/gun_st", "v_grp"),
                     ("v_noblocks/gun_st", "v_noblocks")):
        t = get(path_of(cfg, "gun_st"), rep)
        if t is None:
            rep(f"  {lbl}: input not available")
            continue
        hitcls_table(rep, t, "cfqop", quick, lbl,
                     vgf_exact_gate=(cfg != "v_noblocks"))
        done_st = True
    if not done_st:
        rep.gate("(2) sum_c cfqop_hitv == cfqop_vgf", None, "no single-track file available")


# ==========================================================================
# (3) PER-LEG COVARIANCE
# ==========================================================================
IU, JU = np.triu_indices(6)


def unpack6(v):
    """21-float upper triangle (row-major) -> symmetric 6x6."""
    C = np.zeros((6, 6))
    C[IU, JU] = v
    C[JU, IU] = v
    return C


def section3(rep, quick):
    rep.head("(3) PER-LEG COVARIANCE  --  Jpsi_covrefmom / Jpsi_jacrefmom and the derived summaries")
    rep("")
    rep("  Jpsi_covrefmom = upper triangle, row-major, of the 6x6 cov of")
    rep("  (q/p, lambda, phi)_mu+ then (q/p, lambda, phi)_mu-;  Jpsi_jacrefmom = dm/d(state).")

    any_run = False
    for lbl, cfg, smoke in (("v_default/gun_tt", "v_default", "gun_tt"),
                            ("v_default/data_tt", "v_default", "data_tt"),
                            ("v_grp/gun_tt", "v_grp", "gun_tt")):
        t = get(path_of(cfg, smoke), rep)
        if t is None:
            rep.gate(f"(3) J C J^T closure [{lbl}]", None, "input not available")
            continue
        names = ["Jpsi_covrefmom", "Jpsi_jacrefmom", "Jpsi_qoprefplus", "Jpsi_qoprefminus",
                 "Jpsi_sigmarelplus", "Jpsi_sigmarelminus", "Jpsi_rhomom", "Jpsi_fang",
                 "Jpsi_sigmamass", "Jpsi_mass"]
        d, miss = arrs(t, names, quick, rep, lbl)
        if "Jpsi_covrefmom" not in d or "Jpsi_jacrefmom" not in d:
            rep.gate(f"(3) J C J^T closure [{lbl}]", None, "covariance branches missing")
            continue
        rep.sub(f"{lbl}  ({t.num_entries} entries)")
        any_run = True
        cov, jac = d["Jpsi_covrefmom"], d["Jpsi_jacrefmom"]
        n = len(cov)
        good = np.array([len(cov[i]) == 21 and len(jac[i]) == 6 for i in range(n)])
        rep(f"  candidates with a filled 21+6 export: {good.sum()}/{n}"
            f"   (empty: {n - int(good.sum())})")
        idx = np.where(good)[0]
        if idx.size == 0:
            rep.gate(f"(3) J C J^T closure [{lbl}]", None, "no filled exports")
            continue
        sm = np.asarray(d["Jpsi_sigmamass"], dtype=np.float64)
        mass = np.asarray(d["Jpsi_mass"], dtype=np.float64)
        ratio = np.full(n, np.nan)
        srp_r = np.full(n, np.nan)
        srm_r = np.full(n, np.nan)
        rho_r = np.full(n, np.nan)
        fang_r = np.full(n, np.nan)
        for i in idx:
            C = unpack6(np.asarray(cov[i], dtype=np.float64))
            J = np.asarray(jac[i], dtype=np.float64)
            s2 = float(J @ C @ J)
            if s2 > 0 and sm[i] > 0:
                ratio[i] = np.sqrt(s2) / sm[i]
            qp = float(d["Jpsi_qoprefplus"][i]) if "Jpsi_qoprefplus" in d else np.nan
            qm = float(d["Jpsi_qoprefminus"][i]) if "Jpsi_qoprefminus" in d else np.nan
            if np.isfinite(qp) and qp != 0:
                srp_r[i] = np.sqrt(max(C[0, 0], 0.0)) / abs(qp)
            if np.isfinite(qm) and qm != 0:
                srm_r[i] = np.sqrt(max(C[3, 3], 0.0)) / abs(qm)
            if (np.isfinite(srp_r[i]) and np.isfinite(srm_r[i]) and srp_r[i] > 0
                    and srm_r[i] > 0 and qp != 0 and qm != 0):
                rho_r[i] = C[0, 3] / (qp * qm) / (srp_r[i] * srm_r[i])
            if sm[i] > 0:
                sk = (J[0] ** 2 * C[0, 0] + 2 * J[0] * J[3] * C[0, 3] + J[3] ** 2 * C[3, 3])
                fang_r[i] = 1.0 - sk / sm[i] ** 2
        r = ratio[np.isfinite(ratio)]
        rep(f"  sqrt(J C J^T)/Jpsi_sigmamass:  median {np.median(r):.8f}"
            f"   p1 {np.percentile(r, 1):.8f}   p99 {np.percentile(r, 99):.8f}"
            f"   max|1-x| {np.max(np.abs(r - 1)):.3e}")
        rep.gate(f"(3) median sqrt(J C J^T)/sigma_m within 1e-4 of 1 [{lbl}]",
                 bool(abs(np.median(r) - 1.0) <= 1e-4),
                 f"median {np.median(r):.8f}, max dev {np.max(np.abs(r - 1)):.2e}")

        # recomputation vs export
        rep("")
        rep(f"  {'quantity':<20s} {'max |recomp - exported|':>24s} {'max rel':>12s}")
        okrec = True
        for nm, rec, ex in (("Jpsi_sigmarelplus", srp_r, d.get("Jpsi_sigmarelplus")),
                            ("Jpsi_sigmarelminus", srm_r, d.get("Jpsi_sigmarelminus")),
                            ("Jpsi_rhomom", rho_r, d.get("Jpsi_rhomom")),
                            ("Jpsi_fang", fang_r, d.get("Jpsi_fang"))):
            if ex is None:
                rep(f"  {nm:<20s} {'--- branch missing ---':>24s}")
                okrec = False
                continue
            e = np.asarray(ex, dtype=np.float64)
            m = np.isfinite(rec) & np.isfinite(e)
            if m.sum() == 0:
                rep(f"  {nm:<20s} {'--- no valid entries ---':>24s}")
                continue
            ad = np.abs(rec[m] - e[m])
            den = np.maximum(np.abs(e[m]), 1e-30)
            rl = ad / den
            rep(f"  {nm:<20s} {ad.max():24.4e} {rl.max():12.4e}")
            okrec &= bool(rl.max() <= 1e-5 or ad.max() <= 1e-7)
        rep.gate(f"(3) recomputed sigmarel / rhomom / fang match exports [{lbl}]",
                 okrec, "float32 precision")

        # distributions
        rep("")
        rho = np.asarray(d["Jpsi_rhomom"], dtype=np.float64) if "Jpsi_rhomom" in d else rho_r
        fang = np.asarray(d["Jpsi_fang"], dtype=np.float64) if "Jpsi_fang" in d else fang_r
        srp = np.asarray(d["Jpsi_sigmarelplus"], dtype=np.float64) if "Jpsi_sigmarelplus" in d else srp_r
        srm = np.asarray(d["Jpsi_sigmarelminus"], dtype=np.float64) if "Jpsi_sigmarelminus" in d else srm_r
        rho, fang, srp, srm = rho[good], fang[good], srp[good], srm[good]
        smrel = (sm / np.where(mass != 0, mass, np.nan))[good]
        smrel = np.where(np.isfinite(smrel), smrel, np.nan)
        # cross-leg correlation structure of C, element by element
        nmv = ["q/p", "lam", "phi"]
        corr = np.full((3, 3, len(idx)), np.nan)
        for k, i in enumerate(idx):
            C = unpack6(np.asarray(cov[i], dtype=np.float64))
            for a in range(3):
                for b in range(3):
                    dd = C[a, a] * C[3 + b, 3 + b]
                    if dd > 0:
                        corr[a, b, k] = C[a, 3 + b] / np.sqrt(dd)
        rep("")
        rep("  CROSS-LEG correlation C[a+, b-]/sqrt(C[a+,a+] C[b-,b-]), median |corr|:")
        rep(f"    {'':<6s}" + "".join(f"{'  ' + n + '(-)':>14s}" for n in nmv))
        for a in range(3):
            rep(f"    {nmv[a] + '(+)':<6s}"
                + "".join(f"{np.nanmedian(np.abs(corr[a, b])):14.3e}" for b in range(3)))
        rep("    -> q/p(+) x q/p(-) is ZERO to double precision: the legs' curvatures are")
        rep("       uncorrelated in this fit, so Jpsi_rhomom carries no information here.")
        rep("")
        rep("  DISTRIBUTIONS   median (16, 84):")
        rep(f"    Jpsi_rhomom          {fmtq(rho, '{:+.3e}')}"
            f"   [max |rho| {np.nanmax(np.abs(rho)):.3e}]")
        rep(f"    Jpsi_fang            {fmtq(fang, '{:+.4f}')}"
            f"   [min {np.nanmin(fang):+.4f} max {np.nanmax(fang):+.4f};"
            f" f_ang < 0 in {np.sum(fang < 0)}/{len(fang)} candidates -- f_ang includes the"
            f" curvature-angle CROSS terms, which are negative here]")
        rep(f"    Jpsi_sigmarelplus    {fmtq(srp, '{:.5f}')}")
        rep(f"    Jpsi_sigmarelminus   {fmtq(srm, '{:.5f}')}")
        rep(f"    sigma_m / m          {fmtq(smrel, '{:.5f}')}")
        mr, mf = float(np.median(rho)), float(np.median(fang))
        inr = REF_RHO[0] <= mr <= REF_RHO[1]
        inf = REF_FANG[0] <= mf <= REF_FANG[1]
        rep(f"    gen reference: rho in [{REF_RHO[0]}, {REF_RHO[1]}] -> median {mr:+.5f}"
            f" is {'INSIDE' if inr else 'OUTSIDE'}")
        rep(f"    gen reference: f_ang in [{REF_FANG[0]}, {REF_FANG[1]}] -> median {mf:+.4f}"
            f" is {'INSIDE' if inf else 'OUTSIDE'}")
        if smoke == "gun_tt":
            rep.gate(f"(3) rho and f_ang medians inside the gen reference ranges [{lbl}]",
                     bool(inr and inf),
                     f"rho {mr:+.3e} (ref {REF_RHO}), f_ang {mf:.4f} (ref {REF_FANG})")

        # quantile bins in sigma_m/m
        v = smrel[np.isfinite(smrel)]
        if v.size >= 5:
            edges = np.percentile(smrel[np.isfinite(smrel)], [0, 20, 40, 60, 80, 100])
            rep("")
            rep(f"  binned in 5 quantiles of sigma_m/m   (reference: f_ang falls 0.142 -> 0.098)")
            rep(f"    {'bin':<4s} {'sigma_m/m range':>24s} {'n':>5s} {'median rho':>12s} {'median f_ang':>13s}")
            for b in range(5):
                lo, hi = edges[b], edges[b + 1]
                m = (smrel >= lo) & (smrel <= hi if b == 4 else smrel < hi)
                m &= np.isfinite(smrel)
                if m.sum() == 0:
                    continue
                rep(f"    {b:<4d} {f'[{lo:.5f}, {hi:.5f}]':>24s} {int(m.sum()):5d}"
                    f" {np.nanmedian(rho[m]):12.3e} {np.nanmedian(fang[m]):13.4f}")
    if not any_run:
        rep("  no two-track file with the covariance export was available")


# ==========================================================================
# (4) PRE-FSR MASS
# ==========================================================================
def section4(rep, quick):
    rep.head("(4) PRE-FSR MASS  --  Jpsigenpre_* and Jpsigen_massdressed")
    rep("")
    rep("  NOTE: the J/psi smoke input is a PARTICLE GUN.  A status-22/62 hard-process")
    rep("  resonance need not exist in such a sample, so a large -99 fraction on")
    rep("  Jpsigenpre_mass is INFORMATION about the sample, not a failure of the export.")
    names = ["Jpsigenpre_mass", "Jpsigenpre_masslep", "Jpsigenpre_status",
             "Jpsigen_massdressed", "Jpsigen_mass"]
    any_run = False
    for lbl, cfg in (("v_default/gun_tt", "v_default"), ("v_grp/gun_tt", "v_grp"),
                     ("v_noblocks/gun_tt", "v_noblocks")):
        t = get(path_of(cfg, "gun_tt"), rep)
        if t is None:
            rep(f"  {lbl}: input not available")
            continue
        d, miss = arrs(t, names, quick, rep, lbl)
        if not d:
            continue
        rep.sub(f"{lbl}  ({t.num_entries} entries)")
        any_run = True
        n = None
        for b in names:
            if b not in d:
                rep(f"  {b:<22s} MISSING")
                continue
            a = np.asarray(d[b], dtype=np.float64)
            n = len(a)
            valid = a != -99
            rep(f"  {b:<22s} valid {valid.sum():5d}/{n:<5d} ({valid.mean() * 100:6.2f}%)"
                + (f"   median {np.median(a[valid]):.6f}" if valid.sum() else ""))
        if "Jpsigenpre_status" in d:
            st = np.asarray(d["Jpsigenpre_status"], dtype=np.int64)
            u, c = np.unique(st, return_counts=True)
            rep(f"  Jpsigenpre_status values: " + ", ".join(f"{int(a)}:{int(b)}" for a, b in zip(u, c)))
        for pre in ("Jpsigenpre_mass", "Jpsigen_massdressed"):
            if pre in d and "Jpsigen_mass" in d:
                a = np.asarray(d[pre], dtype=np.float64)
                g = np.asarray(d["Jpsigen_mass"], dtype=np.float64)
                m = (a != -99) & (g != -99) & np.isfinite(a) & np.isfinite(g)
                if m.sum() == 0:
                    rep(f"  ({pre} - Jpsigen_mass): no candidate has both valid")
                    continue
                dd = a[m] - g[m]
                rep(f"  ({pre} - Jpsigen_mass): n {m.sum():5d}  median {np.median(dd):+.6e}"
                    f"  RMS {np.sqrt(np.mean(dd ** 2)):.6e}  min {dd.min():+.4e} max {dd.max():+.4e}")
        rep.gate(f"(4) pre-FSR branches present [{lbl}]",
                 bool(all(b in d for b in names)),
                 f"missing {[b for b in names if b not in d]}" if miss else "all four present")
    if not any_run:
        rep("  no gun_tt file available")


# ==========================================================================
# (5) BYTES PER CANDIDATE
# ==========================================================================
def section5(rep, quick):
    rep.head("(5) BYTES PER CANDIDATE")
    rep("")
    rep("  measured separately, see NOTES -- this section is a deliberate no-op.")
    rep.gate("(5) bytes/candidate", None, "measured separately, see NOTES")


# ==========================================================================
def wait_for_inputs(rep, wait_s):
    """Poll for the inputs that are still being produced."""
    want = []
    for cfg in CONFIGS:
        for smoke in ("gun_tt", "gun_st", "data_tt"):
            if cfg == "base_prod2" and smoke != "data_tt":
                continue
            if (cfg, smoke) in SKIP_INPUTS:
                continue
            want.append(path_of(cfg, smoke))
    t0 = time.time()
    while True:
        miss = [p for p in want if not ready(p)]
        if not miss or (time.time() - t0) > wait_s:
            break
        rep(f"  waiting for {len(miss)} input(s) "
            f"({int(time.time() - t0)} s of {wait_s} s elapsed) ...")
        for p in miss:
            rep(f"    pending: {p}")
        time.sleep(30)
    miss = [p for p in want if not ready(p)]
    rep("")
    rep(f"  inputs ready : {len(want) - len(miss)}/{len(want)}")
    for p in want:
        st = "READY  " if ready(p) else ("EXISTS-BUT-UNREADABLE" if os.path.exists(p) else "MISSING")
        rep(f"    {st:<22s} {p}")
    return miss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="run only these sections, e.g. '3' or '1,2'")
    ap.add_argument("--quick", action="store_true",
                    help="use at most 20 candidates per file")
    ap.add_argument("--wait", type=float, default=720.,
                    help="seconds to poll for still-missing inputs (default 720)")
    ap.add_argument("--out", default=OUTFILE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rep = Report(args.out)
    rep("CVH EXPORT VALIDATION  --  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    rep(f"root      : {ROOT}")
    rep(f"report    : {args.out}")
    rep(f"quick     : {args.quick}   (at most 20 candidates/file)" if args.quick
        else "quick     : False (all candidates)")
    rep(f"numpy {np.__version__}   uproot {uproot.__version__}"
        + ("   wums logging" if logger is not None else "   (no wums; plain prints)"))

    sel = set(range(1, 6))
    if args.only:
        try:
            sel = set(int(x) for x in args.only.replace(" ", "").split(","))
        except ValueError:
            rep(f"  !! could not parse --only {args.only}; running everything")

    rep.head("INPUT AVAILABILITY")
    missing = wait_for_inputs(rep, args.wait)

    for i, fn in ((1, section1), (2, section2), (3, section3), (4, section4), (5, section5)):
        if i not in sel:
            continue
        try:
            fn(rep, args.quick)
        except Exception as e:
            import traceback
            rep(f"  !! SECTION {i} CRASHED: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                rep("     " + line)
            rep.gate(f"({i}) section completed", False, f"{type(e).__name__}: {e}")

    if missing:
        rep.head("INPUTS STILL MISSING AFTER THE POLL")
        for p in missing:
            rep("  " + p)
    rep.close()
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does the IN-MAKER CF evaluator reproduce the offline reference?

`cvhcf` (TrackPropagation/Geant4e/src/CvhCfExponents.cc) implements the row
functions of `cf_rows` (scattering, ionisation, radiation, the hard knock-on
collision) and their fit-level assembly `cf_rows.fit_families`, with the
pooling of `cf_track_resolution.extract` / `cf_mass_likelihood.
build_pairs_tt`.  This runs both on the SAME tree entries (a maker file with
the raw step records) and reports the maximum absolute difference per family.

WHY AN ABSOLUTE TOLERANCE ON S. It is a centred log-CF: identically 0 at
tau = 0 and O(-1..-60) at the far end, so a relative comparison blows up at
small tau where both are legitimately tiny. What is consumed downstream is
exp(S), so an absolute error on S is the meaningful figure -- and the caches
themselves are float32, which is the floor any comparison of this kind is
measured against.

LEVELS, deliberately separate:
  BLOCK level : per pooled block, the C++ row functions (through
                `cvhcf_rows`) against `cf_rows`' at the SAME entries: Sms, Sio,
                the knock-on map and joint pieces with the block's paired
                angular weight, Srad on the refined spectra, and sq2.  A
                failure here is a formula error.
  TRACK level : the whole `cvhcf::trackExponents` against
                `cf_rows.fit_families` on the extractor's blocks.  A failure
                here that block level passes is a POOLING or PAIRING error.
                With --groups the per-group split must sum to the flat
                families exactly.
  NUCLEAR-ELASTIC family (--nucel, needs the step records with `msmatv`,
                `mspdgv` and the `materials` tree): `cvhcf` NucelExponents
                -- angular, recoil and joint parts -- against
                `ks_nucel_cf.step_family` (the shipped table through
                `nucel_tables.Table.rows`), per MS index, per track (and group).
  BRANCH level (--compare-branches): the `cfqop_*` / `cfmass_*` floats the
                MAKER wrote, against the same reference -- the only level that
                also tests what the maker fed the evaluator.  float32, so its
                floor is ~6e-8 max|S|.

An entry the reference refuses (ValueError: the radiative rows not parallel
to the scattering rows, a record it cannot read) must be refused by the C++ as
well; both are counted.

The reference is evaluated in numpy's baseline (libm) arithmetic, as
`cxx/gate_cvhcf_rows.py` evaluates it and for the same reason: numpy's SVML
and FMA dispatch is not reproducible libm arithmetic, and the reference has
places where one ulp is visible (the knock-on map's cancelling Filon sums; the
refined spectra's fine nodes, `exp(log(v_k))`, compared against the exported
nodes at a spectrum's first nonzero point).  --native keeps the dispatch on.

usage:
  source /work/submit/david_w/ZMass/calibration_studies/setup_env.sh
  ./cxx/build_cvhcf.sh
  python3 cvhcf_validate.py --file <globalcor.root> [--ntracks 30] [--mass] [--groups]
"""
import argparse
import ctypes
import os
import sys
import time

_BASELINE = ("FMA3 AVX2 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX "
             "AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR")
if "--native" not in sys.argv and os.environ.get("NPY_DISABLE_CPU_FEATURES") != _BASELINE:
    os.environ["NPY_DISABLE_CPU_FEATURES"] = _BASELINE
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ksclosure", "nucel"))

import cf_brems_exact                                              # noqa: E402
import cf_knockon                                                  # noqa: E402
import cf_rows                                                     # noqa: E402
import cf_track_resolution as ctr                                  # noqa: E402
import cvhcf_rows                                                  # noqa: E402
import prodfiles                                                   # noqa: E402
from cf_mass_likelihood import IONI_SGN, RAD_SGN                   # noqa: E402

# the reference side is the numpy implementation, whatever CF_ROWS_BACKEND
# says: cf_rows would otherwise dispatch to the very .so under test
cf_rows.BACKEND = "py"

NFAM = 9      # ms, ioRe, ioIm, radRe, radIm, kxRe, kxIm, kjRe, kjIm
MAXG = 256
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        L = cvhcf_rows.lib()
        f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        u32 = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        ci, cd, vp = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
        L.cvhcf_ntau.restype = ci
        L.cvhcf_tau_grid.argtypes = [f64]
        L.cvhcf_model_tag.argtypes = [ctypes.c_char_p, ci]
        L.cvhcf_model_tag.restype = ci
        L.cvhcf_ioni_sq2.argtypes = [f32, ci, ci, vp, ci]
        L.cvhcf_ioni_sq2.restype = cd
        L.cvhcf_track.argtypes = [
            u32, i32, f32, ci,
            u32, f32, ci, ci,
            u32, f32, ci, ci,
            u32, f32, ci,
            u32, f32, ci, ci, f32, f32, ci,
            cd, cd, ci, ci, f64, f64, ci, i32, f64, f64, i32]
        L.cvhcf_track.restype = ci
        L.cvhcf_nucel_ctx_new.restype = vp
        L.cvhcf_nucel_ctx_free.argtypes = [vp]
        L.cvhcf_nucel_add_material.argtypes = [vp, ci, ci, f64, f64, f64]
        L.cvhcf_nucel_add_material.restype = ci
        L.cvhcf_nucel_rows.argtypes = [vp, f64, ci, f64, ci, ci, i32, i32, f64, i32, f64, f64, ci,
                                       f64, f64, f64, f64]
        L.cvhcf_nucel_rows.restype = ci
        L.cvhcf_track_nucel.argtypes = [
            vp,
            u32, i32, f32, ci,
            u32, f32, ci, ci, i32, i32,
            u32, f32, ci, ci,
            u32, f32, ci,
            u32, f32, ci, ci, f32, f32, ci,
            cd, cd, ci, f64, f64,
            f64, ci, i32, f64, i32]
        L.cvhcf_track_nucel.restype = ci
        _LIB = L
    return _LIB


def last_error():
    n = lib().cvhcf_last_error(None, 0)
    b = ctypes.create_string_buffer(n + 1)
    lib().cvhcf_last_error(b, n + 1)
    return b.value.decode()


def nucel_ctx(filemats):
    """A shim material context holding the FILE's own material table."""
    ctx = lib().cvhcf_nucel_ctx_new()
    for m in filemats:
        z = np.ascontiguousarray(m["Z"], dtype=np.float64)
        a = np.ascontiguousarray(m["A"], dtype=np.float64)
        w = np.ascontiguousarray(m["W"], dtype=np.float64)
        if lib().cvhcf_nucel_add_material(ctx, int(m["index"]), len(z), z, a, w):
            raise RuntimeError("cvhcf_nucel_add_material refused a material")
    return ctx


def nucel_rows_cxx(ctx, tau, rows, mat, pdg, mass, rid, wb, frac, wq, wbm):
    """`cvhcf::nucelRows` (the port of `nucel_tables.Table.rows`): (ang, rec,
    jnt, N)."""
    nt = len(tau)
    out = np.zeros(5 * nt, dtype=np.float64)
    N = np.zeros(1, dtype=np.float64)
    r = np.ascontiguousarray(rows, dtype=np.float64)
    c = lambda x, t=np.float64: np.ascontiguousarray(x, dtype=t)
    rc = lib().cvhcf_nucel_rows(ctx, c(tau), nt, r.ravel(), r.shape[1], r.shape[0], c(mat, np.int32),
                                c(pdg, np.int32), c(mass), c(rid, np.int32), c(wb), c(frac), len(rid),
                                c(wq), c(wbm), out, N)
    if rc:
        raise RuntimeError(f"cvhcf_nucel_rows: rc={rc}: {last_error()}")
    return (out[:nt], out[nt:2 * nt] + 1j * out[2 * nt:3 * nt],
            out[3 * nt:4 * nt] + 1j * out[4 * nt:], float(N[0]))


def tau_grid():
    n = lib().cvhcf_ntau()
    t = np.zeros(n, dtype=np.float64)
    lib().cvhcf_tau_grid(t)
    return t


def model_tag():
    n = lib().cvhcf_model_tag(None, 0)
    b = ctypes.create_string_buffer(n + 1)
    lib().cvhcf_model_tag(b, n + 1)
    return b.value.decode()


def ioni_sq2_cxx(rows, qsc):
    q = np.ascontiguousarray(qsc, dtype=np.float32).ravel() if qsc is not None and len(qsc) else None
    return lib().cvhcf_ioni_sq2(
        np.ascontiguousarray(rows, dtype=np.float32).ravel(), rows.shape[1], rows.shape[0],
        q.ctypes.data_as(ctypes.c_void_p) if q is not None else ctypes.c_void_p(0),
        0 if q is None else len(qsc))


# --------------------------------------------------------------------------
def reshape(v, idx, branch, a=None, ic=0):
    """A step-record branch as (nrecord, stride), stride taken from the FILE."""
    return prodfiles.reshape_records(
        v, nrec=len(idx),
        stride=(prodfiles.entry_stride(a, branch, ic) if a is not None else None),
        branch=branch, dtype=np.float32)


class Acc:
    """max |difference| per family, plus where it happened."""

    def __init__(self):
        self.m = {}
        self.n = {}

    def add(self, key, a, b):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if len(np.atleast_1d(a)) else 0.
        if key not in self.m or d > self.m[key]:
            self.m[key] = d
        self.n[key] = self.n.get(key, 0) + 1

    def report(self, title):
        print(f"\n{title}")
        print(f"  {'family':<12} {'n':>7}  {'max |dS|':>12}")
        for k in sorted(self.m):
            print(f"  {k:<12} {self.n[k]:>7}  {self.m[k]:>12.4e}")
        return max(self.m.values()) if self.m else 0.


FAMS = ("ms", "io", "rad", "kx", "kj")
REFKEY = dict(ms="Sms", io="Sio", rad="Srad", kx="Skx", kj="Skj")
BRANCH = dict(ms=("ms",), io=("ioni_re", "ioni_im"), rad=("rad_re", "rad_im"),
              kx=("kx_re", "kx_im"), kj=("kj_re", "kj_im"))


def unpack(out, nt):
    """cvhcf_track's 9 families -> dict of complex arrays."""
    f = out.reshape(NFAM, nt)
    return dict(ms=f[0] + 0j, io=f[1] + 1j * f[2], rad=f[3] + 1j * f[4],
                kx=f[5] + 1j * f[6], kj=f[7] + 1j * f[8])


def block_level(acc, msb, iob, rad, nt):
    """The row functions per block, C++ against python, at fit-level entries."""
    wms = {}
    for g, rows, w in msb:
        wms[int(g)] = abs(w)
        n = len(rows)
        e = (np.arange(n), np.full(n, abs(w)), np.ones(n))
        acc.add("ms", cvhcf_rows.ms_rows(TAU, rows, *e).real, cf_rows.ms_rows(TAU, rows, *e).real)
    beta = {}
    wb_rad = None
    if rad is not None and len(rad["ridx"]):
        wb_rad, beta = cf_rows.fit_pairing(rad["uim"], rad["ridx"], rad["uvm"], wms)
    for g, rows, w in iob:
        n = len(rows)
        wq = np.full(n, w)
        for f, py, cx in (("io", cf_rows.ioni_rows(TAU, rows, wq), cvhcf_rows.ioni_rows(TAU, rows, wq)),):
            acc.add(f + "_re", cx.real, py.real)
            acc.add(f + "_im", cx.imag, py.imag)
        if cf_knockon.active():
            e = (np.arange(n), wq, np.full(n, beta.get(int(g), 0.0)), np.ones(n))
            for part, key in (("map", "kx"), ("joint", "kj")):
                py = cf_rows.knockon_rows(TAU, rows, *e, part=part)
                cx = cvhcf_rows.knockon_rows(TAU, rows, *e, part=part)
                acc.add(key + "_re", cx.real, py.real)
                acc.add(key + "_im", cx.imag, py.imag)
    if wb_rad is not None:
        vp, sp = cf_brems_exact.refine_spectra(rad["rrec"], rad["rspc"], rad["rvg"],
                                               cf_brems_exact.RAD_NSUB)
        vc, sc = cvhcf_rows.refine_spectra(rad["rrec"], rad["rspc"], rad["rvg"],
                                           cf_brems_exact.RAD_NSUB)
        acc.add("refine", sc, sp)
        for g, wr in rad["wrad"].items():
            m = np.flatnonzero(rad["ridx"] == g)
            if not len(m):
                continue
            e = (np.arange(len(m)), np.full(len(m), wr), wb_rad[m], np.ones(len(m)))
            py = cf_rows.rad_rows(TAU, rad["rrec"][m], sp[m], vp, *e)
            cx = cvhcf_rows.rad_rows(TAU, rad["rrec"][m], sp[m], vp, *e)
            acc.add("rad_re", cx.real, py.real)
            acc.add("rad_im", cx.imag, py.imag)


# --------------------------------------------------------------------------
NUC_BENCH = []
TRK_BENCH = []


def bench_report(unit):
    """Wall time per track / candidate of the in-maker families (cvhcf::
    trackExponents through the shim, the arguments of the compared entries
    replayed): all families, the families without the knock-on pair, and --
    with --nucel -- the latter plus the nuclear-elastic family."""
    def run(fn, argl):
        t0 = time.perf_counter()
        for a_ in argl:
            fn(*a_)
        return (time.perf_counter() - t0) / max(len(argl), 1) * 1e3
    if not TRK_BENCH:
        return
    nok = [t[:24] + (0,) + t[25:] for t in TRK_BENCH]
    print(f"\nCOST ({len(TRK_BENCH)} {unit}s, ms per {unit})")
    print(f"  families            {run(lib().cvhcf_track, TRK_BENCH):10.3f}")
    print(f"  without knock-on    {run(lib().cvhcf_track, nok):10.3f}")
    if NUC_BENCH:
        print(f"  without knock-on, with nuclear elastic  {run(lib().cvhcf_track_nucel, NUC_BENCH):10.3f}")


_BASE_PDG = {"pi": 211, "kaon": 321, "proton": 2212}


def nucel_entry(a, ic, pfx, mass, gi, fam, vb, uim, uvm, uii, uvi, qsi, qsv, ridx, rrec, rspc, rvg,
                sig, chg, T, rowmap, ctx, args, acc_nb, acc_nt, acc_nf, acc_ng, spcount,
                have_nucb, have_grpb, bench):
    """The nuclear-elastic family of one entry at the three levels; returns
    max |S_nuc| (for the float32 floor)."""
    import ks_nucel_cf
    nt = len(TAU)
    msmat = np.asarray(a["msmatv"][ic], dtype=np.int32)
    if "mspdgv" in a:
        pdg = np.asarray(a["mspdgv"][ic], dtype=np.int32)
    else:
        q = 1 if float(a["refParms"][ic][0]) >= 0. else -1
        b = _BASE_PDG[args.species]
        pdg = np.full(len(uim), q * b if b != 2212 or q > 0 else -2212, dtype=np.int32)
    for x in np.unique(pdg):
        spcount[int(x)] = spcount.get(int(x), 0) + int((pdg == x).sum())
    rec = dict(gi=gi, fam=fam, vb=vb, M=uvm, midx=uim, I=uvi, iidx=uii, qsi=qsi, qsv=qsv,
               ridx=ridx, R=rrec)
    sign = IONI_SGN if mass else chg
    wang, wrec = ks_nucel_cf.maker_step_weights(rec, sig, sign)
    imat = np.array([rowmap.get(int(i), -1) for i in msmat], dtype=int)
    groups = uvm[:, 9].astype(int) if uvm.shape[1] >= 10 else np.full(len(uim), -1)
    import cf_nucel_exact
    pmass = np.array([cf_nucel_exact.mass_of(int(x)) if x != 0 else 0.0 for x in pdg])
    # ---- block level: every MS global index (the rows of a block share its
    #      angular weight; every row carries its own recoil weight)
    for g in np.unique(uim):
        r = np.where(uim == g)[0]
        py_a, py_r, py_j, py_n = ks_nucel_cf.step_family(T, uvm[r], imat[r], pdg[r], wang[r],
                                                         wrec[r], TAU)
        cx_a, cx_r, cx_j, cx_n = nucel_rows_cxx(ctx, TAU, uvm[r].astype(np.float64), msmat[r],
                                                pdg[r], pmass[r], np.arange(len(r)), wang[r],
                                                np.ones(len(r)), wrec[r], wang[r])
        acc_nb.add("nuc_ang", cx_a, py_a)
        acc_nb.add("nuc_rec_re", cx_r.real, py_r.real)
        acc_nb.add("nuc_rec_im", cx_r.imag, py_r.imag)
        acc_nb.add("nuc_jnt_re", cx_j.real, py_j.real)
        acc_nb.add("nuc_jnt_im", cx_j.imag, py_j.imag)
        acc_nb.add("nuc_N", [cx_n], [py_n])
    # ---- track level, pooling and pairing included
    (py_a, py_r, py_j, py_n), py_g = ks_nucel_cf.step_family(T, uvm, imat, pdg, wang, wrec, TAU,
                                                             groups=groups)
    out = np.zeros(NFAM * nt, dtype=np.float64)
    aux = np.zeros(5, dtype=np.float64)
    nw = 5 * nt + 1
    nuc = np.zeros(nw, dtype=np.float64)
    maxg = 256
    gid = np.zeros(maxg, dtype=np.int32)
    gnuc = np.zeros(maxg * nw, dtype=np.float64)
    ng = np.zeros(1, dtype=np.int32)
    empt32 = np.zeros(0, dtype=np.float32)
    empu32 = np.zeros(0, dtype=np.uint32)
    targs = (ctx,
             np.ascontiguousarray(gi, dtype=np.uint32), np.ascontiguousarray(fam, dtype=np.int32),
             np.ascontiguousarray(vb, dtype=np.float32), len(gi),
             np.ascontiguousarray(uim, dtype=np.uint32), np.ascontiguousarray(uvm.ravel(), dtype=np.float32),
             uvm.shape[0], uvm.shape[1], msmat, pdg,
             np.ascontiguousarray(uii, dtype=np.uint32), np.ascontiguousarray(uvi.ravel(), dtype=np.float32),
             uvi.shape[0], uvi.shape[1],
             np.ascontiguousarray(qsi, dtype=np.uint32) if qsi is not None else empu32,
             np.ascontiguousarray(qsv.ravel(), dtype=np.float32) if qsv is not None else empt32,
             0 if qsv is None else len(qsv),
             np.ascontiguousarray(ridx, dtype=np.uint32), np.ascontiguousarray(rrec.ravel(), dtype=np.float32),
             rrec.shape[0], rrec.shape[1],
             np.ascontiguousarray(rspc.ravel(), dtype=np.float32), np.ascontiguousarray(rvg, dtype=np.float32),
             len(rvg), float(sig), float(sign), 1 if args.groups else 0, out, aux, nuc, maxg, gid, gnuc, ng)
    rc = lib().cvhcf_track_nucel(*targs)
    if rc < 0:
        raise RuntimeError(f"cvhcf_track_nucel: rc={rc}: {last_error()}")
    if bench is not None:
        NUC_BENCH.append(targs)
    parts = (("nuc_ang", lambda v: v[:nt], lambda f: f[0]),
             ("nuc_rec_re", lambda v: v[nt:2 * nt], lambda f: f[1].real),
             ("nuc_rec_im", lambda v: v[2 * nt:3 * nt], lambda f: f[1].imag),
             ("nuc_jnt_re", lambda v: v[3 * nt:4 * nt], lambda f: f[2].real),
             ("nuc_jnt_im", lambda v: v[4 * nt:5 * nt], lambda f: f[2].imag))
    py = (py_a, py_r, py_j, py_n)
    for name, cx, pyf in parts:
        acc_nt.add(name, cx(nuc), pyf(py))
    acc_nt.add("nuc_N", [nuc[5 * nt]], [py_n])
    if args.groups:
        cg = {int(gid[k]): gnuc[k * nw:(k + 1) * nw] for k in range(int(ng[0]))}
        if set(cg) != set(py_g):
            print(f"  entry {ic}: nuclear-elastic groups differ: cxx {sorted(cg)} py {sorted(py_g)}")
        for g in set(cg) & set(py_g):
            for name, cx, pyf in parts:
                acc_ng.add(name, cx(cg[g]), pyf(py_g[g]))
            acc_ng.add("nuc_N", [cg[g][5 * nt]], [py_g[g][3]])
    if have_nucb:
        acc_nf.add("nuc_ang", np.asarray(a[f"{pfx}_nuc_ang"][ic]), py_a)
        acc_nf.add("nuc_rec_re", np.asarray(a[f"{pfx}_nuc_rec_re"][ic]), py_r.real)
        acc_nf.add("nuc_rec_im", np.asarray(a[f"{pfx}_nuc_rec_im"][ic]), py_r.imag)
        acc_nf.add("nuc_jnt_re", np.asarray(a[f"{pfx}_nuc_jnt_re"][ic]), py_j.real)
        acc_nf.add("nuc_jnt_im", np.asarray(a[f"{pfx}_nuc_jnt_im"][ic]), py_j.imag)
        acc_nf.add("nuc_N", [float(a["nuc_N"][ic])], [py_n])
    if have_grpb:
        fg = np.asarray(a[f"{pfx}_grp_nuc"][ic]).astype(int)
        if set(fg.tolist()) != set(py_g):
            print(f"  entry {ic}: exported nuclear-elastic groups differ from the reference")
        fb = {k: np.asarray(a[f"{pfx}_grp_{k}"][ic]).reshape(-1, nt)
              for k in ("nuc_ang", "nuc_rec_re", "nuc_rec_im", "nuc_jnt_re", "nuc_jnt_im")}
        fn = np.asarray(a[f"{pfx}_grp_nuc_N"][ic])
        pyp = {"nuc_ang": lambda f: f[0], "nuc_rec_re": lambda f: f[1].real,
               "nuc_rec_im": lambda f: f[1].imag, "nuc_jnt_re": lambda f: f[2].real,
               "nuc_jnt_im": lambda f: f[2].imag}
        for k, g in enumerate(fg):
            if int(g) in py_g:
                for name, arr in fb.items():
                    acc_nf.add("grp_" + name, arr[k], pyp[name](py_g[int(g)]))
                acc_nf.add("grp_nuc_N", [fn[k]], [py_g[int(g)][3]])
        # the split sums to the flat family (float32 on both sides)
        acc_nf.add("grp_sum_ang", fb["nuc_ang"].astype(np.float64).sum(0),
                   np.asarray(a[f"{pfx}_nuc_ang"][ic]))
    return max(float(np.max(np.abs(py_a))), float(np.max(np.abs(py_r))),
               float(np.max(np.abs(py_j))))


def run(args):
    global TAU
    TAU = tau_grid()
    nt = len(TAU)
    print(f"cvhcf tau grid: {nt} points, [{TAU[0]:.6f}, {TAU[-1]:.6f}]")
    ref = np.linspace(0.0, 14.0, 448)[0:4 * nt:4]
    assert np.array_equal(TAU, ref), "the C++ grid is not the stride-4 subset"
    cvhcf_rows.sync_config()
    print(f"model tag: {model_tag()}")

    f = uproot.open(args.file)
    pt = f["runtree"]["parmtype"].array(library="np")
    t = f["tree"]
    keys = set(t.keys())
    mass = args.mass or ("Jpsi_sigmamass" in keys)
    print(f"{args.file}\n  {'TWO-TRACK (mass functional)' if mass else 'SINGLE TRACK (q/p functional)'}"
          f"  {t.num_entries} entries")

    need = ["reseigidx", "resinfvarv", "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv"]
    need += ["Jpsi_sigmamass", "resinfcov"] if mass else ["refCov", "genParms"]
    for b in ("ioniqscaleidx", "ioniqscalev", "radstepidx", "radstepv", "radstepspecv", "radvgrid"):
        if b in keys:
            need.append(b)
    pfx = "cfmass" if mass else "cfqop"
    cfb = [f"{pfx}_{b}" for fam in FAMS for b in BRANCH[fam]] + [f"{pfx}_vgf", f"{pfx}_ok"]
    have_cf = args.compare_branches and all(b in keys for b in cfb)
    if args.compare_branches and not have_cf:
        print(f"  --compare-branches: {pfx}_* families not all in the tree; skipping")
    if have_cf:
        need += cfb
    have_rad = "radstepv" in need
    nucel = args.nucel
    if nucel:
        import nucel_tables
        for b in ("msmatv", "radstepidx", "radstepv"):
            if b not in keys:
                raise SystemExit(f"--nucel needs the step records with `{b}`")
        need.append("msmatv")
        if "mspdgv" in keys:
            need.append("mspdgv")
        elif not args.species:
            raise SystemExit("--nucel: the file has no `mspdgv`; give --species")
        nucb = [f"{pfx}_nuc_ang", f"{pfx}_nuc_rec_re", f"{pfx}_nuc_rec_im", f"{pfx}_nuc_jnt_re",
                f"{pfx}_nuc_jnt_im", "nuc_N"]
        have_nucb = args.compare_branches and all(b in keys for b in nucb)
        if have_nucb:
            need += nucb
        grpb = [f"{pfx}_grp_nuc", f"{pfx}_grp_nuc_ang", f"{pfx}_grp_nuc_rec_re",
                f"{pfx}_grp_nuc_rec_im", f"{pfx}_grp_nuc_jnt_re", f"{pfx}_grp_nuc_jnt_im",
                f"{pfx}_grp_nuc_N"]
        have_grpb = args.groups and have_nucb and all(b in keys for b in grpb)
        if have_grpb:
            need += grpb
        if not mass:
            need.append("refParms")
        filemats = nucel_tables.read_materials(args.file)
        NTAB = nucel_tables.Table(args.table) if args.table else nucel_tables.Table()
        rowmap = NTAB.register(filemats)
        ctx = nucel_ctx(filemats)
        print(f"  nuclear-elastic reference: {NTAB.path}  ({len(filemats)} materials in the file's table)")
        acc_nb, acc_nt, acc_nf, acc_ng = Acc(), Acc(), Acc(), Acc()
        nuc_maxS = 0.0
        spcount = {}
    have_qsc = "ioniqscalev" in need
    need += prodfiles.stride_keys(keys, ("msmoliv", "ioniurbanv", "radstepv", "radstepspecv"))
    a = t.arrays(need, library="np", entry_stop=min(t.num_entries, 5 * args.ntracks + 50))

    acc_b, acc_t, acc_g, acc_f = Acc(), Acc(), Acc(), Acc()
    nsel = nref_refuse = ncx_refuse = 0
    maxS = 0.0
    for ic in range(len(a["reseigidx"])):
        if mass:
            sig = float(a["Jpsi_sigmamass"][ic])
            if not (np.isfinite(sig) and sig > 0.):
                continue
            chg, rsg = IONI_SGN, RAD_SGN
        else:
            qg = a["genParms"][ic][0]
            c00 = float(a["refCov"][ic][0])
            if qg == 0. or not (c00 > 0.):
                continue
            sig = np.sqrt(c00)
            chg = rsg = float(np.sign(qg))
        gi = np.asarray(a["reseigidx"][ic])
        if not len(gi):
            continue
        vb = np.asarray(a["resinfvarv"][ic], dtype=np.float32)
        fam = pt[gi]
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = reshape(a["msmoliv"][ic], uim, "msmoliv", a, ic)
        uii = np.asarray(a["ioniurbanidx"][ic])
        uvi = reshape(a["ioniurbanv"][ic], uii, "ioniurbanv", a, ic)
        qsi = np.asarray(a["ioniqscaleidx"][ic]) if have_qsc else None
        qsv = np.asarray(a["ioniqscalev"][ic], dtype=np.float32).reshape(-1, 2) if have_qsc else None
        if have_rad:
            ridx = np.asarray(a["radstepidx"][ic])
            rrec = prodfiles.reshape_records(
                a["radstepv"][ic], nrec=len(ridx), stride=prodfiles.entry_stride(a, "radstepv", ic),
                branch="radstepv", dtype=np.float32, min_cols=cf_brems_exact.RADV_NCOLS)
            rspc = prodfiles.reshape_records(
                a["radstepspecv"][ic], nrec=len(ridx), stride=prodfiles.entry_stride(a, "radstepspecv", ic),
                branch="radstepspecv", dtype=np.float32)
            rvg = np.asarray(a["radvgrid"][ic], dtype=np.float32)
        else:
            ridx = rrec = rspc = rvg = None

        # ---------------- the extractor's blocks (float32 rows, widened) ------
        msb, iob, wrad = [], [], {}
        ok = True
        for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
            sel = fam == famcode
            for g in np.unique(gi[sel]):
                vpool = vb[sel & (gi == g)].astype(np.float64).sum()
                if vpool <= 0.:
                    continue
                rows = uv[uidx == g]
                if not len(rows):
                    ok = False
                    break
                steps = rows.astype(np.float64)
                if famcode == 10:
                    sq2 = steps[:, 5].sum()
                    if sq2 <= 0.:
                        continue
                    msb.append((g, steps, np.sqrt(vpool / sq2) / sig))
                else:
                    qs = qsv[qsi == g] if qsv is not None else None
                    sq2 = ctr.ioni_sq2(steps, None if qs is None else qs.astype(np.float64))
                    acc_b.add("sq2", [ioni_sq2_cxx(rows, qs)], [sq2])
                    if sq2 <= 0.:
                        continue
                    wsc = np.sqrt(vpool / sq2) / sig
                    iob.append((g, steps, chg * wsc))
                    wrad[int(g)] = rsg * wsc
            if not ok:
                break
        rad = (dict(uim=uim, ridx=ridx, uvm=uvm.astype(np.float64), rrec=rrec.astype(np.float64),
                    rspc=rspc.astype(np.float64), rvg=rvg.astype(np.float64), wrad=wrad)
               if have_rad else None)

        # ---------------- the reference -------------------------------------
        try:
            F = cf_rows.fit_families(TAU, 1.0, msb, iob, rad) if ok else None
        except ValueError as e:
            F = e

        # ---------------- the whole track through the C++ -------------------
        out = np.zeros(NFAM * nt)
        aux = np.zeros(5)
        gid = np.zeros(MAXG, dtype=np.int32)
        gfam = np.zeros(MAXG * NFAM * nt)
        gvq = np.zeros(2 * MAXG)
        ng = np.zeros(1, dtype=np.int32)
        e32, eu = np.zeros(0, np.float32), np.zeros(0, np.uint32)
        trk = (
            np.ascontiguousarray(gi, np.uint32), np.ascontiguousarray(fam, np.int32),
            np.ascontiguousarray(vb, np.float32), len(gi),
            np.ascontiguousarray(uim, np.uint32), np.ascontiguousarray(uvm.ravel(), np.float32),
            uvm.shape[0], uvm.shape[1],
            np.ascontiguousarray(uii, np.uint32), np.ascontiguousarray(uvi.ravel(), np.float32),
            uvi.shape[0], uvi.shape[1],
            np.ascontiguousarray(qsi, np.uint32) if qsi is not None else eu,
            np.ascontiguousarray(qsv.ravel(), np.float32) if qsv is not None else e32,
            0 if qsv is None else len(qsv),
            np.ascontiguousarray(ridx, np.uint32) if ridx is not None else eu,
            np.ascontiguousarray(rrec.ravel(), np.float32) if rrec is not None else e32,
            0 if rrec is None else rrec.shape[0], 0 if rrec is None else rrec.shape[1],
            np.ascontiguousarray(rspc.ravel(), np.float32) if rspc is not None else e32,
            np.ascontiguousarray(rvg, np.float32) if rvg is not None else e32,
            0 if rvg is None else len(rvg),
            float(sig), float(chg), 1, 1 if args.groups else 0, out, aux, MAXG, gid, gfam, gvq, ng)
        rc = lib().cvhcf_track(*trk)
        if args.bench:
            TRK_BENCH.append(trk)
        if isinstance(F, ValueError):
            nref_refuse += 1
            ncx_refuse += int(rc == -3)
            if rc != -3:
                print(f"  entry {ic}: the reference refuses ({F}) and the C++ does not")
            continue
        if rc == -3:
            raise RuntimeError(f"entry {ic}: cvhcf_track: {last_error()}")
        if bool(aux[0]) != ok:
            print(f"  entry {ic}: ok flag disagrees (cxx {bool(aux[0])} vs py {ok})")
        if not (ok and bool(aux[0])):
            continue
        block_level(acc_b, msb, iob, rad, nt)
        C = unpack(out, nt)
        for fk in FAMS:
            py = F[REFKEY[fk]]
            acc_t.add(fk + "_re", C[fk].real, py.real)
            if fk != "ms":
                acc_t.add(fk + "_im", C[fk].imag, py.imag)
            maxS = max(maxS, float(np.max(np.abs(py))))
        if args.groups:
            G = gfam[:int(ng[0]) * NFAM * nt].reshape(int(ng[0]), NFAM * nt).sum(0)
            acc_g.add("sum_g - flat", G, out)
        if have_cf:
            for fk in FAMS:
                py = F[REFKEY[fk]]
                br = BRANCH[fk]
                acc_f.add(fk + "_re", np.asarray(a[f"{pfx}_{br[0]}"][ic]), py.real)
                if len(br) > 1:
                    acc_f.add(fk + "_im", np.asarray(a[f"{pfx}_{br[1]}"][ic]), py.imag)
            if bool(a[f"{pfx}_ok"][ic]) != ok:
                print(f"  entry {ic}: {pfx}_ok disagrees with the reference")
        if nucel:
            nuc_maxS = max(nuc_maxS, nucel_entry(
                a, ic, pfx, mass, gi, fam, vb, uim, uvm, uii, uvi, qsi, qsv, ridx, rrec, rspc, rvg,
                sig, chg, NTAB, rowmap, ctx, args, acc_nb, acc_nt, acc_nf, acc_ng, spcount,
                have_nucb, have_grpb, True if args.bench else None))
        nsel += 1
        if nsel >= args.ntracks:
            break

    print(f"\n{nsel} entries compared; {nref_refuse} refused by the reference, "
          f"{ncx_refuse} of them refused by the C++ as well")
    worst = max(acc_b.report("BLOCK level (C++ row functions vs cf_rows, fit-level entries)"),
                acc_t.report("TRACK level (cvhcf::trackExponents vs cf_rows.fit_families)"))
    if acc_g.m:
        worst = max(worst, acc_g.report("GROUP split (C++ sum over groups vs its flat families)"))
    if nucel:
        print(f"\nNUCLEAR-ELASTIC family, species of the rows: {spcount}")
        wn = max(acc_nb.report("  BLOCK level (cvhcf::nucelRows vs ks_nucel_cf.step_family), per MS index"),
                 acc_nt.report("  TRACK level (trackExponents nuclear-elastic family vs step_family)"))
        if acc_ng.m:
            wn = max(wn, acc_ng.report("  TRACK level, per material group (C++ vs python)"))
        worst = max(worst, wn)
        if acc_nf.m:
            wf = acc_nf.report("  BRANCH level (the maker's float32 <cf>_nuc_* / nuc_N)")
            print(f"  float32 storage floor here: {6.0e-8 * nuc_maxS:.2e} (max|S_nuc| = {nuc_maxS:.3g})")
            print(f"  nuclear-elastic branch-level worst |dS| = {wf:.4e}")
    if args.bench:
        bench_report("candidate" if mass else "track")
    if nucel:
        lib().cvhcf_nucel_ctx_free(ctx)
    if acc_f.m:
        wf = acc_f.report("BRANCH level (the floats the MAKER wrote, float32)")
        print(f"  float32 storage floor here: {6.0e-8 * maxS:.2e} (max|S| = {maxS:.3g})")
        print(f"  branch-level worst |dS| = {wf:.4e}")
    print(f"\nworst |dS| overall: {worst:.4e}   (tolerance {args.tol:g}, max|S| {maxS:.3g})")
    return 0 if worst <= args.tol else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--ntracks", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--mass", action="store_true",
                   help="force the mass-functional sign convention")
    p.add_argument("--groups", action="store_true",
                   help="evaluate the per-material-group split (and, with --nucel, "
                        "the nuclear-elastic one)")
    p.add_argument("--compare-branches", action="store_true",
                   help="also compare the cfqop_*/cfmass_* branches the maker wrote")
    p.add_argument("--nucel", action="store_true",
                   help="validate the nuclear-elastic family (needs msmatv/mspdgv, "
                        "the radiative rows and the materials tree)")
    p.add_argument("--species", default="",
                   help="with --nucel on a file without `mspdgv`: pi|kaon|proton")
    p.add_argument("--native", action="store_true",
                   help="evaluate the reference with numpy's hardware dispatch on")
    p.add_argument("--bench", action="store_true",
                   help="time the in-maker families per track / candidate on the compared entries")
    p.add_argument("--table", default="",
                   help="with --nucel: the nuclear-elastic table (default: the shipped one)")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()

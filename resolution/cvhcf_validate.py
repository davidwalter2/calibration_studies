#!/usr/bin/env python3
"""Does the IN-MAKER exponent evaluator reproduce the offline reference?

This is the gate for turning the raw step-record export off. `cvhcf`
(TrackPropagation/Geant4e/src/CvhCfExponents.cc) is a port of

    cf_track_resolution.ms_step_exponent      -> S_ms
    cf_track_resolution.ioni_step_exponent    -> S_ioni
    cf_brems_exact.rad_exponent               -> S_rad
    cf_delta_ray.delta_step_exponent - carve  -> S_del

and of `extract()`'s POOLING of blocks by global parameter index. A port is
only worth anything if it is measured against the thing it ports, on the
records that actually occur, so this runs both on the SAME tree entries and
reports the maximum absolute difference per family.

WHY AN ABSOLUTE TOLERANCE ON S. It is a centred log-CF: identically 0 at
tau = 0 and O(-1..-60) at the far end, so a relative comparison blows up at
small tau where both are legitimately tiny. What is consumed downstream is
exp(S), so an absolute error on S is the meaningful figure -- and the caches
themselves are float32, i.e. 7.6e-6 at max|S| = 66, which is the floor any
comparison of this kind is measured against.

TWO LEVELS, deliberately separate:
  BLOCK level : per pooled block, C++ primitive vs python function, at the
                SAME weight. A failure here is a formula error.
  TRACK level : the whole `cvhcf::trackExponents` against `extract()`'s own
                loop. A failure here that block level passes is a POOLING
                error (which index joins which rows, which legs share a scale,
                the order of the sums).
  BRANCH level (--compare-branches): the `cfqop_*` / `cfmass_*` floats the
                MAKER actually wrote, against the same python reference. This
                is the one that gates the slimming switch, because it is the
                only level that also tests what the maker fed the evaluator
                (sigma, the charge sign, which entries it registered).
                It is float32 in the file, so its floor is ~1e-5 at |S| ~ 60,
                not the 1e-11 of the two levels above.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  export CMSSW_SRC=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src
  ./cxx/build_cvhcf.sh
  python3 cvhcf_validate.py --file <globalcor.root> [--ntracks 30] [--mass]
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("CVH_IONI_KOKOULIN", "0")

import cf_brems_exact                                              # noqa: E402
import cf_delta_ray                                                # noqa: E402
import cf_track_resolution as ctr                                  # noqa: E402
from cf_mass_likelihood import IONI_SGN, RAD_SGN                   # noqa: E402

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        so = os.path.join(HERE, "cxx", "libcvhcfshim.so")
        if not os.path.exists(so):
            raise SystemExit(f"{so} missing -- run cxx/build_cvhcf.sh")
        L = ctypes.CDLL(so)
        f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        u32 = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        L.cvhcf_ntau.restype = ctypes.c_int
        L.cvhcf_tau_grid.argtypes = [f64]
        L.cvhcf_model_tag.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.cvhcf_model_tag.restype = ctypes.c_int
        L.cvhcf_ms_block.argtypes = [f32, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_double, f64, ctypes.c_void_p]
        L.cvhcf_ms_block.restype = ctypes.c_int
        L.cvhcf_ioni_sq2.argtypes = [f32, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_int]
        L.cvhcf_ioni_sq2.restype = ctypes.c_double
        L.cvhcf_ioni_block.argtypes = [f32, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_double, f64, f64]
        L.cvhcf_ioni_block.restype = ctypes.c_int
        L.cvhcf_rad_block.argtypes = [f32, ctypes.c_int, ctypes.c_int, f32, f32,
                                      ctypes.c_int, ctypes.c_double, f64, f64]
        L.cvhcf_rad_block.restype = ctypes.c_int
        L.cvhcf_track.argtypes = [
            u32, i32, f32, ctypes.c_int,
            u32, f32, ctypes.c_int, ctypes.c_int,
            u32, f32, ctypes.c_int, ctypes.c_int,
            u32, f32, ctypes.c_int,
            u32, f32, ctypes.c_int, ctypes.c_int, f32, f32, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, f64, f64]
        L.cvhcf_track.restype = ctypes.c_int
        _LIB = L
    return _LIB


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


# --------------------------------------------------------------------------
def ms_block_cxx(rows, wstd, want_del):
    sms = np.zeros(len(TAU), dtype=np.float64)
    sdel = np.zeros(len(TAU), dtype=np.float64) if want_del else None
    p = sdel.ctypes.data_as(ctypes.c_void_p) if want_del else ctypes.c_void_p(0)
    lib().cvhcf_ms_block(rows.ravel(), rows.shape[1], rows.shape[0], wstd, sms, p)
    return sms, sdel


def ioni_block_cxx(rows, wstd):
    re = np.zeros(len(TAU), dtype=np.float64)
    im = np.zeros(len(TAU), dtype=np.float64)
    lib().cvhcf_ioni_block(rows.ravel(), rows.shape[1], rows.shape[0], wstd, re, im)
    return re + 1j * im


def rad_block_cxx(rows, spec, vg, wstd):
    re = np.zeros(len(TAU), dtype=np.float64)
    im = np.zeros(len(TAU), dtype=np.float64)
    lib().cvhcf_rad_block(rows.ravel(), rows.shape[1], rows.shape[0],
                          spec.ravel(), vg, len(vg), wstd, re, im)
    return re + 1j * im


def ioni_sq2_cxx(rows, qsc):
    q = qsc.ravel() if qsc is not None and len(qsc) else None
    return lib().cvhcf_ioni_sq2(
        rows.ravel(), rows.shape[1], rows.shape[0],
        q.ctypes.data_as(ctypes.c_void_p) if q is not None else ctypes.c_void_p(0),
        0 if q is None else len(qsc))


# --------------------------------------------------------------------------
def reshape(v, idx, dflt):
    v = np.asarray(v, dtype=np.float32)
    if not len(idx):
        return v.reshape(0, dflt)
    return v.reshape(-1, len(v) // len(idx))


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


# --------------------------------------------------------------------------
def run(args):
    global TAU
    TAU = tau_grid()
    print(f"cvhcf tau grid: {len(TAU)} points, [{TAU[0]:.6f}, {TAU[-1]:.6f}]")
    ref = np.linspace(0.0, 14.0, 448)[0:4 * len(TAU):4]
    assert np.array_equal(TAU, ref), "the C++ grid is not the stride-4 subset"
    print(f"model tag: {model_tag()}")

    f = uproot.open(args.file)
    pt = f["runtree"]["parmtype"].array(library="np")
    t = f["tree"]
    keys = set(t.keys())
    mass = args.mass or ("Jpsi_sigmamass" in keys)
    print(f"{args.file}\n  {'TWO-TRACK (mass functional)' if mass else 'SINGLE TRACK (q/p functional)'}"
          f"  {t.num_entries} entries")

    need = ["reseigidx", "resinfvarv", "msmoliidx", "msmoliv",
            "ioniurbanidx", "ioniurbanv"]
    if mass:
        need += ["Jpsi_sigmamass", "resinfcov"]
    else:
        need += ["refCov", "genParms"]
    for b in ("ioniqscaleidx", "ioniqscalev", "radstepidx", "radstepv",
              "radstepspecv", "radvgrid"):
        if b in keys:
            need.append(b)
    pfx = "cfmass" if mass else "cfqop"
    cfb = [f"{pfx}_{k}" for k in ("ms", "del", "ioni_re", "ioni_im",
                                  "rad_re", "rad_im", "vgf", "ok")]
    have_cf = args.compare_branches and all(b in keys for b in cfb)
    if args.compare_branches and not have_cf:
        print(f"  --compare-branches: {pfx}_* not in the tree; skipping")
    if have_cf:
        need += cfb
    have_rad = "radstepv" in need
    have_qsc = "ioniqscalev" in need
    a = t.arrays(need, library="np", entry_stop=min(t.num_entries, 5 * args.ntracks + 50))

    acc_b, acc_t, acc_f = Acc(), Acc(), Acc()
    nsel = 0
    bench = []
    maxS = 0.0
    for ic in range(len(a["reseigidx"])):
        if mass:
            sig = float(a["Jpsi_sigmamass"][ic])
            if not (np.isfinite(sig) and sig > 0.):
                continue
            chg = IONI_SGN
        else:
            qg = a["genParms"][ic][0]
            c00 = float(a["refCov"][ic][0])
            if qg == 0. or not (c00 > 0.):
                continue
            sig = np.sqrt(c00)
            chg = float(np.sign(qg))

        gi = np.asarray(a["reseigidx"][ic])
        if not len(gi):
            continue
        vb = np.asarray(a["resinfvarv"][ic], dtype=np.float32)
        fam = pt[gi]
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = reshape(a["msmoliv"][ic], uim, 10)
        uii = np.asarray(a["ioniurbanidx"][ic])
        uvi = reshape(a["ioniurbanv"][ic], uii, 11)
        if have_qsc:
            qsi = np.asarray(a["ioniqscaleidx"][ic])
            qsv = np.asarray(a["ioniqscalev"][ic], dtype=np.float32).reshape(-1, 2)
        else:
            qsi = qsv = None
        if have_rad:
            ridx = np.asarray(a["radstepidx"][ic])
            rrec = np.asarray(a["radstepv"][ic], dtype=np.float32).reshape(
                -1, cf_brems_exact.RADV_STRIDE)
            rspc = np.asarray(a["radstepspecv"][ic], dtype=np.float32).reshape(
                -1, 2 * cf_brems_exact.NRADV)
            rvg = np.asarray(a["radvgrid"][ic], dtype=np.float32)
        else:
            ridx = rrec = rspc = rvg = None

        # the Gaussian share, as the offline caches define it
        if mass:
            vgf_ref = (sig * sig - float(a["resinfcov"][ic])) / (sig * sig)
        else:
            vgf_ref = float(vb[(fam == 8) | (fam == 9)].astype(np.float64).sum()) / c00

        # ---------------- python reference, block by block -----------------
        Sms = np.zeros(len(TAU))
        Sdel = np.zeros(len(TAU))
        Sio = np.zeros(len(TAU), dtype=np.complex128)
        Srad = np.zeros(len(TAU), dtype=np.complex128)
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
                    wstd = np.sqrt(vpool / sq2) / sig
                    py_ms = ctr.ms_step_exponent(steps, wstd, TAU)
                    cf = cf_delta_ray.carve_factor(steps, ctr.DELTA_TCUT,
                                                   ctr.DELTA_TMAXCAP)
                    py_del = (cf_delta_ray.delta_step_exponent(
                        steps, wstd, TAU, ctr.DELTA_TCUT,
                        tmax_cap=ctr.DELTA_TMAXCAP) - cf * py_ms)
                    cx_ms, cx_del = ms_block_cxx(np.ascontiguousarray(rows), wstd, True)
                    acc_b.add("ms", cx_ms, py_ms)
                    acc_b.add("del", cx_del, py_del)
                    Sms += py_ms
                    Sdel += py_del
                else:
                    qs = qsv[qsi == g] if qsv is not None else None
                    sq2 = ctr.ioni_sq2(steps, None if qs is None
                                       else qs.astype(np.float64))
                    cx_sq2 = ioni_sq2_cxx(np.ascontiguousarray(rows),
                                          None if qs is None else np.ascontiguousarray(qs))
                    acc_b.add("sq2", [cx_sq2], [sq2])
                    if sq2 <= 0.:
                        continue
                    w = chg * (np.sqrt(vpool / sq2) / sig)
                    py_io = ctr.ioni_step_exponent(steps, w, TAU)
                    cx_io = ioni_block_cxx(np.ascontiguousarray(rows), w)
                    acc_b.add("ioni_re", cx_io.real, py_io.real)
                    acc_b.add("ioni_im", cx_io.imag, py_io.imag)
                    Sio += py_io
                    if have_rad:
                        rm = ridx == g
                        nrs = int(rm.sum())
                        if nrs:
                            wr = (RAD_SGN * abs(w)) if mass else w
                            py_rad = cf_brems_exact.rad_exponent(
                                TAU, rrec[rm].astype(np.float64),
                                rspc[rm].astype(np.float64),
                                rvg.astype(np.float64),
                                weights=np.full(nrs, wr))
                            cx_rad = rad_block_cxx(np.ascontiguousarray(rrec[rm]),
                                                   np.ascontiguousarray(rspc[rm]),
                                                   rvg, wr)
                            acc_b.add("rad_re", cx_rad.real, py_rad.real)
                            acc_b.add("rad_im", cx_rad.imag, py_rad.imag)
                            Srad += py_rad
            if not ok:
                break

        # ---------------- the whole track, pooling included ----------------
        out = np.zeros(6 * len(TAU), dtype=np.float64)
        aux = np.zeros(5, dtype=np.float64)
        empt32 = np.zeros(0, dtype=np.float32)
        empu32 = np.zeros(0, dtype=np.uint32)
        lib().cvhcf_track(
            np.ascontiguousarray(gi, dtype=np.uint32),
            np.ascontiguousarray(fam, dtype=np.int32),
            np.ascontiguousarray(vb, dtype=np.float32), len(gi),
            np.ascontiguousarray(uim, dtype=np.uint32),
            np.ascontiguousarray(uvm.ravel(), dtype=np.float32),
            uvm.shape[0], uvm.shape[1],
            np.ascontiguousarray(uii, dtype=np.uint32),
            np.ascontiguousarray(uvi.ravel(), dtype=np.float32),
            uvi.shape[0], uvi.shape[1],
            np.ascontiguousarray(qsi, dtype=np.uint32) if qsi is not None else empu32,
            np.ascontiguousarray(qsv.ravel(), dtype=np.float32) if qsv is not None else empt32,
            0 if qsv is None else len(qsv),
            np.ascontiguousarray(ridx, dtype=np.uint32) if ridx is not None else empu32,
            np.ascontiguousarray(rrec.ravel(), dtype=np.float32) if rrec is not None else empt32,
            0 if rrec is None else rrec.shape[0],
            cf_brems_exact.RADV_STRIDE,
            np.ascontiguousarray(rspc.ravel(), dtype=np.float32) if rspc is not None else empt32,
            np.ascontiguousarray(rvg, dtype=np.float32) if rvg is not None else empt32,
            0 if rvg is None else len(rvg),
            float(sig), float(chg), 1, out, aux)
        if bool(aux[0]) != ok:
            print(f"  entry {ic}: ok flag disagrees (cxx {bool(aux[0])} vs py {ok})")
        if ok and bool(aux[0]):
            nt = len(TAU)
            acc_t.add("ms", out[0:nt], Sms)
            acc_t.add("del", out[nt:2 * nt], Sdel)
            acc_t.add("ioni_re", out[2 * nt:3 * nt], Sio.real)
            acc_t.add("ioni_im", out[3 * nt:4 * nt], Sio.imag)
            acc_t.add("rad_re", out[4 * nt:5 * nt], Srad.real)
            acc_t.add("rad_im", out[5 * nt:6 * nt], Srad.imag)
            maxS = max(maxS, float(np.max(np.abs(Sms))),
                       float(np.max(np.abs(Sio.real))))
            nsel += 1
            if have_cf:
                nt = len(TAU)
                acc_f.add("ms", np.asarray(a[f"{pfx}_ms"][ic]), Sms)
                acc_f.add("del", np.asarray(a[f"{pfx}_del"][ic]), Sdel)
                acc_f.add("ioni_re", np.asarray(a[f"{pfx}_ioni_re"][ic]), Sio.real)
                acc_f.add("ioni_im", np.asarray(a[f"{pfx}_ioni_im"][ic]), Sio.imag)
                acc_f.add("rad_re", np.asarray(a[f"{pfx}_rad_re"][ic]), Srad.real)
                acc_f.add("rad_im", np.asarray(a[f"{pfx}_rad_im"][ic]), Srad.imag)
                acc_f.add("vgf", [float(a[f"{pfx}_vgf"][ic])],
                          [float(vgf_ref)])
                if bool(a[f"{pfx}_ok"][ic]) != ok:
                    print(f"  entry {ic}: {pfx}_ok disagrees with the reference")
            if args.bench:
                bench.append((gi, fam, vb, uim, uvm, uii, uvi, qsi, qsv,
                              ridx, rrec, rspc, rvg, sig, chg))
        if nsel >= args.ntracks:
            break

    worst_b = acc_b.report(f"BLOCK level (C++ primitive vs python), {nsel} entries")
    worst_t = acc_t.report(f"TRACK level (cvhcf::trackExponents vs extract loop)")
    worst = max(worst_b, worst_t)
    if acc_f.m:
        # float32 in the file: 1 ulp is 6e-8 RELATIVE, so this level's floor is
        # 6e-8 * max|S| -- ~4e-6 at the largest |S| these samples reach. It is
        # reported against that floor and not against the 1e-6 the evaluator
        # itself is held to.
        wf = acc_f.report("BRANCH level (the floats the MAKER wrote, float32)")
        print(f"  float32 storage floor here: {6.0e-8 * maxS:.2e} "
              f"(max|S| = {maxS:.3g})")
        print(f"  branch-level worst |dS| = {wf:.4e}")
    if bench:
        import time
        out = np.zeros(6 * len(TAU), dtype=np.float64)
        aux = np.zeros(5, dtype=np.float64)
        empt32 = np.zeros(0, dtype=np.float32)
        empu32 = np.zeros(0, dtype=np.uint32)
        pre = []
        for (gi, fam, vb, uim, uvm, uii, uvi, qsi, qsv,
             ridx, rrec, rspc, rvg, sig, chg) in bench:
            pre.append((
                np.ascontiguousarray(gi, dtype=np.uint32),
                np.ascontiguousarray(fam, dtype=np.int32),
                np.ascontiguousarray(vb, dtype=np.float32), len(gi),
                np.ascontiguousarray(uim, dtype=np.uint32),
                np.ascontiguousarray(uvm.ravel(), dtype=np.float32),
                uvm.shape[0], uvm.shape[1],
                np.ascontiguousarray(uii, dtype=np.uint32),
                np.ascontiguousarray(uvi.ravel(), dtype=np.float32),
                uvi.shape[0], uvi.shape[1],
                np.ascontiguousarray(qsi, dtype=np.uint32) if qsi is not None else empu32,
                np.ascontiguousarray(qsv.ravel(), dtype=np.float32) if qsv is not None else empt32,
                0 if qsv is None else len(qsv),
                np.ascontiguousarray(ridx, dtype=np.uint32) if ridx is not None else empu32,
                np.ascontiguousarray(rrec.ravel(), dtype=np.float32) if rrec is not None else empt32,
                0 if rrec is None else rrec.shape[0],
                cf_brems_exact.RADV_STRIDE,
                np.ascontiguousarray(rspc.ravel(), dtype=np.float32) if rspc is not None else empt32,
                np.ascontiguousarray(rvg, dtype=np.float32) if rvg is not None else empt32,
                0 if rvg is None else len(rvg),
                float(sig), float(chg), 1, out, aux))
        L = lib()
        n = args.bench * len(pre)
        unit = 'candidate' if mass else 'track'
        print("")
        # The same call with the radiative rows withheld, so the report says
        # WHERE the time goes rather than only how much there is: the rad
        # channel is nsteps x nv x ntau trigonometric evaluations and is the
        # one term whose cost is not bounded by the block count.
        for label, norad in (("full", False), ("no rad", True)):
            arg = pre
            if norad:
                arg = [list(p_) for p_ in pre]
                for p_ in arg:
                    # indices into the cvhcf_track signature: 15 ridx,
                    # 16 rv, 17 nrad (14 is nqs -- getting this wrong raises an
                    # immediate ctypes ArgumentError)
                    p_[15] = empu32
                    p_[16] = empt32
                    p_[17] = 0
            t0 = time.perf_counter()
            for _ in range(args.bench):
                for p_ in arg:
                    L.cvhcf_track(*p_)
            dt = time.perf_counter() - t0
            print(f"COST {label:<7}: {n} evaluations of cvhcf::trackExponents "
                  f"in {dt:.3f} s = {1e3 * dt / n:.3f} ms per {unit}")
    print(f"\nworst |dS| overall: {worst:.4e}   (tolerance {args.tol:g})")
    return 0 if worst <= args.tol else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--ntracks", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--mass", action="store_true",
                   help="force the mass-functional sign convention")
    p.add_argument("--compare-branches", action="store_true",
                   help="also compare the cfqop_*/cfmass_* branches the maker "
                        "wrote (float32, so the floor is ~1e-5 at |S| ~ 60)")
    p.add_argument("--bench", type=int, default=0,
                   help="time N repeats of cvhcf::trackExponents on the "
                        "entries read, and report ms/track")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()

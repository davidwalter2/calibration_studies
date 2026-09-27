#!/usr/bin/env python3
"""Gate 3: port fidelity of the in-maker knock-on families (`cvhcf::
knockonMapBlock`, `knockonMapRad`, `knockonJointRows` and the track-level
`trackExponents` with wantKnockonMap/Joint) against cf_knockon, on real maker
rows, on the exported tau grid (the stride-4 subset of TG, t <= 8).

  block  per ionisation block (map), per radiative block (map), per track (joint)
  track  trackExponents' kx, kj against the reference sums; the per-group
         split sums to the flat family; switches off -> kx = kj = 0

usage: cxx_gate.py SHIM.so FILE [--entries N]   (SHIM from build_shim.sh)
Run with CVH_IONI_KOKOULIN=0 (the ported configuration)."""
import argparse
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cf_knockon as ck                                   # noqa: E402
import cf_track_resolution as ctr                         # noqa: E402
import gates                                              # noqa: E402

NT = 64
TAU = ctr.TG[::4][:NT]
f32p = np.ctypeslib.ndpointer(np.float32, flags="C")
f64p = np.ctypeslib.ndpointer(np.float64, flags="C")
u32p = np.ctypeslib.ndpointer(np.uint32, flags="C")
i32p = np.ctypeslib.ndpointer(np.int32, flags="C")


def lib(path):
    L = ctypes.CDLL(path)
    c_i, c_d = ctypes.c_int, ctypes.c_double
    L.cvhcf_knockon_map_block.argtypes = [f32p, c_i, c_i, c_i, c_d, f64p, f64p]
    L.cvhcf_knockon_map_rad.argtypes = [f32p, c_i, c_i, f32p, f32p, c_i, c_d, f64p, f64p]
    L.cvhcf_knockon_joint_rows.argtypes = [f32p, c_i, c_i, u32p, u32p, c_i, f64p, f64p, c_i, f64p, f64p]
    L.cvhcf_track_knockon.argtypes = [u32p, i32p, f32p, c_i,
                                      u32p, f32p, c_i, c_i, c_i,
                                      u32p, f32p, c_i, c_i, c_i,
                                      u32p, f32p, c_i, u32p, f32p, c_i, c_i, c_i, f32p, f32p, c_i,
                                      c_d, c_d, c_i, c_i, c_i, f64p, f64p]
    return L


def cplx(re, im):
    return re + 1j * im


def dev(a, b):
    """(max abs difference, max |b|)"""
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))


def raw_entry(fn, ic):
    """the float32 export arrays of entry ic, as the maker hands them over"""
    import uproot
    f = uproot.open(fn)
    t = f["tree"]
    pt = f["runtree"]["parmtype"].array(library="np")
    br = ["refCov", "genParms", "reseigidx", "resinfvarv", "msmoliidx", "msmoliv",
          "msmolistride", "ioniurbanidx", "ioniurbanv", "ioniurbanstride",
          "radstepidx", "radstepv", "radstepstride", "radstepspecv", "radvgrid",
          "ioniqscaleidx", "ioniqscalev"]
    a = t.arrays(br, entry_start=ic, entry_stop=ic + 1, library="np")
    g = lambda k: np.ascontiguousarray(a[k][0])
    gi = g("reseigidx").astype(np.uint32)
    return dict(gi=gi, fam=pt[gi].astype(np.int32), vb=g("resinfvarv").astype(np.float32),
                midx=g("msmoliidx").astype(np.uint32), mv=g("msmoliv").astype(np.float32),
                ms=int(a["msmolistride"][0]),
                iidx=g("ioniurbanidx").astype(np.uint32), iv=g("ioniurbanv").astype(np.float32),
                ist=int(a["ioniurbanstride"][0]),
                ridx=g("radstepidx").astype(np.uint32), rv=g("radstepv").astype(np.float32),
                rst=int(a["radstepstride"][0]),
                rsp=g("radstepspecv").astype(np.float32), rvg=g("radvgrid").astype(np.float32),
                qidx=g("ioniqscaleidx").astype(np.uint32), qv=g("ioniqscalev").astype(np.float32),
                sig=float(np.sqrt(np.float32(a["refCov"][0][0]))),
                chg=float(np.sign(a["genParms"][0][0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shim")
    ap.add_argument("file")
    ap.add_argument("--entries", type=int, default=5)
    args = ap.parse_args()
    assert ctr.IONI_KOKOULIN == 0.0, "run with CVH_IONI_KOKOULIN=0"
    L = lib(args.shim)
    recs = gates.load(args.file, args.entries)
    worst = {}

    def upd(k, d):
        a, m = d
        o = worst.get(k, (0., 0.))
        worst[k] = (max(o[0], a), max(o[1], a / m if m > 0 else 0.))

    for ic, r in enumerate(recs):
        e = raw_entry(args.file, ic)
        I, Ii, R, P = r["I"], r["Iidx"], r["R"], r["P"]
        # ---- block level
        for g, w in r["wio"].items():
            rows = np.ascontiguousarray(e["iv"].reshape(-1, e["ist"])[e["iidx"] == g])
            re, im = np.zeros(NT), np.zeros(NT)
            L.cvhcf_knockon_map_block(rows, e["ist"], len(rows),
                                      e["ist"] - 1 if e["ist"] >= 14 else -1, w, re, im)
            upd("map_block", dev(cplx(re, im), ck.fit_map(I[Ii == g], w, TAU)))
            rm = r["Ridx"] == g
            if rm.any():
                rr = np.ascontiguousarray(e["rv"].reshape(-1, e["rst"])[rm])
                nv = len(e["rvg"])
                sp = np.ascontiguousarray(e["rsp"].reshape(-1, 2 * nv)[rm])
                re, im = np.zeros(NT), np.zeros(NT)
                L.cvhcf_knockon_map_rad(rr, e["rst"], len(rr), sp, e["rvg"], nv, w, re, im)
                upd("map_rad", dev(cplx(re, im),
                                   ck.map_rad(TAU, R[rm], P[rm], r["vg"], np.full(rm.sum(), w))))
        beta, alpha = ck.pair_weights(r["Midx"], r["Ridx"], r["M"], R, r["wms"], r["wio"])
        M = np.ascontiguousarray(e["mv"].reshape(-1, e["ms"]))
        for exact in (0, 1):
            re, im = np.zeros(NT), np.zeros(NT)
            L.cvhcf_knockon_joint_rows(M, e["ms"], len(M), e["midx"], e["ridx"],
                                       ck.MS_GROUP_COL if e["ms"] > ck.MS_GROUP_COL else -1,
                                       np.ascontiguousarray(alpha), np.ascontiguousarray(beta),
                                       exact, re, im)
            upd(f"joint_x{exact}", dev(cplx(re, im),
                                       ck.fit_joint(r["M"], alpha, beta, r["Midx"], r["Ridx"], TAU,
                                                    ctr.DELTA_TCUT, ctr.DELTA_TMAXCAP, bool(exact))))
        # ---- track level
        kx_ref = sum((ck.fit_map(I[Ii == g], w, TAU) for g, w in r["wio"].items()),
                     np.zeros(NT, complex))
        for g, w in r["wio"].items():
            rm = r["Ridx"] == g
            if rm.any():
                kx_ref = kx_ref + ck.map_rad(TAU, R[rm], P[rm], r["vg"], np.full(rm.sum(), w))
        kj_ref = ck.fit_joint(r["M"], alpha, beta, r["Midx"], r["Ridx"], TAU,
                              ctr.DELTA_TCUT, ctr.DELTA_TMAXCAP, True)
        out, gs = np.zeros(10 * NT), np.zeros(4 * NT)

        def track(mp, jt, grp):
            out[:], gs[:] = 0., 0.
            return L.cvhcf_track_knockon(
                e["gi"], e["fam"], e["vb"], len(e["gi"]),
                e["midx"], e["mv"], len(e["midx"]), e["ms"], 9 if e["ms"] > 9 else -1,
                e["iidx"], e["iv"], len(e["iidx"]), e["ist"], e["ist"] - 1 if e["ist"] >= 14 else -1,
                e["qidx"], e["qv"], len(e["qidx"]),
                e["ridx"], e["rv"], len(e["ridx"]), e["rst"], e["rst"] - 1 if e["rst"] >= 12 else -1,
                e["rsp"], e["rvg"], len(e["rvg"]), e["sig"], e["chg"], mp, jt, grp, out, gs)
        track(1, 1, 1)
        kx, kj = cplx(out[6 * NT:7 * NT], out[7 * NT:8 * NT]), cplx(out[8 * NT:9 * NT], out[9 * NT:])
        # the C++ standardizes with its own float sigma and forms its own
        # weights; the reference weights come from gates.load (the same rule)
        upd("track_kx", dev(kx, kx_ref))
        upd("track_kj", dev(kj, kj_ref))
        upd("groups_kx", dev(cplx(gs[:NT], gs[NT:2 * NT]), kx))
        upd("groups_kj", dev(cplx(gs[2 * NT:3 * NT], gs[3 * NT:]), kj))
        base = out[:6 * NT].copy()
        track(0, 0, 0)
        upd("off_kxkj_zero", (float(np.abs(out[6 * NT:]).max()), 1.0))
        upd("off_existing_bitid", (float(np.abs(out[:6 * NT] - base).max()), 1.0))
        print(f"entry {ic}: |kx|={np.abs(kx).max():.3e} |kj|={np.abs(kj).max():.3e}")
    for k, (a, rl) in worst.items():
        print(f"{k:20s} max|d| {a:.2e}  rel {rl:.2e}")


if __name__ == "__main__":
    main()

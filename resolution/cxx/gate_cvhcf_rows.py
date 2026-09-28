#!/usr/bin/env python3
"""Gate: the in-maker C++ row functions (`cvhcf`, CvhCfExponents.cc "THE ROW
FUNCTIONS", through cvhcf_rows and cxx/libcvhcfshim.so) against the Python
reference `cf_rows` (and what it calls), channel by channel, on the rows and
entries that actually occur.

  cleanprop  the real-material clean-propagation legs (cf_propagation_test.
             load_model, every species and the ditrack legs), each channel at
             the exact transport entries of `cf_rows.transport_entries` /
             `end_weights` exactly as `cf_propagation_test.leg_exponent` forms
             them, summed over the legs up to planes k, for several functionals;
             plus refine_spectra on the legs' exported spectra; for the
             hadron legs the nuclear-elastic row function
             (`nucel_tables.Table.rows` vs `cvhcf::nucelRows`) at
             `cf_nucel_exact.leg_entries`, its angular, recoil and joint parts.
  maker      real maker rows (a residual maker's step records): the fit-level
             entries `cf_rows.fit_families` forms -- per block of the q/p (or,
             --mass, the candidate-mass) functional, one entry per record, the
             knock-on's angular weight from `cf_rows.fit_pairing`, the
             radiative rows refined once per track.
  timing     wall time of the C++ families per candidate on maker rows.

THE FIGURE per (channel, species): the worst over the evaluations of
max_tau |S_cxx - S_py| / max_tau |S_py|, both sides in double.  Target 1e-10.

usage (from resolution/, after cxx/build_cvhcf.sh):
  python cxx/gate_cvhcf_rows.py cleanprop [--files model_*.root ...] [--planes 0 9 18]
  python cxx/gate_cvhcf_rows.py maker FILE [--entries N] [--mass]
  python cxx/gate_cvhcf_rows.py timing FILE [--entries N] [--mass]
"""
import argparse
import glob
import json
import os
import sys
import time

# THE REFERENCE'S ARITHMETIC.  numpy dispatches its float64 transcendentals
# (pow, log, log10, exp, expm1, log1p) to SVML kernels on AVX512 hardware and
# contracts its complex products into FMAs on AVX2/FMA3 hardware, neither of
# which is IEEE-reproducible libm arithmetic; the C++ is.  With that dispatch
# off numpy IS libm arithmetic, and the C++ port reproduces the reference's
# quadrature bit for bit (nodes, Filon sums, Simpson sums).  The knock-on map
# piece is the difference of two Filon sums that cancel to ~1e-6 on a thin
# step, so the hardware dispatch alone moves the REFERENCE by ~4e-10 of its
# maximum (`reference-spread`); the gate therefore compares against the
# reference evaluated in numpy's baseline arithmetic, and measures that
# spread separately.  --native keeps the dispatch on.
_BASELINE = ("FMA3 AVX2 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX "
             "AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR")
if "--native" not in os.sys.argv and os.environ.get("NPY_DISABLE_CPU_FEATURES") != _BASELINE:
    os.environ["NPY_DISABLE_CPU_FEATURES"] = _BASELINE
    os.execv(os.sys.executable, [os.sys.executable] + os.sys.argv)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["RES_NO_PHI_CACHE"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                         # noqa: E402

import cf_brems_exact as cbe                               # noqa: E402
import cf_knockon as ck                                    # noqa: E402
import cf_rows                                             # noqa: E402
import cf_track_resolution as ctr                          # noqa: E402
import cvhcf_rows as cx                                    # noqa: E402

# the reference side is the numpy implementation, whatever CF_ROWS_BACKEND
# says: cf_rows would otherwise dispatch to the very .so under test
cf_rows.BACKEND = "py"

MODELDIR = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/realmat_full_260925/model"
CHANNELS = ("refine", "ioni", "ms", "rad", "ko_all", "ko_map", "ko_joint",
            "nuc_ang", "nuc_rec", "nuc_jnt")


def rel(a, b):
    """max |a - b| / max |b| (absolute when b vanishes)."""
    a, b = np.asarray(a), np.asarray(b)
    den = float(np.max(np.abs(b))) if b.size else 0.0
    num = float(np.max(np.abs(a - b))) if b.size else 0.0
    return num / den if den > 0 else num, den


class Table:
    def __init__(self):
        self.w = {}

    def add(self, ch, key, r, scale, where):
        cur = self.w.get((ch, key))
        if cur is None or r > cur[0]:
            self.w[(ch, key)] = (r, scale, where)

    def dump(self):
        return {f"{c}|{k}": v for (c, k), v in self.w.items()}


# --------------------------------------------------------------- cleanprop
def species_tag(path):
    b = os.path.basename(path)[len("model_"):-len(".root")]
    return b


def gate_cleanprop(path, planes, funcs, tau, tab, dump=None):
    import cf_nucel_exact as cne
    import cf_propagation_test as cpt
    key = species_tag(path)
    # the exported spectra, unrefined, for the refinement check; load_model
    # refines them itself (RAD_NSUB), which is what the channels then see
    nsub = cbe.RAD_NSUB
    cbe.RAD_NSUB = 1
    raw = cpt.load_model(path)
    cbe.RAD_NSUB = nsub
    legs = cpt.load_model(path)
    for j, leg in enumerate(raw):
        if leg.get("rad") is None or not len(leg["rad"]):
            continue
        vf_p, sp_p = cbe.refine_spectra(leg["rad"], leg["radspec"], leg["radvgrid"], nsub)
        vf_c, sp_c = cx.refine_spectra(leg["rad"], leg["radspec"], leg["radvgrid"], nsub)
        tab.add("refine", key, max(rel(vf_c, vf_p)[0], rel(sp_c, sp_p)[0]),
                float(np.max(np.abs(sp_p))), f"leg{j}")
    ntot = len(legs)
    for k in planes:
        if k >= ntot:
            continue
        A_ms, A_io, A_ms0, A_io0 = cpt.step_transports(legs, k, ioni_start=True)
        for fn in funcs:
            avec = cpt.FUNCTIONALS[fn]
            sig = np.sqrt(cpt.model_variance(legs, k, avec)[0])
            acc = {c: [np.zeros(len(tau), complex), np.zeros(len(tau), complex)]
                   for c in CHANNELS[1:]}
            for j in range(k + 1):
                leg = legs[j]
                if len(leg["ioni"]):
                    wq, _ = cf_rows.end_weights(leg, A_io[j], avec, sig)
                    acc["ioni"][0] += cf_rows.ioni_rows(tau, leg["ioni"], wq)
                    acc["ioni"][1] += cx.ioni_rows(tau, leg["ioni"], wq)
                    rid, wq, wb, fr = cf_rows.transport_entries(
                        leg, A_io[j], A_io0[j], avec, sig, ck.KNOCKON_NSUB, "vector")
                    for part in ("all", "map", "joint"):
                        acc["ko_" + part][0] += cf_rows.knockon_rows(
                            tau, leg["ioni"], rid, wq, wb, fr, part)
                        acc["ko_" + part][1] += cx.knockon_rows(
                            tau, leg["ioni"], rid, wq, wb, fr, part)
                if leg.get("rad") is not None and len(leg["rad"]):
                    rid, wq, wb, fr = cf_rows.transport_entries(
                        leg, A_ms[j], A_ms0[j], avec, sig, cpt.MS_NSUB, "vector")
                    acc["rad"][0] += cf_rows.rad_rows(tau, leg["rad"], leg["radspec"],
                                                      leg["radvgrid"], rid, wq, wb, fr)
                    acc["rad"][1] += cx.rad_rows(tau, leg["rad"], leg["radspec"],
                                                 leg["radvgrid"], rid, wq, wb, fr)
                if len(leg["ms"]):
                    rid, _, wb, fr = cf_rows.transport_entries(
                        leg, A_ms[j], A_ms0[j], avec, sig, cpt.MS_NSUB, "length")
                    acc["ms"][0] += cf_rows.ms_rows(tau, leg["ms"], rid, wb, fr,
                                                    scale=cpt.KMS_SCALE)
                    acc["ms"][1] += cx.ms_rows(tau, leg["ms"], rid, wb, fr,
                                               scale=cpt.KMS_SCALE)
                pdg = cne.pdg_from_leg(leg) if len(leg["ms"]) else None
                if pdg is not None:
                    # the nuclear-elastic row function at cf_nucel_exact's own
                    # clean-propagation entries, every part on
                    ent = cne.leg_entries(leg, A_ms[j], A_ms0[j], A_io[j], avec, sig,
                                          cpt.MS_NSUB)
                    T = cne.table()
                    im = cne.leg_materials(T, leg)
                    kw = dict(recoil=True, joint=True, exact=bool(ck.QOP_EXACT))
                    for side, f in ((0, T.rows), (1, lambda *a, **k: cx.nucel_rows(T, *a, **k))):
                        _, a_, r_, j_ = f(tau, leg["ms"], im, pdg, cne.mass_of(pdg), *ent, **kw)
                        acc["nuc_ang"][side] += a_
                        acc["nuc_rec"][side] += r_
                        acc["nuc_jnt"][side] += j_
            for c, (py, cc) in acc.items():
                r, sc = rel(cc, py)
                if c.startswith("nuc_") and sc == 0.0 and not np.any(cc):
                    continue
                tab.add(c, key, r, sc, f"k{k} {fn}")
                if dump is not None:
                    dump[f"{key}|{c}|k{k}|{fn}|py"] = py
                    dump[f"{key}|{c}|k{k}|{fn}|cxx"] = cc


# ------------------------------------------------------------------- maker
def load_maker(fn, entries, mass):
    """Per candidate the fit-level blocks exactly as cf_track_resolution.extract
    (q/p) or cf_mass_likelihood.build_pairs_tt (--mass) form them."""
    import uproot
    import prodfiles
    f = uproot.open(fn)
    pt = f["runtree"]["parmtype"].array(library="np")
    t = f["tree"]
    br = ["reseigidx", "resinfvarv", "resinfv", "msmoliidx", "msmoliv",
          "ioniurbanidx", "ioniurbanv", "radstepidx", "radstepv", "radstepspecv",
          "radvgrid", "ioniqscaleidx", "ioniqscalev"]
    br += ["Jpsi_sigmamass"] if mass else ["refCov", "genParms"]
    br += [b for b in prodfiles.stride_keys(t, ("msmoliv", "ioniurbanv", "radstepv",
                                                  "radstepspecv")) if b not in br]
    a = t.arrays(br, library="np", entry_stop=entries)
    from cf_mass_likelihood import IONI_SGN, RAD_SGN
    out = []
    n = len(a["reseigidx"])
    for ic in range(n):
        if mass:
            sig = float(a["Jpsi_sigmamass"][ic])
            chg = IONI_SGN
            rsg = RAD_SGN
        else:
            c00 = float(a["refCov"][ic][0])
            if not c00 > 0 or a["genParms"][ic][0] == 0:
                continue
            sig = np.sqrt(c00)
            chg = rsg = np.sign(a["genParms"][ic][0])
        if not (np.isfinite(sig) and sig > 0):
            continue
        gi = np.asarray(a["reseigidx"][ic])
        vb = np.asarray(a["resinfvarv"][ic], np.float64)
        uw = np.asarray(a["resinfv"][ic], np.float64).reshape(-1, 5)
        fam = pt[gi]
        rec = {}
        for nm, ib, vbn, mc in (("M", "msmoliidx", "msmoliv", 8),
                                ("I", "ioniurbanidx", "ioniurbanv", 11),
                                ("R", "radstepidx", "radstepv", 11),
                                ("P", "radstepidx", "radstepspecv", 1)):
            idx = np.asarray(a[ib][ic])
            rec[nm + "idx"] = idx
            rec[nm] = prodfiles.reshape_records(
                a[vbn][ic], nrec=len(idx), stride=prodfiles.entry_stride(a, vbn, ic),
                branch=vbn, min_cols=mc).astype(np.float64)
        qsi = np.asarray(a["ioniqscaleidx"][ic])
        qsv = np.asarray(a["ioniqscalev"][ic], np.float64).reshape(-1, 2)
        ms_blocks, io_blocks, wrad = [], [], {}
        ok = True
        for famcode, (uidx, uv) in ((10, (rec["Midx"], rec["M"])), (11, (rec["Iidx"], rec["I"]))):
            sel = fam == famcode
            for g in np.unique(gi[sel]):
                m = sel & (gi == g)
                vpool = vb[m].sum()
                if vpool <= 0.:
                    continue
                steps = uv[uidx == g]
                if not len(steps):
                    ok = False
                    break
                sq2 = (steps[:, 5].sum() if famcode == 10
                       else ctr.ioni_sq2(steps, qsv[qsi == g]))
                if sq2 <= 0.:
                    continue
                wsc = np.sqrt(vpool / sq2) / sig
                if famcode == 10:
                    ms_blocks.append((g, steps, wsc))
                else:
                    io_blocks.append((g, steps, chg * wsc))
                    wrad[int(g)] = rsg * wsc
            if not ok:
                break
        if not ok:
            continue
        rad = dict(uim=rec["Midx"], ridx=rec["Ridx"], uvm=rec["M"], rrec=rec["R"],
                   rspc=rec["P"], rvg=np.asarray(a["radvgrid"][ic], np.float64), wrad=wrad)
        out.append((ic, ms_blocks, io_blocks, rad))
    return out


def maker_channels(tau, ms_blocks, io_blocks, rad, impl):
    """cf_rows.fit_families' channel calls, through `impl` (cf_rows or cvhcf_rows)."""
    nt = len(tau)
    S = {c: np.zeros(nt, complex) for c in ("ms", "ioni", "rad", "ko_map", "ko_joint")}
    wms = {}
    for g, rows, w in ms_blocks:
        wms[int(g)] = abs(w)
        n = len(rows)
        S["ms"] += impl.ms_rows(tau, rows, np.arange(n), np.full(n, abs(w)), np.ones(n))
    wb_rad, beta = cf_rows.fit_pairing(rad["uim"], rad["ridx"], rad["uvm"], wms)
    for g, rows, w in io_blocks:
        n = len(rows)
        wq = np.full(n, w)
        S["ioni"] += impl.ioni_rows(tau, rows, wq)
        wbk = np.full(n, beta.get(int(g), 0.0))
        S["ko_map"] += impl.knockon_rows(tau, rows, np.arange(n), wq, wbk, np.ones(n), "map")
        S["ko_joint"] += impl.knockon_rows(tau, rows, np.arange(n), wq, wbk, np.ones(n), "joint")
    vf, spf = impl.refine_spectra(rad["rrec"], rad["rspc"], rad["rvg"], cbe.RAD_NSUB)
    for g, wr in rad["wrad"].items():
        m = np.flatnonzero(rad["ridx"] == g)
        if not len(m):
            continue
        S["rad"] += impl.rad_rows(tau, rad["rrec"][m], spf[m], vf, np.arange(len(m)),
                                  np.full(len(m), wr), wb_rad[m], np.ones(len(m)))
    return S, (vf, spf)


def gate_maker(fn, entries, mass, tau, tab):
    key = os.path.basename(os.path.dirname(fn)) + (" mass" if mass else " qop")
    for ic, msb, iob, rad in load_maker(fn, entries, mass):
        Sp, (vp, sp) = maker_channels(tau, msb, iob, rad, cf_rows_ref)
        Sc, (vc, sc) = maker_channels(tau, msb, iob, rad, cx)
        tab.add("refine", key, max(rel(vc, vp)[0], rel(sc, sp)[0]), float(np.max(np.abs(sp))),
                f"entry {ic}")
        for c in Sp:
            r, s = rel(Sc[c], Sp[c])
            tab.add(c, key, r, s, f"entry {ic}")


class _RefRows:
    """cf_rows' channels with refine_spectra from cf_brems_exact, the object
    `maker_channels` calls on the reference side."""
    ioni_rows = staticmethod(cf_rows.ioni_rows)
    ms_rows = staticmethod(cf_rows.ms_rows)
    rad_rows = staticmethod(cf_rows.rad_rows)
    knockon_rows = staticmethod(cf_rows.knockon_rows)
    refine_spectra = staticmethod(cbe.refine_spectra)


cf_rows_ref = _RefRows()


def timing(fn, entries, mass, tau):
    rows = load_maker(fn, entries, mass)
    t_py = t_cx = 0.0
    for ic, msb, iob, rad in rows:
        t0 = time.perf_counter()
        maker_channels(tau, msb, iob, rad, cx)
        t_cx += time.perf_counter() - t0
    for ic, msb, iob, rad in rows[:3]:
        t0 = time.perf_counter()
        maker_channels(tau, msb, iob, rad, cf_rows_ref)
        t_py += time.perf_counter() - t0
    n = max(len(rows), 1)
    print(json.dumps(dict(file=fn, ncand=len(rows), cxx_ms_per_cand=1e3 * t_cx / n,
                          py_ms_per_cand=1e3 * t_py / max(min(len(rows), 3), 1))))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("cleanprop")
    a1.add_argument("--files", nargs="*", default=None)
    a1.add_argument("--planes", nargs="*", type=int, default=[0, 9, 18])
    a1.add_argument("--funcs", nargs="*", default=["qop", "locx", "dydz"])
    a1.add_argument("--ntau", type=int, default=121)
    a1.add_argument("--out", default=None)
    a1.add_argument("--dump", default=None,
                    help="save every summed channel (py and cxx) to this npz")
    a3 = sp.add_parser("spread", help="the reference's own native-vs-baseline "
                       "spread, and the C++ against each, from two --dump files")
    a3.add_argument("baseline")
    a3.add_argument("native")
    for nm in ("maker", "timing"):
        a2 = sp.add_parser(nm)
        a2.add_argument("file")
        a2.add_argument("--entries", type=int, default=10)
        a2.add_argument("--mass", action="store_true")
        a2.add_argument("--out", default=None)
    ap.add_argument("--native", action="store_true",
                    help="keep numpy's FMA/AVX512 dispatch (the reference's hardware arithmetic)")
    args = ap.parse_args()
    if args.cmd == "spread":
        b, nv = np.load(args.baseline), np.load(args.native)
        worst = {}
        for k in b.files:
            if not k.endswith("|py"):
                continue
            sp_, ch = k.split("|")[0], k.split("|")[1]
            pb, pn, cc = b[k], nv[k], b[k[:-3] + "|cxx"]
            r1, _ = rel(pn, pb)
            r2, _ = rel(cc, pn)
            w = worst.setdefault((ch, sp_), [0.0, 0.0])
            w[0], w[1] = max(w[0], r1), max(w[1], r2)
        for (ch, sp_), (r1, r2) in sorted(worst.items()):
            print(f"{ch:9s} {sp_:28s} reference native vs baseline {r1:.3e}   "
                  f"C++ vs native reference {r2:.3e}")
        return
    cx.lib()
    tab = Table()
    if args.cmd == "cleanprop":
        import cf_propagation_test as cpt
        tau = np.concatenate([[0.0], np.geomspace(1e-3, 40.0, args.ntau - 1)])
        files = args.files or sorted(glob.glob(os.path.join(MODELDIR, "model_*_mat.root"))
                                     + [f for f in glob.glob(os.path.join(MODELDIR, "model_*_all4.root"))
                                        if any(t in f for t in ("_jpsi", "_jpsiee", "_ks", "_lam",
                                                                "_zmm", "_zee"))])
        dump = {} if args.dump else None
        for f in files:
            t0 = time.perf_counter()
            gate_cleanprop(f, args.planes, args.funcs, tau, tab, dump)
            print(f"[gate] {os.path.basename(f)} {time.perf_counter() - t0:.0f} s", flush=True)
        del cpt
        if dump is not None:
            np.savez(args.dump, **dump)
    elif args.cmd == "maker":
        tau = ctr.TG
        gate_maker(args.file, args.entries, args.mass, tau, tab)
    else:
        timing(args.file, args.entries, args.mass, ctr.TG[::4][:64])
        return
    for (c, k), (r, s, w) in sorted(tab.w.items()):
        print(f"{c:9s} {k:28s} worst rel {r:.3e}  (max|S_py| {s:.3e}, {w})")
    if args.out:
        with open(args.out, "w") as fo:
            json.dump(tab.dump(), fo, indent=1)


if __name__ == "__main__":
    main()

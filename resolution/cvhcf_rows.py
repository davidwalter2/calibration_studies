"""The in-maker C++ (`cvhcf`) of the CF row functions, with `cf_rows`' signatures.

`cf_rows` is the reference: ONE implementation of every process-noise channel
as a function of the exported Geant4 step records and a list of entries
(record index, q/p weight, angular weight, fraction).  The makers evaluate the
same functions in C++ (TrackPropagation/Geant4e, `CvhCfExponents.cc`, section
"THE ROW FUNCTIONS"), compiled here UNMODIFIED into `cxx/libcvhcfshim.so`
(`cxx/build_cvhcf.sh`).  This module calls them with the arguments `cf_rows`
takes, so a closure can evaluate its model through the maker's own code, and
`cxx/gate_cvhcf_rows.py` compares the two (`CVHCF_SHIM` points at another
build of the .so, a private one under test).  The nuclear-elastic channel's row
function is `nucel_tables.Table.rows` (`nucel_rows`, and `rows_exponent` with
`cf_nucel_exact.rows_exponent`'s signature), on the same shipped table.

The model's switches live in the reference modules' globals (cf_knockon,
cf_brems_exact, cf_track_resolution, cf_nucel_exact); every call hands their
CURRENT values to the C++ (`sync_config`), so a run that changes a switch
evaluates the same model on both sides.  The scattering channel's diagnostic knobs that the C++
does not carry (the MS_* gauges of cf_track_resolution, cf_ms_exact's
switches) must sit at their production values; `sync_config` refuses anything
else rather than evaluating a different model silently.
"""

import ctypes
import os

import numpy as np

import cf_brems_exact
import cf_knockon
import cf_track_resolution as ctr

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None

# The scattering channel as the C++ carries it (CvhCfExponents.h, "THE PORTED
# SWITCH CONFIGURATION").
_MS_PORTED = dict(MS_ELEC_TMAX=1.0, MS_ELEC_EDGE=1.0, MS_FINE_G=1.0,
                  MS_SNAP_YMAX=0.0, MS_WVI_SPLIT=0.0)
_MSX_PORTED = dict(J0M1_GUARD=True, G4_FF_SQUARED=True, G4_SCREEN_F=1.0,
                   MS_CHI0_G4=True, MS_FF_G4=True)


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = os.environ.get("CVHCF_SHIM") or os.path.join(HERE, "cxx", "libcvhcfshim.so")
    if not os.path.exists(so):
        raise SystemExit(f"{so} missing -- run cxx/build_cvhcf.sh")
    os.environ.setdefault("CMSSW_SRC", "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src")
    L = ctypes.CDLL(so)
    f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    ci, cd = ctypes.c_int, ctypes.c_double
    L.cvhcf_row_config_set.argtypes = [ci, ci, ci, ci, cd, cd, cd, ci, cd, cd, cd, cd, cd, ci, ci]
    L.cvhcf_row_config_set.restype = None
    L.cvhcf_ioni_rows.argtypes = [f64, ci, f64, ci, ci, f64, f64, f64]
    L.cvhcf_ms_rows.argtypes = [f64, ci, f64, ci, ci, i32, f64, f64, ci, cd, f64]
    L.cvhcf_refine_spectra.argtypes = [f64, ci, ci, f64, f64, ci, ci, f64, f64]
    L.cvhcf_rad_rows.argtypes = [f64, ci, f64, ci, ci, f64, f64, ci, i32, f64, f64, f64,
                                 ci, ci, f64, f64]
    L.cvhcf_knockon_rows.argtypes = [f64, ci, f64, ci, ci, i32, f64, f64, f64, ci, ci,
                                     f64, f64]
    vp = ctypes.c_void_p
    L.cvhcf_nucel_ctx_new.restype = vp
    L.cvhcf_nucel_ctx_free.argtypes = [vp]
    L.cvhcf_nucel_add_material.argtypes = [vp, ci, ci, f64, f64, f64]
    L.cvhcf_nucel_rows.argtypes = [vp, f64, ci, f64, ci, ci, i32, i32, f64, i32, f64, f64, ci,
                                   f64, f64, f64, f64]
    for fn in ("cvhcf_ioni_rows", "cvhcf_ms_rows", "cvhcf_refine_spectra",
               "cvhcf_rad_rows", "cvhcf_knockon_rows", "cvhcf_nucel_add_material",
               "cvhcf_nucel_rows"):
        getattr(L, fn).restype = ci
    L.cvhcf_last_error.argtypes = [ctypes.c_char_p, ci]
    L.cvhcf_last_error.restype = ci
    _LIB = L
    return L


def _check(rc):
    if rc != 0:
        buf = ctypes.create_string_buffer(4096)
        lib().cvhcf_last_error(buf, 4096)
        raise RuntimeError(f"cvhcf: {buf.value.decode(errors='replace')} (rc={rc})")


def sync_config(**nucel):
    """Hand the reference modules' current switches to the C++; `nucel`
    (recoil, joint, exact) overrides cf_nucel_exact's NUCEL_RECOIL /
    NUCEL_JOINT and cf_knockon.QOP_EXACT for one nuclear-elastic call."""
    import cf_ms_exact
    import cf_nucel_exact as cne
    for n, v in _MS_PORTED.items():
        if getattr(ctr, n) != v:
            raise ValueError(f"cvhcf carries cf_track_resolution.{n} = {v} only "
                             f"(is {getattr(ctr, n)})")
    for n, v in _MSX_PORTED.items():
        if getattr(cf_ms_exact, n) != v:
            raise ValueError(f"cvhcf carries cf_ms_exact.{n} = {v} only "
                             f"(is {getattr(cf_ms_exact, n)})")
    lib().cvhcf_row_config_set(
        int(bool(cf_knockon.KNOCKON_JOINT)),
        int(bool(nucel.get("exact", cf_knockon.QOP_EXACT))),
        int(cf_knockon.KNOCKON_NPERDEC), int(cf_knockon.KNOCKON_NPERDEC_THIN),
        float(cf_knockon.KNOCKON_THIN), float(cf_knockon.KNOCKON_TCUT),
        float(cf_knockon.PMIN_FRAC), int(cf_brems_exact.RAD_NSUB),
        float(ctr.IONI_KOKOULIN), float(ctr.IONI_KOKOULIN_TCUT), float(ctr.IONI_A3_SCALE),
        float(ctr.IONI_EXC_SCALE), float(ctr.IONI_TMAX_SCALE),
        int(bool(nucel.get("recoil", cne.NUCEL_RECOIL))),
        int(bool(nucel.get("joint", cne.NUCEL_JOINT))))


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _rows(rows, ncols=None):
    r = np.ascontiguousarray(np.asarray(rows, dtype=np.float64))
    if r.ndim != 2:
        r = r.reshape(len(r), -1) if len(r) else r.reshape(0, ncols or 1)
    return r


def ioni_rows(tau, rows, wq):
    """cf_rows.ioni_rows through cvhcf::ioniRows."""
    tau = _f64(tau)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not len(rows):
        return S
    sync_config()
    r = _rows(rows)
    re, im = np.zeros(len(tau)), np.zeros(len(tau))
    _check(lib().cvhcf_ioni_rows(tau, len(tau), r, r.shape[1], len(r),
                                 _f64(np.broadcast_to(wq, (len(r),))), re, im))
    return re + 1j * im


def ms_rows(tau, rows, rid, wb, frac, scale=1.0):
    """cf_rows.ms_rows through cvhcf::msRows."""
    tau = _f64(tau)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not len(rid):
        return S
    sync_config()
    r = _rows(rows)
    out = np.zeros(len(tau))
    _check(lib().cvhcf_ms_rows(tau, len(tau), r, r.shape[1], len(r),
                               np.ascontiguousarray(rid, dtype=np.int32), _f64(wb),
                               _f64(frac), len(rid), float(scale), out))
    return out.astype(np.complex128)


def refine_spectra(recs, spec, vg, nsub):
    """cf_brems_exact.refine_spectra through cvhcf::refineSpectra."""
    sync_config()
    r, sp, vg = _rows(recs), _rows(spec), _f64(vg)
    nf = (len(vg) - 1) * int(nsub) + 1
    vf = np.zeros(nf)
    spf = np.zeros((len(r), 2 * nf))
    if len(r):
        _check(lib().cvhcf_refine_spectra(r, r.shape[1], len(r), sp, vg, len(vg),
                                          int(nsub), vf, spf))
    else:
        _check(lib().cvhcf_refine_spectra(np.zeros((1, 11)), 11, 0, np.zeros((1, 2 * len(vg))),
                                          vg, len(vg), int(nsub), vf, spf))
    return vf, spf


def rad_rows(tau, recs, spec, vg, rid, wq, wb, frac, exact_qop=None):
    """cf_rows.rad_rows through cvhcf::radRows (spectra already refined)."""
    if exact_qop is None:
        exact_qop = bool(cf_knockon.QOP_EXACT)
    tau = _f64(tau)
    if not len(rid):
        return np.zeros(len(tau), dtype=np.complex128)
    sync_config()
    r, sp = _rows(recs), _rows(spec)
    re, im = np.zeros(len(tau)), np.zeros(len(tau))
    _check(lib().cvhcf_rad_rows(tau, len(tau), r, r.shape[1], len(r), sp, _f64(vg),
                                len(vg), np.ascontiguousarray(rid, dtype=np.int32),
                                _f64(wq), _f64(wb), _f64(frac), len(rid),
                                int(bool(exact_qop)), re, im))
    return re + 1j * im


_PART = {"all": 0, "map": 1, "joint": 2}


def knockon_rows(tau, rows, rid, wq, wb, frac, part="all"):
    """cf_rows.knockon_rows through cvhcf::knockonRows."""
    tau = _f64(tau)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not cf_knockon.active() or not len(rows) or not len(rid):
        return S
    sync_config()
    r = _rows(rows)
    re, im = np.zeros(len(tau)), np.zeros(len(tau))
    _check(lib().cvhcf_knockon_rows(tau, len(tau), r, r.shape[1], len(r),
                                    np.ascontiguousarray(rid, dtype=np.int32),
                                    _f64(wq), _f64(wb), _f64(frac), len(rid),
                                    _PART[part], re, im))
    return re + 1j * im


# ------------------------------------------------------------ nuclear elastic
_NUC_CTX = {}


def _nucel_ctx(T, imat):
    """The shim's material context for Table `T`, holding the compositions of
    the table rows `imat` (added on first use); the C++ builds its mixtures
    from them exactly as the maker builds them from G4Material."""
    hit = _NUC_CTX.get(id(T))
    if hit is None or hit[0] is not T:
        hit = (T, lib().cvhcf_nucel_ctx_new(), set())
        _NUC_CTX[id(T)] = hit
    _, ctx, have = hit
    for im in np.unique(np.asarray(imat)):
        im = int(im)
        if im < 0 or im in have:
            continue
        m = T.mats[im]
        _check(lib().cvhcf_nucel_add_material(ctx, im, len(m["Z"]), _f64(m["Z"]), _f64(m["A"]),
                                               _f64(m["W"])))
        have.add(im)
    return ctx


def nucel_rows(T, tau, M, imat, pdg, mass, rid, wb, frac, wq_row, wb_mid,
               recoil=True, joint=True, exact=False):
    """nucel_tables.Table.rows (T.rows(tau, M, ...)) through cvhcf::nucelRows:
    (N, ang, rec, jnt)."""
    tau = _f64(tau)
    nt = len(tau)
    M = _rows(M)
    n = len(M)
    out = np.zeros(5 * nt)
    N = np.zeros(1)
    if n:
        sync_config(recoil=recoil, joint=joint, exact=exact)
        bc = lambda x, t=np.float64: np.ascontiguousarray(np.broadcast_to(np.asarray(x, dtype=t), (n,)))
        im = bc(imat, np.int32)
        _check(lib().cvhcf_nucel_rows(_nucel_ctx(T, im), tau, nt, M, M.shape[1], n, im,
                                      bc(pdg, np.int32), bc(mass),
                                      np.ascontiguousarray(rid, dtype=np.int32), _f64(wb),
                                      _f64(frac), len(rid), bc(wq_row), bc(wb_mid), out, N))
    return (float(N[0]), out[:nt], out[nt:2 * nt] + 1j * out[2 * nt:3 * nt],
            out[3 * nt:4 * nt] + 1j * out[4 * nt:])


def rows_exponent(S, leg, tau, rid, wb, frac, wq_row, wb_mid):
    """cf_nucel_exact.rows_exponent through cvhcf::nucelRows."""
    import cf_nucel_exact as cne
    if not cne.NUCEL_CHANNEL or not len(leg["ms"]):
        return S
    pdg = cne.pdg_from_leg(leg)
    if pdg is None:
        return S
    T = cne.table()
    _, ang, rec, jnt = nucel_rows(T, tau, np.asarray(leg["ms"]), cne.leg_materials(T, leg), pdg,
                                  cne.mass_of(pdg), rid, wb, frac, wq_row, wb_mid,
                                  recoil=cne.NUCEL_RECOIL, joint=cne.NUCEL_JOINT,
                                  exact=bool(cf_knockon.QOP_EXACT))
    S += ang
    S += rec
    S += jnt
    return S

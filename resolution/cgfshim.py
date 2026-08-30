"""ctypes binding to the FIT's own CGF block (cxx/libcgfshim.so).

`ioni_step_exponent` here and `cf_track_resolution.ioni_step_exponent` compute
the same object from the same records. This one calls `cvhcgf::blockExponent`
-- the code the track fit actually runs -- so the offline model and the
estimator cannot drift apart. Build with `cxx/build.sh`.
"""
import ctypes
import os

import numpy as np

_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cxx", "libcgfshim.so")

_ERR = {
    -1: "null pointer passed to the shim",
    -2: "bad shape: ns < 0, nt <= 0, or stride not in (11, 13)",
    -3: ("regime 2/3 record with stride 11: beta^2 and E are not in the file. "
         "Re-export with CVH_IONI_EXACTDELTA (which writes stride 13); they "
         "cannot be recovered from tmax without the particle mass."),
}


class ShimUnavailable(RuntimeError):
    pass


def _load():
    if not os.path.exists(_SO):
        raise ShimUnavailable(
            f"{_SO} not built -- run cxx/build.sh (needs cvmfs for CLHEP headers)")
    lib = ctypes.CDLL(_SO)
    lib.cvhcgf_ioni_step_exponent.restype = ctypes.c_int
    lib.cvhcgf_ioni_step_exponent.argtypes = [
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ]
    lib.cvhcgf_block_exponent_one.restype = ctypes.c_int
    lib.cvhcgf_block_exponent_one.argtypes = (
        [ctypes.c_int] + [ctypes.c_double] * 11 + [ctypes.c_int,
         np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
         ctypes.c_int,
         np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")])
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def available():
    try:
        lib()
        return True
    except Exception:
        return False


def ioni_step_exponent(steps, wstd, tau, kok_nbin=0):
    """Drop-in for cf_track_resolution.ioni_step_exponent.

    `kok_nbin` is the Kokoulin bucket count for regime 2/3 steps (0 = off),
    i.e. the offline IONI_KOKOULIN_NBIN when IONI_KOKOULIN is on.
    """
    steps = np.ascontiguousarray(steps, dtype=np.float64)
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    if steps.ndim != 2:
        raise ValueError(f"steps must be 2-D, got {steps.shape}")
    ns, stride = steps.shape
    out = np.empty(2 * len(tau), dtype=np.float64)
    rc = lib().cvhcgf_ioni_step_exponent(
        steps.ravel(), ns, stride, float(wstd), tau, len(tau), int(kok_nbin), out)
    if rc != 0:
        raise RuntimeError(_ERR.get(rc, f"shim returned {rc}"))
    return out[0::2] + 1j * out[1::2]


def block_exponent_one(tau, regime=2, gsig2=0.0, a1=0.0, e1=0.0, a2=0.0, e2=0.0,
                       a3=0.0, e0=0.0, tmax=0.0, beta2=0.0, etot=0.0, gs=1.0,
                       kok_nbin=0):
    """One synthetic step -- for driving both series branches in validation."""
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    out = np.empty(2 * len(tau), dtype=np.float64)
    rc = lib().cvhcgf_block_exponent_one(
        int(regime), float(gsig2), float(a1), float(e1), float(a2), float(e2),
        float(a3), float(e0), float(tmax), float(beta2), float(etot), float(gs),
        int(kok_nbin), tau, len(tau), out)
    if rc != 0:
        raise RuntimeError(_ERR.get(rc, f"shim returned {rc}"))
    return out[0::2] + 1j * out[1::2]

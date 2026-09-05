#!/usr/bin/env python3
"""Newton / trust-region fit of the unbinned CVH J/psi mass likelihood.

Replaces the (alpha, r) GRID SCAN in ``cf_mass_likelihood.py --demo`` by a
real minimisation of the SAME objective, written as a TensorFlow graph so
that the exact gradient and Hessian are available.  The resolution scale is
additionally split into its three physical families.

THE OBJECTIVE (identical to ``cf_mass_likelihood.demo``)
--------------------------------------------------------
Per candidate i the mass density is the inverse Fourier transform of a
product of characteristic functions -- resonance (delta at M = m_Jpsi),
FSR kernel, and the candidate resolution CF:

    L_i = 1/(pi sigma_i) Int_0^inf Re[ phi_K(t/sigma_i)
                                       exp(S_i(t))
                                       exp(-i t (m_i - M(1+alpha))/sigma_i) ] dt

evaluated by the trapezoid rule on the standardized grid TG = linspace(0, 14,
448) (t is in units of 1/sigma_i; the absolute argument of phi_K is
tgi = TG/sigma_i).  ``m_i - M`` is ``mobs_minus_M = z_i sigma_i + (eta_i - M)``
with ``eta_i`` the candidate's gen mass.  phi_K is the empirical CF of the
FSR kernel samples ``dm``, tabulated on 8192 absolute-t points and linearly
interpolated (np.interp) onto tgi.

The scan used ONE resolution scale r multiplying the whole exponent,

    S_i(t) = r [ -1/2 vgf_i t^2 + Sms_i(t) + Sio_re_i(t) + i Sio_im_i(t) ] ,

with vgf_i the Gaussian (hit) variance fraction, Sms the multiple-scattering
exponent and Sio the (complex, skewed) ionization exponent.  Here the single r
is SPLIT INTO THE THREE FAMILIES that produced the three terms:

    S_i(t) = k_hit (-1/2 vgf_i t^2) + k_ms Sms_i(t)
                                    + k_ioni (Sio_re_i(t) + i Sio_im_i(t))
                                    + k_rad  (Srad_re_i(t) + i Srad_im_i(t)) .

k_hit = k_ms = k_ioni = k_rad = r recovers the scan exactly (and the scan
itself had no radiative term at all, i.e. it is the k_rad = 0 model).

THE RADIATIVE FAMILY (bremsstrahlung + pair production) entered the pairs
caches on 2026-09-03 (`cf_mass_likelihood`, key `rad_model`).  It is the
second channel of the SAME parmtype-11 block as the ionization term -- same
transport weight, same sign (a radiated photon can only take energy off a
muon, so it can only LOWER the pair mass) -- and it is one-sided, so it is
the natural candidate for the 1-3 sigma mass asymmetry the ionization-only
model under-predicts.  It is FIXED AT 1 by default (`--krad`, so `--krad 0`
is the pre-2026-09-03 model exactly) and floated with `--float-krad`.

The extended likelihood adds the uniform combinatoric floor over the 0.7 GeV
mass window,

    NLL = - sum_i log[ (1 - f) max(L_i, 0) + f / 0.7 ] ,   f = FBKG = 0.005 ,

optionally with f floated as a free (prior-free) parameter.

IMPLEMENTATION NOTES
--------------------
* Everything is done in REAL float64 arithmetic.  With
  psi = k_ioni Sio_im - tgi (m_i - M(1+alpha)) and Sre the real exponent,

      Re[ phi_K e^{S} e^{-i tgi delta} ]
          = e^{Sre} ( Re phi_K cos psi - Im phi_K sin psi )

  which is exactly what the complex numpy expression evaluates and avoids
  complex-valued gradients.
* The candidates are processed in chunks (``--chunk``, default 32768) with a
  single traced ``tf.function``; the sample is zero-padded to a whole number
  of chunks and padded entries are killed by a 0/1 weight.  This bounds the
  materialised (chunk x 448) float64 temporaries to ~120 MB each.
* ``np.clip(L_i, 0, None)`` in the scan has identically zero gradient below
  zero.  ``--floor softplus`` (default for fitting) replaces it by
  ``s log(1 + exp(L/s))`` with s = ``--floor-scale`` (default 1e-9), which is
  equal to max(L, 0) to ~1e-9 absolute but is everywhere differentiable.
  ``--floor clip`` reproduces the scan bit-for-bit.
* THE SCAN IS NOT PURE float64.  In numpy >= 2, ``Sms + 1j*0. + Sio_re +
  1j*Sio_im`` on float32 cache arrays is COMPLEX64, and ``r * Sexp`` stays
  complex64, so the scan's exponent carries float32 rounding.  ``--dtype-scan``
  reproduces that (it forms the same complex64 combination on the host and
  hands the rounded real/imaginary parts to the graph); the default
  ``--dtype-fit`` promotes the cached float32 to float64 before any
  arithmetic.  The difference is reported by ``--scan-check``.

ENVIRONMENT
-----------
The rabbit TensorFlow stack (rabbit uses the wmassdev singularity, see
/work/submit/david_w/rabbit_260826_tfMinimizer/README.md).  ``wums`` is not in
the image, so it is added from the mfs venv through a symlink shim:

    export APPTAINERENV_PYTHONPATH=\\
      /work/submit/david_w/ZMass/calibration_studies/env_tf/pypath:\\
      /work/submit/david_w/ZMass/calibration_studies/resolution
    singularity exec -B /work/submit,/home/submit,/ceph/submit,/scratch/submit \\
      /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest \\
      python3 cf_masslik_fit.py ...

(tensorflow 2.21.0, python 3.13, numpy 2.4.6, scipy 1.18.0; CPU only -- this
node has no GPU.  Thread counts are pinned with ``--threads``, default 32.)

WHAT STEP 2 (rabbit UnbinnedTerm) NEEDS FROM THIS CODE
------------------------------------------------------
rabbit currently supports only QUADRATIC external likelihood terms
(``rabbit/external_likelihood.py``: grad + Hessian + two scalars).  An
unbinned term is a new object; what it has to be handed is exactly the state
of :class:`MassNLL`:

  constants (tf, immutable, built once by ``load_inputs``; shapes for n
  candidates on the 448-point t grid):
      sigma        (n,)        float64   per-candidate mass resolution
      mobs         (n,)        float64   m_reco - m_Jpsi
      vgf          (n,)        float64   Gaussian (hit) variance fraction
      Sms          (n, 448)    float32   multiple-scattering exponent
      Sio_re       (n, 448)    float32   ionization exponent, real part
      Sio_im       (n, 448)    float32   ionization exponent, imaginary part
      phiK_re/_im  (n, 448)    float64   FSR-kernel CF at t = TG/sigma_i
      TG, dTG      (448,)/(447,)         quadrature grid and weights
      w            (n,)        float64   0/1 padding mask (chunking)

  interface:
      value(theta)                 -> scalar  (theta: (npar,) float64)
      value_and_grad(theta)        -> scalar, (npar,)
      value_grad_hess(theta)       -> scalar, (npar,), (npar, npar)
      hessp(theta, p)              -> (npar,)     [for trust-krylov / rabbit's
                                                   Krylov path; a forward-over-
                                                   reverse tape on _chunk_nll]
  chunking: the term is a SUM over candidates, so value, gradient and Hessian
  are all additive over chunks -- ``MassNLL.nll_grad_hess`` already accumulates
  them.  rabbit only needs the per-chunk ``tf.function`` and the chunk count.

  parameter wiring: ``_unpack`` is the only place that maps the fit vector to
  (alpha, k_hit, k_ms, k_ioni, f_bkg).  In rabbit these become named POIs /
  nuisances; the scalar-potential B-field coefficients of the CVH global fit
  would enter the same way, replacing ``alpha`` by the per-candidate Jacobian
  contraction d(m)/dc_{n,m}.

USAGE
-----
    python3 cf_masslik_fit.py \\
        --pairs-cache  runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz \\
        --kernel-cache runs/cf_masskernel_jpsigun_ul16_260902_m0.npz \\
        --model families --float-bkg \\
        --scan-check   runs/masslik_demo_scan_jpsigun_260902_fixsign.npz \\
        --profile alpha --out runs/masslikfit_jpsigun.npz --tag jpsigun
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

MJPSI = 3.0969
FBKG = 0.005
MWIN = 0.7  # mass window of the uniform combinatoric floor [GeV]
ASCALE = 1e-3  # alpha is fitted in units of 1e-3
FSCALE = 1e-3  # f_bkg is fitted in units of 1e-3.  PRECONDITIONING, not a
# prior: d2NLL/df2 ~ n/f^2 ~ 1e9 in absolute units against ~1e2 for the k's,
# which collapses the trust region (seen on btojpsix: 61 iterations, no move
# in alpha/k).  In units of 1e-3 the five curvatures are within ~1e4.


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pairs-cache", default=None, help="cf_masspairs_*.npz")
    p.add_argument("--kernel-cache", default=None, help="cf_masskernel_*.npz")
    p.add_argument(
        "--model",
        choices=["r", "families"],
        default="families",
        help="'r': one resolution scale (the scan's model, 2 pars). "
        "'families': k_hit, k_ms, k_ioni split (4 pars).",
    )
    p.add_argument(
        "--float-bkg", action="store_true", help="float the uniform floor f_bkg"
    )
    p.add_argument(
        "--krad", type=float, default=1.0,
        help="value of the radiative-family scale when it is NOT floated "
        "(default 1 = the physics model; 0 = the model before the term "
        "existed). In model 'r' the single r multiplies the radiative family "
        "too and this is ignored.",
    )
    p.add_argument(
        "--float-krad", action="store_true",
        help="float k_rad as a fifth parameter of model 'families'",
    )
    p.add_argument("--out", default=None, help="output npz with the full result")
    p.add_argument(
        "--outpath", default=None, help="plot/table dir (default ~/public_html/cvh/<YYMMDD>_masslikfit)"
    )
    p.add_argument("--tag", default="", help="tag appended to figure/table names")
    p.add_argument(
        "--profile",
        default=None,
        choices=["alpha"],
        help="also produce the profiled DNLL curve (re-minimise the other "
        "parameters at fixed alpha over +-5 sigma)",
    )
    p.add_argument("--profile-n", type=int, default=15, help="profile points")
    p.add_argument("--profile-range", type=float, default=5.0, help="+- N sigma")
    p.add_argument(
        "--scan-check",
        default=None,
        help="masslik_demo_scan_*.npz: verify objective identity on its grid",
    )
    p.add_argument("--scan-check-n", type=int, default=13, help="grid points to check")
    p.add_argument(
        "--floor",
        choices=["clip", "softplus", "none"],
        default="softplus",
        help="positivity treatment of L_i before the mixture",
    )
    p.add_argument("--floor-scale", type=float, default=1e-9)
    p.add_argument(
        "--fbkg", type=float, default=FBKG,
        help="value of the fixed uniform floor (ignored with --float-bkg); "
        "0 removes it",
    )
    p.add_argument(
        "--window-norm", action="store_true",
        help="normalise each candidate's model density over the ACCEPTED mass "
        "window, L_i -> L_i / [F_i(hi) - F_i(lo)] with F the Gil-Pelaez CDF of "
        "the same characteristic function.  This is the exact treatment of a "
        "selection that is a cut on the OBSERVED dimuon mass; it is a no-op "
        "for a selection that is not.  Default OFF (every fit before "
        "2026-09-04 is the un-truncated likelihood).",
    )
    p.add_argument("--subset", default=None,
                   help="npz/npy with a boolean mask aligned with the pairs "
                   "cache (key `mask` for an npz); fit only those candidates")
    p.add_argument("--window-lo", type=float, default=MJPSI - 0.35,
                   help="lower edge of the accepted mass window [GeV]")
    p.add_argument("--window-hi", type=float, default=MJPSI + 0.35,
                   help="upper edge of the accepted mass window [GeV]")
    p.add_argument(
        "--dtype-scan",
        action="store_true",
        help="reproduce the scan's complex64 exponent rounding (see docstring)",
    )
    p.add_argument(
        "--precision",
        choices=["float64", "float32"],
        default="float64",
        help="graph precision (float32 only for the timing/precision study)",
    )
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--threads", type=int, default=32, help="TF intra/inter-op threads")
    p.add_argument("--method", default="trust-exact", help="scipy minimizer")
    p.add_argument("--maxn", type=int, default=0, help="debug: use only N candidates")
    p.add_argument(
        "--fd-check", action="store_true", help="finite-difference gradient check"
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--replot",
        default=None,
        help="regenerate the figures from a previously written --out npz "
        "(no fit); --pairs-cache/--kernel-cache are then ignored",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# host-side input assembly -- reproduces cf_mass_likelihood.demo() line by line
# --------------------------------------------------------------------------
def load_inputs(pairs_cache, kernel_cache, dtype_scan=False, maxn=0, log=print,
                phik_cache=True, subset=None):
    t0 = time.time()
    d = np.load(pairs_cache)
    k = np.load(kernel_cache)
    dm = k["dm"]
    TG = np.asarray(d["tgrid"], dtype=np.float64)
    ref = np.linspace(0.0, 14.0, 448)
    assert TG.shape == ref.shape and np.allclose(TG, ref), "unexpected t grid"

    z, sig = d["z"].astype(np.float64), d["sigma"].astype(np.float64)
    eta = d["eta"].astype(np.float64)
    vgf = d["vgf"].astype(np.float64)
    Sms, Sio_re, Sio_im = d["Sms"], d["Sio_re"], d["Sio_im"]
    # the radiative family (2026-09-03). `rad_model` is the provenance value:
    # 1 = built from the `radstepv` export, 0 = the production predates it and
    # the arrays are identically zero, so the family is dropped rather than
    # given a parameter that cannot move the likelihood.
    rad_model = int(d["rad_model"]) if "rad_model" in d.files else 0
    has_rad = ("Srad_re" in d.files) and rad_model == 1
    Srad_re = d["Srad_re"] if has_rad else None
    Srad_im = d["Srad_im"] if has_rad else None
    n = len(z)
    # CANDIDATE SUBSET (2026-09-04): a boolean mask aligned with the cache,
    # written by `censoring_aux.py` (which proves the alignment by matching z
    # bit-for-bit against the tree).  Used to fit with a class of candidates
    # REMOVED -- e.g. the ones exposed to the 2 GeV momentum-floor clamp of the
    # Gauss-Newton step, which the CF model knows nothing about.
    if subset is not None:
        m = np.load(subset)
        m = m["mask"] if hasattr(m, "files") else m
        m = np.asarray(m).astype(bool)
        if len(m) != n:
            raise ValueError(f"subset mask has {len(m)} rows, cache has {n}")
        z, sig, eta, vgf = z[m], sig[m], eta[m], vgf[m]
        Sms, Sio_re, Sio_im = Sms[m], Sio_re[m], Sio_im[m]
        if has_rad:
            Srad_re, Srad_im = Srad_re[m], Srad_im[m]
        log(f"subset {subset}: {int(m.sum())} of {n} candidates kept "
            f"({100.*m.mean():.2f} %)")
        n = int(m.sum())
    if maxn and maxn < n:
        n = maxn
        z, sig, eta, vgf = z[:n], sig[:n], eta[:n], vgf[:n]
        Sms, Sio_re, Sio_im = Sms[:n], Sio_re[:n], Sio_im[:n]
        if has_rad:
            Srad_re, Srad_im = Srad_re[:n], Srad_im[:n]
    log(f"{n} candidates, {len(dm)} kernel samples, "
        f"rad_model={rad_model} (radiative family "
        f"{'IN' if has_rad else 'ABSENT'})")

    # empirical FSR-kernel CF, tabulated on absolute t (demo: 8192 points, in
    # two row blocks to bound memory; row blocking does not change the axis-1
    # mean, so any block size is bit-identical)
    tmax_abs = TG[-1] / sig.min()
    tabs = np.linspace(0.0, tmax_abs, 8192)
    cf = os.path.join(
        os.path.dirname(os.path.abspath(pairs_cache)),
        "masslikfit_phiKtab_%s_%d_%.12e.npz"
        % (os.path.basename(kernel_cache).replace(".npz", ""), len(dm), tmax_abs))
    if phik_cache and os.path.exists(cf):
        phiK_tab = np.load(cf)["phiK_tab"]
        log(f"phiK table from cache {cf}")
    else:
        t1 = time.time()
        blk = 1024
        phiK_tab = np.concatenate(
            [
                np.mean(np.exp(1j * np.outer(tabs[i : i + blk], dm)), axis=1)
                for i in range(0, len(tabs), blk)
            ]
        )
        log(f"phiK table built in {time.time()-t1:.1f} s")
        if phik_cache:
            try:
                np.savez(cf, tabs=tabs, phiK_tab=phiK_tab)
                log(f"phiK table cached -> {cf}")
            except OSError as e:
                log(f"phiK cache write failed: {e}")

    mobs = z * sig + (eta - MJPSI)  # m_reco - M_Jpsi
    tgi = TG[None, :] / sig[:, None]  # absolute t per candidate
    phiK_re = np.interp(tgi, tabs, phiK_tab.real)
    phiK_im = np.interp(tgi, tabs, phiK_tab.imag)
    del tgi

    if dtype_scan:
        # bit-reproduce the scan: Sexp = Sms + 1j*0. + Sio_re + 1j*Sio_im is
        # COMPLEX64 in numpy>=2 (float32 arrays + weak python complex scalar).
        Sexp = Sms + 1j * 0.0 + Sio_re + 1j * Sio_im
        assert Sexp.dtype == np.complex64, f"unexpected {Sexp.dtype}"
        Sms_out = np.ascontiguousarray(Sexp.real)  # float32(Sms + Sio_re)
        Sio_re_out = np.zeros((1, 1), dtype=np.float32)  # folded into Sms_out
        Sio_im_out = np.ascontiguousarray(Sexp.imag)
        folded = True
        del Sexp
    else:
        Sms_out, Sio_re_out, Sio_im_out = Sms, Sio_re, Sio_im
        folded = False
    if dtype_scan and has_rad:
        # --dtype-scan exists to bit-reproduce a scan that had no radiative
        # term. Keeping the family would compare two different models and
        # call the difference a rounding study.
        log("--dtype-scan: DROPPING the radiative family (the scan it "
            "reproduces predates it)")
        has_rad = False
        Srad_re = Srad_im = None

    log(f"inputs assembled in {time.time()-t0:.1f} s")
    return dict(
        n=n,
        TG=TG,
        sig=sig,
        mobs=mobs,
        vgf=vgf,
        Sms=Sms_out,
        Sio_re=Sio_re_out,
        Sio_im=Sio_im_out,
        has_rad=has_rad,
        rad_model=rad_model,
        Srad_re=Srad_re,
        Srad_im=Srad_im,
        phiK_re=phiK_re,
        phiK_im=phiK_im,
        folded=folded,
    )


# --------------------------------------------------------------------------
# the TF objective
# --------------------------------------------------------------------------
class MassNLL:
    """tf.function objective, gradient and Hessian of the unbinned mass NLL.

    Parameter vector (internal, scaled):
        model 'r'        : [alpha/1e-3, r]          (r multiplies ALL families,
                                                     the radiative one too)
        model 'families' : [alpha/1e-3, k_hit, k_ms, k_ioni]
      + [k_rad] when float_krad, + [f_bkg] when float_bkg, in that order.
    """

    def __init__(self, inp, model="families", float_bkg=False, floor="softplus",
                 floor_scale=1e-9, chunk=32768, precision="float64", log=print,
                 fbkg0=FBKG, krad0=1.0, float_krad=False, window=None):
        import tensorflow as tf

        self.tf = tf
        self.model = model
        # SELECTION WINDOW (absolute dimuon mass, GeV) over which the model
        # density is renormalised; None = the untruncated likelihood, which is
        # what every fit before 2026-09-04 used.  Stored in mobs space
        # (m - m_Jpsi), i.e. the same frame as `self.mobs`.
        self.win = (None if window is None
                    else (float(window[0]) - MJPSI, float(window[1]) - MJPSI))
        # the uniform combinatoric floor is normalised over the SAME window
        self.mwin = MWIN if window is None else float(window[1]) - float(window[0])
        self.float_bkg = float_bkg
        self.has_rad = bool(inp.get("has_rad", False))
        self.krad0 = float(krad0)
        self.float_krad = bool(float_krad) and self.has_rad
        if float_krad and not self.has_rad:
            log("--float-krad ignored: this pairs cache has no radiative "
                "family (rad_model=0)")
        self.floor = floor
        self.floor_scale = floor_scale
        self.fbkg0 = fbkg0
        self.log = log
        self.rdt = tf.float64 if precision == "float64" else tf.float32
        self.npdt = np.float64 if precision == "float64" else np.float32
        n = inp["n"]
        self.n = n
        self.chunk = min(chunk, n)
        self.nchunk = int(np.ceil(n / self.chunk))
        npad = self.nchunk * self.chunk - n
        self.folded = inp["folded"]

        def pad1(a, fill=0.0):
            return np.concatenate([a, np.full(npad, fill, a.dtype)])

        def pad2(a):
            return np.concatenate([a, np.zeros((npad, a.shape[1]), a.dtype)])

        rdt = self.npdt
        self.TG = tf.constant(inp["TG"], self.rdt)
        # trapezoid weights: d = diff(TG); trapz = sum(d*(y[1:]+y[:-1])/2)
        self.dTG = tf.constant(np.diff(inp["TG"]).astype(rdt), self.rdt)
        # 1/TG for the Gil-Pelaez CDF integrand; entry 0 is never used (the
        # TG = 0 point is replaced by its exact limit in _winnorm)
        _inv = np.zeros_like(inp["TG"], dtype=rdt)
        _inv[1:] = 1.0 / inp["TG"][1:]
        self.invTG = tf.constant(_inv, self.rdt)
        self.sig = tf.constant(pad1(inp["sig"], 1.0).astype(rdt), self.rdt)
        self.mobs = tf.constant(pad1(inp["mobs"]).astype(rdt), self.rdt)
        self.vgf = tf.constant(pad1(inp["vgf"]).astype(rdt), self.rdt)
        self.w = tf.constant(
            np.concatenate([np.ones(n), np.zeros(npad)]).astype(rdt), self.rdt
        )
        # the (n, 448) blocks stay float32 (as cached) and are promoted inside
        # the graph -- 3x1.6 GB saved on the J/psi-gun sample
        self.Sms = tf.constant(pad2(inp["Sms"]))
        self.Sio_im = tf.constant(pad2(inp["Sio_im"]))
        if self.has_rad:
            self.Srad_re = tf.constant(pad2(inp["Srad_re"]))
            self.Srad_im = tf.constant(pad2(inp["Srad_im"]))
        else:
            self.Srad_re = self.Srad_im = None
        if self.folded:
            self.Sio_re = None
        else:
            self.Sio_re = tf.constant(pad2(inp["Sio_re"]))
        self.pKre = tf.constant(pad2(inp["phiK_re"]).astype(rdt), self.rdt)
        self.pKim = tf.constant(pad2(inp["phiK_im"]).astype(rdt), self.rdt)
        self.npar = ((2 if model == "r" else 4)
                     + (1 if self.float_krad else 0)
                     + (1 if float_bkg else 0))
        self.parnames = (
            ["alpha[1e-3]", "r"] if model == "r"
            else ["alpha[1e-3]", "k_hit", "k_ms", "k_ioni"]
        ) + (["k_rad"] if self.float_krad else []) + (
            ["f_bkg[1e-3]"] if float_bkg else [])
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self._alpha_cache = None

        # -------- traced graph pieces (index passed as a TENSOR -> 1 trace) --
        self._g_nll = tf.function(self._chunk_nll)
        self._g_lg = tf.function(self._chunk_lg)
        self._g_lgh = tf.function(self._chunk_lgh)
        self._g_nll_cached = tf.function(self._chunk_nll_cached)
        log(f"MassNLL: n={n} pad={npad} chunk={self.chunk} x {self.nchunk} "
            f"npar={self.npar} precision={precision} floor={floor} "
            f"rad={'floated' if self.float_krad else (('fixed %.3f' % self.krad0) if self.has_rad else 'absent')} "
            f"window={'OFF' if self.win is None else ('[%.4f, %.4f] GeV' % (self.win[0]+MJPSI, self.win[1]+MJPSI))}")

    # ---- parameter unpacking (inside the graph, differentiable) ----
    def _unpack(self, th):
        tf = self.tf
        alpha = th[0] * self.npdt(ASCALE)
        if self.model == "r":
            khit = kms = kioni = th[1]
            # the single r multiplies EVERY family, the radiative one too
            krad = th[1]
            nxt = 2
        else:
            khit, kms, kioni = th[1], th[2], th[3]
            nxt = 4
            if self.float_krad:
                krad = th[4]
                nxt = 5
            else:
                krad = tf.constant(self.npdt(self.krad0), self.rdt)
        fb = (th[nxt] * self.npdt(FSCALE)
              if self.float_bkg else tf.constant(self.fbkg0, self.rdt))
        return alpha, khit, kms, kioni, krad, fb

    def _slice(self, x, i, two_d=True):
        tf = self.tf
        C = self.chunk
        if two_d:
            return tf.slice(x, [i * C, 0], [C, tf.shape(x)[1]])
        return tf.slice(x, [i * C], [C])

    def _chunk_nll(self, i, th):
        _, _, _, _, _, fb = self._unpack(th)
        Li = self._raw_li(i, th)
        w = self._slice(self.w, i, False)
        return self._mix(Li, fb, w)

    def _raw_li(self, i, th):
        tf = self.tf
        alpha, khit, kms, kioni, krad, fb = self._unpack(th)
        C = self.chunk
        sig = self._slice(self.sig, i, False)
        mobs = self._slice(self.mobs, i, False)
        vgf = self._slice(self.vgf, i, False)
        w = self._slice(self.w, i, False)
        Sms = tf.cast(tf.slice(self.Sms, [i * C, 0], [C, 448]), self.rdt)
        Sio_im = tf.cast(tf.slice(self.Sio_im, [i * C, 0], [C, 448]), self.rdt)
        pKre = tf.slice(self.pKre, [i * C, 0], [C, 448])
        pKim = tf.slice(self.pKim, [i * C, 0], [C, 448])
        TG = self.TG
        tgi = TG[None, :] / sig[:, None]
        Sre = khit * (-0.5 * vgf[:, None] * TG[None, :] ** 2) + kms * Sms
        if not self.folded:
            Sio_re = tf.cast(tf.slice(self.Sio_re, [i * C, 0], [C, 448]), self.rdt)
            Sre = Sre + kioni * Sio_re
        Sim = kioni * Sio_im
        if self.has_rad:
            Sre = Sre + krad * tf.cast(
                tf.slice(self.Srad_re, [i * C, 0], [C, 448]), self.rdt)
            Sim = Sim + krad * tf.cast(
                tf.slice(self.Srad_im, [i * C, 0], [C, 448]), self.rdt)
        delta = mobs - self.npdt(MJPSI) * alpha
        psi = Sim - tgi * delta[:, None]
        eS = tf.exp(Sre)
        integ = eS * (pKre * tf.cos(psi) - pKim * tf.sin(psi))
        Li = tf.reduce_sum(
            self.dTG[None, :] * (integ[:, 1:] + integ[:, :-1]) * self.npdt(0.5), axis=1
        ) / (self.npdt(np.pi) * sig)
        if self.win is not None:
            Li = Li / self._winnorm(eS * pKre, eS * pKim, Sim, tgi, sig, alpha)
        return Li

    # ---- window (selection) truncation: L_i -> L_i / [F_i(hi) - F_i(lo)] ----
    def _winnorm(self, A, B, Sim, tgi, sig, alpha):
        """Model probability inside the accepted mass window, per candidate.

        Gil-Pelaez: F(x) = 1/2 - (1/pi) Int_0^inf Im[Phi(t) e^{-itx}]/t dt, so

            F(hi) - F(lo) = (1/pi) Int_0^inf { Im[Phi e^{-i t x_lo}]
                                             - Im[Phi e^{-i t x_hi}] } dt/t

        with Phi(t) = phi_K(t) e^{S(sigma t)} the SAME characteristic function
        whose real part gives the density, and x_e = m_e - M(1+alpha) the
        window edges seen in the candidate's own frame.  Written on the
        standardized grid TG = sigma t the explicit sigma cancels in dt/t and

            Im[Phi e^{-i t x}] = A sin(psi) + B cos(psi),
            A = e^{Sre} Re phi_K,  B = e^{Sre} Im phi_K,  psi = Sim - TG x/sigma

        (the same A, B, psi as the density, which uses A cos psi - B sin psi).

        THE t -> 0 END POINT IS EXACT AND MODEL INDEPENDENT.  The DIFFERENCE of
        the two edge integrands behaves as

            Im[Phi (e^{-i t x_lo} - e^{-i t x_hi})]/t
                = Im[Phi . i t (x_hi - x_lo) + O(t^2)]/t -> (x_hi - x_lo) Re Phi(0)
                = (m_hi - m_lo),

        i.e. in TG units (m_hi - m_lo)/sigma, whatever the kernel mean or any
        residual non-centring of S.  Taking the difference BEFORE dividing by t
        is what makes the 1/t integrable without an analytic expansion of
        phi_K; the two edges never appear separately.
        """
        tf = self.tf
        dlo = self.npdt(self.win[0]) - self.npdt(MJPSI) * alpha
        dhi = self.npdt(self.win[1]) - self.npdt(MJPSI) * alpha
        plo = Sim - tgi * dlo
        phi_ = Sim - tgi * dhi
        num = (A * tf.sin(plo) + B * tf.cos(plo)) - (
            A * tf.sin(phi_) + B * tf.cos(phi_))
        # 1/TG with the (removable) TG = 0 point replaced by its exact limit
        D = tf.concat(
            [(self.npdt(self.win[1] - self.win[0]) / sig)[:, None],
             num[:, 1:] * self.invTG[None, 1:]], axis=1)
        norm = tf.reduce_sum(
            self.dTG[None, :] * (D[:, 1:] + D[:, :-1]) * self.npdt(0.5), axis=1
        ) / self.npdt(np.pi)
        return norm

    def _chunk_liraw(self, i, th):
        """per-candidate raw L_i (before the positivity floor), for diagnostics"""
        return self._raw_li(i, th)

    def _mix(self, Li, fb, w):
        tf = self.tf
        if self.floor == "clip":
            Lp = tf.maximum(Li, self.npdt(0.0))
        elif self.floor == "softplus":
            s = self.npdt(self.floor_scale)
            Lp = s * tf.math.softplus(Li / s)
        else:
            Lp = Li
        L = (self.npdt(1.0) - fb) * Lp + fb / self.npdt(self.mwin)
        return -tf.reduce_sum(w * tf.math.log(L))

    def _chunk_lg(self, i, th):
        tf = self.tf
        with tf.GradientTape() as t1:
            t1.watch(th)
            L = self._chunk_nll(i, th)
        return L, t1.gradient(L, th)

    def _chunk_lgh(self, i, th):
        tf = self.tf
        with tf.GradientTape() as t2:
            t2.watch(th)
            with tf.GradientTape() as t1:
                t1.watch(th)
                L = self._chunk_nll(i, th)
            g = t1.gradient(L, th)
        H = t2.jacobian(g, th, experimental_use_pfor=False)
        return L, g, H

    # ---- alpha-cached path: exp(Sre)*phiK and Sim do not depend on alpha ----
    def set_k_cache(self, khit, kms, kioni, krad=None):
        """Materialise A = e^{Sre} Re phiK, B = e^{Sre} Im phiK, Sim, tgi."""
        tf = self.tf
        t0 = time.time()
        A, B, S, T = [], [], [], []
        C = self.chunk
        for i in range(self.nchunk):
            sl = slice(i * C, (i + 1) * C)
            sig = self.sig[sl]
            Sms = tf.cast(self.Sms[sl], self.rdt)
            Sre = self.npdt(khit) * (
                -0.5 * self.vgf[sl][:, None] * self.TG[None, :] ** 2
            ) + self.npdt(kms) * Sms
            if not self.folded:
                Sre = Sre + self.npdt(kioni) * tf.cast(self.Sio_re[sl], self.rdt)
            Sim = self.npdt(kioni) * tf.cast(self.Sio_im[sl], self.rdt)
            # the radiative family enters the REAL exponent (so it must be
            # added before the exp) and the phase offset alike; k_rad is not
            # an alpha-dependent quantity, so it belongs in this cache
            if self.has_rad:
                kr = self.npdt(self.krad0 if krad is None else krad)
                Sre = Sre + kr * tf.cast(self.Srad_re[sl], self.rdt)
                Sim = Sim + kr * tf.cast(self.Srad_im[sl], self.rdt)
            e = tf.exp(Sre)
            A.append(e * self.pKre[sl])
            B.append(e * self.pKim[sl])
            S.append(Sim)
            T.append(self.TG[None, :] / sig[:, None])
        if self.win is not None:
            self.log("  (alpha cache: the window normalisation is recomputed "
                     "per alpha from the same A, B, Sim, tgi)")
        self._alpha_cache = (
            tf.concat(A, 0), tf.concat(B, 0), tf.concat(S, 0), tf.concat(T, 0)
        )
        # RE-WRAP: `_chunk_nll_cached` reads `self._alpha_cache` as a python
        # attribute, so the tensors are baked in as captured constants at TRACE
        # time.  Without a fresh tf.function every later set_k_cache() would be
        # silently ignored and every k would return the FIRST k's NLL.
        self._g_nll_cached = self.tf.function(self._chunk_nll_cached)
        self.log(f"  alpha cache built in {time.time()-t0:.1f} s")

    def _chunk_nll_cached(self, i, alpha, fb):
        tf = self.tf
        C = self.chunk
        A, B, S, T = self._alpha_cache
        sl = slice(i * C, (i + 1) * C)
        delta = self.mobs[sl] - self.npdt(MJPSI) * alpha
        psi = S[sl] - T[sl] * delta[:, None]
        integ = A[sl] * tf.cos(psi) - B[sl] * tf.sin(psi)
        Li = tf.reduce_sum(
            self.dTG[None, :] * (integ[:, 1:] + integ[:, :-1]) * self.npdt(0.5), axis=1
        ) / (self.npdt(np.pi) * self.sig[sl])
        if self.win is not None:
            Li = Li / self._winnorm(A[sl], B[sl], S[sl], T[sl],
                                    self.sig[sl], alpha)
        return self._mix(Li, fb, self.w[sl])

    def nll_cached(self, alpha, fb=None):
        tf = self.tf
        a = tf.constant(alpha, self.rdt)
        f = tf.constant(self.fbkg0 if fb is None else fb, self.rdt)
        return float(
            sum(
                self._g_nll_cached(tf.constant(i, tf.int32), a, f).numpy()
                for i in range(self.nchunk)
            )
        )

    # ---- public: accumulate over chunks ----
    def _theta(self, x):
        return self.tf.constant(np.asarray(x, self.npdt))

    def nll(self, x):
        tf = self.tf
        th = self._theta(x)
        self.nfev += 1
        return float(sum(self._g_nll(tf.constant(i, tf.int32), th).numpy()
                         for i in range(self.nchunk)))

    def nll_grad(self, x):
        tf = self.tf
        th = self._theta(x)
        self.ngev += 1
        L = 0.0
        g = np.zeros(self.npar)
        for i in range(self.nchunk):
            l_, g_ = self._g_lg(tf.constant(i, tf.int32), th)
            L += float(l_.numpy())
            g += g_.numpy().astype(np.float64)
        return L, g

    def nll_grad_hess(self, x):
        tf = self.tf
        th = self._theta(x)
        self.nhev += 1
        L = 0.0
        g = np.zeros(self.npar)
        H = np.zeros((self.npar, self.npar))
        for i in range(self.nchunk):
            l_, g_, h_ = self._g_lgh(tf.constant(i, tf.int32), th)
            L += float(l_.numpy())
            g += g_.numpy().astype(np.float64)
            H += h_.numpy().astype(np.float64)
        return L, g, H


# --------------------------------------------------------------------------
# minimisation
# --------------------------------------------------------------------------
def minimize(obj, x0, method="trust-exact", fixed=None, log=print, gtol=1e-5):
    """scipy trust-region minimisation with the exact TF gradient + Hessian.

    ``fixed`` is a dict {index: value} of parameters held constant (used for
    the profile); the reduced problem keeps the exact reduced Hessian.
    """
    from scipy.optimize import minimize as spmin

    fixed = fixed or {}
    free = [i for i in range(obj.npar) if i not in fixed]
    x0 = np.asarray(x0, float).copy()
    cache = {}

    def full(xr):
        x = x0.copy()
        for j, i in enumerate(free):
            x[i] = xr[j]
        for i, v in fixed.items():
            x[i] = v
        return x

    BIG = 1e30

    def lgh(xr):
        key = tuple(xr)
        if key not in cache:
            cache.clear()
            L, g, H = obj.nll_grad_hess(full(xr))
            if not (np.isfinite(L) and np.all(np.isfinite(g))
                    and np.all(np.isfinite(H))):
                # A trial step that pushes f_bkg out of (0, 1) makes the
                # mixture negative and log() NaN.  scipy's trust-region loop
                # compares rho against thresholds, and EVERY comparison with
                # NaN is False, so it neither shrinks the radius nor moves --
                # it just burns maxiter in place (seen on btojpsix +
                # --float-bkg).  Returning a huge FINITE value makes rho very
                # negative, which shrinks the radius as intended.
                L, g, H = BIG, np.zeros(obj.npar), np.eye(obj.npar)
            cache[key] = (L, g, H)
        return cache[key]

    t0 = time.time()
    res = spmin(
        lambda xr: lgh(xr)[0],
        np.array([x0[i] for i in free]),
        jac=lambda xr: lgh(xr)[1][free],
        hess=lambda xr: lgh(xr)[2][np.ix_(free, free)],
        method=method,
        options=dict(gtol=gtol, maxiter=200),
    )
    x = full(res.x)
    L, g, H = obj.nll_grad_hess(x)
    # Newton polish: trust-exact often stops with status 2 ("bad approximation
    # ... to predict improvement") on an essentially exactly quadratic
    # objective while the gradient is already at 1e-5.  A couple of explicit
    # Newton steps drive it to machine level and give the Newton decrement
    # lambda^2/2 = 1/2 g^T H^-1 g as an unambiguous convergence measure.
    npol = 0
    Hf = H[np.ix_(free, free)]
    dec = 0.5 * float(g[free] @ np.linalg.solve(Hf, g[free]))
    while dec > 1e-10 and npol < 40:
        step = np.linalg.solve(Hf, g[free])
        # damped Newton: halve until the step is finite AND decreases the NLL.
        # This is what rescues a start from which the undamped step leaves the
        # physical region (f_bkg < 0).
        t = 1.0
        ok = False
        while t > 1e-6:
            xn = x.copy()
            xn[free] -= t * step
            Ln, gn, Hn = obj.nll_grad_hess(xn)
            if np.isfinite(Ln) and Ln < L - 1e-12:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        x, L, g, H = xn, Ln, gn, Hn
        Hf = H[np.ix_(free, free)]
        dec = 0.5 * float(g[free] @ np.linalg.solve(Hf, g[free]))
        npol += 1
    wall = time.time() - t0
    return dict(x=x, free=free, nll=L, grad=g, hess=H, res=res, wall=wall,
                npolish=npol, decrement=dec)


def cov_from_hess(H, free=None):
    if free is not None:
        H = H[np.ix_(free, free)]
    return np.linalg.inv(H)


def corr_from_cov(C):
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s), s


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def fmt_matrix(M, names, fmt="{:9.4f}"):
    w = max(len(n) for n in names)
    out = [" " * (w + 2) + " ".join(f"{n:>9s}" for n in names)]
    for i, n in enumerate(names):
        out.append(f"{n:>{w}s}  " + " ".join(fmt.format(M[i, j]) for j in range(len(names))))
    return "\n".join(out)


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 8))
    gpus = tf.config.list_physical_devices("GPU")

    today = datetime.date.today().strftime("%y%m%d")
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{today}_masslikfit/"
    )
    os.makedirs(outdir, exist_ok=True)
    try:
        import pubhtml

        pubhtml.ensure_index(outdir)
    except Exception as e:  # pragma: no cover
        print(f"pubhtml.ensure_index skipped: {e}")
    tag = ("_" + args.tag) if args.tag else ""
    # a --replot run must NOT truncate the results table written by the fit
    log = Tee(os.path.join(
        outdir, f"masslikfit{tag}{'_replot' if args.replot else ''}.txt"))
    log(f"# cf_masslik_fit.py  {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"# tf {tf.__version__}  np {np.__version__}  GPUs {gpus}  "
        f"threads intra={args.threads}")
    log(f"# pairs  {args.pairs_cache}")
    log(f"# kernel {args.kernel_cache}")
    log(f"# model={args.model} float_bkg={args.float_bkg} fbkg0={args.fbkg} "
        f"floor={args.floor} precision={args.precision} "
        f"dtype_scan={args.dtype_scan} krad={args.krad} "
        f"float_krad={args.float_krad} window_norm={args.window_norm}"
        + (f" window=[{args.window_lo:.4f}, {args.window_hi:.4f}]"
           if args.window_norm else ""))
    log("")

    if args.replot:
        z = np.load(args.replot, allow_pickle=True)
        res = {k: z[k] for k in z.files if k != "meta"}
        res.update(json.loads(str(z["meta"])))
        for k in ("fit_x", "fit_err", "scan_alpha_hat", "fit_nll"):
            if k in res and np.ndim(res[k]) == 0:
                res[k] = float(res[k]) if k != "fit_x" else res[k]
        make_plots(res, outdir, tag, log)
        log(f"replotted from {args.replot} into {outdir}")
        log.close()
        return

    assert args.pairs_cache and args.kernel_cache, "--pairs-cache/--kernel-cache required"
    inp = load_inputs(args.pairs_cache, args.kernel_cache,
                      dtype_scan=args.dtype_scan, maxn=args.maxn, log=log,
                      subset=args.subset)
    obj = MassNLL(inp, model=args.model, float_bkg=args.float_bkg,
                  floor=args.floor, floor_scale=args.floor_scale,
                  chunk=args.chunk, precision=args.precision, log=log,
                  fbkg0=args.fbkg, krad0=args.krad, float_krad=args.float_krad,
                  window=((args.window_lo, args.window_hi)
                          if args.window_norm else None))

    def x_start(k=1.0, alpha=0.2):
        """Starting vector for the CURRENT parameter layout (one place, so a
        new family cannot be forgotten in one of the four call sites)."""
        x = [alpha, k] if args.model == "r" else [alpha, k, k, k]
        if obj.float_krad:
            x = x + [args.krad]
        if args.float_bkg:
            x = x + [args.fbkg / FSCALE]
        return np.array(x, float)
    result = dict(pairs=args.pairs_cache, kernel=args.kernel_cache,
                  model=args.model, float_bkg=args.float_bkg, floor=args.floor,
                  precision=args.precision, n=obj.n, parnames=obj.parnames,
                  window_norm=bool(args.window_norm), subset=args.subset or "",
                  window=[args.window_lo, args.window_hi],
                  krad=args.krad, float_krad=bool(args.float_krad))

    # ---------------- 1. objective identity vs the scan ----------------
    if args.scan_check:
        sc = np.load(args.scan_check)
        alphas, rs, nllg = sc["alphas"], sc["rs"], sc["nll"]
        irb, iab = np.unravel_index(np.argmin(nllg), nllg.shape)
        log(f"== scan check ==  {args.scan_check}")
        log(f"   scan grid {nllg.shape} r={rs[0]}..{rs[-1]} "
            f"alpha={alphas[0]:.2e}..{alphas[-1]:.2e}; grid min at "
            f"r={rs[irb]:.2f} alpha={alphas[iab]:.3e}")
        # points: 10 alphas along the best-r row + 3 off-row
        ia_list = np.unique(np.clip(
            np.linspace(iab - 12, iab + 12, 10).round().astype(int),
            0, len(alphas) - 1))
        pts = [(irb, int(ia)) for ia in ia_list]
        pts += [(max(0, irb - 4), iab), (min(len(rs) - 1, irb + 4), iab),
                (irb, max(0, iab - 24))]
        pts = list(dict.fromkeys(pts))[: args.scan_check_n]
        # alpha cache per r value
        rows = {}
        for ir, ia in pts:
            rows.setdefault(ir, []).append(ia)
        log(f"{'r':>6s} {'alpha':>12s} {'scan NLL':>18s} {'TF NLL':>18s} "
            f"{'diff':>12s}")
        diffs = []
        t_cached = []
        for ir in sorted(rows):
            # model 'r': the single r multiplies the radiative family too
            obj.set_k_cache(rs[ir], rs[ir], rs[ir],
                            rs[ir] if args.model == "r" else None)
            for ia in rows[ir]:
                t0 = time.time()
                v = obj.nll_cached(float(alphas[ia]))
                t_cached.append(time.time() - t0)
                dd = v - nllg[ir, ia]
                diffs.append(dd)
                log(f"{rs[ir]:6.2f} {alphas[ia]:12.5e} {nllg[ir,ia]:18.6f} "
                    f"{v:18.6f} {dd:12.3e}")
        obj._alpha_cache = None
        log(f"   max |dNLL| = {np.max(np.abs(diffs)):.4e} over {len(diffs)} points "
            f"(NLL ~ {abs(nllg[irb,iab]):.3e});  "
            f"mean cached eval {np.mean(t_cached):.2f} s")
        # uncached timing for reference
        x_ref = x_start(k=rs[irb], alpha=alphas[iab] / ASCALE)
        t0 = time.time(); v_un = obj.nll(x_ref); t_un = time.time() - t0
        log(f"   uncached full eval {t_un:.2f} s, NLL {v_un:.6f} "
            f"(diff to scan {v_un - nllg[irb,iab]:.3e})")
        t0 = time.time(); obj.nll_grad(x_ref); t_g = time.time() - t0
        t0 = time.time(); obj.nll_grad_hess(x_ref); t_h = time.time() - t0
        log(f"   timings: nll {t_un:.2f} s | nll+grad {t_g:.2f} s | "
            f"nll+grad+hess {t_h:.2f} s")
        log("")
        result.update(scan_check_pts=np.array(pts), scan_check_diff=np.array(diffs),
                      t_nll=t_un, t_grad=t_g, t_hess=t_h)
        result.update(scan_alphas=alphas, scan_rs=rs, scan_nll=nllg,
                      scan_alpha_hat=float(sc["alpha_hat"]),
                      scan_err=float(sc["err"]))

    # ---------------- gradient finite-difference check ----------------
    if args.fd_check:
        x0 = x_start()
        L0, g0 = obj.nll_grad(x0)
        log("== gradient finite-difference check ==")
        log(f"{'par':>12s} {'analytic':>16s} {'central FD':>16s} {'rel':>10s}")
        fds = []
        for i, nm in enumerate(obj.parnames):
            h = 1e-4 * max(abs(x0[i]), 1e-2)
            xp, xm = x0.copy(), x0.copy()
            xp[i] += h
            xm[i] -= h
            fd = (obj.nll(xp) - obj.nll(xm)) / (2 * h)
            fds.append(fd)
            rel = (g0[i] - fd) / max(abs(fd), 1e-12)
            log(f"{nm:>12s} {g0[i]:16.6f} {fd:16.6f} {rel:10.2e}")
        result.update(fd_analytic=g0, fd_numeric=np.array(fds), fd_x0=x0)
        log("")

    # ---------------- 2/3. the fit ----------------
    x0 = x_start()
    log(f"== fit ({args.model}"
        + (", k_rad floated" if obj.float_krad
           else (f", k_rad fixed {args.krad:g}" if obj.has_rad
                 else ", no radiative family"))
        + (", f_bkg floated" if args.float_bkg else "") + ") ==")
    obj.nfev = obj.ngev = obj.nhev = 0
    fit = minimize(obj, x0, method=args.method, log=log)
    C = cov_from_hess(fit["hess"])
    R, S = corr_from_cov(C)
    ev = np.linalg.eigvalsh(fit["hess"])
    log(f"   {args.method}: success={fit['res'].success} "
        f"niter={fit['res'].nit} nhev={obj.nhev} status={fit['res'].status} "
        f"({fit['res'].message})")
    log(f"   + {fit['npolish']} Newton polish step(s); "
        f"Newton decrement lambda^2/2 = {fit['decrement']:.3e}"
        + ("" if fit["decrement"] < 1e-6 else "   *** NOT CONVERGED ***"))
    log(f"   wall {fit['wall']:.1f} s (nfev={obj.nfev} ngev={obj.ngev} "
        f"nhev={obj.nhev})   NLL = {fit['nll']:.6f}   "
        f"|grad|inf = {np.max(np.abs(fit['grad'])):.3e}")
    for i, nm in enumerate(obj.parnames):
        u = " e-3" if "1e-3" in nm else ""
        log(f"   {nm:>13s} = {fit['x'][i]:12.6f} +- {S[i]:.6f}{u}")
    log("   correlation matrix:")
    log(fmt_matrix(R, obj.parnames))
    log(f"   Hessian eigenvalues: " + "  ".join(f"{e:.4e}" for e in ev))
    log(f"   positive definite: {bool(np.all(ev > 0))}   "
        f"condition number {ev.max()/ev.min():.3e}")
    log("")
    result.update(fit_npolish=fit["npolish"], fit_decrement=fit["decrement"],
                  fit_nfev=obj.nfev, fit_ngev=obj.ngev,
                  fit_x=fit["x"], fit_nll=fit["nll"], fit_grad=fit["grad"],
                  fit_hess=fit["hess"], fit_cov=C, fit_corr=R, fit_err=S,
                  fit_eig=ev, fit_wall=fit["wall"], fit_nit=fit["res"].nit,
                  fit_nhev=obj.nhev)

    # ---------------- 5. positivity diagnostics ----------------
    import tensorflow as _tf
    _th = obj._theta(fit["x"])
    _neg = 0
    _min = np.inf
    _g_raw = _tf.function(obj._chunk_liraw)
    for _i in range(obj.nchunk):
        _li = _g_raw(_tf.constant(_i, _tf.int32), _th).numpy()
        _w = obj.w[_i * obj.chunk:(_i + 1) * obj.chunk].numpy()
        _li = _li[_w > 0]
        _neg += int((_li < 0).sum())
        if len(_li):
            _min = min(_min, float(_li.min()))
    log("== positivity of the raw L_i at the minimum ==")
    log(f"   L_i < 0 for {_neg}/{obj.n} candidates ({100.*_neg/obj.n:.4f} %), "
        f"min L_i = {_min:.4e};  mixture floor f/0.7 = {FBKG/MWIN:.4e}")
    log("")
    result.update(n_neg_Li=_neg, min_Li=_min)

    # ---------------- 5b. floor variation ----------------
    if args.floor != "clip":
        obj_c = MassNLL(inp, model=args.model, float_bkg=args.float_bkg,
                        floor="clip", chunk=args.chunk, fbkg0=args.fbkg,
                        precision=args.precision, log=lambda *a: None,
                        krad0=args.krad, float_krad=args.float_krad)
        fit_c = minimize(obj_c, x0, method=args.method, log=log)
        log("== floor variation (softplus -> clip) ==")
        for i, nm in enumerate(obj.parnames):
            log(f"   {nm:>12s}: {fit['x'][i]:12.6f} -> {fit_c['x'][i]:12.6f}  "
                f"(d = {fit_c['x'][i]-fit['x'][i]:+.3e}, "
                f"{(fit_c['x'][i]-fit['x'][i])/max(S[i],1e-12):+.3f} sigma)")
        log(f"   dNLL = {fit_c['nll']-fit['nll']:+.4e}")
        log("")
        result.update(fit_x_clip=fit_c["x"], fit_nll_clip=fit_c["nll"])
        del obj_c

    # ---------------- profile ----------------
    if args.profile == "alpha":
        log("== profiled DNLL(alpha) ==")
        sa = S[0]
        agrid = fit["x"][0] + np.linspace(-args.profile_range, args.profile_range,
                                          args.profile_n) * sa
        prof = np.zeros(len(agrid))
        xw = fit["x"].copy()
        t0 = time.time()
        for j, a in enumerate(agrid):
            if obj.npar == 1:
                prof[j] = obj.nll([a])
            else:
                f = minimize(obj, xw, method=args.method, fixed={0: a}, log=log)
                prof[j] = f["nll"]
                xw = f["x"].copy()
            log(f"   alpha = {a:9.5f} e-3   DNLL = {prof[j]-fit['nll']:10.5f}")
        prof -= fit["nll"]
        log(f"   profile in {time.time()-t0:.1f} s")
        log("")
        result.update(prof_alpha=agrid, prof_dnll=prof)

    # ---------------- output ----------------
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, **{k: v for k, v in result.items()
                              if not isinstance(v, (str, bool, list))},
                 meta=json.dumps({k: v for k, v in result.items()
                                  if isinstance(v, (str, bool, list, int))}))
        log(f"wrote {args.out}")

    if not args.no_plots and args.scan_check:
        make_plots(result, outdir, tag, log)
    log(f"tables/figures in {outdir}")
    log.close()


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_plots(res, outdir, tag, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.ROOT)
    try:
        from wums import plot_tools

        def save(fig, nm):
            plot_tools.save_pdf_and_png(outdir, nm, fig)
            plt.close(fig)
    except Exception:
        def save(fig, nm):
            fig.savefig(os.path.join(outdir, nm + ".pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(outdir, nm + ".png"), bbox_inches="tight")
            plt.close(fig)

    alphas, rs, nllg = res["scan_alphas"], res["scan_rs"], res["scan_nll"]
    scan_prof = nllg.min(axis=0) - nllg.min()

    # (a) profiled DNLL vs alpha, fit vs scan
    if "prof_alpha" in res:
        fig, ax = plt.subplots(figsize=(9.5, 6.4), constrained_layout=True)
        ax.plot(alphas * 1e3, scan_prof, "o-", color="0.45", lw=1.6, ms=4,
                label=r"scan, profiled over $r$ (grid $\Delta r=0.05$)")
        ax.plot(res["prof_alpha"], res["prof_dnll"], "-", color="crimson", lw=2.6,
                label="fit, profiled (re-minimised)")
        ax.plot(res["prof_alpha"], res["prof_dnll"], "o", color="crimson", ms=5)
        # parabola from the fit covariance
        sa = res["fit_err"][0]
        xg = np.linspace(res["prof_alpha"][0], res["prof_alpha"][-1], 200)
        ax.plot(xg, 0.5 * ((xg - res["fit_x"][0]) / sa) ** 2, ":", color="navy",
                lw=2.0, label=r"fit parabola ($H^{-1}$)")
        for lv in (0.5, 2.0):
            ax.axhline(lv, color="0.8", lw=0.8, ls="--")
        ax.set_xlim(res["prof_alpha"][0], res["prof_alpha"][-1])
        ax.set_ylim(0, max(3.0, res["prof_dnll"].max() * 1.05))
        ax.set_xlabel(r"momentum scale $\alpha$ [$10^{-3}$]")
        ax.set_ylabel(r"$\Delta$NLL")
        ax.legend(fontsize="small", loc="upper center")
        ax.grid(alpha=0.25)
        save(fig, f"masslikfit_profile_alpha{tag}")
        log(f"wrote {outdir}/masslikfit_profile_alpha{tag}.pdf")

    # (b) alpha-r contours.  The scan grid is Da = 2.5e-5 (~1.5 sigma_alpha)
    # and Dr = 0.05 (~18 sigma_r), so its raw DNLL=1 contour is an artefact of
    # linear interpolation between nodes; the fair comparison is the local
    # 2D parabola through the 3x3 grid block around the grid minimum, which is
    # what the scan's own parabolic error estimate uses in 1D.
    fig, ax = plt.subplots(figsize=(8.6, 6.8), constrained_layout=True)
    A, Rg = np.meshgrid(alphas * 1e3, rs)
    D = nllg - nllg.min()
    ax.contour(A, Rg, D, levels=[1.0, 4.0], colors="0.55",
                    linewidths=[2.0, 1.4], linestyles=["-", "--"])
    ax.plot([], [], color="0.55", lw=2.0,
            label=r"scan grid, $\Delta$NLL = 1, 4 (linear interp.)")

    irb, iab = np.unravel_index(np.argmin(nllg), nllg.shape)
    ell_scan = None
    if 0 < irb < len(rs) - 1 and 0 < iab < len(alphas) - 1:
        # local quadratic through the 3x3 block: NLL ~ c + b.u + 0.5 u^T H u
        ha = (alphas[1] - alphas[0]) / ASCALE
        hr = rs[1] - rs[0]
        blk = D[irb - 1:irb + 2, iab - 1:iab + 2]
        Haa = (blk[1, 2] + blk[1, 0] - 2 * blk[1, 1]) / ha ** 2
        Hrr = (blk[2, 1] + blk[0, 1] - 2 * blk[1, 1]) / hr ** 2
        Har = (blk[2, 2] - blk[2, 0] - blk[0, 2] + blk[0, 0]) / (4 * ha * hr)
        Hs = np.array([[Haa, Har], [Har, Hrr]])
        gs = np.array([(blk[1, 2] - blk[1, 0]) / (2 * ha),
                       (blk[2, 1] - blk[0, 1]) / (2 * hr)])
        if np.all(np.linalg.eigvalsh(Hs) > 0):
            cen = np.array([alphas[iab] / ASCALE, rs[irb]]) - np.linalg.solve(Hs, gs)
            ell_scan = (cen, np.linalg.inv(Hs))

    def ellipse(ax_, cen, C2, lev, color, ls, lw, label=None):
        w, V = np.linalg.eigh(C2)
        th = np.linspace(0, 2 * np.pi, 400)
        k = np.sqrt(2 * lev)
        pts = V @ (k * np.sqrt(w)[:, None] * np.array([np.cos(th), np.sin(th)]))
        ax_.plot(cen[0] + pts[0], cen[1] + pts[1], ls, color=color, lw=lw,
                 label=label)

    if ell_scan is not None:
        ellipse(ax, ell_scan[0], ell_scan[1], 1.0, "black", "-", 1.8,
                r"scan 3$\times$3 parabola, $\Delta$NLL = 1")
        ellipse(ax, ell_scan[0], ell_scan[1], 4.0, "black", "--", 1.2)

    C = res["fit_cov"]
    pn = list(res["parnames"])
    have_fit_ell = C.shape[0] >= 2 and len(pn) >= 2 and pn[1] == "r"
    if have_fit_ell:
        c2 = np.asarray(C)[:2, :2]
        ellipse(ax, res["fit_x"][:2], c2, 1.0, "crimson", "-", 2.6,
                r"fit ($H^{-1}$), $\Delta$NLL = 1")
        ellipse(ax, res["fit_x"][:2], c2, 4.0, "crimson", "--", 1.8,
                r"fit ($H^{-1}$), $\Delta$NLL = 4")
        ax.plot(res["fit_x"][0], res["fit_x"][1], "*", color="crimson", ms=17,
                label="fit minimum", zorder=5)
        sa, sr = res["fit_err"][0], res["fit_err"][1]
        ax.set_xlim(res["fit_x"][0] - 6 * sa, res["fit_x"][0] + 6 * sa)
        ax.set_ylim(res["fit_x"][1] - 6 * sr, res["fit_x"][1] + 6 * sr)
        rho = np.asarray(res["fit_corr"])[0, 1]
        ax.set_title(rf"$\sigma_\alpha = {sa:.4f}\times10^{{-3}}$, "
                     rf"$\sigma_r = {sr:.4f}$, $\rho = {rho:+.3f}$; "
                     rf"grid $\Delta\alpha = {(alphas[1]-alphas[0])*1e3:.3f}"
                     rf"\times10^{{-3}}$, $\Delta r = {rs[1]-rs[0]:.2f}$",
                     fontsize=11)
    ax.plot(res["scan_alpha_hat"] * 1e3, rs[irb], "P", color="0.2", ms=11,
            label="scan grid minimum", zorder=4)
    # the scan nodes themselves, so the sampling density is visible
    ax.plot(A.ravel(), Rg.ravel(), "+", color="0.75", ms=6, mew=0.9, zorder=1,
            label="scan grid nodes")
    ax.set_xlabel(r"momentum scale $\alpha$ [$10^{-3}$]")
    ax.set_ylabel(r"resolution scale $r$")
    ax.legend(fontsize="xx-small", loc="lower left", framealpha=0.95,
              borderpad=0.5, handlelength=1.8, labelspacing=0.35)
    ax.grid(alpha=0.25)
    save(fig, f"masslikfit_contour_alpha_r{tag}")
    log(f"wrote {outdir}/masslikfit_contour_alpha_r{tag}.pdf")


if __name__ == "__main__":
    main()

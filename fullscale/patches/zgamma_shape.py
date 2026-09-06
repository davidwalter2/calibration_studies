#!/usr/bin/env python3
"""Add the smooth multiplicative K(m) to `ZGammaLineshape` (idempotent patch).

WHY. The provider is LO in both the hard matrix element and the parton
luminosity, while the sample is POWHEG MiNNLO. The generated/model ratio runs
1.7 at 55 GeV -> 1.0 at the peak -> 1.2 at 150 GeV, and a 2-parameter fit to
the pre-FSR spectrum comes out **+75.8 MeV on Gamma_Z** (zchannel/README.md,
"Step 3"). Five Legendre terms close it to +1.1 MeV, cost 1.25x/1.21x on the
two POI errors, keep every rho(POI, c_k) below 0.40, and make the fit
independent of the PDF set, the PDF order and mu_F (0.25 / 0.03 MeV across five
luminosity tables). It is therefore MANDATORY, not optional.

It exists in `zchannel/fit_gen.py` (`GenFit(shape=N)`) but not in rabbit, so
the detector-level card could not float it. This moves the identical
construction into the provider, where it applies to the Born spectrum BEFORE
the FSR fold -- the same place `GenFit` applies it -- so `pdf`, the CF and
`MassCFTerm` all inherit it.

usage: python3 zgamma_shape.py <path to rabbit/lineshapes/zgamma.py>
"""
import sys


def patch(path):
    s = open(path).read()
    if "self.shape = int(shape)" in s:
        print(f"{path}: already patched")
        return 0

    # ---- 1. constructor keyword ------------------------------------------
    old = """        terms=DEFAULT_TERMS,
        acceptance=None,
        fsr=None,
        fsr_mmax=None,
        dtype=tf.float64,
    ):"""
    new = """        terms=DEFAULT_TERMS,
        acceptance=None,
        fsr=None,
        fsr_mmax=None,
        shape=0,
        shape_window=None,
        dtype=tf.float64,
    ):"""
    assert old in s, "constructor signature not found"
    s = s.replace(old, new, 1)

    # ---- 2. parameter names ----------------------------------------------
    old = """        self.param_names = tuple(
            p for p in (mz_param, gz_param, sin2_param) if p is not None
        )
"""
    new = """        self.shape = int(shape)
        if self.shape < 0:
            raise ValueError("shape must be >= 0")
        self.shape_params = tuple(f"shape{k}" for k in range(1, self.shape + 1))
        self.param_names = tuple(
            [p for p in (mz_param, gz_param, sin2_param) if p is not None]
            + list(self.shape_params)
        )
"""
    assert old in s, "param_names block not found"
    s = s.replace(old, new, 1)

    # ---- 3. the basis, built once the Born grid exists --------------------
    # `m_born` is set at the end of the grid section; anchor on the acceptance
    # tabulation, which is the last thing built from it.
    anchor = "    def config(self):"
    assert anchor in s
    helper = '''    def _build_shape_basis(self, shape_window):
        """Legendre basis of the smooth K(m), tabulated on the Born grid.

        ``K(m) = exp(sum_{k>=1} c_k P_k(u))``, ``u = 2(m - lo)/(hi - lo) - 1``
        over ``shape_window`` (the FIT window, not the Born support -- that is
        where the coefficients are meant to be orthogonal). ``k`` starts at 1:
        ``P_0`` is a constant and the overall normalisation is already free.
        Identical construction to ``zchannel/fit_gen.py`` ``GenFit``.
        """
        self.shape_window = (
            tuple(self.window)
            if shape_window is None
            else (float(shape_window[0]), float(shape_window[1]))
        )
        if not self.shape:
            self._shape_basis = None
            return
        lo, hi = self.shape_window
        if not lo < hi:
            raise ValueError(f"empty shape window {self.shape_window}")
        from numpy.polynomial import legendre

        uu = 2.0 * (self.m_born - lo) / (hi - lo) - 1.0
        basis = np.stack(
            [legendre.legval(uu, [0] * k + [1]) for k in range(1, self.shape + 1)]
        )
        self._shape_basis = tf.constant(basis, self.dtype)

    def _shape_factor(self, values):
        """``exp(sum_k c_k P_k)`` on the Born grid, or None."""
        if not self.shape or self._shape_basis is None:
            return None
        if not values:
            return None
        c = tf.stack(
            [
                tf.cast(values[p], self.dtype)
                for p in self.shape_params
                if p in values
            ]
        )
        if int(c.shape[0]) != self.shape:
            return None
        arg = tf.tensordot(c, self._shape_basis, axes=1)
        return tf.exp(tf.clip_by_value(arg, self.npdt(-30.0), self.npdt(30.0)))

'''
    s = s.replace(anchor, helper + anchor, 1)

    # call it at the end of __init__: the last statement of the grid section
    # is the acceptance tabulation, `self._acc = ...`
    import re

    m = re.search(r"\n(        self\._acc = [^\n]*\n)", s)
    assert m, "self._acc assignment not found"
    s = s[: m.end(1)] + "        self._build_shape_basis(shape_window)\n" + s[m.end(1):]

    # ---- 4. apply it in born_pdf -----------------------------------------
    old = """        y = self.dsigma_dm(values, in_pb=False, **kw)
        if self._acc is not None:
            y = y * self._acc
        return y"""
    new = """        y = self.dsigma_dm(values, in_pb=False, **kw)
        if self._acc is not None:
            y = y * self._acc
        k = self._shape_factor(values)
        if k is not None:
            y = y * k
        return y"""
    assert old in s, "born_pdf body not found"
    s = s.replace(old, new, 1)

    # ---- 5. declarations --------------------------------------------------
    old = """    def param_declarations(self, mz_prior=None, gz_prior=None, sin2_prior=None):"""
    new = """    def param_declarations(
        self, mz_prior=None, gz_prior=None, sin2_prior=None, shape_prior=None
    ):"""
    assert old in s
    s = s.replace(old, new, 1)
    old = """        if self.sin2_param is not None:
            rows[self.sin2_param] = (self.sin2_ref_row() if False else (0.0, sin2_prior, 0.0, 0))"""
    if old not in s:
        old = """        if self.sin2_param is not None:
            rows[self.sin2_param] = (0.0, sin2_prior, 0.0, 0)"""
    new = """        if self.sin2_param is not None:
            rows[self.sin2_param] = (0.0, sin2_prior, 0.0, 0)
        for p in self.shape_params:
            rows[p] = (0.0, shape_prior, 0.0, 0)"""
    assert old in s, "param_declarations rows not found"
    s = s.replace(old, new, 1)

    # ---- 6. config round trip --------------------------------------------
    old = """            "fsr_mmax": self.fsr_mmax,
            "terms": list(self.terms),"""
    new = """            "fsr_mmax": self.fsr_mmax,
            "shape": self.shape,
            "shape_window": list(self.shape_window),
            "terms": list(self.terms),"""
    assert old in s, "config block not found"
    s = s.replace(old, new, 1)

    open(path, "w").write(s)
    print(f"{path}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(patch(sys.argv[1]))

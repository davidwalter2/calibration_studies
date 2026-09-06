#!/usr/bin/env python3
"""Add the EXACT second-order (Jensen) correction to `MassCFTerm`.

WHY. The CF propagates the fitted blocks' fluctuations LINEARLY, but the mass
is a nonlinear function of the fitted parameters. The missing quadratic term
has a nonzero mean, 1/2 tr(H Sigma)/m = (3A + B)/8 = 1.5 (sigma_m/m)^2 for
m ~ (kappa1 kappa2)^{-1/2} with uncorrelated legs. At Z resolutions that is
15-43 MeV -- see MASSCFTERM_SPEC.md sec. 4b.

The branch `material-resolution` ships the three HOOKS this needs
(`_chunk_mean_shift`, `_chunk_residual`, `_chunk_logjac`) and no
implementation. This is the implementation, transcribed from the validated
offline reference `resolution/oddmoment/masslik_np.py` (`_delta`, `_chunk_L`),
so that the two compute the same function.

TWO FORMS, AND ONLY ONE IS ACCEPTABLE AT THE Z.

`shift`   treat 1.5 s^2 m as a deterministic location shift. This assumes the
          MLE responds to it with weight 1. It does not: the missing term is a
          QUADRATIC form and the measured response is F_J = 0.73 +- 0.14 at
          J/psi resolution and 0.56 +- 0.22 at Z-like sigma_m/m = 1.85 %, so
          the form OVER-corrects by 27 % / ~44 %.
`exact`   invert the second-order map per candidate. With u the linear
          relative fluctuation and s^2 = Var(u),

              m_hat/m - 1 = u + u^2 + s^2/2        (mean 1.5 s^2)
              u_i         = 1/2 (sqrt(1 + 4(r - s^2/2)) - 1),  r = delta/m
              delta_eff   = m u_i ,   L_i -> L_i / (1 + 2 u_i)

          No response factor is assumed and none is needed. This is the
          default here; `shift` exists only to reproduce the comparison.

THE JACOBIAN 1/(1+2u) IS NOT OPTIONAL -- it is the change of measure that
makes the transform a reparameterisation rather than a rescaling.

s^2 IS AN INPUT, not a formula. The card passes `jensen_s2` per candidate.
Fold the angular share in there: the exact map's mean is 1.5 s^2, so a
candidate whose true mean shift is (1.5 - f_ang) s_true^2 gets
`jensen_s2 = s_true^2 (1.5 - f_ang)/1.5`. On the Z the measured `Jpsi_fang`
has median 9e-5, i.e. the two agree to 1e-4 relative; on the J/psi gun
f_ang = 0.086-0.106 and it matters.

The DENOMINATOR is the OBSERVED mass, `mobs + m_ref` -- truth-free, so the
same code runs on data. (The offline reference divides by the gen mass; the
two differ by O(sigma/m) ~ 1 %, which enters the correction at second order.)

usage: python3 unbinned_jensen.py <path to rabbit/unbinned.py>
"""
import re
import sys


def patch(path):
    s = open(path).read()
    if "jensen_mode" in s:
        print(f"{path}: already patched")
        return 0

    # ---- 1. constructor keyword ------------------------------------------
    m = re.search(r"\n(        self_consistent_sigma=True,\n)", s)
    if m is None:
        raise SystemExit("the self_consistent_sigma keyword was not found -- "
                         "this file is not the material-resolution MassCFTerm")
    s = (s[: m.end(1)]
         + "        jensen_s2=None,\n"
           "        jensen_mode=\"exact\",\n"
           "        jensen_scale=1.0,\n"
           "        jensen_disc_floor=0.1,\n"
         + s[m.end(1):])

    # ---- 2. store it, next to a_res --------------------------------------
    # anchor on the END of the `_dyn_sigma` assignment, whose body spans
    # several lines under black
    i = s.find("self._dyn_sigma = (")
    if i < 0:
        raise SystemExit("the _dyn_sigma gate was not found")
    j = s.find("\n        )\n", i)
    if j < 0:
        raise SystemExit("the _dyn_sigma gate is not closed as expected")

    class _M:
        pass
    m = _M()
    m.end = lambda _=None, _j=j + len("\n        )\n"): _j
    block = '''
        # ---- the second-order (Jensen) correction ------------------------
        if jensen_mode not in ("off", "shift", "exact"):
            raise ValueError(
                f"jensen_mode must be 'off', 'shift' or 'exact', "
                f"got '{jensen_mode}'"
            )
        self.jensen_mode = jensen_mode
        self.jensen_scale = float(jensen_scale)
        self.jensen_disc_floor = float(jensen_disc_floor)
        self._jensen_s2_np = None
        if jensen_s2 is not None:
            arr = np.asarray(jensen_s2, dtype=np.float64).ravel()
            if arr.shape != (self.n,):
                raise ValueError(
                    f"jensen_s2 has shape {arr.shape}, expected {(self.n,)}"
                )
            if np.any(arr < 0.0):
                raise ValueError("jensen_s2 must be non-negative (it is a variance)")
            self._jensen_s2_np = arr
            self.jensen_s2 = tf.constant(arr, dtype)
        else:
            self.jensen_s2 = None
        self._jensen = (
            self.jensen_mode != "off"
            and self.jensen_s2 is not None
            and self.jensen_scale != 0.0
            and bool(np.any(self._jensen_s2_np != 0.0))
        )
        # the OBSERVED mass, the denominator of r = delta/m. Truth-free.
        self._jensen_m = tf.constant(mobs + float(m_ref), dtype)
        # `_chunk_residual` computes u and `_chunk_logjac` needs it; both are
        # called once per chunk, residual first, inside one graph.
        self._jensen_u = {}
'''
    s = s[: m.end(1)] + block + s[m.end(1):]

    # ---- 3. the mean-shift hook (form `shift`) ---------------------------
    old = '''        Reserved for a deterministic per-candidate offset of the predicted mass
        that is not a global parameter -- e.g. the second-order Jensen term of
        the mass functional, ``0.5 tr(H Sigma)``.  ``None`` means zero.
        """
        return None'''
    new = '''        The second-order (Jensen) term of the mass functional,
        ``0.5 tr(H Sigma) = 1.5 s^2 m``, in the ``shift`` form: a deterministic
        location offset the MLE is ASSUMED to respond to with weight 1.  It
        does not (the measured response is 0.73 at J/psi resolution and 0.56 at
        Z-like), so this form over-corrects by 27-44 % and exists only for
        comparison.  ``jensen_mode="exact"`` is the default and does the work
        in :meth:`_chunk_residual` instead.  ``None`` means zero.
        """
        if not self._jensen or self.jensen_mode != "shift":
            return None
        lo, hi = self._chunks[ci]
        return (self.npdt(1.5 * self.jensen_scale)
                * self.jensen_s2[lo:hi] * self._jensen_m[lo:hi])'''
    assert old in s, "the _chunk_mean_shift body was not found"
    s = s.replace(old, new, 1)

    # ---- 4. the exact map, at the end of _chunk_residual ------------------
    old = '''        ms = self._chunk_mean_shift(values, ci)
        if ms is not None:
            delta = delta - ms
        return delta'''
    new = '''        ms = self._chunk_mean_shift(values, ci)
        if ms is not None:
            delta = delta - ms
        return self._jensen_exact(delta, ci)

    def _jensen_exact(self, delta, ci):
        """Invert the second-order mass map; identity unless mode is 'exact'.

        ``m_hat/m - 1 = u + u^2 + s^2/2`` (uncorrelated equal legs,
        ``m ~ (k1 k2)^{-1/2}``), so with ``r = delta/m``

            u = 1/2 (sqrt(max(1 + 4(r - s^2/2), floor)) - 1)

        and the density picks up ``du/dr = 1/(1 + 2u)``, which
        :meth:`_chunk_logjac` supplies.  The discriminant floor only bites
        where ``r < -1/4``, i.e. a candidate more than a quarter of its own
        mass below the pole -- the far tail, where the second-order expansion
        has no meaning either way.
        """
        if not self._jensen or self.jensen_mode != "exact":
            self._jensen_u.pop(ci, None)
            return delta
        lo, hi = self._chunks[ci]
        m = self._jensen_m[lo:hi]
        s2 = self.npdt(self.jensen_scale) * self.jensen_s2[lo:hi]
        r = delta / m
        disc = tf.maximum(
            self.npdt(1.0) + self.npdt(4.0) * (r - self.npdt(0.5) * s2),
            self.npdt(self.jensen_disc_floor),
        )
        u = self.npdt(0.5) * (tf.sqrt(disc) - self.npdt(1.0))
        self._jensen_u[ci] = u
        return u * m'''
    assert old in s, "the _chunk_residual tail was not found"
    s = s.replace(old, new, 1)

    # ---- 5. the log-Jacobian ---------------------------------------------
    old = '''        A nonlinear ``_chunk_residual`` changes the measure, and the density
        the likelihood needs is ``p_x(x_i) |dx_i/dm_i|``.  ``None`` means the
        transform is the identity (unit Jacobian), which is the linear default.
        """
        return None'''
    new = '''        A nonlinear ``_chunk_residual`` changes the measure, and the density
        the likelihood needs is ``p_x(x_i) |dx_i/dm_i|``.  ``None`` means the
        transform is the identity (unit Jacobian), which is the linear default.

        For the exact Jensen map that is ``-log(1 + 2u)``.  It must not be
        dropped: without it the transform is a rescaling, not a
        reparameterisation, and the correction is wrong at its own order.
        """
        if not self._jensen or self.jensen_mode != "exact":
            return None
        u = self._jensen_u.get(ci)
        if u is None:
            raise RuntimeError(
                "_chunk_logjac was called before _chunk_residual for chunk "
                f"{ci}; the Jensen Jacobian has nothing to report"
            )
        return -tf.math.log(self.npdt(1.0) + self.npdt(2.0) * u)'''
    assert old in s, "the _chunk_logjac body was not found"
    s = s.replace(old, new, 1)

    # ---- 6. config round trip --------------------------------------------
    old = '''            "self_consistent_sigma": self.self_consistent_sigma,'''
    new = '''            "self_consistent_sigma": self.self_consistent_sigma,
            "jensen_mode": self.jensen_mode,
            "jensen_scale": self.jensen_scale,
            "jensen_disc_floor": self.jensen_disc_floor,'''
    assert old in s, "the config() block was not found"
    s = s.replace(old, new, 1)

    # ---- 7. the reader ----------------------------------------------------
    old = '''            a_res=data.pop("a_res", None),'''
    new = '''            a_res=data.pop("a_res", None),
            jensen_s2=data.pop("jensen_s2", None),'''
    assert old in s, "the read_unbinned_terms_from_h5 constructor call was not found"
    s = s.replace(old, new, 1)

    # ---- 8. the dataset table in the module docstring ---------------------
    s = s.replace(
        "``a_res``",
        "``a_res``",
    )
    open(path, "w").write(s)
    print(f"{path}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(patch(sys.argv[1]))

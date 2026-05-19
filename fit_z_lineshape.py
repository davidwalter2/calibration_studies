"""
Differentiable post-FSR Z lineshape model in JAX — unbinned likelihood fit.

Workflow:
  1. build_pdf_grid(m_range)   — precompute PDF luminosities once (slow, numpy)
  2. postfsr_pdf(event_masses, ...)  — evaluate normalised PDF at arbitrary masses (JAX)
  3. nll(params, event_masses, ...)  — unbinned NLL: -Σ log f(xᵢ; θ), jit/grad-able
  4. fit(event_masses, ...)    — minimise NLL w.r.t. (sin2theta_w, mass_z, width_z)

The model PDF is computed on a dense internal grid and then interpolated to
the event masses, so event_masses can be any array of mass values in [m_min, m_max].
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import constants
import drell_yan_xsec as dy

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1.  Precompute PDF luminosities  (numpy/lhapdf, called once)
# ---------------------------------------------------------------------------

# (lhapdf flavor id, Q_f, I3)
_QUARKS = [
    (1, -1/3, -1/2),   # d
    (2,  2/3,  1/2),   # u
    (3, -1/3, -1/2),   # s
    (4,  2/3,  1/2),   # c
    (5, -1/3, -1/2),   # b
]


def build_pdf_grid(m_min, m_max, n_internal=2000):
    """Precompute PDF luminosities on the internal summation grid.

    Parameters
    ----------
    m_min, m_max  : mass range in GeV (should bracket all event masses)
    n_internal    : number of points in the internal log(m²) grid before
                    the ×10 upsampling done inside the convolution

    Returns
    -------
    m_s   : (2N-1,) numpy array  — mass values on extended log(m²) grid
    lumis : (5, 2N-1) numpy array — one row per quark flavor
    N     : int  (= n_internal * 10)
    ds    : float
    """
    N = n_internal * 10
    s_min = 2 * np.log(m_min)
    s_max = 2 * np.log(m_max)
    ds = (s_max - s_min) / (N - 1)

    s_ext = s_min + np.arange(2 * N - 1) * ds
    m_s = np.exp(s_ext / 2)
    m_s = np.clip(m_s, None, constants.s ** 0.5)
    Q2_s = m_s ** 2

    lumis = np.zeros((len(_QUARKS), 2 * N - 1))
    for i, (flavor, _, _) in enumerate(_QUARKS):
        print(f"  PDF lumi: flavor {flavor} ({2*N-1} points)...")
        lumis[i] = np.array([
            dy.integrate_sigma_hat_prime_sm(constants.cme, flavor, Q2)
            for Q2 in Q2_s
        ])

    return m_s, lumis, N, ds


# ---------------------------------------------------------------------------
# 2.  JAX hard matrix element  (differentiable w.r.t. EW params)
# ---------------------------------------------------------------------------

def _term1(Q2, e_f):
    return e_f ** 2 / (2 * Q2 ** 2)


def _term2(Q2, e_f, g, mass_z, width_z, sin2theta_w):
    prop_re = (1 - mass_z ** 2 / Q2) / ((Q2 - mass_z ** 2) ** 2 + mass_z ** 2 * width_z ** 2)
    ew_factor = (1 - 4 * sin2theta_w) / (4 * sin2theta_w * (1 - sin2theta_w))
    return prop_re * ew_factor * e_f * g


def _term3(Q2, e_f, g, mass_z, width_z, sin2theta_w):
    prop_mod2 = 1 / ((Q2 - mass_z ** 2) ** 2 + mass_z ** 2 * width_z ** 2)
    ew_factor = (1 + (1 - 4 * sin2theta_w) ** 2) / (32 * sin2theta_w ** 2 * (1 - sin2theta_w) ** 2)
    return prop_mod2 * ew_factor * g ** 2


def _hard_me(Q2, lumis, sin2theta_w, mass_z, width_z):
    """dσ/dQ at all grid points. Q2 and lumis[i] have shape (2N-1,)."""
    alpha = constants.alpha_ew
    gev2pb = 4 * alpha ** 2 * jnp.pi / (3 * constants.Nc) * 0.389379e9

    d_sigma = jnp.zeros_like(Q2)
    for i, (_, Q_f, I3) in enumerate(_QUARKS):
        e_f = float(Q_f)
        g_fR = -Q_f * sin2theta_w
        g_fL = I3 - Q_f * sin2theta_w

        me = (_term1(Q2, e_f)
              + _term2(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)
              + _term3(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)
              + _term2(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)
              + _term3(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w))

        d_sigma = d_sigma + me * lumis[i]

    return 2 * jnp.sqrt(Q2) * gev2pb * d_sigma   # dσ/dQ, shape (2N-1,)


# ---------------------------------------------------------------------------
# 3.  JAX FSR convolution  (differentiable w.r.t. F_ext values)
# ---------------------------------------------------------------------------

def _radiator_kernel(z, m):
    L = jnp.log(m ** 2 / constants.mass_muon ** 2)
    beta = (2 * constants.alpha_ew / jnp.pi) * (L - 1)
    return beta * (1 - z) ** (beta - 1) * (1 + z ** 2) / 2


def _convolve_fsr(F_ext, m_s_jax, N, ds):
    """Direct O(N²) summation in log(m²) space.

    F_ext   : (2N-1,) pre-FSR spectrum on extended grid
    m_s_jax : (2N-1,) mass values (constant w.r.t. EW params)
    Returns (N,) post-FSR spectrum on the first N grid points.
    """
    L_j    = jnp.log(m_s_jax[:N] ** 2 / constants.mass_muon ** 2)
    beta_j = (2 * constants.alpha_ew / jnp.pi) * (L_j - 1)
    soft   = (ds ** beta_j) * F_ext[:N]

    tau_k = jnp.arange(1, N) * ds
    z_k   = jnp.exp(-tau_k)
    q_jk  = m_s_jax[:N, None] * jnp.exp(tau_k / 2)
    G_jk  = _radiator_kernel(z_k, q_jk) * z_k * ds

    k_idx   = jnp.arange(1, N)[None, :]
    j_idx   = jnp.arange(N)[:, None]
    F_shift = F_ext[j_idx + k_idx]
    hard    = jnp.sum(F_shift * G_jk, axis=1)

    return soft + hard


# ---------------------------------------------------------------------------
# 4.  Full differentiable forward model: PDF evaluated at event masses
# ---------------------------------------------------------------------------

def postfsr_pdf(event_masses, m_s, lumis, N, ds, sin2theta_w, mass_z, width_z):
    """Evaluate the normalised post-FSR PDF at arbitrary event masses.

    The PDF is computed on the dense internal log(m²) grid and then
    interpolated to the requested evaluation points.

    Parameters
    ----------
    event_masses  : (n_events,) masses at which to evaluate f(m; θ)  [GeV]
    m_s           : (2N-1,) internal grid from build_pdf_grid
    lumis         : (5, 2N-1) precomputed PDF luminosities (fixed)
    N, ds         : from build_pdf_grid
    sin2theta_w, mass_z, width_z : differentiable JAX scalars

    Returns
    -------
    (n_events,) JAX array — PDF values f(xᵢ; θ), normalised so that
                             ∫ f dm ≈ 1 over the grid range
    """
    m_s_jax  = jnp.array(m_s)
    lumi_jax = jnp.array(lumis)
    Q2_s     = m_s_jax ** 2

    # Pre-FSR spectrum on internal grid
    F_ext = _hard_me(Q2_s, lumi_jax, sin2theta_w, mass_z, width_z)

    # Post-FSR via FSR convolution → (N,) on uniform s = log(m²) grid
    result_s = _convolve_fsr(F_ext, m_s_jax, N, ds)

    # Normalise: the grid spacing in m is non-uniform, so divide by
    # the trapezoidal integral ∫ f(m) dm over the internal grid
    m_grid_internal = m_s_jax[:N]
    norm = jnp.trapz(result_s, m_grid_internal)
    pdf_s = result_s / norm   # now ∫ f dm ≈ 1

    # Interpolate onto event_masses (linear in log(m²) space)
    s_internal = 2 * jnp.log(m_grid_internal)
    s_events   = 2 * jnp.log(jnp.array(event_masses, dtype=jnp.float64))
    return jnp.interp(s_events, s_internal, pdf_s)


# ---------------------------------------------------------------------------
# 5.  Unbinned NLL and fitter
# ---------------------------------------------------------------------------

def nll(params, event_masses, m_s, lumis, N, ds):
    """Unbinned negative log-likelihood:  -Σᵢ log f(xᵢ; θ).

    Parameters
    ----------
    params        : (sin2theta_w, mass_z, width_z)
    event_masses  : (n_events,) individual event masses [GeV]
    """
    sin2theta_w, mass_z, width_z = params
    pdf_vals = postfsr_pdf(event_masses, m_s, lumis, N, ds, sin2theta_w, mass_z, width_z)
    return -jnp.sum(jnp.log(jnp.clip(pdf_vals, 1e-300)))


def fit(event_masses, m_s, lumis, N, ds, init_params=None, method="BFGS"):
    """Minimise unbinned NLL w.r.t. (sin2theta_w, mass_z, width_z).

    Parameters
    ----------
    event_masses  : (n_events,) array of event masses [GeV]
    m_s, lumis, N, ds : from build_pdf_grid()
    init_params   : starting values; defaults to CMS MiNNLO constants

    Returns
    -------
    result : jax.scipy.optimize.OptimizeResults  (.x = best-fit params)
    """
    if init_params is None:
        init_params = jnp.array([
            constants.sin2theta_w,
            constants.mass_z,
            constants.width_z,
        ])
    else:
        init_params = jnp.array(init_params, dtype=jnp.float64)

    masses_jax = jnp.array(event_masses, dtype=jnp.float64)
    loss = jit(lambda p: nll(p, masses_jax, m_s, lumis, N, ds))

    return jax.scipy.optimize.minimize(fun=loss, x0=init_params, method=method)


# ---------------------------------------------------------------------------
# Quick test / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Precomputing PDF luminosities...")
    m_s, lumis, N, ds = build_pdf_grid(m_min=60.0, m_max=120.0, n_internal=200)
    print(f"  Grid: N={N}, ds={ds:.5f}, m_s range [{m_s[0]:.2f}, {m_s[-1]:.2f}] GeV")

    params0 = jnp.array([constants.sin2theta_w, constants.mass_z, constants.width_z])

    # Use a dense mass grid as proxy for event masses
    proxy_masses = np.linspace(70.0, 110.0, 500)

    pdf_vals = postfsr_pdf(proxy_masses, m_s, lumis, N, ds, *params0)
    print(f"  PDF integral over proxy grid ≈ {float(jnp.trapz(pdf_vals, proxy_masses)):.4f}")

    # Gradient at nominal params
    g = grad(lambda p: nll(p, proxy_masses, m_s, lumis, N, ds))(params0)
    print(f"  ∂NLL/∂sin2θ = {g[0]:.4e}")
    print(f"  ∂NLL/∂m_Z   = {g[1]:.4e}")
    print(f"  ∂NLL/∂Γ_Z   = {g[2]:.4e}")

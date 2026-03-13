import numpy as np
import constants

def non_relativistic_breit_wigner(m, mass, width):
    return (1 / np.pi) * (width / 2) / ((m - mass)**2 + (width / 2)**2)


def radiator_kernel(z, m, alpha=constants.alpha_ew, mass_radiator=constants.mass_muon):
    """LO Altarelli-Parisi radiator function for mu -> mu + gamma"""
    # L is the 'big log' determining the scale of the radiation
    L = np.log(m**2 / mass_radiator**2)
    beta = (2 * alpha / np.pi) * (L - 1)

    # Using the exponentiated form to handle the z->1 singularity
    # This represents the probability density of the muon retaining fraction z of energy
    return beta * (1 - z)**(beta - 1) * (1 + z**2) / 2


def convolution_fft(f, m_grid, _mass=None, _width=None):
    """BW ⊗ AP kernel via direct summation in s = log(m²) space.

    The substitution s = log(m'²), τ = log(q²/m'²) converts the integral
        dσ/dm' = ∫_{m'}^∞ BW(q) K(m'²/q², q) · 2m'²/q³ dq
    into a sum over shifts:
        result(s_j) = Σ_k F(s_j + τ_k) · G(τ_k, q_jk) · dτ
    with F(s) = BW(e^{s/2}), G(τ, q) = K(e^{-τ}, q) · e^{-τ},
    and the exact scale q_jk = m_s[j] · exp(τ_k / 2).

    The soft singularity G(τ) ~ β·τ^{β-1} at τ→0 is handled analytically:
        ∫₀^{ds} G(τ) dτ ≈ ds^β  (leading-power approximation, error < 0.1%)
    Both soft and hard parts use the exact position-dependent (and for the hard
    part, shift-dependent) scale, which matters when the mass range spans
    decades (e.g. 0-120 GeV).
    """
    N = len(m_grid)*10
    s_min = 2 * np.log(m_grid[0])
    s_max = 2 * np.log(m_grid[-1])
    ds = (s_max - s_min) / (N - 1)

    # Uniform log(m²) grid, extended (N-1) points beyond s_max for boundary treatment.
    # The extension ensures the sum at high-mass bins can access BW values
    # above m_grid[-1] rather than zero-padding.
    s_ext = s_min + np.arange(2 * N - 1) * ds
    m_s = np.exp(s_ext / 2)

    m_s[m_s>constants.s**0.5] = constants.s**0.5

    F_ext = f(m_s)

    # Soft part: τ ∈ [0, ds] — analytical, using F(s+τ) ≈ F(s) and ∫₀^{ds} G dτ ≈ ds^β
    # β is evaluated at each grid point m_s[j] (exact position-dependent scale).
    L_j = np.log(m_s[:N] ** 2 / constants.mass_muon ** 2)
    beta_j = (2 * constants.alpha_ew / np.pi) * (L_j - 1)
    soft = (ds ** beta_j) * F_ext[:N]

    # Hard part: τ ∈ [ds, (N-1)·ds] — direct O(N²) summation.
    # The scale q_jk = m_s[j] · exp(τ_k / 2) depends on both position j and shift k,
    # which breaks shift-invariance and prevents FFT. For N~100-1000 the O(N²)
    # cost is negligible.
    tau_k = np.arange(1, N) * ds                          # (N-1,)
    z_k   = np.exp(-tau_k)                                # (N-1,)

    # Exact scale: q[j, k] = m_s[j] * exp(tau_k / 2)
    q_jk = m_s[:N, np.newaxis] * np.exp(tau_k / 2)       # (N, N-1)
    G_jk = radiator_kernel(z_k, q_jk) * z_k * ds         # (N, N-1)

    # F_ext[j + k] for k=1..N-1, j=0..N-1
    k_idx   = np.arange(1, N)[np.newaxis, :]              # (1, N-1)
    j_idx   = np.arange(N)[:, np.newaxis]                 # (N, 1)
    F_shift = F_ext[j_idx + k_idx]                        # (N, N-1)

    hard = np.sum(F_shift * G_jk, axis=1)                 # (N,)

    # Interpolate from uniform log(m²) grid back to original m_grid
    return np.interp(m_grid, m_s[:N], soft + hard)


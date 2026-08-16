import numpy as np
from scipy.integrate import quad
import constants

def get_quark_couplings(sin2theta_w):
    # Quarks
    quarks = [
        (1, -1/3, 'd', -1/2),
        (2, 2/3, 'u', 1/2),
        (3, -1/3, 's', -1/2),
        (4, 2/3, 'c', 1/2),
        (5, -1/3, 'b', -1/2),
        # (6, 2/3, 't', 1/2),
    ]

    # List of quark properties and couplings
    quark_couplings = []

    for flavor, Q_f, name, I3 in quarks:
        e_f = Q_f  # fractional charge in units of e; constants.e is already in the 4*pi*alpha^2 prefactor
        g_fR = -Q_f * sin2theta_w
        g_fL = I3 - Q_f * sin2theta_w
        
        quark_couplings.append((flavor, e_f, g_fR, g_fL))
    return quark_couplings


def convert_gev2pb(d_sigma):
    # Conversion from GeV^-2 to Pb
    factor = 4 * constants.alpha_ew**2*np.pi / (3 * constants.Nc)
    return factor * 0.389379 * 1e9* d_sigma


def f_s(x, tau, flavor, Q2, pdf):
    tau_x = tau / x

    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_flavor_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_anti_flavor_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2


# def integrate_sigma_hat_prime_sm(s, flavor, Q2):
#     def integrate_one(Q2_i):
#         tau_i = Q2_i / s
#         result, _ = quad(lambda x: f_s(x, tau_i, flavor, Q2_i) * (tau_i / x), tau_i, 1)
#         return result

#     if np.isscalar(Q2):
#         return integrate_one(Q2)

#     return np.array([integrate_one(iQ2) for iQ2 in Q2])

def integrate_sigma_hat_prime_sm(s, flavor, Q2, pdf):
    if not np.isscalar(Q2):
        return np.array([integrate_sigma_hat_prime_sm(s, flavor, iQ2, pdf) for iQ2 in Q2])

    tau = Q2 / s
    def integrand1(x):
        tau_x = tau/x
        return f_s(x, tau, flavor, Q2, pdf) * tau_x

    result1, _ = quad(integrand1, tau, 1, epsrel=1e-5)
    return result1


def integrate_sigma_hat_prime_sm_Ycut(s, flavor, Q2, pdf, Y_cut):
    """Per-flavor PDF luminosity integrated over |Y| < ``Y_cut``.

    Equivalent to ``integrate_sigma_hat_prime_sm`` restricted to
    ``|Y| < min(Y_cut, Y_max(Q^2))`` where ``Y_max = 0.5*log(s/Q^2)``.
    For ``Y_cut`` >= ``Y_max`` it reproduces the inclusive 1D function.

    The integration uses the per-Y density ``sigma_hat_prime_sm_at_Y`` and the
    Y -> -Y symmetry of pp collisions: the [0, Y_hi] integral is doubled.
    """
    if not np.isscalar(Q2):
        return np.array(
            [integrate_sigma_hat_prime_sm_Ycut(s, flavor, iQ2, pdf, Y_cut) for iQ2 in Q2]
        )

    Y_max = 0.5 * np.log(s / Q2)
    Y_hi = min(Y_cut, Y_max)
    if Y_hi <= 0:
        return 0.0
    val, _ = quad(
        lambda y: sigma_hat_prime_sm_at_Y(s, flavor, Q2, y, pdf),
        0.0, Y_hi, epsrel=1e-5,
    )
    return 2.0 * val  # symmetric in Y


def sigma_hat_prime_sm_at_Y(s, flavor, Q2, Y, pdf):
    """Per-flavor parton-flux density at fixed (Q^2, Y).

    Defined so that the 1D PDF luminosity is recovered by integrating in Y:

        integrate_sigma_hat_prime_sm(s, flavor, Q^2, pdf)
            = int_{-Y_max}^{+Y_max} sigma_hat_prime_sm_at_Y(s, flavor, Q^2, Y, pdf) dY

    with ``Y_max = 0.5*log(s/Q^2)``. Derivation: substitute ``x = sqrt(tau)*exp(Y)``
    in the existing integrand ``f_s(x)*(tau/x) dx``; ``dx = x dY``, so the per-Y
    integrand is ``f_s(x1=sqrt(tau)*exp(Y), tau, flavor, Q^2, pdf) * tau``.
    Returns 0 outside the kinematic boundary ``|Y| > Y_max``.

    For pp collisions (``ih1 = ih2 = +1``) f_s is symmetric in (x1, x2), so the
    result is even in Y; ``d sigma / d|Y| = 2 * (this function)`` for ``|Y| > 0``.

    Q2 and Y can be scalar or array-like and will be broadcast.
    """
    if np.isscalar(Q2) and np.isscalar(Y):
        tau = Q2 / s
        x1 = np.sqrt(tau) * np.exp(Y)
        x2 = tau / x1
        if not (0.0 < x1 <= 1.0 and 0.0 < x2 <= 1.0):
            return 0.0
        return tau * f_s(x1, tau, int(flavor), Q2, pdf)

    Q2_arr = np.atleast_1d(np.asarray(Q2, dtype=float))
    Y_arr = np.atleast_1d(np.asarray(Y, dtype=float))
    Q2_b, Y_b = np.broadcast_arrays(Q2_arr, Y_arr)
    out = np.zeros(Q2_b.shape, dtype=float)
    for idx in np.ndindex(*Q2_b.shape):
        Q2_i = float(Q2_b[idx])
        Y_i = float(Y_b[idx])
        tau = Q2_i / s
        x1 = np.sqrt(tau) * np.exp(Y_i)
        x2 = tau / x1
        if 0.0 < x1 <= 1.0 and 0.0 < x2 <= 1.0:
            out[idx] = tau * f_s(x1, tau, int(flavor), Q2_i, pdf)
    return out



def d_sigma_sm(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w, integrals=None):
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:

        termL = summation_terms(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)
        termR = summation_terms(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)
        
        if integrals is None:
            integral = integrate_sigma_hat_prime_sm(s, flavor, Q2)
        else:
            integral = integrals[flavor-1]

        d_sigma +=  (termL+ termR )* integral
    
    d_sigmasm =  convert_gev2pb(d_sigma)    # Conversion from GeV^-2 to Pb
    return d_sigmasm


# hard marrix element
def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g, mass_z, width_z, sin2theta_w):
    return (((1 - (mass_z**2 / Q2)) / ((Q2 - mass_z**2)**2 + mass_z**2 * width_z**2)) *
            (1 - 4 * sin2theta_w) / (4 * sin2theta_w * (1- sin2theta_w ))* e_f * g)
            
def term_3(Q2, e_f, g, mass_z, width_z, sin2theta_w):
    return (1 / ((Q2 - mass_z**2)**2 + mass_z**2 * width_z**2) * 
            (1 + (1 - 4 * sin2theta_w)**2) / (32 * sin2theta_w**2 * (1-sin2theta_w)**2)) * g**2

def summation_terms(Q2, e_f, g, mass_z, width_z, sin2theta_w):
    return  (term_1(Q2, e_f) + term_2(Q2, e_f, g, mass_z, width_z, sin2theta_w) + term_3(Q2, e_f, g, mass_z, width_z, sin2theta_w))


def dsigma_dQ(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w, integrals=None):
    Q= np.sqrt(Q2)
    sm = d_sigma_sm(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w, integrals=integrals)

    return 2*Q*sm


def d_sigma_sm_dY(Q2, Y, quark_couplings, s, mass_z, width_z, sin2theta_w,
                  pdf=None, integrals=None):
    """``d^2 sigma / (dQ^2 dY)`` in pb/GeV^2 at signed boson rapidity ``Y``.

    The hard matrix element terms depend on Q^2 only; all Y dependence is in
    the per-flavor PDF luminosity ``sigma_hat_prime_sm_at_Y``.

    Pass either ``pdf`` (the per-flavor luminosity is evaluated on the fly at
    every (Q^2, Y)) or ``integrals`` (precomputed list of 5 arrays, one per
    quark flavor, each matching the broadcast shape of (Q2, Y)).
    """
    d_sigma = 0.0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        termL = summation_terms(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)
        termR = summation_terms(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)
        if integrals is None:
            if pdf is None:
                raise ValueError("d_sigma_sm_dY: pass either pdf or integrals")
            lum = sigma_hat_prime_sm_at_Y(s, flavor, Q2, Y, pdf)
        else:
            lum = integrals[flavor - 1]
        d_sigma += (termL + termR) * lum
    return convert_gev2pb(d_sigma)


def dsigma_dQ_dY(Q2, Y, quark_couplings, s, mass_z, width_z, sin2theta_w,
                 pdf=None, integrals=None):
    """``d^2 sigma / (dQ dY)`` in pb/GeV at signed (Q, Y).

    Equals ``2 * Q * d_sigma_sm_dY``. Even in Y for pp collisions, so
    ``d^2 sigma / (dQ d|Y|) = 2 * dsigma_dQ_dY`` for ``|Y| > 0``.
    """
    Q = np.sqrt(Q2)
    sm = d_sigma_sm_dY(Q2, Y, quark_couplings, s, mass_z, width_z, sin2theta_w,
                       pdf=pdf, integrals=integrals)
    return 2 * Q * sm


def dsigma_dQ_1(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w):
    Q= np.sqrt(Q2)
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(s, flavor, Q2)

        sum_terms = 2*term_1(Q2, e_f)

        d_sigma +=  sum_terms * integral
    
    d_sigmasm =  convert_gev2pb(d_sigma)
    return 2*Q*d_sigmasm


def dsigma_dQ_2(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w):
    Q= np.sqrt(Q2)
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(s, flavor, Q2)

        sum_terms1 = term_2(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)+ term_2(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)
        d_sigma += sum_terms1 * integral
    
    d_sigmasm =  convert_gev2pb(d_sigma) 
    return 2*Q*d_sigmasm


def dsigma_dQ_3(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w):
    Q= np.sqrt(Q2)
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(s, flavor, Q2)

        sum_terms =  term_3(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)+ term_3(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)

        d_sigma +=  sum_terms * integral
    
    d_sigmasm =  convert_gev2pb(d_sigma) 
    return 2*Q*d_sigmasm

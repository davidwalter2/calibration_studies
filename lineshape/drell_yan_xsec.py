import numpy as np
from scipy.integrate import quad_vec
import constants
import lhapdf
pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)


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
        
        # Rounding to 4 decimal places
        e_f = round(e_f, 10)
        g_fR = round(g_fR, 10)
        g_fL = round(g_fL, 10)
        
        quark_couplings.append((flavor, e_f, g_fR, g_fL))
    return quark_couplings


def convert_gev2pb(d_sigma):
    # Conversion from GeV^-2 to Pb
    factor = 4 * constants.alpha_ew**2*np.pi / (3 * constants.Nc)
    return factor * 0.389379 * 1e9* d_sigma


def f_s(x, tau, flavor, Q2):
    tau_x = tau / x

    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_flavor_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_anti_flavor_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2


def integrate_sigma_hat_prime_sm(s, flavor, Q2):
    scalar = np.isscalar(Q2)
    Q2_arr = np.atleast_1d(np.asarray(Q2, dtype=float))
    tau = Q2_arr / s

    # Change of variables: x = tau_i + t*(1 - tau_i), mapping [tau_i, 1] -> [0, 1]
    # for each component, so all components share a smooth domain with no discontinuities.
    def integrand(t):
        vals = np.zeros(len(Q2_arr))
        for i, (tau_i, Q2_i) in enumerate(zip(tau, Q2_arr)):
            x_i = tau_i + t * (1 - tau_i)
            vals[i] = f_s(x_i, tau_i, flavor, Q2_i) * (tau_i / x_i) * (1 - tau_i)
        return vals

    result, _ = quad_vec(integrand, 0, 1)
    return result[0] if scalar else result


def d_sigma_sm(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w):
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:

        termL = summation_terms(Q2, e_f, g_fL, mass_z, width_z, sin2theta_w)
        termR = summation_terms(Q2, e_f, g_fR, mass_z, width_z, sin2theta_w)
        
        integral = integrate_sigma_hat_prime_sm(s, flavor, Q2)

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


def dsigma_dQ(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w):
    Q= np.sqrt(Q2)
    sm = d_sigma_sm(Q2, quark_couplings, s, mass_z, width_z, sin2theta_w)
    
    return 2*Q*sm


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

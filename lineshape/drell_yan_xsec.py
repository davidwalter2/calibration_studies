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

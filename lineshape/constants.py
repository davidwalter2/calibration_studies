import math

# Constants
s = (13e3)**2 # Center-of-mass energy squared in GeV^2
Nc = 3  # Number of colors in QCD

# Electroweak inputs for the FO Z/gamma* calculation.
#
# Gmu scheme (DYTurbo ewscheme=1 in CT18ZNNLO_1GeV.in): independent inputs
# are G_F, M_W, M_Z; sin^2(theta_W) and alpha(M_Z) are derived.
# Numerical values match the DYTurbo .in file verbatim so the FO calculation
# uses the same EW inputs as the DYTurbo CT18Z reference.
G_F   = 1.1663787e-5            # Fermi constant in GeV^-2
mass_w = 79.906853549493746     # W mass (GeV), constant-width scheme
# mass_z = 91.1876               # PDG value (s-dependent-width input)
mass_z = 91.153509740726733     # constant-width scheme (AN A.1 / DYTurbo .in)

# Derived in Gmu:
#   sin^2(theta_W) = 1 - M_W^2 / M_Z^2          (on-shell)
#   alpha(M_Z)     = sqrt(2) G_F M_W^2 sin^2 / pi
sin2theta_w = 1.0 - mass_w**2 / mass_z**2                                # ~0.23152
alpha_ew    = math.sqrt(2) * G_F * mass_w**2 * sin2theta_w / math.pi     # ~1/128.83
e           = math.sqrt(4 * math.pi * alpha_ew)

# Decay width of the Z boson in GeV
# width_z = 2.4952
# Constant-width-scheme value used by POWHEG/MiNNLO and DYTurbo
# (AN2020_008_v5 App. A.1: SM-predicted 2.4941 GeV adjusted by 1/sqrt(1+gamma_Z^2)).
width_z = 2.4932

# 
mass_muon = 0.1056583755

# parameters from PDG
mass_j = 3.096900
width_j = 0.0000929

mass_u1 = 9.46030
width_u1 = 0.00005402

mass_u2 = 10.02326
width_u2 = 0.00003198

mass_u3 = 10.3552
width_u3 = 0.00002032
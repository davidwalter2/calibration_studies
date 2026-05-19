import constants, drell_yan_xsec as dy
import numpy as np
import h5py
from wums import ioutils
import lhapdf

Q_val = arr = np.linspace(10,120,441)[:-1]+0.125
Q2 = Q_val**2

pdf_sets = {"NNPDF31_nnlo_as_0118": 101}

for pdf_set, members in pdf_sets.items():
    
    results =  {}
    for member in range(members):
        print(f"Now at member {member}")
        pdf = lhapdf.mkPDF(pdf_set, member)

        results[member] = [dy.integrate_sigma_hat_prime_sm(constants.s, f+1, Q2, pdf) for f in range(5)]

    fout = h5py.File(f"data/{pdf_set}.hdf5", "w")
    ioutils.pickle_dump_h5py("results", results, fout)

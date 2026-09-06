#!/usr/bin/env python3
"""Parton-luminosity tables at a rescaled factorisation scale mu_F = k Q.

`rabbit/lineshapes/make_lumi_table.py` always uses mu_F = Q.  The generator-
level closure needs mu_F = Q/2 and 2Q as well, to separate "the LO luminosity
is evaluated at the wrong scale" from "the sample has genuine NNLO corrections
that no luminosity choice can reproduce".  The only change is a thin PDF
wrapper that shifts the scale at which LHAPDF is interrogated; the luminosity
integral itself is the same `drell_yan_xsec.integrate_sigma_hat_prime_sm`.

    ./run_tf_z.sh python3 make_lumi_scale.py --kf 0.5 --tag nnpdf31_nnlo_muf05_13tev
"""
import argparse, datetime, json, os, sys
import numpy as np

REF = "/work/submit/david_w/ZMass/calibration_studies/lineshape"
QUARKS = ((1, -1/3, -1/2), (2, 2/3, 1/2), (3, -1/3, -1/2),
          (4, 2/3, 1/2), (5, -1/3, -1/2))


class ScaledPDF:
    """`pdf` interrogated at ``mu_F^2 = k^2 Q^2`` instead of ``Q^2``."""

    def __init__(self, pdf, k):
        self._pdf = pdf
        self._k2 = float(k) ** 2

    def xfxQ2(self, pid, x, q2):
        return self._pdf.xfxQ2(pid, x, self._k2 * q2)

    def __getattr__(self, name):
        return getattr(self._pdf, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfset", default="NNPDF31_nnlo_as_0118")
    p.add_argument("--pdf-member", type=int, default=0)
    p.add_argument("--kf", type=float, required=True, help="mu_F = kf * Q")
    p.add_argument("--sqrt-s", type=float, default=13000.0)
    p.add_argument("--m-lo", type=float, default=40.0)
    p.add_argument("--m-hi", type=float, default=200.0)
    p.add_argument("--n-anchor", type=int, default=300)
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default="data/lumi")
    a = p.parse_args()

    sys.path.insert(0, REF)
    import drell_yan_xsec as dy
    import lhapdf

    pdf = ScaledPDF(lhapdf.mkPDF(a.pdfset, a.pdf_member), a.kf)
    s = a.sqrt_s ** 2
    m = np.exp(np.linspace(np.log(a.m_lo), np.log(a.m_hi), a.n_anchor))
    log_lumi = np.empty((len(QUARKS), a.n_anchor))
    for i, (fl, _, _) in enumerate(QUARKS):
        v = np.asarray(dy.integrate_sigma_hat_prime_sm(s, fl, m ** 2, pdf), float)
        if not np.all(v > 0):
            raise RuntimeError(f"non-positive luminosity for flavour {fl}")
        log_lumi[i] = np.log(v)
    prov = dict(pdfset=a.pdfset, pdf_member=a.pdf_member,
                pdf_description=f"mu_F = {a.kf} Q",
                sqrt_s_gev=a.sqrt_s,
                factorisation_scale=f"mu_F = {a.kf} Q",
                order="LO parton luminosity", y_cut=None,
                acceptance="none", m_lo=a.m_lo, m_hi=a.m_hi,
                n_anchor=a.n_anchor, source=os.path.join(REF, "drell_yan_xsec.py"),
                source_mtime="", source_function="integrate_sigma_hat_prime_sm",
                generator=os.path.abspath(__file__),
                created=datetime.date.today().isoformat(),
                lhapdf_version=lhapdf.version())
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"zlumi_{a.tag}.npz")
    np.savez(out, log_m=np.log(m), log_lumi=log_lumi,
             flavors=np.array([q[0] for q in QUARKS]),
             provenance=np.array([json.dumps(prov)]))
    print(f"[make_lumi_scale] mu_F = {a.kf} Q -> {out}")


if __name__ == "__main__":
    main()

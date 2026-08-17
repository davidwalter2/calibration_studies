#!/usr/bin/env python3
"""Compare the nuclear-elastic kernel the DRIVER draws against the one the
SIMULATION actually applies.

THE POINT
---------
`cf_nucel_exact` builds its angular kernel from `nucel_g4driver`, which calls
the G4 elastic model's `ApplyYourself()` standalone.  Every check on that so far
has constrained the driver's INPUTS -- cross-section class, model class,
dispatch, energy ranges, XS factors -- and all of them match the physics list.
Yet the antiproton over-corrects ~10x in locx and four hypotheses have died.

Only a direct comparison of OUTPUTS can catch a discrepancy in how
`ApplyYourself` behaves when driven standalone versus inside
`G4HadronicProcess`: target isotope sampling, Fermi motion, final-state
validity checks.  None of those are replicated by the driver and none can be
settled by reading source.

INPUT
-----
`HadElasticKernelWatcher` binaries, 6 x float32 per record:
    [tag, theta_rad, ekin_pre_MeV, dE_MeV, steplen_mm, r_cm]
tag = 1 -> the step ended in hadElastic; tag = 0 -> prescaled control.

THE CONTROL MATTERS.  The recorded angle is pre->post momentum direction over a
step, so it also contains the field bend and the msc kick of that same step.
Neither was disabled -- doing so would change the stepping and hence the physics
the closure is judged on -- so the control distribution IS that contamination,
measured under identical conditions.  Read the signal against it; do not assume
it away.

Usage:
    python nucel_kernel_compare.py --glob '<scratch>/kern/k_*.bin' --nev 160000
"""

import argparse
import glob as globmod
import os

import numpy as np

# the modelled path: 14 legs of ToyLayerMat out to the outermost scoring plane
XG_MODEL = 12.4544        # g/cm^2
RCUT = 107.0              # cm; beyond this is ToyShellOut + the real CMS
LAB = {-211: "pi-", -321: "K-", 2212: "p", -2212: "pbar"}


def load(pattern):
    fs = sorted(globmod.glob(pattern))
    if not fs:
        raise SystemExit(f"no files match {pattern}")
    a = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 6) for f in fs])
    return a, len(fs)


def qtab(x, qs=(0.10, 0.50, 0.90, 0.99, 0.999)):
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [np.nan] * len(qs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", default=os.environ.get("KERN_DIR", ""))
    p.add_argument("--nev", type=int, required=True, help="primaries per species")
    p.add_argument("--driver", default="", help="dir holding cmp_<pdg>.theta.bin")
    p.add_argument("--rcut", type=float, default=RCUT)
    a = p.parse_args()

    print("=" * 100)
    print("SIM vs DRIVER: the nuclear-elastic kernel, measured on both sides")
    print("=" * 100)

    for pdg in (-2212, 2212):
        pat = os.path.join(a.scratch, f"k_{pdg}_s*.bin")
        try:
            rec, nf = load(pat)
        except SystemExit as e:
            print(f"  {LAB[pdg]}: {e}")
            continue
        sig = rec[rec[:, 0] > 0.5]
        ctl = rec[rec[:, 0] < 0.5]
        # restrict to the MODELLED path
        sigr = sig[sig[:, 5] < a.rcut]
        ctlr = ctl[ctl[:, 5] < a.rcut]

        n_sim = len(sigr) / float(a.nev)                 # collisions per track
        print(f"\n### {LAB[pdg]}   ({nf} files, {a.nev} primaries, r < {a.rcut} cm)")
        print(f"  collisions: {len(sigr)}   ->  N_sim = {n_sim:.4f} per track")

        # ---- RATE, against the driver's sigma_el
        if a.driver:
            recf = os.path.join(a.driver, f"cmp_{pdg}.rec")
            if os.path.exists(recf):
                mu = None
                for line in open(recf):
                    if line.startswith("rate"):
                        sigma_per_mm = float(line.split()[1])
                        mu = sigma_per_mm * 10.0 / 0.1073   # cm^2/g
                n_drv = mu * XG_MODEL
                print(f"  N_driver = mu*xg = {mu:.5f} * {XG_MODEL} = {n_drv:.4f}"
                      f"   ->  sim/driver = {n_sim / n_drv:.3f}")

        # ---- SHAPE
        th_s = sigr[:, 1]
        th_c = ctlr[:, 1]
        qs = (0.10, 0.50, 0.90, 0.99, 0.999)
        hdr = "".join(f"{100*q:>9.1f}%" for q in qs)
        print(f"  theta [mrad]        {hdr}      max")
        print(f"    SIM  hadElastic  " + "".join(f"{1e3*v:>10.2f}" for v in qtab(th_s, qs))
              + f"{1e3*th_s.max():>10.2f}" if len(th_s) else "    (none)")
        print(f"    SIM  control     " + "".join(f"{1e3*v:>10.2f}" for v in qtab(th_c, qs))
              + f"{1e3*th_c.max():>10.2f}" if len(th_c) else "    (none)")
        if a.driver:
            tb = os.path.join(a.driver, f"cmp_{pdg}.theta.bin")
            if os.path.exists(tb):
                th_d = np.fromfile(tb)
                print(f"    DRIVER           " + "".join(f"{1e3*v:>10.2f}" for v in qtab(th_d, qs))
                      + f"{1e3*th_d.max():>10.2f}")
                print(f"    ratio sim/drv    "
                      + "".join(f"{qtab(th_s,qs)[i]/qtab(th_d,qs)[i]:>10.3f}" for i in range(len(qs))))

        # ---- the far tail, which is where pbar's driver kernel is anomalous
        for thr in (0.1, 0.2, 0.5, 1.0):
            fs = float((th_s > thr).mean()) if len(th_s) else 0.0
            line = f"    frac(theta > {thr:4.1f} rad)   SIM {fs:.3e}"
            if a.driver and os.path.exists(os.path.join(a.driver, f"cmp_{pdg}.theta.bin")):
                fd = float((th_d > thr).mean())
                line += f"   DRIVER {fd:.3e}"
                if fd > 0:
                    line += f"   sim/drv {fs/fd:6.2f}"
                elif fs > 0:
                    line += "   *** SIM HAS THEM, DRIVER DOES NOT ***"
            print(line)

        # ---- recoil energy loss, the term the channel drops
        dE = sigr[:, 3]
        print(f"    recoil dE [MeV]  <dE> {dE.mean():.3f}  rms {dE.std():.3f}"
              f"  99% {np.quantile(dE,0.99):.3f}  max {dE.max():.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GATE: at k = 0 a `--material` card IS the phase-2 card.

`make_joint_card.py --material` replaces the four ad-hoc resolution knobs by
the parmtype-15 amounts and the 18 hit classes. The replacement is only a
reparameterisation if, at the point where every amount is 1 and every hit
scale is 1, the two terms are the SAME likelihood:

    material:  S_f = sum_g A(0) S_{f,g} = sum_g S_{f,g}     v = vg_other + sum_c v_c
    phase 2:   S_f = k_f S_f^flat  with k_f = 1             v = k_hit vgf

and the two right-hand sides are equal by construction of the cache -- the
per-group and the flat exponents are the same per-step sums associated
differently, and `vg_other` is DEFINED as `vgf - sum_c v_c`. So the comparison
measures the STORAGE, not the model: float32 for the group rows against
float32 for the flat rows, which is a 7.3e-8 relative floor
(`matres/validate_inmaker_groups.py`), and float64-exact for the Gaussian
share.

It is run on two cards built from the SAME caches with the SAME `--*-maxn`
and `--seed`, so the candidates are identical and any difference is the
parameterisation alone.

Three points are compared, per term:

* the DEFAULT point (k = 0 / k_* = 1);
* a step in the parameters that enter both terms IDENTICALLY -- the 50 field
  modes and the Z lineshape POIs, which reach the likelihood only through the
  mean and the kernel -- which exercises the sparse `D`, the self-consistent
  sigma and the Jensen map on top of the resolution;
* the per-candidate densities at the default point, so a difference is
  localised rather than only totalled.

The gradient with respect to a `material_<group>` parameter is NOT expected to
agree: in phase 2 that parameter moves the mean only, in phase 3 it moves the
mean AND the width. That difference is the point of phase 3, and it is
reported rather than compared.

usage:
    python3 gate_material_k0.py --material cards/joint_mat_smoke.hdf5 \\
        --reference cards/joint_p2_smoke.hdf5
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def vg(term):
    @tf.function
    def _vg(xt):
        with tf.GradientTape() as t:
            t.watch(xt)
            v = term.nll(xt)
        return v, t.gradient(v, xt)
    return _vg


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--gtol", type=float, default=1e-5,
                   help="tolerance on the gradient of the shared parameters. "
                        "It is LOOSER than --rtol on purpose: d(NLL)/d(field "
                        "mode) is a near-cancelling sum over candidates, so a "
                        "given relative perturbation of the density shows up "
                        "amplified there, and the amplification is a property "
                        "of the sample, not of the parameterisation.")
    p.add_argument("--rtol", type=float, default=1e-7,
                   help="tolerance on the NLL and on the gradient of the "
                        "shared parameters. The float32 storage floor of the "
                        "group rows is 7.3e-8 RELATIVE ON THE EXPONENT, which "
                        "is not the same thing as a relative tolerance on the "
                        "NLL; the default is loose enough to pass on that and "
                        "tight enough to catch a real model difference.")
    p.add_argument("--step", type=float, default=0.2,
                   help="size of the shared-parameter step, in prior sigmas")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    from rabbit import inputdata

    ref = {t.name: t for t in inputdata.FitInputData(a.reference).unbinned_terms}
    mat = {t.name: t for t in inputdata.FitInputData(a.material).unbinned_terms}
    print(f"[gate] reference {a.reference}\n[gate] material  {a.material}")
    ok = True
    rng = np.random.default_rng(a.seed)

    for name in sorted(set(ref) & set(mat)):
        tr, tm = ref[name], mat[name]
        if tr.n != tm.n:
            print(f"  {name}: {tr.n} vs {tm.n} candidates -- NOT the same "
                  "sample, the gate is meaningless")
            ok = False
            continue
        pr = list(tr.param_names)
        pm = list(tm.param_names)
        gone = [q for q in pr if q not in pm]
        added = [q for q in pm if q not in pr]
        print(f"\n=== {name}: {tr.n} candidates, {len(pr)} -> {len(pm)} "
              f"parameters ===")
        print(f"  dropped {gone}")
        print(f"  added   {len(added)} ({added[:3]} ...)")

        xr = np.array(tr.param_defaults, np.float64)
        xm = np.array(tm.param_defaults, np.float64)
        # the parameters that enter BOTH terms in exactly the same way: not
        # the material amounts (they gain a width dependence) and not the hit
        # classes (they do not exist in the reference)
        shared = [q for q in pr if q in set(pm)
                  and not q.startswith("material_")]
        ir = {q: i for i, q in enumerate(pr)}
        im = {q: i for i, q in enumerate(pm)}

        # ---- the DIRECT statement: the two resolution exponents ---------
        # `_chunk_resolution_parts` is the one method `MaterialCFTerm`
        # overrides, so comparing its output at the default point separates
        # the STORAGE (float32 group rows vs float32 flat rows) from the
        # MODEL. Everything downstream -- the quadrature, the kernel, the
        # corrections, the normalisation -- is the same code on both sides.
        vr0 = tr._values(tf.constant(np.array(tr.param_defaults, np.float64)))
        vm0 = tm._values(tf.constant(np.array(tm.param_defaults, np.float64)))
        sre_r, sim_r, gv_r = tr._chunk_resolution_parts(vr0, 0)
        sre_m, sim_m, gv_m = tm._chunk_resolution_parts(vm0, 0)
        for lab, ar, am in (("Re S", sre_r, sre_m), ("Im S", sim_r, sim_m),
                            ("Gaussian v", gv_r, gv_m)):
            if ar is None and am is None:
                continue
            ar = np.zeros_like(am.numpy()) if ar is None else ar.numpy()
            am = np.zeros_like(ar) if am is None else am.numpy()
            dd = np.abs(am - ar)
            sc = max(np.abs(ar).max(), 1e-300)
            print(f"  chunk 0 {lab:<11s} max |diff| {dd.max():.4e}, "
                  f"rel to max|ref| {dd.max()/sc:.4e}  "
                  f"(float32 floor {np.finfo(np.float32).eps*sc:.2e} x "
                  f"sqrt(rows))")

        fr, fm = vg(tr), vg(tm)
        for label in ("default", f"+{a.step:g} sigma on the shared"):
            if label != "default":
                sig = np.asarray(tr.param_prior_sigmas, np.float64)
                d = rng.normal(size=len(shared))
                for k, q in enumerate(shared):
                    s = sig[ir[q]]
                    s = s if np.isfinite(s) and s > 0 else 1e-3
                    step = a.step * s * d[k]
                    xr[ir[q]] += step
                    xm[im[q]] += step
            vr, gr = fr(tf.constant(xr))
            vm, gm = fm(tf.constant(xm))
            vr, vm = float(vr.numpy()), float(vm.numpy())
            gr, gm = gr.numpy(), gm.numpy()
            dv = abs(vm - vr) / max(abs(vr), 1.0)
            gsh_r = np.array([gr[ir[q]] for q in shared])
            gsh_m = np.array([gm[im[q]] for q in shared])
            dg = (np.max(np.abs(gsh_m - gsh_r))
                  / max(np.max(np.abs(gsh_r)), 1e-300))
            flag = "OK" if (dv < a.rtol and dg < a.gtol) else "MISMATCH"
            ok &= flag == "OK"
            print(f"  [{label}] NLL {vr:.9f} vs {vm:.9f}  rel {dv:.3e}; "
                  f"grad over {len(shared)} shared params rel {dg:.3e}   {flag}")
            worst = int(np.argmax(np.abs(gsh_m - gsh_r)))
            print(f"      worst shared gradient: {shared[worst]} "
                  f"{gsh_r[worst]:.8g} vs {gsh_m[worst]:.8g}")

        # per-candidate density at the default point (undo the step)
        xr0 = np.asarray(tr.param_defaults, np.float64)
        xm0 = np.asarray(tm.param_defaults, np.float64)
        lr = tr.raw_density(tf.constant(xr0)).numpy()
        lm = tm.raw_density(tf.constant(xm0)).numpy()
        rel = np.abs(lm - lr) / np.maximum(np.abs(lr), 1e-300)
        print(f"  per-candidate density: max rel {rel.max():.3e}, "
              f"median {np.median(rel):.3e}, "
              f"q99 {np.quantile(rel, 0.99):.3e}")

        # what is DIFFERENT by design
        mats = [q for q in pr if q.startswith("material_")]
        if mats:
            gr_m = np.array([gr[ir[q]] for q in mats])
            gm_m = np.array([gm[im[q]] for q in mats])
            j = int(np.argmax(np.abs(gm_m - gr_m)))
            print(f"  BY DESIGN different -- d(NLL)/d(material): the mean-only "
                  f"gradient vs mean+width, worst {mats[j]} "
                  f"{gr_m[j]:.6g} -> {gm_m[j]:.6g} "
                  f"(ratio {gm_m[j]/gr_m[j] if gr_m[j] else float('nan'):.4g})")

    print("\n" + ("GATE PASSED" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

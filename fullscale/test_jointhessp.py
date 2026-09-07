"""JointObjective.hessp == JointObjective.hess @ p, external term included."""
import sys
import numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/fullscale")
sys.path.insert(0, "/work/submit/david_w/ZMass/rabbit-material")
import fit_joint as FJ

args = FJ.parse_args(["--card", "/work/submit/david_w/ZMass/calibration_studies/fullscale/cards/joint_smoke.hdf5",
                      "--fix", "k_hit", "k_ms", "k_ioni", "k_rad",
                      "--hess-mode", "pfor", "--no-fit", "--no-sandwich"])
terms, external, aux = FJ.load(args, log=lambda *a, **k: None)
names = list(terms[0].param_names)
fixed = set(args.fix)
free = [i for i, nm in enumerate(names) if nm not in fixed]
obj = FJ.JointObjective(terms, external, free, hess_mode="pfor", chunk=None,
                        log=lambda *a, **k: None)
x0 = np.asarray(obj.x0, float)[free]
H = obj.hess(x0)
rng = np.random.default_rng(11)
worst = 0.0
for k in range(3):
    p = rng.standard_normal(len(obj.free))
    a, b = obj.hessp(x0, p), H @ p
    rel = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)
    worst = max(worst, rel)
    print(f"  tangent {k}: {rel:.3e}")
print(f"  worst {worst:.3e}  ({len(obj.free)} free parameters, external term "
      f"{'ON' if obj.ext is not None else 'OFF'})")
assert worst < 1e-9, worst
print("PASS")

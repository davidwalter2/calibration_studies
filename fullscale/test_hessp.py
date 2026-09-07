"""hessp must equal hess @ p, and trust-krylov must find the same minimum."""
import os, sys, time
import numpy as np, tensorflow as tf
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/fullscale")
sys.path.insert(0, "/work/submit/david_w/ZMass/rabbit-material")
import h5py
from rabbit import unbinned
from chunkfit import ChunkedObjective
from scipy.optimize import minimize

CARD = "/work/submit/david_w/ZMass/calibration_studies/fullscale/cards/z_n300k_fl.hdf5"
with h5py.File(CARD, "r") as f:
    t = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
t.chunk = 32768
names = list(t.param_names)
free = [i for i, n in enumerate(names) if n not in ("k_hit", "k_ms", "k_ioni", "k_rad")]
# keep it cheap: 3 chunks
t._chunks = t._chunks[:3]
t.nchunk = 3
obj = ChunkedObjective([t], free, hess_mode="pfor")
x0 = np.asarray(t.param_defaults, float)[free]
rng = np.random.default_rng(7)

H = obj.hess(x0)
worst = 0.0
for k in range(4):
    p = rng.standard_normal(len(free))
    a = obj.hessp(x0, p)
    b = H @ p
    rel = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)
    worst = max(worst, rel)
    print(f"  tangent {k}: max|hessp - H@p| / max|H@p| = {rel:.3e}")
print(f"  worst {worst:.3e}")
assert worst < 1e-9, worst

for m in ("trust-exact", "trust-krylov"):
    t0 = time.time()
    if m == "trust-exact":
        r = minimize(obj.value_grad, x0, jac=True, hess=obj.hess, method=m,
                     options={"maxiter": 200, "gtol": 1e-6})
    else:
        r = minimize(obj.value_grad, x0, jac=True, hessp=obj.hessp, method=m,
                     options={"maxiter": 200, "gtol": 1e-6})
    print(f"  {m:13s} nit {r.nit:3d}  NLL {r.fun:.8f}  |grad|inf "
          f"{np.max(np.abs(r.jac)):.3g}  {time.time()-t0:.0f} s")
    if m == "trust-exact":
        ref = r
    else:
        d = np.max(np.abs(r.x - ref.x) / np.maximum(np.abs(ref.x), 1e-12))
        print(f"  max relative parameter difference {d:.3e}, dNLL {r.fun-ref.fun:+.3e}")
        assert abs(r.fun - ref.fun) < 1e-6, (r.fun, ref.fun)
print("PASS")

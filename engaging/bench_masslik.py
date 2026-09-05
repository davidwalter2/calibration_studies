#!/usr/bin/env python3
"""Time NLL / NLL+grad / NLL+grad+Hessian of the unbinned mass likelihood.

Site-independent: it imports ``cf_masslik_fit`` and drives the same ``MassNLL``
object the fit uses, so the submit-CPU and Engaging-GPU numbers are of the
identical graph.  It also prints the three values, which is the cheapest
possible cross-site numerical check (they must match to float64 round-off
before any fitted parameter can).

  python bench_masslik.py --pairs-cache runs/cf_masspairs_....npz \
                          --kernel-cache runs/cf_masskernel_....npz \
                          [--model families] [--krad 0] [--repeat 3]
"""
import argparse, json, os, sys, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--pairs-cache", required=True)
p.add_argument("--kernel-cache", required=True)
p.add_argument("--model", default="families", choices=["r", "families"])
p.add_argument("--krad", type=float, default=0.0)
p.add_argument("--chunk", type=int, default=32768)
p.add_argument("--threads", type=int, default=32)
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--json-out", default=None)
a = p.parse_args()

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(a.threads)
tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads // 8))
gpus = tf.config.list_physical_devices("GPU")

import cf_masslik_fit as M

t0 = time.time()
inp = M.load_inputs(a.pairs_cache, a.kernel_cache, log=print)
t_load = time.time() - t0
obj = M.MassNLL(inp, model=a.model, chunk=a.chunk, precision="float64",
                log=print, krad0=a.krad)

x = np.array([0.2, 1.0] if a.model == "r" else [0.2, 1.0, 1.0, 1.0], float)

def bench(fn, n):
    fn(x)                                   # trace + warm up
    ts = []
    for _ in range(n):
        t = time.time(); out = fn(x); ts.append(time.time() - t)
    return min(ts), out

t_nll, v_nll = bench(obj.nll, a.repeat)
t_grad, v_grad = bench(obj.nll_grad, a.repeat)
t_hess, v_hess = bench(obj.nll_grad_hess, a.repeat)

nll = float(np.asarray(v_nll).reshape(()))
grad = np.asarray(v_grad[1], float)
hess = np.asarray(v_hess[2], float)

print(f"\n# bench_masslik  tf {tf.__version__}  np {np.__version__}")
print(f"# GPUs {gpus}  threads {a.threads}  n {obj.n}  model {a.model} krad {a.krad}")
print(f"# load {t_load:.1f} s")
print(f"timings [s, best of {a.repeat}]: nll {t_nll:.3f} | nll+grad {t_grad:.3f} "
      f"| nll+grad+hess {t_hess:.3f}")
print(f"NLL       {nll!r}")
print("grad      " + " ".join(f"{g!r}" for g in grad))
print("hess_diag " + " ".join(f"{h!r}" for h in np.diag(hess)))

if a.json_out:
    json.dump(dict(site=os.environ.get("BENCH_SITE", os.uname().nodename),
                   gpus=[g.name for g in gpus], tf=tf.__version__,
                   n=int(obj.n), model=a.model, krad=a.krad,
                   t_nll=t_nll, t_grad=t_grad, t_hess=t_hess, t_load=t_load,
                   nll=nll, grad=grad.tolist(), hess=hess.tolist()),
              open(a.json_out, "w"), indent=1)
    print(f"wrote {a.json_out}")

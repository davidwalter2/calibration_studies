# `qmsmodel` — why the hit-chi2 and the mass CF term disagreed on `material_tec_services`

Reproduce (all offline; ~25 min on 24 cores):

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
F='/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_jpsigun_ul16_260905d_m0/task_*/globalcor_0.root'

# 1. Q(Rossi) vs Moliere, per material group, from the msmoliv step records
python3 qms_steps.py    --files "$F" --max-files 24 -j 24 -o tmp/gun_qms_24f.npz

# 2. the quadratic term's per-candidate gradient / Hessian diagonal
python3 qms_quadgrad.py --files "$F" -j 6                      -o tmp/gun_quadgrad.npz
python3 qms_quadgrad.py --files "$F" -j 6 --max-chi2-ndof 0 --max-grad 0 --max-hess 0 \
                                                               -o tmp/gun_quadgrad_nocut.npz
python3 qms_quadgrad.py --files "$F" -j 6 --fr-cuts 0.03 0.01  -o tmp/gun_quadgrad_frcut.npz

# 3. the per-candidate step census of one group (36 = tec_services)
python3 qms_census.py   --files "$F" -j 6 --group 36 --refgroup 37 -o tmp/gun_census36.npz

# 4. tables + figures
python3 qms_report.py                 # -> ~/public_html/cvh/<YYMMDD>_qmsmodel/
python3 qms_toy.py -n 20000 --sigh 0.05
```

The result is in `/work/submit/david_w/Documents/Resolution/NOTES.md`, entry
2026-09-06.  In one line: the hit-chi2 term's `k_g` derivative is
**mean-energy-loss only** (`dqopdxi` is the only non-zero row of the `dxi`
column), so it cannot see a wrong `Q`; what it does see is a mean-loss bias
that grows with `dE_ref/p`, and `tec_services` is simply the group with 58 % of
its Fisher information above `dE_ref/p = 0.1`.

# Truth-assisted corrections on realistic-alignment MC and their single-track closure

> Correction sets are numbered v719, v720, ... and registered in `truthassist/CORRECTIONS.md` (+ `corrections_registry.json`). Old labels: corv718 = v722 (realistic) / v725 (ideal), cor6dofT = v724 / v726, in-fit 6dof = v723 (do not use).

Workflow (2026-09-24):

1. `task.sbatch` + `tasks_260924.txt` (slurm array, 63 tasks): `runCvhTruthAssisted.py`
   (CMSSW_15_0_19_patch2, branch nano15) on J/psi (UL16 JPsiToMuMu Pt8toInf TkAlJpsiMuMu)
   and DY (UL16 MiniAODv2) MC. `mode=derive`: gen-matched muons fitted with the five
   reference parameters frozen to gen (fitFromGenParms), packed gradients/Hessian.
   `mode=closure`: free single-track fits with the reference jacobian, on DISJOINT files.
   Module-level model (Bz 6, material 7 per module), pixel classes 16-21, realistic MC
   alignment, the SIM's field (default MC field: 160812 + OAE tracker parametrisation).
   Outputs: /ceph/submit/data/user/d/david_w/ZMass/cvh/truthassist_260924/ (159 GB).
2. `solve_truth.py`: C++ (std::atomic_ref) accumulation into the upper triangle of one dense
   matrix on the free parameters, in-place Cholesky. Bz and material frozen. Variants:
   `--mask v718` (in-plane dofs only), `--prior TYPE=SIGMA`. Local run ~11 min, 65 GB.
3. `plot_closure.py`: (q/p)reco/(q/p)gen vs q*p_gen, with/without corrections applied through
   jacrefv; charge-odd/even split vs p; fitted alignment vs true MC misalignment (runtrees of
   an ideal and a realistic gun run, copied to solve/truth_runtrees/).

Results (4.9M derivation tracks, 1.9M closure tracks): the corrections remove the
charge-odd (alignment-like) response of Z muons, e.g. endcap A(250 GeV) -0.78e-3 -> -0.04e-3;
the charge-even offset (-0.1e-3 plateau for Z, -0.3e-3 for J/psi, rising at low p) is
untouched (field/material frozen). Fitted local-x alignment follows the truth (corr 0.85 BPix,
0.96 TID, 0.8 TEC; TOB 0.17). Pixel class parameters on realistic alignment agree with the
ideal-geometry July values (edge-y-diff BPix +134/+99/+77 um, edge-x ~+10, sizeX1 -3.4/-1.3/-0.5).
Plots: ~/public_html/calibration_studies/260924_truthassist_plot_closure/

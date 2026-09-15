#!/bin/bash
# The per-leg fit benchmark: every row on the same events, same window, same
# five shape terms, so the differences are same-run differences.
# Runs on a node that can mount /ceph (the container's mount hook binds it).
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
NODE=${NODE:-submit50}
WHICH=${1:-physics}

if [ "$WHICH" = "physics" ]; then
ALT=(
 "per-leg, empirical D + h=data/kern_perleg_emp_1gev.npz:data/acc_perleg_emp_1gev.json"
 "per-leg, analytic D (data cfg)=data/kern_perleg_data_1gev.npz:data/acc_perleg_1gev.json"
 "per-leg, analytic D exp1=data/kern_perleg_exp1.npz:data/acc_perleg_exp1.json"
 "per-leg, analytic D no pairs=data/kern_perleg_nopair.npz:data/acc_perleg_nopair.json"
 "per-leg, analytic D eikonal e,mu pairs=data/kern_perleg_paireik.npz:data/acc_perleg_paireik.json"
 "MC-conditional + per-leg A(m)=data/kern_fid_sc3.3e-4.npz:data/acc_perleg_1gev.json"
 "inclusive empirical kernel=data/kern_incl_sc3.3e-4.npz"
 "inclusive analytic (data cfg)=data/kern_cfg_data_vb6e-10.npz"
)
OUT=data/fit_perleg_physics.json
elif [ "$WHICH" = "machinery" ]; then
# The machinery-only benchmark: the same QED as the MC on both sides, so the
# difference is the collinear factorisation and nothing else.  `cond:` rows are
# the MC's OWN conditional kernel read in the model's mass and selection
# variables, which decomposes the residual into its two approximations.
ALT=(
 "per-leg, mc D=data/kern_perleg_mc_1gev.npz:data/acc_perleg_mc_1gev.json"
 "per-leg, analytic D (data cfg)=data/kern_perleg_data_1gev.npz:data/acc_perleg_1gev.json"
 "per-leg, mc D, du 4e-5=data/kern_perleg_mc_du4.npz:data/acc_perleg_mc_du4.json"
 "per-leg, mc D, 2 GeV bands=data/kern_perleg_mc_2gev.npz:data/acc_perleg_mc_2gev.json"
 "MC-conditional, banded=data/kern_fid_band3.3e-4.npz"
 "cond: true mass, true selection=data/kern_cond_true.npz:data/acc_cond_true.json"
 "cond: collinear mass=data/kern_cond_collmass.npz:data/acc_cond_collmass.json"
 "cond: collinear selection=data/kern_cond_collsel.npz:data/acc_cond_collsel.json"
 "cond: collinear mass + selection=data/kern_cond_coll.npz:data/acc_cond_coll.json"
 "inclusive mc standalone=data/kern_cfg_mc_sc3.3e-4.npz"
 "inclusive empirical kernel=data/kern_incl_sc3.3e-4.npz"
)
OUT=data/fit_perleg_machinery.json
elif [ "$WHICH" = "corr" ]; then
# The correlated two-leg density: the exact O(alpha) recoil sharing replaces
# D(x_+) D(x_-) at fixed z, with K(z) untouched.  The `cond:` rows are the same
# references the machinery benchmark uses, so the residual is read off the same
# target; `collinear selection` is the model's own target, because the model
# takes the pT decision on x pT^pre.
ALT=(
 "corr, mc K=data/kern_corr_mc_1gev.npz:data/acc_corr_mc_1gev.json"
 "corr, data K=data/kern_corr_data_1gev.npz:data/acc_corr_data_1gev.json"
 "corr, mc K, matched u_c 0.03=data/kern_corr_mc_uc003.npz:data/acc_corr_mc_uc003.json"
 "corr, mc K, matched u_c 0.01=data/kern_corr_mc_uc001.npz:data/acc_corr_mc_uc001.json"
 "corr, mc K, matched u_c 0.003=data/kern_corr_mc_uc0003.npz:data/acc_corr_mc_uc0003.json"
 "per-leg, mc D=data/kern_perleg_mc_1gev.npz:data/acc_perleg_mc_1gev.json"
 "per-leg, analytic D (data cfg)=data/kern_perleg_data_1gev.npz:data/acc_perleg_1gev.json"
 "cond: true mass, true selection=data/kern_cond_true.npz:data/acc_cond_true.json"
 "cond: collinear selection=data/kern_cond_collsel.npz:data/acc_cond_collsel.json"
 "cond: collinear mass + selection=data/kern_cond_coll.npz:data/acc_cond_coll.json"
 "MC-conditional, banded=data/kern_fid_band3.3e-4.npz"
)
OUT=data/fit_perleg_corr.json
else
ALT=(
 "per-leg, 1 GeV bands=data/kern_perleg_data_1gev.npz:data/acc_perleg_1gev.json"
 "per-leg, 0.5 GeV bands=data/kern_perleg_data_0p5gev.npz:data/acc_perleg_0p5gev.json"
 "per-leg, 2 GeV bands=data/kern_perleg_data_2gev.npz:data/acc_perleg_2gev.json"
 "per-leg, coarse (a+,a-) grid=data/kern_perleg_data_thin4.npz:data/acc_perleg_thin4.json"
 "per-leg, n_leg 1500=data/kern_perleg_data_nleg1500.npz:data/acc_perleg_nleg1500.json"
 "per-leg, n_leg 6000=data/kern_perleg_data_nleg6000.npz:data/acc_perleg_nleg6000.json"
 "per-leg, var_budget 6e-9=data/kern_perleg_data_vb6e-9.npz:data/acc_perleg_vb6e-9.json"
 "per-leg, var_budget 6e-11=data/kern_perleg_data_vb6e-11.npz:data/acc_perleg_vb6e-11.json"
 "per-leg, A(m) Bernstein 8=data/kern_perleg_data_1gev.npz:data/acc_perleg_1gev_b8.json"
 "per-leg, empirical D 1 GeV legs=data/kern_perleg_emp1_1gev.npz:data/acc_perleg_emp1_1gev.json"
 "per-leg, empirical D 20 GeV legs=data/kern_perleg_emp20_1gev.npz:data/acc_perleg_emp20_1gev.json"
 "per-leg, empirical D 2 GeV bands=data/kern_perleg_emp_2gev.npz:data/acc_perleg_emp_2gev.json"
)
OUT=data/fit_perleg_disc.json
fi

ssh "$NODE" "cd $Z && ./run_tf_z.sh python3 -u fit_gen.py fit \
  --gen data/genmerged_full.npz --suite perleg \
  --kernel data/kern_fid_sc3.3e-4.npz --acc data/acc_d8.json --nm 8192 \
  --kernel-alt $(printf "'%s' " "${ALT[@]}") -o $OUT"

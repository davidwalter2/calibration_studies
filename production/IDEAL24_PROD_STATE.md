# Ideal-geometry closure productions on the fixed CVH code (`260924`) — final state

The four closure samples of `IDEAL_PROD_STATE.md` (260917) repeated on
`CMSSW_15_0_19_patch2_dev2 @ 3d4c926ff461` (MS within-step correlation sign,
reference EDM on the free indices, step records under the mass constraint):
same chunks (task NNNN = same events as 260917 and v2), same options, 4 threads,
5000 MB. Numbers, spot checks and caches: `PRODUCTIONS.md` §1 and §9.

| tag | tasks | slurm array | output |
|---|---:|---|---|
| `dymc_8p5M_260924_ideal` | 380 | 6536140 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260924_ideal/` |
| `dymc_8p5M_260924_alignctl` | 40 | 6536101 | `…/dymc_8p5M_260924_alignctl/` |
| `jpsimc_20M_260924_ideal` | 600 | 6536141 (0–499), 6536186 (500–599) | `…/jpsimc_20M_260924_ideal/` |
| `jpsimc_20M_260924_alignctl` | 60 | 6536102 | `…/jpsimc_20M_260924_alignctl/` |

All complete, integrity clean, every array task COMPLETED on its first attempt.

## Payload
One overlay tarball, md5 `fa77325ac1b1b37925ae00b2055f1c68`, in every
`<tag>/payload/`. Built multi-target (`scram b enable-multi-targets`): v3
libraries (md5-identical to the 2026-09-18 build) plus
`lib/el9_amd64_gcc12/scram_x86-64-v2/`. Checked in a `USER_SCRAM_TARGET=x86-64-v2`
smoke that every local library mapped by cmsRun comes from the v2 directory
(the area's `biglib/pluginSimulation.so` has no v2 twin and is not loaded by
these jobs); v2 vs v3 on 40 DY events: 16/16 candidates, identical `niter`,
|Δm/m| ≤ 1.0e-6.

## Batch
The condor clusters (3805576–9) were removed unstarted: the flock collector
`t3serv009:11000` was down. The productions ran on slurm through
`slurm_task_260924.sbatch`, which executes the condor job script of the
`condor_*24/` directory unchanged in a node-local sandbox.

    cd calibration_studies/production
    ./slurm_submit_260924.sh condor_<leg>24 [--time H:MM:SS] [--throttle N]   # submit / resume
    ./integrity_260917.sh <tag> <ntasks>
    ./build_pairs_260924.sh zideal zctl jideal jctl

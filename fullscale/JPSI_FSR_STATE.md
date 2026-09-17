# The two FSR single changes of the phase-2 card — state, jobs and collection

Two independent single changes, one per fit, in a 2x2:

| | Z fold = banded ATOMS (as P2XP) | Z fold = the `mc` kernel TABLE |
|---|---|---|
| J/psi = **delta** at MJPSI | `P2N` (`joint_nok`) — the same-code reference | `P2XT` (`joint_ztab`) |
| J/psi = the **`mc` kernel** | `P2K` (`joint_fsrmc`) | `P2B` (`joint_both`) — built only after both singles certify |

plus the J/psi-only trio `J0`/`JK`/`JD`, in which `m_Z` and `Gamma_Z` are not
in the likelihood at all so a J/psi-side change is attributable to the J/psi
leg alone.

Closes `SUMMARY.md` open item 2's precondition: *"before its first quotable
number, the J/psi term's FSR treatment (a delta at the PDG mass against a
radiating MC) must be quantified"*.

## What changed

| | |
|---|---|
| `zchannel/jpsi_fsr_kernel.py` | NEW. Builds `phi_K(t) = <exp(i t dm)>`, `dm = m' - M`, the kernel CF a **delta** lineshape takes exactly. `mc` = the sample's own, from the production's gen record; `analytic` = exponentiated exact QED (`fsr_analytic.FSRKernel`) |
| `zchannel/fsr_analytic.py` | `coll_log_exact` + `FSRKernel(mass_exact=)`: the muon-mass-exact O(alpha) spectrum. -2.8e-3 on `<u>` at the J/psi, -2.7e-4 at the Upsilon, below 1e-5 at the Z |
| `fullscale/make_joint_card.py` | `--jpsi-fsr <npz>` (default OFF), `--z-pairs` now optional (quadratic + J/psi card), `load_fsr_kernel` + `verify_kernel` |
| `fullscale/gate_jpsi_fsr.py` | NEW. The `phik` path against exact quadrature |
| `fullscale/build_jpsi_fsr.sh` | NEW. Kernels + the three cards |
| `zchannel/plot_jpsi_fsr.py` | NEW. Figures -> `~/public_html/ZMass/cvh/260916_jpsi_fsr/` |
| `fullscale/precond_refit.sbatch` | rows `P2K`, `J0`, `JK` |

## Cards (all built and verified on submit82)

| card | terms | size | kernel |
|---|---|---:|---|
| `cards/joint_ok_full.hdf5` | quad + J/psi + Z | 10.733 GB | delta — the HISTORICAL reference (`P2XP`), built 2026-09-07 |
| `cards/joint_nok.hdf5` | quad + J/psi + Z | 10.733 GB | delta — the SAME-CODE reference (`P2N`) |
| `cards/joint_fsrmc.hdf5` | quad + J/psi + Z | 10.733 GB | `mc` |
| `cards/joint_ztab.hdf5` | quad + J/psi + Z | 10.736 GB | delta J/psi, **Z fold = the `mc` TABLE** (`P2XT`) |
| `cards/jpsi_nok.hdf5` | quad + J/psi | 4.764 GB | delta |
| `cards/jpsi_fsrmc.hdf5` | quad + J/psi | 4.764 GB | `mc` |
| `cards/jpsi_fsrdata.hdf5` | quad + J/psi | 4.764 GB | `data` (exact QED, truncated to the cache's gen window) |

Kernels: `zchannel/data/jpsi_kern_{mc,data,data_trunc}.npz`.

## Jobs (Engaging, submitted 2026-09-16 08:0x)

| job | rows | card(s) | resources |
|---:|---|---|---|
| **22823944** | `J0 JK` | `jpsi_nok`, `jpsi_fsrmc` | `-G h200:1 --time=1-00:00:00`, `mit_preemptable` |
| **22824029** | `P2K` | `joint_fsrmc` | `-G h200:1 --time=1-12:00:00`, `mit_preemptable` |
| ~~22824626~~ -> **22847975** | `P2N` | `joint_nok` | `mit_preemptable`, `-G h200:1 --time=1-00:00:00`, `FRESH=0` from its iteration-15 snapshot (22824626 was preempted at 13:57, one Hessian before its collapse) |
| **22825078** | `JD` | `jpsi_fsrdata` | `-G h200:1 --time=0-06:00:00`, `mit_preemptable` |
| ~~22830065~~ -> **22847976** | `P2XT` | `joint_ztab` | `mit_preemptable`, `FRESH=0` from its 3-Hessian snapshot; **done 02:03, rc=0, 9 h 04** |
| ~~22886298~~ -> **22903561** | `P2B` | `joint_both` | `mit_preemptable`, `-G h200:1 --time=1-12:00:00`, `FRESH=0` from its iteration-13 snapshot (22886298 was preempted at ~12:58 after 13 Hessians and slurm requeued it COLD) -- BOTH single changes |

**`P2N` reproduces `P2XP` digit for digit while it descends**, which settles
the question the row was built to answer. Its EDM sequence
`... 7361.428509941325, 137225.37080812445, 134131.79006991247 ...` is
`P2XP`'s iterations 5-7 to ten digits (`7361.428509941325`,
`137225.37080812445`, `134131.79006991247`), so the positivity-floor
difference between the two cards really is inert on the trajectory -- it
touches 2 of 3 000 000 candidates -- and `P2N` may be read as the same-code
`P2XP`.

**Why `P2N` and not just `P2XP`.** `joint_ok_full` was written on 2026-09-07,
**before** `65319ab` gave every card a real positivity floor, so its J/psi leg
ran at rabbit's `1e-9` default instead of `make_card`'s `1e-7`. That is a
second difference against `joint_fsrmc`, and one that makes the two NLLs
incomparable outright. `joint_nok` is the same card rebuilt with today's code,
so **`P2N` vs `P2K` differ by the kernel and nothing else**; `P2XP` stays in
the table as the historical row. (The J/psi-only pair `J0`/`JK` was built
today on both sides and needs no such companion: the only code difference
between their two builds is the `datasets["phik_*"]` assignment and the
`verify_kernel` call, both of which are unreachable without `--jpsi-fsr`.)

Submitted as

```bash
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-00:00:00 --export=ALL,ROWS="J0 JK" precond_refit.sbatch'
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-12:00:00 --export=ALL,ROWS="P2K" precond_refit.sbatch'
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-12:00:00 --export=ALL,ROWS="P2N" precond_refit.sbatch'
```

## Collect

```bash
eng 'squeue -u david_w -o "%.10i %.9P %.8j %.2t %.10M %R"'
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
mkdir -p $FS/runs/engaging_260916
rsync -a engaging:orcd/pool/zmass/fitresults/native/rabbit_{J0,JK,JD,P2K,P2N,P2XT}.hdf5 \
      $FS/runs/engaging_260916/
rsync -a 'engaging:orcd/pool/zmass/engaging/zprecond_2282*.out' \
      $FS/runs/engaging_260916/
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
# `-o` is an OUTPUT DIRECTORY: one json per result, named after the hdf5
./run_tf_z.sh python3 $FS/native_dump.py \
    "$FS/runs/engaging_260916/rabbit_*.hdf5" -o $FS/runs/engaging_260916

# the comparison table
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
python3 $FS/jpsi_fsr_table.py --ref P2XP \
    P2XP=$FS/runs/engaging_260913/rabbit_P2XP.json \
    P2N=$FS/runs/engaging_260916/rabbit_P2N.json \
    P2K=$FS/runs/engaging_260916/rabbit_P2K.json \
    P2XT=$FS/runs/engaging_260916/rabbit_P2XT.json \
    J0=$FS/runs/engaging_260916/rabbit_J0.json \
    JK=$FS/runs/engaging_260916/rabbit_JK.json \
    JD=$FS/runs/engaging_260916/rabbit_JD.json
```

**Certification**: `EDM < 1e-3` AND a finite positive-definite Hessian (the
fit's own covariance step completed). A row that fails either is reported as
NOT certified, with its log.

The reference row `P2XP` is already collected:
`runs/engaging_260913/rabbit_P2XP.json`, EDM 1.05e-11, NLL 5616702.75815184,
`m_Z` +20.692 +- 2.134 MeV, `Gamma_Z` -4.377 +- 3.787, `bfield_mode0`
+1.40124e-3 +- 2.563e-5.

## Kernel-level numbers, already measured

| | MC (`mc`) | analytic (`data`) |
|---|---:|---:|
| candidates / construction | 7 853 326 | exp2nll + `e`/`mu` pairs, mass-exact |
| `<dm>` in `+-0.35` GeV | **-7.1969 MeV** = **-2.3239e-3** | **-8.0089 MeV** = **-2.5861e-3** |
| difference | | **+0.812 MeV = +2.62e-4** |
| `<u>` inclusive | 2.3887e-3 (truncated at the cache's gen window) | 1.0842e-2 (untruncated) |
| `P(u > 0.0566)` | 0.013496 | 0.038700 (untruncated) / 0.015704 (truncated) |

## ONE change, verified dataset by dataset

Read back with `hdf5plugin` registered and compared array by array:

| pair | result |
|---|---|
| `jpsi_nok` vs `jpsi_fsrmc` | **23/23** numeric datasets bit-identical (`mobs`, `sigma`, `vgf`, `a_res`, `jensen_s2`, `norm_*`, `weights`, `tgrid`, the 218 914 119 sparse-`D` values and their 437 828 238 indices, all five resolution-family blocks and their norm-class versions); the term `config` string identical; the ONLY difference is that B carries `phik_t/phik_re/phik_im` |
| `joint_nok` vs `joint_fsrmc` | **54/54** across BOTH mass terms, both `config` strings identical, and the external `hitchi2` gradient and dense Hessian identical (`param_prior_sigmas` compares unequal only because 54 of its 96 entries are `NaN`) |

## Gates passed

`fullscale/gate_jpsi_fsr.py` (`logs/gate_jpsi_fsr.log`):

* the density IS the convolution: `max |L_K/L_conv - 1|` = 2.39e-7 at the
  card's `dt = 0.02`, 5.20e-8 at 0.01, 1.38e-8 at 0.005 — exactly `dt^2`, i.e.
  the tabulation's linear interpolation and nothing else;
* against the empirical `dm` sample directly: median 1.3e-4, at the
  200 k-draw direct sum's own noise floor;
* `_norm_z` (Gil-Pelaez) against Simpson on the mass grid, kernel ON: worst
  8.4e-5 over eight resolution classes, unchanged between 4001 and 8001 mass
  points (so it is `_norm_z`'s own `norm_tpoints = 8192`, not the quadrature).

`make_joint_card.py --verify` now also checks the kernel ROUND TRIP. It had to:
`phik` is a **dataset**, not part of `config()`, and a term built with a kernel
but written without `phik_t/phik_re/phik_im` comes back with a delta and
nothing says so — the first `jpsi_fsrmc` card was byte-identical to
`jpsi_nok`.

## What the kernel is worth to the likelihood, at theta = 0

200 000 of the card's own candidates, the card's 64 resolution classes and its
truncation normalisation, corrections off so the two differ by the kernel
alone:

| | `-sum log(L/Z)` | per candidate | min `L` |
|---|---:|---:|---:|
| delta at `MJPSI` | -354 553.65 | -1.772768 | 1.53e-4 |
| `mc` kernel | **-363 236.27** | **-1.816181** | 7.58e-4 |

`2 dNLL = 17 365` over 200 000 candidates = **0.0868 per candidate**, ~260 000
over the card's 3 M.

## PRE-REGISTERED prediction, written before the fits landed

From the card's own J/psi Jacobian on the 3 000 000 selected candidates
(`D_card = -dm/dtheta`, whitened; `pscale[bfield_mode0] = 3.1254e-3`):

```
<D_card[:, bfield_mode0]> = -816.52 MeV per card unit   (rms 16.63)
```

With a **delta** the J/psi term has to put `<delta> = <m_obs - M - D theta>` at
zero, and `<m_obs> - M` is `<dm> = -7.1969 MeV`. Along `bfield_mode0` **alone**,
with nothing resisting, that is

```
dtheta_0 = -7.1969e-3 / -0.81652 = +8.81e-3 card units
```

The measured `P2XP` value is `+1.40124e-3 +- 2.563e-5`, i.e. **16 %** of the
free response — the rest is taken by the hit-chi2 curvature and shared over the
other 91 directions (the mass-coherent direction is not `mode0` alone:
`|<D_card>|` over the 92 is 3285 MeV/unit, with `mode38` +1882, `mode40` +1728,
`mode24` +1390, `mode42` +1093 ahead of `mode0`).

**The better projection**, also computed before the fits landed
(`jpsi_Dbar.npz`, `jpsi_fsr_table.py`): the fitted calibration vector's mean
predicted J/psi mass shift is

```
<D_card> . theta  =  -1.5105 MeV  =  -4.877e-4 of the momentum scale   (P2XP)
```

against the `-7.1969 MeV` a **mean-matching** estimator would have to absorb.
A likelihood is not mean-matching against a one-sided tail -- it sits nearer
the mode -- so 21 % of the bound is what this one realises, and `-7.20 MeV`
is an upper bound rather than a prediction (the same distinction
`zchannel/README.md` draws for `<u | in window>`). `-4.88e-4` is the number to
compare against, and it is twice the `m_Z` pull's own `+2.27e-4`.

So, **if the FSR treatment is the explanation of the phase-2 pulls**:

1. `<D_card> . theta` moves toward **zero** when the kernel is switched on, by
   about `+1.5 MeV`, in both the J/psi-only pair and phase 2. `bfield_mode0`
   moves DOWN with it; its J/psi-only difference `JK - J0` should be of order
   `-1.4e-3` and at most `-8.8e-3`.
2. `m_Z` in phase 2 moves DOWN from `+20.69 +- 2.13 MeV` toward zero. In the
   reference the whole pull is `+2.27e-4` of `m_Z`, an order below the
   kernel's own `-2.32e-3`, which is the same ~1/6 response the field mode
   shows.
3. The material pulls, which run to `+2.7 %` at 50 sigma on `bpix L1/L2` and
   `support7`, are NOT predicted to go away: they are 50-sigma effects against
   a truth of zero and a mean mass shift is one direction out of 92.

Anything that contradicts 1 or 2 says the FSR treatment is **not** the
explanation and the pulls have another source.

## What one kernel cannot carry (measured, before the fits)

`MassCFTerm` holds ONE `phi_K` tabulation. The sample's kernel is class
dependent: over the card's 64 norm classes `<dm>` spans **-12.15 to -5.00 MeV**
(rms 1.343). The driver is the generator filter — `PythiaFilter(443, status 2,
MinPt = 8.0)` is on the **gen** J/psi, so a candidate whose *reconstructed* `pT`
is below 8 GeV is one that radiated, and the lowest reco-`pT` octile has
`<dm> = -11.90 MeV` against `-6.5` for the other seven.

What a single kernel gets wrong is not that rms — the inclusive mean is right
by construction — but the correlation between a class's offset and its weight
in the scale estimate:

```
plain mean            <dm> = -7.1969 MeV = -2.3239e-3
1/sigma^2-weighted    <dm> = -7.8012 MeV = -2.5190e-3
residual                   = -0.6043 MeV = -1.95e-4
```

**One kernel therefore removes ~92 % of the FSR effect on the scale and leaves
-1.95e-4 as an upper bound.** The remedy is a per-class kernel, which rabbit
does not have (`phik_grid` is per candidate and is refused together with a
parameter-dependent `sigma`, which this leg has).

## How the phase-2 rows descend, and how long they take

`P2XP` needed **24 Hessians / 8 h 01** on an H200, and its EDM sequence is
`137291, 136935, 135996, 134389, 7361, 137225, 134132, 133710, 132486,
130304, 126006, 117673, 102013, 74415, 30063, 37.0, 27.6, 21.1, 2.81, 0.230,
2.35e-5, 1.05e-11, 2.47e-22, 1.05e-11`. **The first ten iterations are a
shelf**, the descent starts at 11 and the collapse at 15. A flat EDM before
iteration 11 on this card family is therefore expected and is not a stall;
`--stallRelTol 1e-4` and the spectral preconditioner (condition number
5.21e14 -> 11 on every phase-2 row) are what carry it off the shelf.
A single NEGATIVE EDM en route (`P2XP` iteration 5, `P2K` iteration 7) is the
trust-region step meeting a direction of negative curvature; both cards
recover on the next Hessian.

Budget from that: ~21 min per Hessian, ~8 h per phase-2 row.

**Preemption.** `mit_preemptable` is the only partition with an H200 and a
walltime above 6 h (`mit_normal_gpu` has H200s but caps at 6 h, which does not
fit an 8 h row without a handover), so a phase-2 row can be killed mid-flight:
`P2XT` was preempted at 12:19 after 3 Hessians and requeued. The sbatch writes
a snapshot every 0.25 h and `FRESH=0` turns it into
`--externalPostfit <snapshot>`, so the recovery for a repeat is

```bash
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-12:00:00 --export=ALL,FRESH=0,ROWS="P2XT" precond_refit.sbatch'
```

`FRESH` defaults to **1** in this script, so a plain requeue restarts from
scratch.

**`--externalPostfit <snapshot>` CARRIES ON MINIMISING** from the snapshot's
parameter values; it is only with `--noFit` that it runs the postfit alone
(`rabbit/snapshot.py`, `bin/rabbit_fit.py:653`). So `FRESH=0` is a true warm
start and a preempted row loses only the Hessian it was in the middle of. The
snapshot stores PHYSICAL values, not the preconditioner's internal
coordinates, so the resume is exact.

**`mit_normal_gpu` is not the answer**, though it is the only non-preemptable
partition with H200s: its walltime caps at 6 h, three quarters of a phase-2
row, and it was **442 jobs deep** -- both rows sat there 50 minutes without
starting. `mit_preemptable` with `FRESH=0` is strictly better once snapshots
exist: it starts in minutes and a preemption now costs **one Hessian**, ~21
min, rather than the run.

The standing recovery for any preempted phase-2 row is therefore

```bash
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-12:00:00 --export=ALL,FRESH=0,ROWS="<TAG>" precond_refit.sbatch'
```

## Results as they land

| row | card | kernel | EDM | wall | `bfield_mode0` [1e-3] | `<D_card>.theta` [MeV] | `m_Z` [MeV] |
|---|---|---|---:|---:|---:|---:|---:|
| `P2XP` | `joint_ok_full` | delta | 1.05e-11 | 8 h 01 | +1.40124 +- 0.02563 | -1.5105 | +20.692 +- 2.134 |
| `J0` | `jpsi_nok` | delta | 1.10e-12 | **43 min** | +1.91544 +- 0.02562 | -1.9049 | — (no Z term) |
| `JK` | `jpsi_fsrmc` | `mc` | 2.06e-11 | **39 min** | -0.75376 +- 0.02633 | **+0.4537** | — (no Z term) |
| `JD` | `jpsi_fsrdata` | `data` | 7.23e-13 | **69 min** | -0.86590 +- 0.02639 | +0.5578 | — (no Z term) |
| `P2N` | `joint_nok` | delta | 4.29e-11 | 5 h 50 (+ 5 h 03 before its preemption) | +1.40124 +- 0.02563 | -1.5105 | +20.692 +- 2.134 |
| `P2K` | `joint_fsrmc` | `mc` | 8.49e-13 | **8 h 34** | -1.31131 +- 0.02556 | +0.8804 | **-45.594 +- 2.132** |

`J0`: converged at EDM 6.5e-23 in the minimiser and 1.10e-12 as reported,
condition number 4.31e7 -> 1 under the preconditioner, `rc=0`, postfit
covariance computed. Its material pulls are the reference's to two digits
(`bpix_active_L2 +2.66 %` at 324 sigma, `bpix_support7 +2.66 %`,
`tec_structure +0.32 %`), as pre-registered.

### THE ATTRIBUTABLE NUMBER: `JK - J0`

The two cards are bit-identical apart from the kernel CF, both certified, so
the difference is the FSR treatment and nothing else.

| | `J0` (delta) | `JK` (`mc` kernel) | difference |
|---|---:|---:|---:|
| `bfield_mode0` [1e-3] | +1.91544 +- 0.02562 | **-0.75376 +- 0.02633** | **-2.66920** (104 sigma of its own error) |
| `<D_card>.theta` [MeV] | -1.9049 | **+0.4537** | **+2.3586** |
| the same, relative | -6.151e-4 | +1.465e-4 | **+7.616e-4** |
| NLL | -5 460 317.0356 | -5 574 341.0470 | `2 dNLL = 228 048` for the kernel |
| largest material pulls | `bpix_active_L2 +2.66 %` (324 sigma), `bpix_support7 +2.66 %`, `bpix_active_L1 +2.10 %` | +2.66 %, +2.64 %, +2.07 % | **unchanged** |

**A delta at the PDG mass costs the extracted momentum scale
`7.62e-4`** on this MC. Both pre-registered predictions hold: the
mean predicted mass shift moves toward zero (prediction 1, `+2.36 MeV`
against the `+1.5 MeV` estimated from the reference's realised fraction and
inside the `+1.4` to `+8.8e-3` band quoted for `bfield_mode0`), and the
material pulls do NOT go away (prediction 3) -- they are a 50-sigma feature of
this MC that a mean mass shift cannot reach.

### The KERNEL MODEL dependence: `JD - JK`

`JD` is the same card with the exponentiated exact-QED kernel (truncated to
the same gen window), so `JD - JK` is the model dependence of the kernel
itself, measured on the extracted scale:

| | `JK` (`mc`) | `JD` (`data`, exact QED) | difference |
|---|---:|---:|---:|
| the kernel's own `<dm>` [MeV] | -7.1969 | -8.0089 | -0.8120 |
| `bfield_mode0` [1e-3] | -0.75376 +- 0.02633 | -0.86590 +- 0.02639 | -0.11214 (4.3 sigma) |
| `<D_card>.theta` [MeV] | +0.4537 | +0.5578 | +0.1041 |
| the same, relative | +1.465e-4 | +1.801e-4 | **+3.36e-5** |
| NLL | -5 574 341.0470 | -5 574 635.2937 | 588 for `data` |

**The kernel-model systematic on the scale is `3.4e-5`, not the `2.6e-4` the
two kernels' means differ by**: the likelihood realises 12.8 % of a
mean-matching response against a one-sided tail, the same fraction the
`JK - J0` step shows, so a kernel-mean difference is suppressed by ~8 in the
answer.

**The NLL is NOT the arbiter of which kernel is right here.** `data` is
preferred by 588 units with no extra parameter, and that cannot mean exact QED
describes this MC's radiation better than the MC's own measured radiation.
What it means is that with ONE kernel the model is not exact for either: the
sample's kernel is class dependent (rms 1.343 MeV over the 64 norm classes,
above), and a kernel with a heavier hard tail partly absorbs that. The
quantity that IS interpretable is the scale, and it moves by 3.4e-5.


## The Z fold representation (`P2XT`)

`make_joint_card.py --fsr` already accepted either kernel representation --
`ZGammaLineshape` dispatches on the npz keys -- so the change is the file:

| | `P2N` / `P2K` / `P2XP` | `P2XT` |
|---|---|---|
| file | `kern_loose_band3.3e-4.npz` | `ktab_mc_dm10.npz` |
| form | banded **atoms** `(r, w, m_lo, m_hi)`, 14 315 atoms in 11 bands, `sigma_cap = 3.3e-4` | cell-integrated **table** `(m_nodes, u_edges, K, u_mean, p0)`, 151 x 1268 |
| source | the DY production's OWN gen record, 13 032 784 events | the `mc` configuration: standalone Photos++ 3.61 reproducing the sample at unlimited statistics (`data/photos/gen_mcMix.npz`) |
| nodes / bands | 11 bands, 70-130 GeV, outermost opened to `(0, inf)` | 1 GeV nodes, **50-200 GeV**, i.e. the whole Born grid |
| `p0` | — (the atom at `r = 1`) | a genuine delta, 0.366-0.445 across the nodes |

**So this row changes two things at once and the report says so**: the
representation (atoms -> cells) and the statistical precision of the source
(13 M events -> unlimited). The pure-representation part is the atom-minus-table
bias already measured at gen level, `+0.66 / +0.89 MeV` on `m_Z` / `Gamma_Z`
for the inclusive `mc` kernel with these very discretisations.

**The Born support is unchanged.** `m_hi_born = min(window_hi / r_min, cap)`
with `cap` the luminosity table's upper edge: the atoms give `130/0.004607`
and the table `130/exp(-7)`, both far above the cap, so both saturate it and
both build the SAME extended Born grid -- measured, `m_hi_born = 200.0 GeV`,
`n_born = 15360`, `nm = 8192`, `dm = 9.767 MeV` for each. The table's nodes
cover 50-200 GeV exactly, so nothing is continued as a constant.

**`--fsr-inline` (new, default ON) is REQUIRED for a card that travels.**
`ZGammaLineshape._table_config` writes the table **by reference** -- just the
npz path -- whenever that file is on disk, and on Engaging
`/work/submit/david_w/...` does not resolve, so the card could not be read
back. `make_card.py --fsr-inline` passes the arrays instead of the path, which
puts the table in the card as zlib'd base64: **3.756 MB** of config JSON for
151 x 1268, and the round trip is exact -- `m_nodes`, `u_edges` and `u_mean`
bit-identical, `K` and `p0` to **4.4e-16** relative, which is the provider's
own row renormalisation and nothing else. Atom kernels were always inline and
are unaffected.


## THE Z FOLD (`P2XT`) -- certified, and it is SMALL

`rc=0`, EDM **4.283e-16**, 27 Hessians, 9 h 04 with one warm restart.
Against `P2N`, whose only difference is the Z term's kernel file:

| | `P2N` (banded atoms) | `P2XT` (the `mc` TABLE) | difference |
|---|---:|---:|---:|
| `m_Z` [MeV] | +20.692 +- 2.134 | +20.937 +- 2.135 | **+0.245** |
| `Gamma_Z` [MeV] | -4.377 +- 3.787 | -4.264 +- 3.799 | **+0.112** |
| `bfield_mode0` [1e-3] | +1.40124 +- 0.02563 | +1.40087 +- 0.02563 | -0.00037 |
| `<D_card>.theta` [MeV] | -1.5105 | -1.5101 | +0.0004 |
| NLL | 5 616 702.7582 | 5 616 698.7003 | -4.058 |

**+0.245 MeV on `m_Z`** -- a ninth of the statistical error and a third of the
`+0.66 MeV` gen-level atom-minus-table bias of the inclusive `mc` kernel. The
J/psi leg does not notice it at all (`bfield_mode0` moves by 0.0004 of its own
error, `<D_card>.theta` by 0.4 keV), which is what a change confined to the Z
term's lineshape must do.

Two things go in the caveat rather than the number. The row changes the
representation AND the source (the DY production's own 13.03 M-event gen
record -> standalone Photos at unlimited statistics), so `+0.245 MeV` is their
sum, not the representation alone; and the gen-level comparison that gave
`+0.66` was atoms and table of the SAME standalone run, which this is not.
What the row does establish is that **the fold representation is not where the
`-20` MeV Z-side offset of `P2K` lives**.

## `P2N` == `P2XP`, exactly

`P2N` -- the same card rebuilt with today's code, warm-started once after a
preemption -- certifies at EDM **4.294e-11**, `rc=0`, and returns

```
m_Z  +20.692 +- 2.134 MeV      Gamma_Z  -4.377 +- 3.787 MeV
NLL  5 616 702.758152          bfield_mode0  +1.40124 +- 0.02563 e-3
<D_card> . theta  -1.5105 MeV
```

which is `P2XP`'s answer **to every printed digit** (`P2XP - P2N` is
`-0.000 MeV` on `m_Z`, `-0.000` on `Gamma_Z`, `+0.00000e-3` on
`bfield_mode0`, and the NLLs agree to 1e-6). The positivity-floor difference
between `joint_ok_full` and `joint_nok` is therefore not merely inert on the
trajectory but on the answer, and **`P2K - P2N` = `P2K - P2XP` = -66.286 MeV
with the kernel as the only difference**.

## PHASE 2 WITH THE J/psi KERNEL (`P2K`) -- certified

`rc=0`, EDM **8.49e-13**, postfit covariance computed, 24 Hessians in
**8 h 34** on a preemptable H200.

| | `P2XP` (delta) | `P2K` (`mc` kernel) | difference |
|---|---:|---:|---:|
| `m_Z` [MeV] | +20.692 +- 2.134 | **-45.594 +- 2.132** | **-66.286** |
| `Gamma_Z` [MeV] | -4.377 +- 3.787 | -6.053 +- 3.776 | -1.676 |
| `bfield_mode0` [1e-3] | +1.40124 +- 0.02563 | -1.31131 +- 0.02556 | -2.71255 |
| `<D_card>.theta` [MeV] | -1.5105 | +0.8804 | +2.3909 = **+7.72e-4** |
| NLL | 5 616 702.7582 | 5 499 700.3370 | (not comparable, see below) |
| material pulls | `bpix_active_L2 +2.65 %` (324 sigma) | +2.65 % (324 sigma) | unchanged |

**The transfer is 1:1 and that is the check.** The J/psi leg's scale moves by
`+7.72e-4` and `m_Z` moves by `-66.286 / 91 188 = -7.27e-4`: a multiplicative
momentum-scale change has to appear on `m_Z` with the opposite sign and the
same magnitude, and it does, to 6 %. The J/psi-only pair gives the same step
independently (`+7.616e-4`), on a card with no `m_Z` in it at all.

**So a delta at the PDG mass was costing `m_Z` +66.3 MeV, and the `+20.69` was
not the FSR -- it was the SUM of an FSR-induced `+66.3` and a residual
`-45.6`.** Decomposing with the measured J/psi scale, `-(scale) x m_Z` predicts
`+44.5` for `P2XP` and `-25.9` for `P2K` against the fitted `+20.69` and
`-45.59`: a common Z-side offset of **-23.8 / -19.7 MeV** that does not move
when the J/psi leg changes, which is where `SUMMARY.md` open item 1 (the `K(m)`
truncation) lives.

The two NLLs are not comparable: the J/psi term's model changed, so its
normalisation did.

## The residual scale after the kernel, and what it is not

`JK` leaves `<D_card>.theta = +0.4537 MeV = +1.465e-4` where truth is 0, and
`bfield_mode0` overshoots to `-0.754e-3`. Two cheap, certain measurements
narrow it, neither needing a new fit.

**It is not the hit-chi2 term.** That term's own minimum is a 92x92 linear
solve on the card's own `grad_values`/`hess_dense` plus the card's priors, and
it sits at

```
<D_card> . theta = -41.20 MeV = -1.330e-2     (bfield_mode0 alone: -41.45 MeV)
```

i.e. the hit-chi2 curvature on this MC wants a momentum scale **1.3 % off**,
and the mass term drags it back by a factor ~90. `bfield_mode0` is pulled from
`+50.7e-3` to `-0.75e-3`; its overshoot past zero is 1.5 % of that pull and is
not a separate effect.

**It is intrinsic to the J/psi mass term.** At the joint minimum
`dL_mass/dtheta = -(g + H_quad theta*)` and `H_mass = H_total - H_quad` with
`H_total` the inverse of the certified postfit covariance, so the mass term's
OWN stationary point follows without refitting:

| | fitted `<D>.theta` | the MASS TERM alone |
|---|---:|---:|
| `J0` (delta) | -1.9049 MeV | -2.2267 MeV = -7.190e-4 |
| `JK` (`mc` kernel) | +0.4537 MeV | **+0.5300 MeV = +1.711e-4** |

The hit-chi2 term contributes only `-0.076 MeV` (`-2.5e-5`) of the residual.
**The J/psi mass likelihood's own answer is `+1.71e-4` where truth is 0.**

What is left, with sizes:

| candidate | size | sign | status |
|---|---:|---|---|
| the two mass-likelihood corrections' own residual | **+5.1e-5** (the gun closure with both first-principles terms, `+0.051 +- 0.017e-3`) | same | ~30 % of it, measured elsewhere |
| ONE kernel for 64 resolution classes | <= 1.95e-4 on the kernel MEAN, ~2.5e-5 realised | **opposite** | makes the unexplained part larger, not smaller |
| `k_ms` frozen at 1 where the MC truth is 1.0298 | 0.5 MeV on `m_Z` = 5.5e-6 | — | negligible |
| `f_ang` | — | — | excluded: the J/psi leg reads `Jpsi_fang` per candidate |
| the field-mode basis | — | — | excluded: on MC the truth of all 92 is 0 and the basis is the production's own |
| **the correction FORM under a kernel** | unbounded analytically | — | **the open one** |

The last is the caveat already in `zchannel/README.md`:
`--unbinnedDeltaKernelForm auto` keys on `term.kernel.kind == "delta"`, a
`phik` does not change that, so a kernelled J/psi term still takes the
**residual** form -- which is exact only where `delta_i` IS the resolution
fluctuation, and with a kernel it is the fluctuation PLUS the FSR
displacement.

### `JKF`: the flag cannot be run, and that is itself the answer's first half

`JKF` (**22854490**) is the same `jpsi_fsrmc` card with
`EXTRA="--unbinnedDeltaKernelForm fluctuation"`. It **FAILED, rc=1, NOT
certified**: `Condition number: nan`, `edmval: nan`,
`Minimizer raised: array must not contain infs or NaNs`, and the postfit died
on `Cholesky decomposition failed, Hessian is not positive-definite`.

That is the failure mode the `auto` rule exists to prevent, and it says
something: **the fluctuation form's first-order Fourier truncation is NOT
small on the kernelled J/psi term.** So the residual form is the only usable
one today, the mismatch it carries under a kernel cannot be removed with a
flag, and removing it needs a code change -- the exact map applied to the
FLUCTUATION inside the convolution rather than to the residual.

### The fit-free scan that decides the rest

`jpsi_scale_pref.py` scans the term's own truncated `-sum log(L_i/Z_i)` in a
common predicted-mass shift `s` -- the one number a momentum-scale error is --
and reads off the minimum. 300 000 candidates, the card's own `a_res`,
`jensen_s2`, 64 norm classes and truncation window, no fit anywhere:

| | preferred shift [MeV] | relative |
|---|---:|---:|
| delta, corrections ON | -2.7794 | -8.97e-4 |
| **kernel, corrections ON** | **-0.5589** | **-1.80e-4** |
| delta, corrections OFF | -2.3945 | -7.73e-4 |
| **kernel, corrections OFF** | **-0.0545** | **-1.76e-5** |

| the pieces | | |
|---|---:|---:|
| the FSR kernel, corrections ON | +2.2205 MeV | **+7.17e-4** |
| the two corrections, kernel ON | -0.5043 MeV | **-1.63e-4** |
| the two corrections, delta | -0.3849 MeV | -1.24e-4 |

**The FSR step agrees with the fits**: +7.17e-4 here, +7.616e-4 from
`JK - J0`, +7.72e-4 from `P2K - P2XP`.

**And with the kernel in and the two corrections OFF, the J/psi term's
preferred scale is `-1.8e-5`** -- consistent with zero at the level this scan
resolves. So the residual is **the two mass-likelihood corrections as applied
in the RESIDUAL form**: they move the preferred scale by `-1.63e-4` with the
kernel and by `-1.24e-4` with the delta, which is the correction-form caveat
above, not a separate effect.

*Scope of the scan*: it measures the best COMMON shift, while the fit
optimises 92 directions against the hit-chi2 curvature as well, so the
absolute preferred shift here (`-0.56 MeV`) is not the fit's
`<D_card>.theta` (`+0.53 MeV`, mass term alone) and should not be read as
such. The DIFFERENCES are what the scan establishes, and those are what the
fits reproduce.

**Verdict on the residual**: the J/psi FSR is fully accounted for by the
kernel; what remains is the two corrections in the residual form, worth
`-1.6e-4`, and the fix is the code change named above rather than a
configuration choice.

## `P2B` -- the combination

`joint_both.hdf5` (10.736 GB, built and verified: the J/psi `mc` kernel round
trips and the Z `mc` table is carried inline) was submitted as **22886298**
once all three singles certified, on `mit_preemptable`, `-G h200:1
--time=1-12:00:00`. Collect and certify exactly as the others:

```bash
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
rsync -a engaging:orcd/pool/zmass/fitresults/native/rabbit_P2B.hdf5 \
      $FS/runs/engaging_260916/
rsync -a engaging:orcd/pool/zmass/engaging/zprecond_22886298.out \
      $FS/runs/engaging_260916/
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
./run_tf_z.sh python3 $FS/native_dump.py \
    "$FS/runs/engaging_260916/rabbit_*.hdf5" -o $FS/runs/engaging_260916
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
python3 $FS/jpsi_fsr_table.py --ref P2N \
    P2N=$FS/runs/engaging_260916/rabbit_P2N.json \
    P2K=$FS/runs/engaging_260916/rabbit_P2K.json \
    P2XT=$FS/runs/engaging_260916/rabbit_P2XT.json \
    P2B=$FS/runs/engaging_260916/rabbit_P2B.json
```

Recovery if preempted: `--export=ALL,FRESH=0,ROWS="P2B"` on
`mit_preemptable` -- **not** `mit_normal_gpu`, which caps at 6 h and was 442
jobs deep.

**A slurm requeue after preemption is COLD.** `FRESH` defaults to 1 in this
script, so the requeued job restarts from scratch AND its first periodic
snapshot (0.25 h in) overwrites the good one. The recovery is therefore:
copy the snapshot aside, `scancel` the requeued job, and resubmit with
`FRESH=0` -- inside the first 15 minutes. That is what
`rabbit_P2B.snapshot.i13.hdf5` is.

**PRE-REGISTERED**: if the two changes are additive, `P2B` lands at
`P2K + (P2XT - P2N)` = `-45.594 + 0.245` = **-45.35 MeV** on `m_Z`, with
`bfield_mode0` at `P2K`'s `-1.311e-3` and `<D_card>.theta` at `+0.880 MeV`.
The two touch different terms, so a departure from additivity above the
0.2 MeV level would be a genuine interaction and worth chasing.

## What was NOT needed

**rabbit needed no change.** The `phik` path of `MassCFTerm` and the table
dispatch of `ZGammaLineshape` both already existed and both round-trip; the
staged commit is `d36ba27` on `vmass-conditioning` throughout, unchanged, and
`stage_native.sh code` was never re-run. Every change is in
`calibration_studies` on `resolution-energy-loss-corrections`.

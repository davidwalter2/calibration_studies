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
| **22824626** | `P2N` | `joint_nok` | `-G h200:1 --time=1-12:00:00`, `mit_preemptable` |
| **22825078** | `JD` | `jpsi_fsrdata` | `-G h200:1 --time=0-06:00:00`, `mit_preemptable` |
| **22830065** | `P2XT` | `joint_ztab` | `-G h200:1 --time=1-12:00:00`, `mit_preemptable` |

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

## Results as they land

| row | card | kernel | EDM | wall | `bfield_mode0` [1e-3] | `<D_card>.theta` [MeV] | `m_Z` [MeV] |
|---|---|---|---:|---:|---:|---:|---:|
| `P2XP` | `joint_ok_full` | delta | 1.05e-11 | 8 h 01 | +1.40124 +- 0.02563 | -1.5105 | +20.692 +- 2.134 |
| `J0` | `jpsi_nok` | delta | 1.10e-12 | **43 min** | +1.91544 +- 0.02562 | -1.9049 | — (no Z term) |
| `JK` | `jpsi_fsrmc` | `mc` | 2.06e-11 | **39 min** | -0.75376 +- 0.02633 | **+0.4537** | — (no Z term) |
| `JD` | `jpsi_fsrdata` | `data` | 7.23e-13 | **69 min** | -0.86590 +- 0.02639 | +0.5578 | — (no Z term) |
| `P2N` | `joint_nok` | delta | | | | | |
| `P2K` | `joint_fsrmc` | `mc` | | | | | |

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

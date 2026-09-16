# J/psi FSR kernel — state, jobs and collection commands

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
| `cards/joint_ok_full.hdf5` | quad + J/psi + Z | 10.733 GB | delta (the REFERENCE, = `P2XP`) |
| `cards/joint_fsrmc.hdf5` | quad + J/psi + Z | 10.733 GB | `mc` |
| `cards/jpsi_nok.hdf5` | quad + J/psi | 4.764 GB | delta |
| `cards/jpsi_fsrmc.hdf5` | quad + J/psi | 4.764 GB | `mc` |

Kernels: `zchannel/data/jpsi_kern_{mc,data,data_trunc}.npz`.

## Jobs (Engaging, submitted 2026-09-16 08:0x)

| job | rows | card(s) | resources |
|---:|---|---|---|
| **22823944** | `J0 JK` | `jpsi_nok`, `jpsi_fsrmc` | `-G h200:1 --time=1-00:00:00`, `mit_preemptable` |
| **22824029** | `P2K` | `joint_fsrmc` | `-G h200:1 --time=1-12:00:00`, `mit_preemptable` |

Submitted as

```bash
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-00:00:00 --export=ALL,ROWS="J0 JK" precond_refit.sbatch'
eng 'cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -p mit_preemptable \
     -G h200:1 --time=1-12:00:00 --export=ALL,ROWS="P2K" precond_refit.sbatch'
```

## Collect

```bash
eng 'squeue -u david_w -o "%.10i %.9P %.8j %.2t %.10M %R"'
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
mkdir -p $FS/runs/engaging_260916
rsync -a engaging:orcd/pool/zmass/fitresults/native/rabbit_{J0,JK,P2K}.hdf5 \
      $FS/runs/engaging_260916/
rsync -a 'engaging:orcd/pool/zmass/engaging/zprecond_2282*.out' \
      $FS/runs/engaging_260916/
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
for t in J0 JK P2K; do
  ./run_tf_z.sh python3 $FS/native_dump.py \
      $FS/runs/engaging_260916/rabbit_$t.hdf5 -o $FS/runs/engaging_260916/rabbit_$t.json
done
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

So, **if the FSR treatment is the explanation of the phase-2 pulls**:

1. `bfield_mode0` moves DOWN (toward zero) when the kernel is switched on, in
   both the J/psi-only pair and phase 2. Its J/psi-only difference
   `JK - J0` should be of order `-1.4e-3` and at most `-8.8e-3`.
2. `m_Z` in phase 2 moves DOWN from `+20.69 +- 2.13 MeV` toward zero. In the
   reference the whole pull is `+2.27e-4` of `m_Z`, an order below the
   kernel's own `-2.32e-3`, which is the same ~1/6 response the field mode
   shows.
3. The material pulls, which run to `+2.7 %` at 50 sigma on `bpix L1/L2` and
   `support7`, are NOT predicted to go away: they are 50-sigma effects against
   a truth of zero and a mean mass shift is one direction out of 92.

Anything that contradicts 1 or 2 says the FSR treatment is **not** the
explanation and the pulls have another source.

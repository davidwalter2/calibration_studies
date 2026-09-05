# Z → μμ channel of the unbinned CVH mass likelihood

The J/ψ channel of `rabbit.unbinned.MassCFTerm` fits a momentum scale `alpha`
against a resonance of known, essentially zero-width mass. The Z channel fits
`m_Z` and `Γ_Z` themselves, against a lineshape that has to be computed. This
directory is that channel: the FSR kernel, the datacard builder, the fit driver
and a systematics scan, prototyped on the 459-candidate smoke of the
`dymc_8p5M_260905` production and ready to point at the full output.

Everything downstream of `MassCFTerm` lives on the rabbit branch
`z-lineshape-kernel` (worktree `/work/submit/david_w/ZMass/rabbit-zlineshape`):
the `ZGammaLineshape` provider, `TabulatedLineshapeKernel`, and the truncated
likelihood (`norm_window`) this channel needs.

---

## The pieces

| file | what it does |
|---|---|
| `fwlite.sh` | run a script under CMSSW 15_0 FWLite (read-only use of the release area) |
| `dump_gen_fsr.py` | pre-FSR / post-FSR / dressed gen dimuon masses from DY MiniAOD |
| `zfsr_kernel.py` | those samples → the empirical FSR kernel CF `phi_K(t)`, plus diagnostics |
| `make_z_card.py` | pairs cache + kernel → a rabbit datacard with one `MassCFTerm` |
| `fit_z.py` | read the card back, fit, and project the covariance to full statistics |
| `z_variants.py` | rebuild the term under each modelling choice and report the bias |
| `data/` | gen dumps, kernels, caches, cards — all regenerable, git-ignored |

The rabbit side:

| file | what it does |
|---|---|
| `rabbit/lineshapes/zgamma.py` | Born Z/γ\* lineshape, POIs `m_Z`, `Γ_Z`; its CF |
| `rabbit/unbinned.py` | `MassCFTerm` + `TabulatedLineshapeKernel` + `norm_window` |
| `tests/test_zgamma_kernel.py` | the provider's six tests |
| `tests/test_unbinned_norm.py` | the truncated likelihood's five tests |

---

## The generator record (read this before touching `dump_gen_fsr.py`)

`DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos`, UL16
MiniAODv2. The obvious pre-FSR selection finds nothing — **no muon in
`prunedGenParticles` carries `fromHardProcessBeforeFSR`**. What is there,
verified on 4000 events:

| object | status | note |
|---|---|---|
| hard-process Z | 22 (first copy) and 62 (last copy) | masses bit-identical in 4000/4000 |
| **pre-Photos muon pair** | **746** | CMS' Photos interface keeps the unradiated copies; present only in the 59 % of events that radiated |
| post-FSR bare muons | 1, `isPrompt`, `isHardProcess` | what the tracker sees |
| FSR photons | 1, `isPrompt` | all kept in `prunedGenParticles`; `packedGenParticles` is not needed (and its `statusFlags` are unreliable) |

`m(μμ, status 746) == m(Z, status 62)` to an RMS of 4×10⁻⁶ GeV, so **the Z mass
is the pre-FSR mass** and it exists in every event. That is what the script
uses; the 746 pair is written out as `m_pre746` for the cross-check.

`prunedGenParticles` is stored at reduced float precision, which puts a ~0.1 MeV
floor on `dm` (the median |m_Z62 − m_μμ| in unradiated events is 0.13 MeV).

**The maker already exports the post-FSR mass.** `Jpsigen_mass` is a ΔR < 0.1,
charge-matched pair of status-1 muons (`ResidualGlobalCorrectionMakerTwoTrackG4e.cc`
:4455-4527); over the smoke's 449 gen-matched candidates it agrees with the
status-1 `isHardProcess` pair to an RMS of **2.3 keV**. So a separate gen pass is
needed **only for the pre-FSR mass** — see "C++ to-do" below.

---

## Running on the full production

The production is `dymc_8p5M_260905`: 380 slurm tasks over the 104-file /
8.5 M-event priority head, output at

```
/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260905/task_XXXX/globalcor_0.root
```

expected ~3.90 M candidates (0.460 attempted/event), ~140 GB. `/ceph/submit` is
evicted on submit82 — use submit50/51/52 (submit60 has ceph but **no AVX2**, so
CMSSW 15_0 dies there with an illegal instruction).

### 1. FSR kernel

The kernel is generator-level, so it does not wait for the production. Shard
over the same 104 MiniAOD files (~25 k events per file, ~35 s each):

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
FL=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
sed 's|^file:||' $FL > $Z/data/paths_dymc.txt
mkdir -p $Z/data/genparts
seq 0 103 | xargs -P 12 -I{} ssh submit51 \
  "$Z/fwlite.sh $Z/dump_gen_fsr.py --filelist $Z/data/paths_dymc.txt \
   --skip-files {} --nfiles 1 -o $Z/data/genparts/gen_{}.npz"

python zfsr_kernel.py -i data/genparts/gen_*.npz -o data/zfsr_kernel_reco.npz \
    --acc-pt 5 --acc-eta 2.4 --tmax 20 --npoints 16385 --report
```

`--acc-pt/--acc-eta` must match the **reconstruction** acceptance of the sample,
not a physics-analysis cut. The `dymc_8p5M` selection has *no* explicit muon pT
cut (`TrackProducerFromPatMuons ptMin=-1`; the smoke's sub-leading muon goes down
to 4.7 GeV, only 72 % are above 26 GeV), so `--acc-pt 5 --acc-eta 2.4` is the
right proxy. Do **not** pass `--mass-window`: the mass selection is applied once,
by the likelihood's `norm_window`, and applying it here as well double-counts it.

The empirical `phi_K` carries the gen sample's statistical error: a half-sample
split gives 1.0×10⁻³ (mean over `t`) at 126 k events, so 1×10⁻⁴ needs ~13 M gen
events. The FSR kernel is generator-level, so the whole 1731-file dataset is
usable, not just the 104-file production head — 35 s per file, embarrassingly
parallel.

`--tmax` must exceed `max(tgrid)/min(sigma)` = `7.8926 / σ_min`; the smoke's
`σ_min = 0.494 GeV` needs 16 GeV⁻¹, so 20 is the working default. `make_z_card.py`
checks this and refuses otherwise.

### 2. Pairs cache

`cf_inmaker.py pairs` reads all files matching one glob and writes one npz;
`--ntasks` caps how many files it takes. At 3.9 M candidates × 64 τ points × 5
float32 family arrays the cache is ≈ 5 GB, ~10 GB peak while concatenating —
fine on a submit node, but shard if you would rather:

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution
D=/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260905
python cf_inmaker.py pairs --files "$D/task_*/globalcor_0.root" \
    --cache runs/zpairs_dymc8p5M.npz --mass-window 91.1876 60
```

**`--mass-window` is mandatory and it cuts on the *gen* mass**, `|Jpsigen_mass −
M| ≤ HW`. Two consequences:

* the default `(3.0969, 0.35)` selects **zero** Z candidates;
* the halfwidth must stay below 190.2 so the `−99` no-gen-match sentinel is
  rejected, and should otherwise be as loose as possible — `60` gives
  `m_gen ∈ [31, 151]`, wide enough to be inactive once `make_z_card.py` applies
  the real `m_obs ∈ [60, 120]` cut. A tight gen-mass window would be a
  selection on a *latent* variable that the likelihood cannot renormalise over.

### 3. Datacard

```bash
cd $Z
python make_z_card.py --pairs ../resolution/runs/zpairs_dymc8p5M.npz \
    --kernel data/zfsr_kernel_reco.npz -o cards/zcard_dymc8p5M.hdf5 \
    --norm-classes 64 --gz-prior 2.3
```

Leave `--upsample` at 1 and set the term's `upsample` at *fit* time instead
(see "The 64-point tau grid" below) -- resampling at build time multiplies the
card size.

Expect ≈ 5 GB (the five `(n, 64)` float32 family blocks dominate; the truncation
block adds 32 × 257 rows, i.e. nothing). Run it in the rabbit TF environment:

```bash
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/rabbit-zlineshape:\
/work/submit/david_w/WRemnants_dev/wums:\
/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath
singularity exec -B /work/submit,/home/submit,/scratch/submit,/tmp,/ceph/submit "$IMG" python ...
```

(`wums.sparse_hist` is needed by `rabbit.tensorwriter` and is absent from the
`env_tf/pypath` shim, hence the `WRemnants_dev/wums` entry. Drop the
`/ceph/submit` bind on a node where ceph is evicted.)

### 4. Fit

Locally, for a small card:

```bash
python fit_z.py --card cards/zcard_dymc8p5M.hdf5 --project 3.9e6
```

At 3.9 M candidates use Engaging (see `../engaging/README.md`) and an **H200**,
not an L40S: `rabbit_fit.py` builds the Hessian with `tf.vectorized_map` over
the parameters, so the whole batch is live and a 300 k-candidate J/ψ card
already OOMs on 46 GB.

```bash
cd ../engaging && ./stage_engaging.sh code
rsync -a cards/zcard_dymc8p5M.hdf5 engaging:orcd/pool/zmass/cards/
eng 'bash -lc "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -G h200:1 \
     --export=ALL,STAGES=2,CARD=$HOME/orcd/pool/zmass/cards/zcard_dymc8p5M.hdf5 \
     masslik_fit_gpu.sbatch z --rabbit"'
```

which ends in `rabbit_fit.py <card> -t 0 --unblind --paramModel UnbinnedParams`.

### 5. Systematics scan

```bash
python z_variants.py --pairs ../resolution/runs/zpairs_dymc8p5M.npz \
    --kdir data --project 3.9e6
```

---

## The 64-point tau grid is not enough (read this one too)

The in-maker exports the CF exponents on 64 points of `[0, 7.8926]`
(`tau=stride4of448<=8`; it computes 448 internally and strides by 4). The
density is an inverse Fourier transform whose integrand oscillates
`|m_obs - m_pred| / sigma` times across that grid. For a J/psi in a +-0.35 GeV
window that is a handful of periods. For a Z in a 60-120 GeV window it is up to
**61** -- 0.8 samples per period:

| pull = \|m_obs - m_ref\|/sigma | candidates (smoke) | periods | points/period | median rel. error on `L_i` |
|---|---|---|---|---|
| 0-1   | 158 |  0.6 | 100 | 1.4e-3 |
| 1-2   |  87 |  1.8 |  35 | 2.7e-3 |
| 2-3   |  50 |  3.0 |  21 | 4.6e-3 |
| 3-5   |  51 |  5.0 |  13 | 8.5e-3 |
| 5-8   |  34 |  8.2 | 7.8 | 3.7e-2 |
| 8-12  |  25 | 12.4 | 5.2 | 6.8e-2 |
| 12-30 |  31 | 23.4 | 2.7 | 2.1e-1 (max 2.7) |

The whole NLL moves by **-33.7** over 449 candidates when the same term is
rebuilt on a 16x finer grid (converged: the 16x -> 64x change is 3e-4). The
candidates near the peak are fine; it is the FSR tail -- where the width
information lives -- that is misrepresented.

The exponents are *smooth* in tau (largest second difference below 2 % of the
range; a cubic spline through every other in-maker point reproduces them to
~1e-4 absolute), so resampling is faithful, not a guess. `MassCFTerm` does it
in the graph (`upsample=N`), which keeps the datacard at 64 points and only
grows the per-chunk intermediates -- at 3.9 M candidates a 16x-finer *stored*
array would be a 79 GB card. `make_z_card.py --upsample N` also exists and
resamples at build time; use it only for small samples.

The same arithmetic is why the truncation normalisation integrates in Fourier
space (`_norm_z`, Gil-Pelaez) rather than by sampling the density on a mass
grid: both need a fine tau grid, but Fourier needs a `(K, nt)` tensor where the
mass grid needs `(K, n_mass, nt)`. The mass-grid version produced `Z > 1`
(up to 2.8) on this sample -- an integral of a density over a sub-interval,
larger than one, which is how the problem was found.

**C++ to-do (cheap):** export more of the 448 points the maker already
computes. The six `cfmass_*` branches are 1.4 % of the 34.7 kB event; all 448
points would make them 9.7 %, i.e. +28 % on the output -- affordable, and it
removes the resampling step (though not the memory cost of integrating on a
fine grid).

---

## Two windows, and why they differ

`--born-window` is the support of the **pre-FSR** lineshape the provider models.
`--window` is the window the **observed** candidates were selected in, and is
what the likelihood renormalises over (`norm_window`). They must not be equal:

* setting the Born support to the selection window forbids the Born masses that
  FSR moves *into* the window — and FSR only ever lowers the mass, so that is a
  real population;
* leaving the observed density unnormalised is the bigger error. It is not a
  constant offset: `Z` moves with `m_Z`, so the missing term biases the mass.
  `tests/test_unbinned_norm.py` test 5 makes it concrete on a Voigt toy cut to
  60–120 GeV with only 2.7 % outside: Γ comes out 479 MeV low (48.7 σ) and the
  resolution scale 46 % high.

The DY sample is generated with `m_pre > 50 GeV`, so 50 is the natural lower
Born edge; the upper edge only has to sit far enough above the window.

---

## What is still missing for a *data* Z channel

* **Acceptance `A(m)`.** The likelihood has no acceptance at all. The muon
  pT/η cuts sculpt the observed spectrum, and — because a muon that radiates
  hard falls out of acceptance — they also sculpt the FSR kernel, which is why
  the kernel has to be rebuilt whenever the selection changes. Folding the gen
  acceptance into the kernel (what `--acc-pt/--acc-eta` do) captures part of it;
  a proper `A(m)` multiplying the density is the real fix.
* **The FSR kernel is multiplicative, and `MassCFTerm` convolves.** `dm/m_pre`
  is nearly independent of `m_pre` while `dm` is not (see the table in the
  report). The clean fix is to fold FSR into the *provider* — have
  `ZGammaLineshape` return the CF of the **post**-FSR spectrum, since
  `p_post(m) = ∫ p_born(m′) K(m|m′) dm′` is a function of the POIs alone. The
  LL collinear radiator in `../lineshape/zwidth_sensitivity.py` (`FSRKernel`)
  is the natural implementation. `MassCFTerm` also accepts a per-candidate
  `phik_grid` of shape `(n, nt)`, which would allow an `m_obs`-dependent
  kernel, but at 3.9 M × 64 × float64 × 2 that is 4 GB and does not scale.
* **Background.** `UniformBackground` / `BernsteinBackground` are wired up with
  a fixed or floating fraction, but the shape and normalisation of the real
  background (Z→ττ, top, QCD) are not measured.
* **The kernel's own statistical error.** The empirical `phi_K` carries
  `1/√N_gen` noise (2.8×10⁻³ at 126 k gen events); at the target precision the
  kernel needs either the full 8.5 M MC or a smooth parametric form.
* **Theory nuisances.** PDF (regenerate `zlumi_*.npz` per replica and add a
  luminosity-shape nuisance), EW loop corrections, running α and sin²θ_W, the
  fixed-vs-running width convention.
* **`m_Z` and `alpha` are exactly degenerate** in a single-resonance fit. The Z
  channel measures `m_Z` only jointly with a channel that pins the momentum
  scale — which is the whole point of the unified likelihood.
* **The `t`-grid cost at scale.** Integrating on a 16x grid multiplies the
  per-chunk graph tensors by 16. The Hessian tape already peaks at ~88 GB for
  200 k candidates at `nt = 256`; `rabbit_fit` on an H200 will need `--chunk`
  tuning, or `trust-krylov` with Hessian-vector products instead of a
  materialised Hessian.
* **The selection variable is not the fitted observable.** The maker cuts on
  `Jpsitrk_mass` (the input-track dimuon mass), not on the CVH-refit
  `Jpsi_mass`; 459/459 smoke candidates pass on the former, 458/459 on the
  latter. The truncation normalisation assumes the cut is on the observable.

### C++ to-do for after the productions

Add **the pre-FSR pair mass per candidate** to the two-track maker, so the FSR
kernel can be built from the production output itself instead of a separate
FWLite pass over MiniAOD. In `ResidualGlobalCorrectionMakerTwoTrackG4e.cc`, next
to the existing `Jpsigen_*` block (~line 4520), take the `|pdgId| == 23` entry
with `status() == 62` (equivalently 22) from the same `genPartCollection` and
store its `mass()` as `Jpsigenpre_mass` — one float per candidate. Optionally
store the status-746 pair as a cross-check while the branch is being validated.
With that branch present, `zfsr_kernel.py` runs straight off the pairs cache and
the FSR kernel automatically inherits the exact analysis selection, which is the
part that is hardest to reproduce by hand.

Note the gen block is filled from an `edm::View<reco::Candidate>`, which does
**not** expose `GenStatusFlags`; `status()` and `pdgId()` are enough for the Z,
but anything needing the flags would have to change the token type to
`edm::View<reco::GenParticle>`.

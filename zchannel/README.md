# Z → μμ channel of the unbinned CVH mass likelihood

The J/ψ channel of `rabbit.unbinned.MassCFTerm` fits a momentum scale `alpha`
against a resonance of known, essentially zero-width mass. The Z channel fits
`m_Z` and `Γ_Z` themselves, against a lineshape that has to be computed. This
directory is that channel: the FSR kernel, the acceptance, the datacard builder,
the fit driver, a systematics scan and the generator-level closure.

Everything downstream of `MassCFTerm` lives on the rabbit working branch
`vmass-conditioning` (worktree `/work/submit/david_w/ZMass/rabbit-vmass`):
the `ZGammaLineshape` provider, `TabulatedLineshapeKernel`, and the truncated
likelihood (`norm_window`) this channel needs.

The **full-statistics** results — σ(m_Z), σ(Γ_Z), the v-form conditioning, the
joint J/ψ + Z fit and what is still open — are in
`../fullscale/SUMMARY.md`. This README is the channel's own machinery and the
generator-level validation it rests on.

---

## The pieces

| file | what it does |
|---|---|
| `fwlite.sh` | run a script under CMSSW 15_0 FWLite (read-only use of the release area) |
| `dump_gen_fsr.py` | pre-FSR / post-FSR / dressed gen dimuon masses from DY MiniAOD |
| `run_gen_dump.sh` | shard `dump_gen_fsr.py` over the full DY MiniAOD filelist |
| `merge_gen.py` | merge the per-file gen dumps into one compact npz |
| `zfsr_kernel.py` | those samples → the empirical FSR kernel CF `phi_K(t)`, plus diagnostics |
| `fsr_analytic.py` | the **analytic** QED FSR kernel: exact O(α) + exponentiation + O(α²)LL+NLL + the exact O(α²) pair radiator, and the exact matrix elements both are validated against |
| `cmp_fsr.py` | the analytic kernel against the Photos++ generator record (figures + moment tables) |
| `fsr_perleg.py` | the **per-leg** radiator `D` (the Mellin square root of `K`), the `h(a_+,a_-\|m)` table and the selection-conditional kernel + `A(m)` |
| `dump_gen_perleg.py` | the same gen dump with the two legs matched **by charge** (`x = E'/E` per leg) |
| `run_perleg_dump.sh`, `merge_perleg.py` | shard and merge it |
| `build_perleg.sh` | build every `h` table and selection-conditional kernel, including the discretisation variants |
| `run_perleg_fit.sh` | the per-leg fit benchmark, `physics` and `disc` suites |
| `cmp_perleg.py` | the per-leg construction against the generator record (figures + tables) |
| `ptres.py` | the **detector side**: the CVH-refit muon `p_T` resolution on the DY reco MC, its 4-parameter form and its tails, the per-leg pass probability, and the `sigma`-conditioning of the acceptance |
| `build_selection.sh` | the tables and kernels of the asymmetric cuts and of the resolution in the acceptance |
| `run_selection_fit.sh` | their fit benchmark, one suite per selection |
| `fit_gen.py` | **generator-level closure**: FSR kernel, acceptance, and the fit |
| `kern_from_selected.py` | rebuild the kernel *and* `A(m)` from the gen record of the SELECTED reconstructed candidates |
| `fit_gensel.py` | the same closure on the selected candidates' own gen masses, so the fold is isolated from the detector |
| `make_z_card.py` | pairs cache + kernel → a rabbit datacard with one `MassCFTerm` |
| `fit_z.py` | read the card back, fit, project the covariance to full statistics |
| `z_variants.py` | rebuild the term under each modelling choice and report the bias |
| `check_tgrid.py` | is the in-maker's 64-point τ grid fine enough? (no — see below) |
| `make_lumi_scale.py` | parton-luminosity tables at μ_F = k Q (scale systematic) |
| `plot_gen.py` | the closure figures |
| `plot_pairs.py` | the pair-emission figures: exact vs eikonal vs leading log vs Photos |
| `run_tf_z.sh` | run a script in the rabbit TF image with this branch on the path |
| `data/generator_settings_*.md` | **the generator's own parameters** (committed) |
| `data/` | gen dumps, kernels, acceptances, caches, cards — all regenerable, git-ignored |

On the rabbit side: `rabbit/lineshapes/zgamma.py` (Born Z/γ* lineshape, POIs
`m_Z`, `Γ_Z`, and its CF), `rabbit/unbinned.py` (`MassCFTerm` +
`TabulatedLineshapeKernel` + `norm_window`), and the tests
`tests/test_zgamma_kernel.py`, `tests/test_unbinned_norm.py`.

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

**The maker exports both masses per candidate**: `Jpsigen_mass` (a ΔR < 0.1,
charge-matched pair of status-1 muons; it agrees with the status-1
`isHardProcess` pair to an RMS of 2.3 keV) and, since the 2026-09-06 export set,
`Jpsigenpre_mass` — the `|pdgId| == 23`, `status() == 62` entry. So on a v2
production the kernel and the acceptance can be built **from the production's
own pairs cache**, inheriting the exact analysis selection, and the separate
FWLite pass over MiniAOD is only needed for pre-v2 samples or for
generator-level work over files the production never read.

---

## Running on the full production

`dymc_8p5M_260906_v2`: 380 tasks over the 104-file / 8.5 M-event priority head,
**3 799 624 candidates**, ~267 GB. It runs `numberOfThreads=4`, so **a task is
FOUR files**:

```
/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2/task_XXXX/globalcor_0.root
                                                              ...      /globalcor_1.root … _3.root
```

An event never splits across streams and the candidate content is bit-identical
to a single-thread run after sorting on (run, lumi, event), so the four files
simply concatenate — but **naming `globalcor_0.root` takes a quarter of the
candidates and says nothing**. Every reader here goes through
`resolution/prodfiles.py`, which widens the stream index, skips a task whole if
its `.complete` sentinel is missing or a stream is empty, and — the part that
bites — makes `--ntasks` a cap on **tasks**, not files.

`/ceph/submit` is evicted on submit82 — use submit50/51/52 (submit60 has ceph
but **no AVX2**, so CMSSW 15_0 dies there with an illegal instruction).

### 1. FSR kernel

The kernel is generator-level, so it does not wait for the production. Shard
over the 104 MiniAOD files (~25 k events per file, ~35 s each):

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
cut (`TrackProducerFromPatMuons ptMin=-1`; the sub-leading muon goes down to
4.7 GeV, only 72 % are above 26 GeV), so `--acc-pt 5 --acc-eta 2.4` is the right
proxy. Do **not** pass `--mass-window`: the mass selection is applied once, by
the likelihood's `norm_window`, and applying it here too double-counts it.

`--tmax` must exceed `max(tgrid)/min(sigma)` = `7.8926 / σ_min` (σ_min = 0.494
GeV needs 16 GeV⁻¹, so 20 is the working default); `make_z_card.py` checks this
and refuses otherwise. The empirical `phi_K` carries the gen sample's
statistical error — a half-sample split gives 1.0×10⁻³ (mean over `t`) at 126 k
events, so 1×10⁻⁴ needs ~13 M gen events. The kernel is generator-level, so the
whole 1731-file dataset is usable, not just the 104-file production head.

> **A gen-fiducial kernel is not the selected sample's kernel.** Measured on
> this sample, `⟨u⟩ = ⟨−ln(m_post/m_pre)⟩` is 0.027147 inclusive, 0.023456 under
> the gen fiducial (pT > 5, |η| < 2.4), 0.024032 for the kernel npz in use — and
> **0.014447 on the reconstructed, selected candidates**. The fold in the
> likelihood then describes a sample radiating 1.66× more than the one it is
> folded against, and `A(m_pre)` cannot absorb it, because `A` is a function of
> the Born mass while the defect is a correlation between the selection and `u`
> at fixed `m_pre`. `kern_from_selected.py` builds both `k` and `A` from the
> selected candidates' own gen record, which is the construction the model
> assumes: `p(m_obs) ~ ∫ dm_pre BW(m_pre) A(m_pre) k(m_obs/m_pre | m_pre)`.

### 2. Pairs cache

`cf_inmaker.py pairs` reads every stream of every usable task and writes one
npz; `--ntasks` caps how many **tasks** it takes (0 = all). At 3.8 M candidates
× 64 τ points × 5 float32 family arrays the cache is ≈ 5 GB, ~10 GB peak while
concatenating:

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution
D=/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2
python cf_inmaker.py pairs --files "$D/task_*/globalcor_*.root" --ntasks 0 \
    --cache runs/zpairs_dymc8p5M.npz --mass-window 91.1876 60
python prodfiles.py "$D" --stats          # what a spec resolves to, before spending the hours
./masspairs_parallel.sh "$D" runs/zpairs_dymc8p5M.npz 40    # sharded, explicit file lists
```

**`--mass-window` is mandatory and it cuts on the *gen* mass**,
`|Jpsigen_mass − M| ≤ HW`. Two consequences: the default `(3.0969, 0.35)`
selects **zero** Z candidates; and the halfwidth must stay below 190.2 so the
`−99` no-gen-match sentinel is rejected, and should otherwise be as loose as
possible — `60` gives `m_gen ∈ [31, 151]`, wide enough to be inactive once
`make_z_card.py` applies the real `m_obs ∈ [60, 120]` cut. A tight gen-mass
window would be a selection on a *latent* variable the likelihood cannot
renormalise over.

### 3. Datacard

```bash
cd $Z
python make_z_card.py --pairs ../resolution/runs/zpairs_dymc8p5M.npz \
    --kernel data/zfsr_kernel_reco.npz -o cards/zcard_dymc8p5M.hdf5 \
    --norm-classes 64 --gz-prior 2.3
```

`--fit-upsample` (default 4) is the τ upsampling; it is stored in the card's
config and applied in the graph at fit time, so the card stays at 64 points.
Leave the build-time `--upsample` at 1 — resampling there multiplies the card
size. Expect ≈ 5 GB (the five `(n, 64)` float32 family blocks dominate; the
truncation block adds 32 × 257 rows, i.e. nothing). Run it in the rabbit TF
environment:

```bash
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:\
/work/submit/david_w/WRemnants_dev/wums:\
/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath
singularity exec -B /work/submit,/home/submit,/scratch/submit,/tmp,/ceph/submit "$IMG" python ...
```

(`wums.sparse_hist` is needed by `rabbit.tensorwriter` and is absent from the
`env_tf/pypath` shim, hence the `WRemnants_dev/wums` entry. Drop the
`/ceph/submit` bind on a node where ceph is evicted.)

### 4. Fit

```bash
python fit_z.py --card cards/zcard_dymc8p5M.hdf5 --project 3.9e6      # small cards, locally
```

At full statistics use Engaging (`../engaging/README.md`) and an **H200**, not
an L40S: `rabbit_fit.py` builds the Hessian with `tf.vectorized_map` over the
parameters, so the whole batch is live and a 300 k-candidate J/ψ card already
OOMs on 46 GB.

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

## The 64-point τ grid is not enough

The in-maker exports the CF exponents on 64 points of `[0, 7.8926]`
(`tau=stride4of448<=8`; it computes 448 internally and strides by 4). The
density is an inverse Fourier transform whose integrand oscillates
`|m_obs − m_pred| / σ` times across that grid. For a J/ψ in a ±0.35 GeV window
that is a handful of periods. For a Z in a 60–120 GeV window it is up to **61** —
0.8 samples per period:

| pull = \|m_obs − m_ref\|/σ | candidates | periods | points/period | median rel. error on `L_i` |
|---|---|---|---|---|
| 0–1 | 158 | 0.6 | 100 | 1.4e-3 |
| 1–2 | 87 | 1.8 | 35 | 2.7e-3 |
| 2–3 | 50 | 3.0 | 21 | 4.6e-3 |
| 3–5 | 51 | 5.0 | 13 | 8.5e-3 |
| 5–8 | 34 | 8.2 | 7.8 | 3.7e-2 |
| 8–12 | 25 | 12.4 | 5.2 | 6.8e-2 |
| 12–30 | 31 | 23.4 | 2.7 | 2.1e-1 (max 2.7) |

The whole NLL moves by **−33.7** over 449 candidates when the same term is
rebuilt on a 16× finer grid (converged: 16× → 64× changes it by 3e-4). The
candidates near the peak are fine; it is the FSR tail — where the width
information lives — that is misrepresented.

The exponents are *smooth* in τ (largest second difference below 2 % of the
range; a cubic spline through every other in-maker point reproduces them to
~1e-4 absolute), so resampling is faithful, not a guess. `MassCFTerm` does it in
the graph (`upsample=N`), which keeps the datacard at 64 points and only grows
the per-chunk intermediates — at 3.9 M candidates a 16×-finer *stored* array
would be a 79 GB card. `make_z_card.py --upsample N` resamples at build time
instead; use it only for small samples.

The same arithmetic is why the truncation normalisation integrates in Fourier
space (`_norm_z`, Gil-Pelaez) rather than by sampling the density on a mass
grid: both need a fine τ grid, but Fourier needs a `(K, nt)` tensor where the
mass grid needs `(K, n_mass, nt)`. The mass-grid version produced `Z > 1` (up to
2.8) on this sample — an integral of a density over a sub-interval, larger than
one, which is how the problem was found.

**Cheap C++ option if resampling ever has to go:** export more of the 448 points
the maker already computes. The six `cfmass_*` branches are 64 float32 each,
~1.4 kB; all 448 points would take them to ~9.9 kB, i.e. +8.5 kB/candidate —
**+25 %** on a pre-v2 34.7 kB event and **~+12 %** on the v2 format's ~70 kB
(267 GB → ~300 GB for this production). Halfway, `stride2of448` = 224 points,
costs half that. It removes the resampling step, though not the memory cost of
integrating on a fine grid.

---

## Two windows, and why they differ

`--born-window` is the support of the **pre-FSR** lineshape the provider models.
`--window` is the window the **observed** candidates were selected in, and is
what the likelihood renormalises over (`norm_window`). They must not be equal:

* setting the Born support to the selection window forbids the Born masses that
  FSR moves *into* the window — and FSR only ever lowers the mass, so that is a
  real population;
* leaving the observed density unnormalised is the bigger error, and it is not a
  constant offset: `Z` moves with `m_Z`, so the missing term biases the mass.
  `tests/test_unbinned_norm.py` test 5 makes it concrete on a Voigt toy cut to
  60–120 GeV with only 2.7 % outside: Γ comes out 479 MeV low (48.7 σ) and the
  resolution scale 46 % high.

The DY sample is generated with `m_pre > 50 GeV`, so 50 is the natural lower
Born edge; the upper edge only has to sit far enough above the window.

---

## Modelling shifts, measured on the 459-candidate prototype

Fit with the resolution scales held at 1, τ upsampling 4×, truncated on
60–120 GeV, FSR kernel from 126 k gen events in the reconstruction acceptance:
`m_Z` = **−137.1 ± 140.0 MeV** and `Γ_Z` = **+293.2 ± 288.4 MeV** against a
truth of 0 (i.e. 91.15351 GeV, the fixed-width scheme) and ~+1 MeV. Projected to
3.9 M candidates by scaling the reference-point information, σ(`m_Z`) = 1.450
MeV and σ(`Γ_Z`) = 2.763 MeV with ρ = −0.010 — **and profiling the four
resolution scales costs the Z nothing** (free / ±1e-2 / ±1e-3 / fixed all give
the same numbers), because the FSR kernel and the lineshape carry the shape
information `k_res` was competing for. The Z alone does not *determine* them: at
449 candidates the 6-parameter information is not even positive definite.

Each row is the change in the fitted `m_Z`; at 3.9 M candidates 1 σ = 1.45 MeV:

| choice | Δ`m_Z` | σ(`m_Z`)/σ(`Γ_Z`) at 3.9 M | note |
|---|---|---|---|
| **baseline** | — | 1.450 / 2.763 | reco-like kernel, Born 50–130, upsample 4 |
| no τ upsampling (64 points) | **+29.3** | 1.463 / 2.770 | 20 σ. 4× and 16× agree exactly |
| no window normalisation | **+12.7** | 1.462 / 2.771 | 8.7 σ |
| no FSR kernel at all | **−313.7** | 1.397 / 2.054 | the effect the kernel exists to describe |
| FSR kernel from pT > 26 GeV | **−25.7** | 1.447 / 2.622 | a *wrong* acceptance for this sample |
| FSR kernel, multiplicative form | **−25.3** | 1.447 / 2.623 | the additive-vs-multiplicative ambiguity |
| FSR kernel, no acceptance at all | −2.0 | 1.450 / 2.759 | vs the loose reco-like cut |
| Born support 50–200 instead of 50–130 | +0.1 | 1.450 / 2.763 | |
| Born support = the selection window | −0.1 | 1.450 / 2.766 | |
| σ < 5 GeV (drops 5 of 449) | −3.6 | 1.443 / 2.748 | |

**The Born-window choice does not matter at all** once the *observed*-mass
truncation is normalised (±0.1 MeV between 50–130, 50–200 and 60–120). What
matters is the quadrature (29 MeV), the truncation (13 MeV), and the FSR kernel:
getting its acceptance grossly wrong, or treating the multiplicative kernel as
additive, each cost ~25 MeV — 17 σ at full statistics. That 25 MeV is the price
of `MassCFTerm` convolving a kernel that is really a rescaling, and it is the
strongest argument for folding FSR into the lineshape provider (which is what
`ZGammaLineshape(fsr=…)` now does).

---

## Generator-level closure of the kernel

Asked at generator level — no resolution, no detector — against 29.3 M events of
the same sample the detector-level channel is fitted on. Figures:
`~/public_html/ZMass/cvh/260905_zgen/`.

### The generator's own parameters

`data/generator_settings_DYJetsToMuMu_powhegMiNNLO_UL16.md` and
`data/prov/pwhg_main_init.log`: the gridpack's `powheg.input` sets **no** EW
inputs, so running the gridpack's own `pwhg_main` for one initialisation prints
what POWHEG actually used. It reads the PDG (running-width) values and
**converts them to the constant-width scheme**, unconditionally:

```
input Z mass = 91.187600  Z width = 2.4941343245745466   (PDG, running width)
      Z mass = 91.153509740726733   Z width = 2.4932018986110700   (used)
      W mass = 79.906853549493746   sthw2 = 0.23153999447822571
```

which is `m/sqrt(1+(Γ/m)²)` to 3e-15 GeV. **The provider's
`width_scheme="fixed"` default, and every one of its EW constants, is the
generator's own** (`GZ_FIXED = 2.4932` is 1.9 keV below POWHEG's 2.4932018986).
The fixed-vs-running ambiguity is *resolved*, not bounded: fitting in either
convention against its own reference gives the same answer to 0.2 MeV.

### The pre-FSR spectrum needs a smooth `K(m)`

`fit_gen.py fit --suite prefsr`. Fine-binned weighted likelihood on the
provider's own 2.44 MeV grid, truncated to the fit window, MiNNLO weights
clipped at 100× the modal |w| (two events in 29 M carry |w| ~ 1e19 and would
take N_eff from 20 M to 2), sandwich covariance.

The 2-parameter fit does **not** close: `m_Z` −2.47 ± 0.40 MeV, `Γ_Z`
**+75.8 ± 0.85 MeV**, and the width bias grows monotonically with the window
(+5.5 MeV at 89–93, +129 MeV at 51–150). The cause is visible without any fit
(`01_born_truth.png`): the generated/model ratio runs 1.7 at 55 GeV → 1.0 at the
peak → 1.2 at 150 GeV. That is the **NNLO K-factor** — the provider's hard ME
*and* its parton luminosity are both LO.

It is not a luminosity choice: NNPDF3.1 NNLO (the generator's `lhaid 306000`),
its replica 1, NNPDF3.1 LO, CT18 NNLO, μ_F = Q/2 and μ_F = 2Q give Δ`m_Z` from
−6.05 to +0.44 MeV and Δ`Γ_Z` from +71.45 to +79.49 MeV — the full PDF-set +
PDF-order + scale spread is ±3 MeV on `m_Z` and ±4 MeV on `Γ_Z`, **5 % of the
effect**.

And it is not degenerate with the POIs. Multiplying the Born spectrum by
`exp(Σ_k c_k P_k(m))` (Legendre, orthogonal over the window, the constant
absorbed by the normalisation) and floating the `c_k`:

| terms | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| 0 | −2.47 ± 0.40 | +75.83 ± 0.85 |
| 1 | +18.87 ± 0.43 | +57.46 ± 0.85 |
| 2 | +8.07 ± 0.42 | +2.85 ± 0.88 |
| 3 | +3.07 ± 0.46 | +6.85 ± 0.89 |
| 4 | +1.41 ± 0.46 | −0.69 ± 0.95 |
| **5** | **−0.45 ± 0.50** | **+1.14 ± 0.97** |
| 6 | −0.56 ± 0.50 | +0.74 ± 1.03 |

Five terms close and 5 → 6 moves nothing, at a cost of **1.25× on σ(`m_Z`) and
1.21× on σ(`Γ_Z`)**, with every ρ(POI, c_k) below 0.40; with the shape floating
the five luminosities agree to **0.25 MeV / 0.03 MeV**. That is the degeneracy
statement: a smooth K(m) is *orthogonal enough* to the resonance that it can be
floated for free. (At full statistics the K(m) truncation is the one open item —
`../fullscale/SUMMARY.md`.)

Cross-checks, 3 shape terms unless stated: `m_prelep` (the status-746 lepton
pair) agrees with `m_pre` (the status-62 Z) to 0.0001 MeV; the running-width
scheme to 0.2 MeV; `nm` = 8192 / 32768 / 65536 to 0.002 MeV; the unweighted fit
agrees within its error and its sandwich error equals its unit-weight error to
0.0002 MeV. Floating `sin²θ_W` **without** a shape drives it to +0.072
(nonsense — it acts as a shape parameter), so it is not identifiable from the
mass spectrum alone.

### The multiplicative FSR fold

The FSR kernel is *exactly* multiplicative in this sample: the distribution of
`u = −ln(m_post/m_pre)` is the same at every `m_pre` from 60 to 200 GeV to
within a few per cent; ⟨u⟩ moves from 26.17e-3 (60–80 GeV) to 28.84e-3
(110–150 GeV), 10 % over the whole range and 2.5 % over 80–110.

So the fold belongs in the *provider*, not in `MassCFTerm`'s additive `phi_K`:
`ZGammaLineshape(fsr=…)` returns

```
p_post(m) = Σ_j w_j p_born(m / r_j) / r_j          r_j = m_post/m_pre <= 1
```

built on an extended Born grid (the fold needs the density *above* the window),
so `pdf`, the CF and `MassCFTerm` all model the post-FSR mass as a function of
the POIs alone. FSR is not optional and cannot be absorbed by anything else:
without the fold the fit is `m_Z` −227 MeV / `Γ_Z` +741 MeV, and a 5-term smooth
shape only pulls that to −31 / +317 MeV.

**The fold is a midpoint quadrature and its discretisation is the leading
systematic.** Atoms are groups of the empirical `u` spectrum placed at their
weighted mean, so the residual is `(1/2) Var(u|group) m² p''`; capping the
in-group standard deviation of `u` at `sigma_cap` gives a bias that scales as
`sigma_cap²`:

| `sigma_cap` | atoms | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|---|
| 1e-2 | 144 | +101.67 | +151.76 |
| 3.3e-3 | 400 | +7.10 | +28.98 |
| 1e-3 | 1183 | +0.77 | +3.98 |
| **3.3e-4** (`build_kernel`'s default) | 3215 | **+0.15 ± 0.56** | **+1.31 ± 1.14** |
| 1e-4 | 9189 | −0.59 ± 0.59 | +0.72 ± 1.14 |

A global `Var(u)` comparison of the atoms against the data does *not* diagnose
this (it is a difference of two ~1.5e-2 numbers); the `sigma_cap` scan does.
Splitting the gen sample in half moves the fit by 0.4 MeV on `Γ_Z`; building the
kernel from only `m_pre` ∈ 60–88 or 94–130 moves it by ±5.4 MeV, which is the
price of assuming `m_pre`-independence and is largely removed by the **banded**
kernel (per-atom `m_lo`/`m_hi`, `fit_gen.py kernel --bands …`; banding moves the
inclusive fit by 0.3 / 0.7 MeV). `nm` = 4096 / 8192 / 16384 and a Born grid
capped at 160 or 200 GeV all agree to 0.06 MeV.

The closure, default kernel and 5 shape terms:

| model, window 60–120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| post-FSR, **no** fold, no `K(m)` | −227.12 ± 0.48 | +741.35 ± 1.19 |
| post-FSR, **no** fold, + `K(m)` | −30.66 ± 0.56 | +317.02 ± 1.19 |
| **post-FSR, folded, + `K(m)`** | **+0.15 ± 0.56** | **+1.31 ± 1.14** |
| pre-FSR control, + `K(m)` | −0.45 ± 0.50 | +1.14 ± 0.97 |

So FSR is *not* absorbable by a smooth nuisance (it broadens the peak; a smooth
ratio cannot), and once folded the post-FSR fit returns what the pre-FSR fit
returns.

The residual `m_pre` dependence of `u` is not an accident of this sample: it is
`beta(m) = (2 alpha/pi)(ln(m^2/m_mu^2) - 1)`, and the **analytic kernel** below
tracks it exactly instead of banding a measured table.

### Acceptance, and where the multiplicative kernel breaks

`A(m_pre) = P(selected | m_pre)` for `pT > 25` GeV, `|η| < 2.4` on the post-FSR
muons is a smooth rise from 0 at 50 GeV to 0.5 at 200 GeV, fitted by a degree-8
Bernstein to ±0.3 % over 70–120, and folded into the provider as `acceptance=` —
applied to the Born spectrum *before* the FSR fold, i.e. the factorisation
`P(m_post, pass) = p_born(m_pre) A(m_pre) K_sel(m_post|m_pre)`.

But `K_sel` is **not** `m_pre`-independent. Conditioning on the selection makes
⟨u⟩ run from 7.1e-3 (60–80 GeV) to 14.9e-3 (110–150 GeV) — a factor two, because
the `pT` cut removes hard emission at an `m_pre`-dependent rate; the shape ratio
between slices reaches 190 %. A single multiplicative kernel is therefore
*wrong* under a tight fiducial selection and the banded kernel is mandatory. The
looser selection the CVH production actually applies (`pT > 5`, `|η| < 2.4`) is
much better behaved: ⟨u⟩ = 19.5e-3 versus 21.4e-3 inclusive.

With the fiducial kernel and `A(m)` both in place the selected spectrum closes:

| model, fiducial `pT` > 25, \|η\| < 2.4, window 60–120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| no FSR fold, no `A(m)`, + `K(m)` | −29.34 ± 0.88 | +320.55 ± 1.82 |
| FSR folded, no `A(m)`, + `K(m)` | −0.05 ± 0.87 | +1.69 ± 1.75 |
| **FSR folded + `A(m)` (Bernstein 8), + `K(m)`** | **−0.31 ± 0.87** | **+2.61 ± 1.75** |
| … with the *inclusive* kernel instead | −3.48 ± 0.88 | +3.39 ± 1.81 |
| pre-FSR + `A(m)` control | −1.01 ± 0.78 | +1.86 ± 1.50 |

Bernstein degrees 4 / 6 / 8 / 10 agree to 0.2 / 0.6 MeV. At generator level
**`A(m)` is largely degenerate with the smooth `K(m)`** — dropping it entirely
changes the fit by 0.3 / 0.9 MeV — so what matters about `A(m)` is only whatever
part of it is *not* smooth.

### Reproducing the closure

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
# 1. gen dump: 550 MiniAOD files, 8 workers per node, ~35 min, resumable
$Z/run_gen_dump.sh   0 274 submit50 8 &
$Z/run_gen_dump.sh 275 549 submit51 8 &
# 2. merge (on a node with ceph)
python3 $Z/merge_gen.py -i "/ceph/submit/data/user/d/david_w/ZMass/zgen/gen_*.npz" \
        -o $Z/data/genmerged_full.npz
# 3. kernels and acceptance (numpy only)
python3 fit_gen.py kernel --gen data/genmerged_full.npz -o data/kern_incl_sc3.3e-4.npz
python3 fit_gen.py kernel --gen data/genmerged_full.npz -o data/kern_fid_sc3.3e-4.npz \
        --acc-pt 25 --acc-eta 2.4
python3 fit_gen.py acceptance --gen data/genmerged_full.npz -o data/acc_d8.json --degree 8
# 4. the three fit suites (~2-10 min each) and the figures
./run_tf_z.sh python3 -u fit_gen.py fit --gen data/genmerged_full.npz --suite prefsr \
        -o data/fit_prefsr.json
./run_tf_z.sh python3 -u fit_gen.py fit --gen data/genmerged_full.npz --suite postfsr \
        --kernel data/kern_incl_sc3.3e-4.npz --nm 8192 -o data/fit_postfsr2.json
./run_tf_z.sh python3 -u fit_gen.py fit --gen data/genmerged_full.npz --suite fiducial \
        --kernel data/kern_fid_sc3.3e-4.npz --acc data/acc_d8.json --nm 8192 \
        -o data/fit_fiducial.json
./run_tf_z.sh python3 -u plot_gen.py --gen data/genmerged_full.npz \
        --fits data/fit_prefsr.json --fits-post data/fit_postfsr2.json \
        --fits-fid data/fit_fiducial.json --kernel data/kern_incl_sc3.3e-4.npz \
        --kernel-fid data/kern_fid_sc3.3e-4.npz --acc data/acc_d8.json --nm 8192
```

---

## Analytic FSR kernel

`fsr_analytic.py` builds the same `(r_j, w_j, m_lo_j, m_hi_j)` atoms from a
closed-form QED radiator instead of from the Photos++ record, so the kernel is
a **function of the fitted mass** rather than a table measured at one generator
setting, and its mass dependence is exact by construction. `cmp_fsr.py` is the
validation against the generator; figures
`~/public_html/ZMass/cvh/260913_fsr_analytic/`.

### The radiator

`z = (m_post/m_pre)^2`, `x = 1-z`, `u = -ln(m_post/m_pre) = -(1/2) ln z`;
for a single emission `x = 2 E_gamma/m` in the Z rest frame.
`L(m) = ln(m^2/m_mu^2)`, `beta(m) = (2 alpha/pi)(L-1)` with the **on-shell**
`alpha = 1/137.035999` (the photons are real; the running of `alpha` belongs to
the propagator, not the emission vertex) and `m_mu = 0.1056583745` GeV.
`beta(91.19) = 0.058168`.

The exact O(alpha) spectrum, integrated over all angles at fixed `z`, is

```
R1(z; m) = (alpha/pi) (1+z^2)/(1-z) [ ln(z m^2/m_mu^2) - 1 ]                (1)
```

— the Altarelli-Parisi splitting function times the collinear logarithm
**evaluated at the outgoing pair mass** `s' = z m^2`, minus one. The `-1` is the
non-logarithmic part of the soft eikonal of a back-to-back pair; the `ln z` is
the *entire* non-logarithmic hard remainder. (Berends-Kleiss-Jadach,
Nucl. Phys. B202 (1982) 63; the ZFITTER final-state radiator, Bardin et al.,
Comput. Phys. Commun. 133 (2001) 229; Bardin-Passarino, *The Standard Model in
the Making*.)

`fsr_analytic.py validate` checks (1) against a **numerical evaluation of the
exact spin-summed matrix element** for `V* -> mu+ mu- gamma` — Dirac traces with
the exact muon mass, the transverse projector `-g + QQ/s` on the current indices
and `-g` on the photon index, quadrature over the muon direction with the
collinear region resolved by `1 -+ beta_mu cos = (1-beta_mu) e^t`:

| | z = 0.99 | z = 0.9 | z = 0.5 | z = 0.02 |
|---|---|---|---|---|
| `R_exact/R1 - 1`, m = 91.19 | −2.4e−7 | −5.4e−7 | −2.5e−6 | −5.9e−6 |
| `R_exact/R1 - 1`, m = 9.46 | −3.4e−5 | −6.3e−5 | −2.5e−4 | −6.2e−4 |
| `R_exact/R1 - 1`, m = 3.097 | −4.3e−4 | −7.1e−4 | −2.6e−3 | −8.9e−3 |
| `\|R_A/R_V - 1\|`, m = 91.19 | 5.5e−10 | 6.1e−8 | 2.2e−6 | — |

The deviation scales as `m_mu^2/s` (dividing `m_mu` by 30 divides it by 900), and
the soft limit `R_exact (1-z)/beta` reaches 1 to 1e−6. So at the Z the muon mass
terms and the vector/axial (i.e. `gamma*`/`Z`) decomposition are irrelevant at
the 1e−5 level — **the massless closed form (1) is exact for this purpose**, and
only at the J/psi do mass terms reach 1e−3.

### The kernels

Every variant is a probability density in `z` normalised to 1 (the O(alpha)
*rate* correction `3 alpha/4pi` is z- and m-independent and drops out).

**`exp1`, exponentiated exact O(alpha)** (YFS / Kuraev-Fadin):

```
K(z) = C beta (1-z)^{beta-1} + h(z)
h(z) = -(beta/2)(1+z) + (alpha/pi) (1+z^2) ln z / (1-z)                     (2)
C    = 1 - int_0^1 h dz = 1 + 3 beta/4 + (alpha/pi)(pi^2/3 - 5/4)
```

`h` is (1) minus the soft term `beta/(1-z)` the exponentiated factor already
supplies, so the O(alpha) expansion
`beta(1-z)^{beta-1} = delta(1-z) + beta[1/(1-z)]_+ + O(beta^2)` reproduces (1)
exactly. `int_0^1 (1+z^2) ln z/(1-z) dz = 5/4 - pi^2/3`.

**`exp2`, `exp1` + the O(alpha^2) leading log.** The LL radiator for the
pair-mass fraction is, in Mellin space, `R~(n) = exp[(beta/2) g(n)]` with
`g(n) = 3/2 - 2 S_{n-1} - 1/n - 1/(n+1)` the QED non-singlet anomalous dimension
(two radiating legs, each carrying `beta/2`). Its O(beta^2) term is
`(beta^2/8)[P (x) P](z)` with `P = [(1+z^2)/(1-z)]_+`, and the convolution is

```
[P (x) P](z) = (9/4 - 2pi^2/3) delta(1-z) + 6 [1/(1-z)]_+
             + 8 [ln(1-z)/(1-z)]_+
             + (1+z)[3 ln z - 4 ln(1-z)] - 4 ln z/(1-z) - 5 - z            (3)
```

(derived here from `P = 2[1/(1-z)]_+ - (1+z) + (3/2) delta(1-z)`; its Mellin
moments reproduce `g(n)^2` to 5e−13 for n = 1…8, and `int [P (x) P] = 0` as
probability conservation requires — this is the Kuraev-Fadin second-order
structure function, Sov. J. Nucl. Phys. 41 (1985) 466, written for the two-leg
exponent `beta`). **Only the regular part of (3) is new**: the exponentiated
factor already supplies `beta^2 [ln(1-z)/(1-z)]_+`, matching `(beta^2/8) x 8`
exactly, and the `3 beta/4` in `C` already supplies `(3 beta^2/4)[1/(1-z)]_+`,
matching `(beta^2/8) x 6` exactly. The `delta` coefficient follows from
`int K = 1`.

**`exp2nll`, `exp2` + the O(alpha^2) next-to-leading log** — every term of order
`alpha^2 L` (`alpha^2 L^2` is already in `exp2`). The pair-mass fraction
factorises, `z = z_+ z_-`, so in Mellin space the kernel is the *square* of the
muon's **time-like (fragmentation) QED structure function** `D(z; L)`, which
evolves with `a = alpha/2pi` as `dD/dL = a P0 (x) D + a^2 P1T (x) D` from
`D(z; 0) = delta(1-z) + a C1 + ...`. Solving to O(a^2) and squaring,

```
K~(n) = exp[ (alpha/pi)(L g0 + c1) + (alpha/pi)^2 (L/2) g1 ] + NNLL        (5)
```

(lower case = Mellin moment of the corresponding capital). The O(alpha) term of
(5) *is* the exact spectrum (1), which therefore **defines** the O(alpha)
coefficient function: with `p(z) = (1+z^2)/(1-z)` and
`P0 = [p]_+ = 2[1/(1-z)]_+ - (1+z) + (3/2) delta(1-z)`,

```
C1(z)   = [ p(z) (ln z - 1) ]_+ = Chat(z) - P0(z)                          (6)
Chat(z) = p(z) ln z + (pi^2/3 - 5/4) delta(1-z)
```

and the O(alpha^2) term of (5) is
`(alpha/pi)^2 { (L^2/2) P0 (x) P0 + L [ P1T/2 + P0 (x) C1 ] }`. `exp2` already
carries the **whole** `L^2` coefficient and part of the `L` one: its O(alpha^2)
content is exactly
`(beta^2/8) P0 (x) P0 + (alpha/pi) beta (pi^2/3 - 5/4) [1/(1-z)]_+`, and because
`beta = (2 alpha/pi)(L-1)` the `(L-1)^2` of the first piece supplies
`-(alpha/pi)^2 L P0 (x) P0`, which is precisely the `-P0 (x) P0` inside
`P0 (x) C1`. The remainder — the whole of `exp2nll` — is

```
Delta K(z) = (alpha/pi)^2 L [ G(z) + g_delta delta(1-z) ]                  (7)
G(z) = 3 p(z) ln z ln(1-z) + [ 9/(2(1-z)) - 5 - 2z ] ln z
     + [ (11/4)(1+z) - 4/(1-z) ] ln^2 z - (5/2)(1-z)
     - (pi^2/3 - 5/4)(1+z) - p(-z) S_2(z)
g_delta = P1_delta/2 + (3/2)(pi^2/3 - 5/4) = 3 zeta_3 + pi^2/4 - 27/16
        = 4.386072 = - int_0^1 G dz
```

so `int Delta K dz = 0` analytically and the normalisation is untouched — no
numerical rescale anywhere. `G = P1T/2 + A - (pi^2/3 - 5/4)(1+z)`, built from:

* **`P1T`, the two-loop time-like non-singlet splitting function in its abelian
  part** (`C_F^2 -> 1`, `C_A = 0`, `n_f = 0`), normalised as
  `P = a P0 + a^2 P1`:

  ```
  P1T(z) = 2 p ln z ln(1-z) + [ 3/(1-z) - 7 - 5z ] ln z
         + [ (5/2)(1+z) - 4/(1-z) ] ln^2 z - 9(1-z) - 2 p(-z) S_2(z)
         + (3/8 - pi^2/2 + 6 zeta_3) delta(1-z)                            (8)
  p(-z) = 2/(1+z) - 1 + z
  S_2(z) = ln^2 z/2 - pi^2/6 - 2 Li_2(-z) - 2 ln z ln(1+z),  S_2(1) = 0
  ```

  It is the **valence (C-odd) combination** `P_NS,- = P_qq,V - P_qqbar,V`: the Z
  couples to the C-odd vector current and what is counted on the `mu-` leg is
  `mu - mubar`, and only this combination obeys `int P dz = 0`, i.e. muon-number
  conservation, which the photonic sector must satisfy exactly. The `S_2(z)`
  term is the **crossed-photon** interference in `mu -> mu gamma gamma`
  (`T^a T^b T^a T^b = C_F(C_F - C_A/2) -> 1` in QED), not a pair effect. The
  space-like kernel is Curci-Furmanski-Petronzio (Nucl. Phys. B175 (1980) 27)
  and Floratos-Kounnas-Lacaze (Nucl. Phys. B192 (1981) 417); its QED form is
  de Florian-Sborlini-Rodrigo, JHEP 10 (2016) 056, eqs. (57), (58), (63), (64).
  The time-like difference is the Drell-Levy-Yan / Gribov-Lipatov-violating term

  ```
  P1T - P1S = 2 [ ln z P0 ] (x) P0                                         (9)
  ```

  (Curci-Furmanski-Petronzio; Mitov-Moch-Vogt, Phys. Lett. B638 (2006) 61, whose
  `tlike-ns.h` gives `diffP1ns` = 4 x (9) in the `alpha_s/4pi` normalisation).

* **`A(z) = (P0 (x) Chat_reg)(z)`**, the convolution needed for `P0 (x) C1`, is
  the *same* object: `Chat_reg = p ln z`, so `A = ([p]_+ (x) [p ln]) =
  (P1T - P1S)/2`, and its closed form is

  ```
  A(z) = 2 p ln z ln(1-z) + (3/2) p ln z - p ln^2 z
       + (z-1) ln z + (1+z) ln^2 z / 2                                    (10)
  ```

**Scheme.** The abelian kernel above is the complete *photonic* two-loop
splitting function of QED: `C_F C_A` has no QED analogue, and the `n_f T_F`
terms need a real fermion pair. Those belong to the pair sector, which the
kernel carries **exactly at O(α²)** through `pair_radiator` — nothing is double
counted, and the `n_f` sector is then complete to the same order as the
photonic one.

**`oalpha`**, fixed-order O(alpha) with a soft cutoff `x_cut`: a delta at `z = 1`
carrying `1 - P(x > x_cut)` plus (1) above it. Not a model — it measures the
size of the exponentiation.

**Pair emission.** A virtual photon of mass² `q^2` radiated off the muon line
converts to a fermion pair. The photon propagator with one self-energy
insertion, cut, is exactly a dispersive integral over the emission of a **vector
of mass² `q^2`** carrying the same coupling `e`:

```
R_pair(z; s) = int (dq^2/q^2) rho(q^2) R_gamma*(z; q^2, s)                  (4)
rho_lepton   = (alpha/3pi) (1 + 2 m_l^2/q^2) sqrt(1 - 4 m_l^2/q^2)
rho_had      = (alpha/3pi) R(q^2)
```

(Kniehl, Krawczyk, Kühn, Stuart, Phys. Lett. B209 (1988) 337; the same
statement in a form that can actually be read is Hoang and Teubner,
Nucl. Phys. B519 (1998) 285, hep-ph/9707496 eq. (37), and Hoang, Kühn and
Teubner, hep-ph/9505262 eq. (1).) `R_gamma*` is
the **exact** spin-summed matrix element of `V* -> mu+ mu- gamma*(q^2)` — both
attachments and their interference, exact `m_mu`, photon polarisation sum `-g`,
which equals `-g + kk/q^2` because the muon emission current is conserved
(checked to 1e−15) — evaluated with the same Dirac-trace machinery as (1) and
reproducing it to 6e−12 as `q^2 -> 0`. (4) is therefore exact at O(α²) for
everything except the *singlet* channel, in which the observed muon pair is not
the one the current produced; that is a background to the dimuon spectrum, not
FSR, and its rate with the pair mass inside a 60-120 GeV window is **~6e−9 per
event**.

The mass loss of a pair of mass `q` and energy fraction `x = 2E/sqrt(s)` is
`1 - z = x - q^2/s`, and the massive-photon phase space closes at
`q^2 < s (1 - sqrt z)^2`, so the pair spectrum has a hard threshold at
`1 - sqrt z = 2 m_l/m` and **no soft singularity at all**. It is therefore not
exponentiated; the kernel is the convolution

```
K = K_photonic (x) [ (1 - N_pair) delta(1-z) + R_pair(z) ]                  (5)
```

carried out on the atoms, so mean mass losses add exactly and the photon-pair
cross term (worth 17 % of the pair `<u>`) is kept.

At `m` = 91.19 GeV, with `<u>` the *unconditional* mean mass loss per event:

| species | `N_pair` | `<u>` | % of the photonic radiator |
|---|---|---|---|
| `e⁺e⁻` | 2.530e−3 | 2.899e−4 | +1.10 % |
| `μ⁺μ⁻` | 3.309e−4 | 8.015e−5 | +0.30 % |
| `τ⁺τ⁻` | 3.209e−5 | 1.413e−5 | +0.05 % |
| hadrons | 6.472e−4 | 1.826e−4 | +0.69 % |
| **all** | **3.541e−3** | **5.668e−4** | **+2.15 %** |

**The dispersive leading log is 46 % too high for `e⁺e⁻`** (34 % for `μ⁺μ⁻`,
71 % for `τ⁺τ⁻`, and 36 % *too low* for hadrons, where its crude `R` dominates
the error). Replacing `R_gamma*` by
`(alpha/pi) (1+z^2)/(1-z) [ln(s/q^2) - 1]` — the photon radiator with its
collinear log moved from `m_mu^2` to `q^2`, which is what `beta_pair_ll` still
computes as the reference — gives `<u>` = 4.239e−4 (`e`), 1.073e−4 (`μ`),
2.412e−5 (`τ`) against the exact 2.898e−4 / 8.015e−5 / 1.413e−5. Three things
are wrong with it, in order of size:

* for `q^2 < m_mu^2` — most of the `dq^2/q^2` range of an `e⁺e⁻` pair — the
  collinear log is cut off by the **muon** mass and saturates at `L`. In ladder
  language the photon's transverse momentum must exceed `m_mu^2` before the
  conversion can happen, so the double log is `(Y^2 - Δ^2)/2` with
  `Y = ln(s/4m_e^2)` and `Δ = ln(m_mu^2/4m_e^2)`, not `Y^2/2`: **−18 %**;
* it has no `ln z`, the non-logarithmic hard remainder the photon radiator (1)
  carries (and which alone is worth −10 % there too): **−10 %**;
* it ignores the massive-photon phase-space limit, which removes the whole
  `z -> 1` region: **−7 %**.

The first item is the one the literature hides. The standard O(α²) pair
radiator is written with a single log `L_l = ln(s/m_l^2)` of the *emitted* pair,
because in `e+e-` annihilation the emitted pair (`mu`, `tau`, hadrons) is always
heavier than the radiator. Here it is the other way round, and the correct
leading log carries both masses,

```
(L_l^2 - Delta^2)/2 = L_R L_l - L_R^2/2 ,
L_R = ln(s/m_mu^2) ,   Delta = L_l - L_R = ln(m_mu^2/m_l^2)
```

— one log from the photon emission off the muon, one from the conversion, minus
the region `q^2 < m_mu^2` where the emission log is quenched. Hoang, Kühn and
Teubner (Nucl. Phys. B452 (1995) 173, hep-ph/9505262) computed exactly this
configuration — a light pair radiated off a *heavy* fermion, and the formula
ZFITTER uses for final-state pair corrections — and their leading logs are
`rho^R = (L_l^3 - Delta^3)/18` real and `-(L_l^3 - Delta^3)/36` virtual, whose
`Delta = 0` case is the familiar Burgers `-(1/36) L^3` that the general-purpose
codes carry. At the Z the naive `L_l^2/2` overstates the `e+e-` term by 24 %
and the `tau+tau-` term by a factor 2.

`R(q^2)` matters as much as any of these, and in the other direction: the old
step model (`R = 2` above 1 GeV²) is half of the true `int R dln q^2`, so the
old hadronic term was 26 % *low* even before the three errors above. The model
here is the PDG parton-model continuum `3 sum Q_q^2 (1 + alpha_s/pi)` with the
**physical** open-flavour thresholds (`2 m_D0` = 3.73 GeV and `2 m_B` =
10.56 GeV; using `2 m_q` instead overshoots by 2.4 %), a flat `R = 3.40` across
the open-charm region, a linear ramp onto the non-resonant plateau between 1 and
1.5 GeV, and the nine narrow vector resonances (`rho`, `omega`, `phi`, `J/psi`,
`psi(2S)`, `Y(1S-4S)`) as discrete `q^2` nodes of weight
`(9 pi/alpha^2) Gamma_ee/M`, which is exact for `Gamma << M` and follows from
`int sigma_had ds = 12 pi^2 Gamma_ee/M`. That gives `int R dln q^2` = **36.0**,
against **35.6 ± 0.2** implied by `Delta alpha_had^(5)(m_Z^2)` = 0.02766 ±
0.00007 — the dispersion kernel `s_0/(s_0-s)` differs from a plain `dln s`
measure by only 0.09 % once the integral is truncated at `s = m_Z^2`, the 12 %
enhancement below the Z cancelling what the truncation drops above it. The
residual +1 % is the vacuum-polarisation-dressed `Gamma_ee` of the resonances
and the narrow-width formula applied to the `rho`; it is the dominant
uncertainty of the hadronic pair term. (PDG *Quantum Chromodynamics* review
eqs. (9.7)-(9.9) for the continuum, KNT19 arXiv:1911.00367 and DHMZ19
arXiv:1908.00921 for the sub-2 GeV region.)

The exact `R_pair(z; m)` costs a few hundred ms per `z`, so the kernel reads a
table of `B(u; m) = R_pair(z) / [(alpha/pi)(1+z^2)/(1-z)]` — 13 masses ×
200 log-spaced `u` × 4 species, linear in `ln m` — built once by
`fsr_analytic.py pairtable`. Dividing out the Altarelli-Parisi pole is what
makes the interpolation accurate at both ends; `fsr_analytic.py pair` checks
the table's rate and first moment against a direct quadrature done in the other
order (over the emitted energy at fixed `q^2`) and closes to 1e−4.

### Discretisation

The provider's fold is a midpoint quadrature, so each atom sits at the *exact*
conditional mean of `u` in its group and the residual is
`w_j Var(u|group) m^2 |p''| / 2`. Groups grow while `w_j Var_j` stays below
`--var-budget` (default 6e−10, ~420 atoms per band). Both the weights and the
conditional moments come from Gauss-Legendre quadrature in `t = (1-z)^beta`, the
substitution that removes the `(1-z)^{beta-1}` endpoint singularity exactly
(`dt = beta (1-z)^{beta-1} dz`, so the integrand is bounded); the integrand is
evaluated as `C + (h+q2) x^{1-beta}/beta` rather than as `K |dz/dt|` because `x`
reaches 1e−100 near `t = 0`, where the two factors individually overflow and
underflow while their product does not. `<u>` is stable to 1e−11 against the
panel count (500…8000) and the Gauss order (8…32).

Banding in `m_pre` is free for an analytic kernel: 75 bands of 2 GeV over
50-200 GeV, each with its own `beta(m)`, ~32 k atoms.

### Validation of the O(alpha^2) NLL term

`fsr_analytic.py nll` prints every check below; figures
`~/public_html/ZMass/cvh/260913_fsr_nll/`.

| check | residual |
|---|---|
| Mitov-Moch-Vogt `diffP1ns` (HPLs, with `Li_2`) vs `8 A(z)` of (10) | 1.4e−13 |
| `A(z)` vs a direct numerical `([p]_+ (x) [p ln])(z)`, z = 1e−5…0.9 | 1.0e−14 |
| `int P1S dz + delta` and `int P1T dz + delta` (muon number, NS_−) | 3.6e−15 |
| `int A dz`, `int G dz + g_delta` | 1.8e−15 |
| `M[P1S](n) + gamma^(1)_{NS,−}(n)/4`, n = 1…8 (harmonic sums) | 8.4e−15 |
| `L^2` coefficient of `int z^{n-1} K dz` − `g0(n)^2/2`, n = 1…8 | 5.3e−15 |
| `L` coefficient − `[ g1(n)/2 + g0(n) c1(n) ]`, n = 1…8 | 1.4e−14 |
| `int K dz − 1` at `m_Z`, unclipped quadrature | 8.8e−8 |
| `<u>` of the atoms, `n_fine` 4000→16000, `ng` 16→32 | 1.2e−12 |

The Mellin check is the decisive one: the `L^2` coefficient of the **full**
kernel reproduces `P0 (x) P0 / 2` and the `L` coefficient reproduces the
complete NLL combination `P1T/2 + P0 (x) C1`, moment by moment, to 1e−14. The
anomalous-dimension row is against `gamma^(1)_{NS,−}` of Moch-Vermaseren-Vogt
(Nucl. Phys. B688 (2004) 101) eq. (3.6) in harmonic-sum form, which fixes the
`delta(1-z)` coefficient `3/8 − pi^2/2 + 6 zeta_3` independently of the sum rule.
`(1-z) P1T -> 7e−7` at `1-z = 1e−8`: the **abelian two-loop cusp is zero**, so
the NLL term has no `[1/(1-z)]_+` at all.

Its only exponentiable piece is therefore the `delta(1-z)`, and (7) puts it
inside `C`, i.e. under the exponentiated soft factor. Adding it as an explicit
atom at `z = 1` instead changes the kernel by
`c_nll g_delta [ beta (1-z)^{beta-1} - delta(1-z) ] = c_nll g_delta beta
[1/(1-z)]_+ + O(alpha^4)` — **O(alpha^3)**, the same O(alpha^2) expansion, worth
−5.6e−4 on `<u>`, i.e. 3.4 % of the NLL term's own effect.

### What the NLL term does

`c_nll = (alpha/pi)^2 L = 7.29e−5` at the Z. Ratios `exp2nll / exp2`
(`00_nll.txt`):

| `m_pre` band | 50-60 | 70-80 | 86-96 | 110-130 | 150-200 |
|---|---|---|---|---|---|
| `<u>` | 0.98325 | 0.98342 | 0.98351 | 0.98363 | 0.98379 |
| `<u^2>` | 0.94936 | 0.95017 | 0.95061 | 0.95120 | 0.95199 |
| `<1-z>` | 0.99154 | 0.99162 | 0.99167 | 0.99172 | 0.99180 |

| `u` | 1e−4 | 1e−3 | 1e−2 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| density ratio, peak band | 1.00032 | 1.00041 | 1.00076 | 1.00105 | 1.00065 | 0.99894 | 0.99192 | 0.97723 | 0.91264 |
| `P(u > u_0)` ratio | 0.99946 | 0.99916 | 0.99830 | 0.99613 | 0.99379 | 0.98935 | 0.97621 | 0.94895 | — |

**The NLL term is a +0.3…+1.1e−3 effect on the density where the fit lives**
(`u < 0.1`, i.e. `m_post > 0.9 m_pre`) and a few-per-cent one in the hard tail;
the unradiated fraction moves by +3.1e−4 and the mass-dependence handle
`<u>(110-150)/<u>(60-80)` by 3e−4 (1.11062 → 1.11095), so it does not touch what
the `beta(m)` banding is for. `<u>` moves by −1.65 % only because `<u>` is
dominated by the tail; restricted to `u < 2` the shift is −1.12 %.

**Where the fixed order stops working.** `G(z)` carries `-(7/4) ln^2 z` at small
`z`, and `(α/π) L ln z = −0.36` at `z = 1e−5`: the NLL truncation is not sufficient
once the effective collinear log `ln(z m^2/m_mu^2)` closes, and the `alpha^2 L^2`
term of `exp2` is no better there (it doubles the kernel at the two-muon
threshold). `exp2nll` goes negative for `u > 4.85` at the Z (`z < 6.2e−5`) and is
clipped to zero; that region carries `P = 5.5e−7` of the kernel and the clipping
costs 1.9e−7 of the norm.

### Validation against Photos++

The sample's Photos++ 3.61 configuration was read out of the installed library
and confirmed in the event record (`data/generator_settings_*.md`): **soft-photon
exponentiation ON** (`IEXP = 1`, unbounded multiplicity — up to 7 photons per
event survive in `prunedGenParticles`), **`XPHCUT = 1e-7`** on `x = 2 E_gamma/M`
(measured directly: the single-photon `E_gamma/m_ll` edge converges to 5.0e−8 =
XPHCUT/2 as the Z boost is removed, i.e. `E_gamma > 4.6` keV at the Z),
`alpha = 1/137.036`, **the exact O(alpha) Z matrix-element correction OFF**
(`meCorrectionWtForZ = false`), **real pair emission OFF** (`IfPair = false`),
and Pythia's QED shower off leptons disabled by the interface. Since
`1-z = 2 E_gamma/m` exactly for one emission, `x_cut = 1e-7` is directly
comparable with the model.

**Unradiated fraction** — a parameter-free test of the exponentiated soft factor.
The MC's `npre == 0` flag (no status-746 pre-Photos muon copy; those events have
`u <= 6.8e-7`, pure MiniAOD float noise) against the model's `P(1-z < 1e-7)`:

| `m_pre` band | 50-60 | 70-80 | 86-96 | 110-130 | 150-200 |
|---|---|---|---|---|---|
| MC | 0.44254 | 0.42172 | 0.41029 | 0.39514 | 0.37551 |
| exp. O(alpha) | 0.44180 | 0.42169 | 0.41058 | 0.39579 | 0.37575 |
| ratio | 1.0017 | 1.0001 | 0.9993 | 0.9984 | 0.9994 |
| O(alpha), no exp. | 0.18409 | 0.13758 | 0.11094 | 0.07435 | 0.02251 |

**0.2 % or better in every band**, with no free parameter — and the fixed order
is wrong by a factor 2.3 to 17.

**Spectrum.** In the peak band the density `(1/N) dN/du` agrees over four decades
in `u` (`01_u_density`), and the IR-safe tail `P(u > u_0)` (`02_u_tail`) gives

| `u_0` | 1e−5 | 1e−4 | 1e−3 | 1e−2 | 5e−2 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| MC / exp1 | 1.0044 | 1.0066 | 1.0100 | 1.0155 | 1.0189 | 1.0126 | 0.9961 | 0.9840 |
| MC / exp2 | 1.0022 | 1.0036 | 1.0054 | 1.0086 | 1.0120 | 1.0168 | 1.0187 | 1.0128 |
| MC / O(alpha) | 0.7629 | 0.8134 | 0.8699 | 0.9354 | 0.9891 | 1.0409 | 1.0730 | 1.0966 |

The MC sits **0.4-1.9 % above** the exponentiated exact O(alpha) in the region
that matters, with a shape that is not a rescaling of `beta`. That is the size
and the character of the missing exact-ME correction in Photos, not a defect of
(1)-(3): the O(alpha^2) NLL terms the model does drop are `alpha/(2pi)` =
1.2e−3 of the radiator, ten times too small to account for it.

**Mass dependence.** `<u>` per `m_pre` band (`03_mpre_mean_u`, `06_mpre_mean_x`,
`00_moments.txt`): MC/exp1 is 1.025 at 50-60 GeV, 1.008 at the peak, 1.022 at
130-150 GeV — flat to ~1.5 % over a factor 3 in mass, while `beta` itself moves
by 16 %. The ratio the fold is actually sensitive to,

```
<u>(110-150) / <u>(60-80):   MC  1.1021 +- 0.0079     exp. O(alpha)  1.1019
```

agrees to 0.02 %; the naive collinear-log ratio `(L-1)|_{138.6}/(L-1)|_{75.6}` =
1.0912 is 1 % off, because `<u>` is not linear in `beta`. Freezing `beta` at
`m_Z` makes `<u>` flat and is wrong by 12 % at 50 GeV. The per-band *shape*
ratio (`07_band_shape`) is flat to a couple of per cent in both the 60-80 and
110-150 bands.

### The fit-level test

`fit_gen.py fit --suite postfsr`, 27.7 M gen events in 60-120 GeV, 5 Legendre
shape terms, `nm = 8192`; offsets from the generator's own
`m_Z = 91.153510`, `Gamma_Z = 2.493202` (constant-width scheme).

| kernel | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| no FSR fold at all | −227.12 ± 0.48 | +741.35 ± 1.19 |
| **empirical, inclusive** (`kern_incl_sc3.3e-4.npz`) | **+0.15 ± 0.56** | **+1.31 ± 1.14** |
| empirical, banded 5 GeV | +0.60 ± 0.55 | −0.21 ± 1.13 |
| analytic exp. O(α), banded 2 GeV | +1.25 ± 0.54 | +6.35 ± 1.13 |
| analytic exp. O(α), single band | +0.89 ± 0.54 | +6.33 ± 1.13 |
| analytic exp. O(α) + O(α²)LL, banded | +1.07 ± 0.54 | +2.67 ± 1.13 |
| analytic exp. O(α) + O(α²)LL, single band | +0.70 ± 0.54 | +2.65 ± 1.13 |
| … `β` frozen at `m_Z` | +0.86 ± 0.54 | +2.52 ± 1.13 |
| … `L` instead of `L−1` in `β` (+8.0 % on `β`) | +5.60 ± 0.55 | −24.27 ± 1.13 |
| analytic exp. O(α) + O(α²)LL **+ O(α²)NLL**, banded | +1.17 ± 0.54 | +2.34 ± 1.13 |
| … + exact `e⁺e⁻` pairs | +1.79 ± 0.54 | −0.28 ± 1.13 |
| … + exact `e`, `μ` pairs | +1.63 ± 0.54 | −0.51 ± 1.13 |
| **… + exact `e`, `μ`, `τ`, hadron pairs** | **+1.38 ± 0.54** | **−1.04 ± 1.13** |
| … + `e`, `μ` pairs in the **eikonal** limit (= Photos) | +1.69 ± 0.54 | −0.93 ± 1.13 |
| … + `e`, `μ`, `τ`, hadron pairs, dispersive **leading log** | +2.35 ± 0.54 | −5.16 ± 1.13 |
| O(α), **no exponentiation**, `x_cut` = 1e−7 | +24.94 ± 0.54 | −39.12 ± 1.13 |

Kernel-to-kernel **differences** are far more precise than the rows themselves —
same events, same nuisances, so the statistical fluctuation cancels. Repeating
the whole comparison on half the sample and taking `|Δ_half − Δ_full|` (whose
variance is exactly the variance of `Δ_full`) gives:

| O(α²) NLL, on top of | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| `exp2`, no pairs | **+0.103 ± 0.003** | **−0.332 ± 0.003** |
| `exp2` + `e`, `μ`, `τ`, hadron pairs | +0.074 ± 0.002 | −0.299 ± 0.002 |

Read like for like — banded analytic against banded empirical, single-band
analytic against inclusive empirical — **the analytic kernel reproduces the
MC-derived one to 0.5 MeV on `m_Z`** (+0.47 banded, +0.55 unbanded) and, with
the O(α²)LL term, to 1.3-2.9 MeV on `Γ_Z`. The empirical reference carries
±0.7 MeV of its own `sigma_cap` quadrature bias, so the two are equivalent
within the precision of the comparison. The residual is the same +1.5 % of
radiative strength the tail table shows, and it points at Photos (exact-ME
correction off), not at the analytic form.

What each ingredient is worth on `m_Z`:

| ingredient | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| the fold itself | −227 | +741 |
| exponentiation vs fixed order O(α) | **23.7** | −45.5 |
| O(α²) LL | −0.18 | −3.67 |
| the mass dependence of `β` | 0.21 | 0.15 |
| exact `e⁺e⁻` pairs | 0.62 | −2.6 |
| exact pairs, all species | 0.21 | −3.4 |
| … as the dispersive leading log instead | 1.18 | −7.5 |
| … in the eikonal limit instead (`e`, `μ`; = Photos) | 0.52 | −3.3 |
| ±1 % on `β` (from the `L` vs `L−1` slope) | **0.57** | −3.3 |
| O(α²) NLL | **+0.103 ± 0.003** | **−0.332 ± 0.003** |
| … its additive-vs-exponentiated O(α³) ambiguity | 0.004 | 0.011 |

**A relative error `ε` on the radiative strength moves `m_Z` by `ε × 57` MeV,
not `ε × 250` MeV**: the floating 5-term smooth `K(m)` absorbs most of a uniform
rescaling of the radiator (it also turns the −227 MeV of "no fold at all" into
−31 MeV). `Γ_Z` is 5.8× more sensitive, `ε × 330` MeV.

Numerics, on the banded exp. O(α) kernel: band width 1 / 2 / 5 GeV gives
+1.32 / +1.25 / +1.44 MeV; `--var-budget` 6e−11 / 6e−10 / 6e−9 gives
+1.10 / +1.25 / +1.33; truncating the support at `u = 2` instead of the two-muon
threshold gives +1.13. Every discretisation choice is inside ±0.25 MeV, i.e.
below the statistical error of the closure.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
python3 fsr_analytic.py validate                       # exact ME vs eq. (1)
python3 fsr_analytic.py nll                            # every O(alpha^2) check
python3 fsr_analytic.py moments --u-cut 0.113013       # inclusive + windowed
python3 fsr_analytic.py pairtable --procs 40           # the exact pair table
python3 fsr_analytic.py pairtable --procs 40 --eikonal \
        -o data/fsr/pairkern_eik.npz                   # what Photos generates
python3 fsr_analytic.py pair                           # every pair check
python3 fsr_analytic.py kernel -o data/fsr/kan_exp2nll_pairx_all.npz \
        --variant exp2nll --pair e mu tau had          # the recommended kernel
python3 fsr_analytic.py kernel -o data/fsr/kan_exp2nll.npz --variant exp2nll
./run_tf_z.sh python3 -u fit_gen.py fit --gen data/genmerged_full.npz \
        --suite postfsr --kernel data/kern_incl_sc3.3e-4.npz --nm 8192 \
        --kernel-alt data/fsr/kan_*.npz -o data/fsr/fit_postfsr_analytic.json
./run_tf_z.sh python3 -u fit_gen.py fit --gen data/genmerged_full.npz \
        --suite postfsr --kernel data/fsr/kan_exp2.npz --nm 8192 \
        --kernel-alt data/fsr/kan_exp2.npz data/fsr/kan_exp2nll.npz \
        data/fsr/kan_exp2nll_pair_all.npz data/fsr/kan_exp2_pair_all.npz \
        -o data/fsr/fit_nll.json          # add --nmax 14637131 for the half
./run_tf_z.sh python3 -u cmp_fsr.py --gen data/genmerged_full.npz
./run_tf_z.sh python3 -u cmp_fsr.py --nll              # the NLL figures
```

### Narrow resonances

The same radiator at the J/psi and the Upsilon, where the mass terms of the
exact matrix element reach 1e−3 and 6e−4 (table above) and should be kept if
these numbers are ever needed below the per-mille level:

| | `β` | `<u>` incl. | `P(u > 0.113)` | window | `<u \| in window>` | mean mass shift |
|---|---|---|---|---|---|---|
| J/psi, 3.0969 | 0.026740 | 10.80e−3 | 0.0241 | ±0.35 GeV | 2.596e−3 | **−8.04 MeV** |
| Υ(1S), 9.4603 | 0.037115 | 16.12e−3 | 0.0349 | ±0.35 GeV | 1.270e−3 | **−12.01 MeV** |
| Υ(1S), 9.4603 | 0.037115 | 16.12e−3 | 0.0349 | ±0.60 GeV | 2.117e−3 | −20.02 MeV |
| Z, 91.1876 | 0.058168 | 26.98e−3 | 0.0568 | — | — | — |

`<u | in window>` is the mean of `u` over the events that stay inside the
selection window; `-m <u | in window>` is therefore what an unmodelled FSR
kernel costs a *mean*-based mass estimator, and an upper bound on what it costs
a peak-based one. 2.3 % of J/psi decays radiate out of a ±0.35 GeV window
altogether. Against a `delta` at the PDG mass this is a **−2.6e−3** relative
scale shift at the J/psi and **−1.3e−3** at the Upsilon — both far above the
1e−5 target, so the J/psi term cannot keep a `delta` lineshape once the MC it is
fitted against radiates (`../fullscale/SUMMARY.md`, open item 2).

### Recommendation for the data likelihood

* Use `exp2nll` + pair emission (`e`, `mu`, `tau`, hadrons), banded at 2 GeV. It
  is first-principles throughout — no fitted constant, no scale factor — it is
  complete through O(α²) NLL, and its mass dependence is exact, which the
  tabulated kernel's cannot be.
* Quote as theory systematics: the O(α³) truncation (~`β/2` of the O(α²) terms,
  i.e. ≲0.1 MeV on `m_Z`), the hadronic pair term's `R(q^2)` model (4 % of the
  hadronic `<u>`, 0.01 MeV on `m_Z` and 0.1 MeV on `Γ_Z`), and the
  additive-vs-exponentiated O(α³) ambiguity of the NLL term (0.004 MeV). The
  `L` vs `L−1` row is a **sensitivity slope, not an uncertainty**: `β` is known
  exactly at O(α).
* Do **not** take the +0.5 MeV difference against the Photos kernel as a
  systematic on the analytic one. Photos in this sample runs with the exact
  O(α) Z matrix-element correction switched off and with no pair emission —
  exactly the two pieces the analytic kernel supplies.
* **The analytic kernel is the *unconditional* kernel** `p(m_post | m_pre)`. It
  does not and cannot describe the selection-conditional kernel `K_sel`: a
  lepton `p_T` cut removes hard emission at an `m_pre`-dependent rate (⟨u⟩ runs
  from 7.1e−3 to 14.9e−3 across the Born range under `p_T > 25`, and is 19.5e−3
  vs 21.4e−3 inclusive under the CVH production's `p_T > 5`). The factorisation
  to use is
  `P(m_post, pass) = p_born(m_pre) A(m_pre) K(m_post|m_pre) x [K_sel/K](u, m_pre)`,
  with the analytic `K` carrying all the QED and only the *ratio* `K_sel/K`
  taken from MC — a geometry statement, not a QED one, and therefore insensitive
  to the generator's radiative accuracy. That ratio, not the kernel, is now the
  leading FSR-related modelling issue for a data Z channel.

---

## Photos++ standalone and the two kernel configurations

`photos_standalone/` runs **Photos++ 3.61 outside CMSSW** — the same build
`CMSSW_10_6_26` links,
`/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/photospp/3.61-pafccj` — on
`q qbar -> Z -> mu+ mu-` events generated at the sample's own `m_pre`
distribution. It answers what the sample alone cannot: what Photos does at
unlimited statistics, and what each of its switches is worth. `cmp_photos.py`
is the comparison; figures `~/public_html/ZMass/cvh/260913_fsr_photos/`.

### What the sample actually ran

The EDM provenance shows only `parameterSets = {'Photospp'}` because the
`Photospp` PSet is a `cms.untracked.PSet` and untracked parameters are not
stored — **not** because the configuration was left at the Photos defaults.
The configuration is the GEN request's fragment
(`SMP-RunIISummer20UL16wmLHEGEN-00496`), and every non-default switch in it is
visible in the generated events:

| switch | sample | Photos 3.61 default | evidence in the sample's gen record |
|---|---|---|---|
| `setExponentiation` | True | True | photon multiplicity under the Z reaches 7; 5.9 % of events have ≥ 3 |
| `setInfraredCutOff` | **1e-7** | 0.01 | minimum photon energy in the Z rest frame 2.6e−6 GeV; 75.4 % of photons below the default cutoff `0.01·m_Z/2` |
| `setMeCorrectionWtForZ` | **True** | False | the closure below |
| `setPairEmission` | **True** | False | 2.430e−3 of events carry **exactly two** status-1 `e±` under the Z (never one) and 3.066e−4 **exactly four** muons (never three); the pair masses start at 1.028 and 216.5 MeV against `2m_e` = 1.022 and `2m_mu` = 211.3 |
| `setPhotonEmission` | True | True | |
| `setStopAtCriticalError` | False | True | |
| `suppressAll` + `forceBremForDecay(23, ±24)` | on | off | 100.00 % of the status-746 Photos history entries sit under the Z branch |

`setMomentumConservationThreshold(0.1)` is a HepMC-only knob; on the HEPEVT path
the tolerance is hard-coded at 1e−4.

Four traps in the Photos 3.61 API, all read out of the upstream source
(`PHOTOS.3.61`; `cmsdist` applies no patches, so upstream == the cvmfs build):

* **`Photos::initialize()` overwrites settings.** It re-applies
  `setExponentiation(true)` — which resets `xphcut` to 1e−7, `isec`/`itre` to 0
  and the kinematic corrections to `PHCORK(5)` — and unconditionally re-applies
  `maxWtInterference(2.0)`. Anything it touches has to be set **after** it.
* **`phokey.fint` is an envelope, not physics.** The crude emission probability
  is multiplied by it (`photosC.cxx:1928`) and the accept/reject weight divided
  by it (`photosC.cxx:2328`); Photos aborts when the ratio still exceeds 1,
  which the Z matrix-element correction (weight up to 2.12) and pair emission
  (2.04) do at the default 2.0. `--fint=8` clears both, and the kernel is
  independent of it to 0.07 %, below its 0.09 % statistical error.
* `Photos::iniInfo()` prints *"emission of photons is inactive"* when photon
  emission is **active** — `if(IfPhot)` where `initialize()` correctly has
  `if(!IfPhot)`. A display bug; ignore the line.
* Photos is a static singleton and is not thread safe: parallelism is by
  separate processes.

### The exact-ME correction fires on 39 % of the sample, not all of it

`HEPEVT_struct::check_ME_channel()` applies the Z correction only when the
decaying particle has **exactly two mothers, of opposite sign, both with
`|pdgId|` in 1–6 or 11–16**. The mothers come from
`PhotosParticle::findProductionMothers()`, which walks up through Z self-copies
to the two status-21 incoming hard-process partons. A gluon on either leg kills
it — and the sample is POWHEG-BOX `Zj` + MiNNLO, whose underlying Born is Z + 1
parton. Measured in 1.35 M events of the sample's own gen record (100 % return
exactly two status-21 mothers, so no pruning ambiguity):

| initial state | fraction | ME correction |
|---|---|---|
| `q qbar` | 0.3528 | **fires** |
| `q g` | 0.5710 | no |
| `g g` | 0.0358 | no |
| `q q` same sign | 0.0404 | no |

Weighted by the MiNNLO weight — which is what a kernel histogram carries, and
which differs because 7.6 % of events have negative weights concentrated in the
non-`q qbar` channels — the ME-corrected fraction is **f_ME = 0.3886 ± 0.0008**
in the peak band, rising monotonically from 0.357 at 50–60 GeV to 0.477 at
150–200 GeV (`data/photos/f_me.json`). The sample's kernel is therefore a
*mixture* of Photos-with-ME and Photos-without-ME at that measured, unfitted
weight, and `mix.py` builds it.

### The generator

`photos_gen.cc` builds `q qbar -> Z -> mu+ mu-` with the quarks as the Z's
mothers (which is what the ME channel test needs), the muons distributed as the
Born `(1+cos²θ) + 2R cosθ` with `R` from the standard γ*/Z couplings at that
mass, and `m_pre` drawn from the sample's own conditional distribution inside
each band on a 1 MeV grid (`prep_input.py`). Generation is **flat in band
index** — every band's kernel is normalised on its own, so band populations
never enter — and a band is recombined into a wider one with the sample's
weights, which reproduces `<m_pre>` per band to 1e−6. It accumulates the same
`(s0, s1, s2)` fine-`u` sums at 2e−5 that `fit_gen.build_kernel` merges, so the
same atom builder runs on the sample and on any standalone run.

**Nothing in the kernel depends on the production model.** Varying it, at 5.1e8
events per variation (peak band, `05_syst_*`):

| variation | `<u>` ratio | `P(u>1e-3)` | `P(u>0.1)` | `P(u>0.5)` |
|---|---|---|---|---|
| flat `cosθ` | 0.9991 ± 0.0009 | 0.9998 | 0.9991 | 0.9971 |
| `(1+cosθ)²` | 0.9991 ± 0.0009 | 0.9998 | 0.9991 | 0.9971 |
| up quarks only | 0.9992 ± 0.0009 | 1.0002 | 0.9998 | 0.9977 |
| down quarks only | 0.9988 ± 0.0009 | 1.0001 | 1.0004 | 0.9969 |
| Z boosted to the sample's `(p_T, y)` | 1.0004 ± 0.0009 | 1.0002 | 0.9997 | 1.0009 |
| `fint` 2 → 8 | 1.0007 ± 0.0009 | 1.0007 | 0.9999 | 1.0004 |

With the ME correction **on** the same variations move `<u>` by at most 0.25 %
and `P(u>0.5)` by 0.6 % — Photos's per-event ME weight is a ratio of matrix
elements at the generated production angle, so it is production-dependent,
while the switched-off kernel is not, as the exact O(α) result integrated over
angles requires.

### Validation gate

2.06e9 events per configuration. Peak band, `<m_pre>` = 91.105 GeV; the
sample's errors are `sqrt(p(1-p)/N_eff)` with `N_eff` = 1.63e7 from the clipped
MiNNLO weights.

**P(Photos emitted nothing at all)** — the sample's `npre == 0`, i.e. no
status-746 pre-Photos muon copy, which Photos writes whenever it touched the
muons, by a photon *or* by a pair. A parameter-free test of the emission rate:

| | 50-60 | 70-80 | **86-96** | 110-130 | 150-200 |
|---|---|---|---|---|---|
| sample | 0.442541 | 0.421723 | **0.410287 ± 0.00012** | 0.395140 | 0.375510 |
| **Photos, sample cfg** | 0.441076 | 0.420804 | **0.409836 (−3.6σ)** | 0.394908 | 0.374904 |
| ME on for 100 % | 0.441657 | 0.421291 | 0.410543 (+2.0σ) | 0.395169 | 0.375077 |
| ME on for 0 % | 0.440751 | 0.420510 | 0.409386 (−7.0σ) | 0.394723 | 0.374749 |
| ME off, pairs off | 0.441810 | 0.421725 | 0.410657 (+3.1σ) | 0.395859 | 0.376026 |
| ME on, pairs off | 0.442712 | 0.422416 | 0.411712 (+12σ) | 0.396387 | 0.376317 |

**Tail `P(u > u0)`**, standalone / sample in the peak band:

| `u0` | Photos, sample cfg | ME on 100 % | ME on 0 % | ME off, pairs off | ME on, pairs off |
|---|---|---|---|---|---|
| 1e−4 | 1.0016 (+4.7σ) | 0.9996 (−1.1σ) | 1.0029 (+8.2σ) | 0.9980 (−5.8σ) | 0.9950 (−15σ) |
| 1e−3 | 1.0021 (+5.1σ) | 0.9995 (−1.2σ) | 1.0038 (+8.8σ) | 0.9971 (−6.9σ) | 0.9929 (−17σ) |
| 1e−2 | 1.0028 (+4.9σ) | 0.9985 (−2.5σ) | 1.0054 (+9.3σ) | 0.9962 (−6.5σ) | 0.9893 (−18σ) |
| 5e−2 | 1.0036 (+4.5σ) | 0.9957 (−5.2σ) | 1.0085 (+10σ) | 0.9976 (−2.9σ) | 0.9849 (−19σ) |
| 0.1 | 1.0034 (+3.4σ) | 0.9922 (−7.7σ) | 1.0105 (+10σ) | 0.9990 (−1.0σ) | 0.9810 (−19σ) |
| 0.2 | 1.0045 (+3.5σ) | 0.9868 (−9.8σ) | 1.0158 (+12σ) | 1.0048 (+3.6σ) | 0.9764 (−18σ) |
| 0.5 | 1.0067 (+3.0σ) | 0.9727 (−12σ) | 1.0283 (+12σ) | 1.0213 (+9.2σ) | 0.9652 (−15σ) |
| 1.0 | 1.0082 (+1.9σ) | 0.9563 (−10σ) | 1.0412 (+9.4σ) | 1.0385 (+8.7σ) | 0.9531 (−11σ) |

The sample's configuration reproduces its own kernel **to 0.3 % everywhere and
to 0.2 % below `u` = 1e−2**, and it is the only configuration that does: with
pair emission off the kernel is 0.3–1.1 % low in the soft half and 2–4 % high in
the hard tail; with the ME correction applied to every event it is 1–4 % low
above `u` = 0.1; with it applied to none, 0.3–4 % high throughout. The residual
of the mixture is a uniform +0.2 % of radiative strength, of the same size as
the 0.4 % on the pair rate, and it is partly the sample's own dumper: it
requires exactly two hard-process status-1 muons, so the 3.07e−4 of events in
which Photos emitted a `mu+ mu-` pair are dropped, biased against the largest
energy loss and worth about half of it.

The **pair rate** is an independent confirmation that this is the right
configuration: standalone Photos gives 2.727e−3 pairs per event against
2.737e−3 counted in the sample's gen record, agreeing to 0.4 %.

### Where the Photos-minus-analytic residual comes from

The generator sits 1.5–1.9 % above the exponentiated exact O(α) in the tail
(`MC/exp1` in `../260913_fsr_analytic/00_moments.txt`). Switching Photos's
pieces on and off one at a time decomposes it. Peak band, `P(u > u0)` relative
to the analytic `exp1`:

| `u0` | Photos, photons only, no ME | + exact-ME on every event | + pairs instead | **the sample** |
|---|---|---|---|---|
| 1e−3 | +0.70 % | +0.28 % | +1.38 % | +1.00 % |
| 1e−2 | +1.17 % | +0.47 % | +2.11 % | +1.56 % |
| 5e−2 | +1.65 % | +0.35 % | +2.76 % | +1.89 % |
| 0.2 | +1.75 % | −1.13 % | +2.86 % | +1.26 % |

So:

* **Photos's own exponentiated shower is +1.7 % above the exponentiated exact
  O(α)** at `u` = 0.05 when the exact matrix-element correction is off, and
  **+0.35 % when it is applied to every event.** The correction is worth −1.3 %,
  and with it Photos and the analytic form agree to a few per mille over four
  decades in `u` — which is the check that the analytic radiator is right and
  that the crude Photos kernel is what is high.
* The sample *has* the correction switched on, but it fires on only 39 % of
  events, so its photonic part keeps about +1.1 % of that excess.
* **Real pair emission supplies the rest, +1.1 %.** `exp1`/`exp2` have no pair
  term; the `data` configuration does.

The two effects are the same size and the same sign, which is why a single
"missing exact-ME correction" explanation fitted the inclusive number but not
the shape: the ME piece grows with `u` and turns over, the pair piece does not.

### Pair emission against the analytic pair term

Photos 3.61 emits **`e+e-` and `mu+mu-` pairs only** — `PHOPAR(..., 11,
0.000511, ...)` and `PHOPAR(..., 13, 0.1057, ...)` in `photosC.cxx`, called once
before and once after the photons with `STRENG = 0.5`. No `tau`, no hadrons, so
the matching analytic species set is `("e", "mu")`.

`setPhotonEmission(false)` isolates the pair kernel. Photos's pair matrix
element is eq. (1) of Jadach, Skrzypek and Ward, Phys. Rev. D49 (1994) 1178 —
the **soft** pair current `J = p_-/(p_-.k) - p_+/(p_+.k)` contracted with the
pair tensor `(4 k_1^mu k_2^nu - q^2 g^{mu nu})/2q^4` — of which its own authors
write that it is "valid for the soft pairs emissions but is applied, at present,
in PHOTOS Monte Carlo algorithm over the entire phase space" (Antropov, Arbuzov,
Sadykov, Wąs, Acta Phys. Polon. B48 (2017) 1469, arXiv:1706.05571, sec. 2). The
Altarelli-Parisi hard factor sits in `pairs.cxx` one line below it, commented
out. Against the exact O(α²) pair radiator and against that **eikonal limit** —
the same dispersive integral (4) with the exact matrix element replaced by
`T_born x (-J^2)`, which is what `YOT1` computes — at a fixed `m` = 91.1876 GeV,
1e9 standalone events:

| | Photos | eikonal | Photos/eik | exact | Photos/exact |
|---|---|---|---|---|---|
| `e⁺e⁻` rate | 2.41855e−3 | 2.41811e−3 | **1.0002** | 2.53055e−3 | 0.956 |
| `e⁺e⁻` `<u>` | 1.95999e−4 | 1.95676e−4 | **1.0016** | 2.89912e−4 | 0.676 |
| `μ⁺μ⁻` rate | 3.12600e−4 | 3.13249e−4 | **0.9979** | 3.30929e−4 | 0.945 |
| `μ⁺μ⁻` `<u>` | 5.68119e−5 | 5.68428e−5 | **0.9995** | 8.01532e−5 | 0.709 |

**Photos is the soft limit of the pair matrix element, and nothing else.** The
eikonal calculation reproduces it to 0.2 % in rate and in mass loss, over the
whole `u` spectrum (`01_pair_uspec`) and over four decades of pair-mass cut
(`03_rate_vs_qcut`). Its normalisation is therefore correct — including the
`WT = YOT1*YOT2*YOT3/8/FREJECT` line the author flagged with "origin must be
understood": the `8` is the physics constant, not an uncompensated envelope,
even though it also happens to bound the weight (`sup(YOT1 YOT2 YOT3)` = 7.92).
The bookkeeping is clean too, checked with an instrumented build of the upstream
source: all four `trypar` calls per species fire every event (two blocks × two
charged legs), the `STRENG -> STRENG/(1-PRHARD)` veto compensation makes the
crude probability exactly additive (4.6177e−2/event measured against 4.6178e−2
ideal), the `0.5` marked "for 1-leg only" is exactly what the two-leg
multichannel sum needs, the two legs radiate equally (`05_per_leg`), and **no
rate is lost to weight truncation** (`sup(WT)` = 0.495, overweight fraction 0).

What Photos misses is **hard** pair emission: the eikonal current has no
Altarelli-Parisi numerator, so it falls below the exact spectrum for
`u > 0.2` (`01_pair_uspec`) and its mean mass loss is **32 % low**. This is the
same conclusion its authors reach from their own exact-phase-space evaluation of
the same soft matrix element, which "reproduce[s] well results of PHOTOS" and
leaves differences "dominated ... by non leading terms and of rather hard pair
emission" (arXiv:1706.05571, secs. 4 and 6). Adding the
`tau` and hadronic species it does not generate at all, the total is
2.528e−4 against the exact 5.422e−4, i.e. **47 %**.

The dispersive leading-log term the kernel used before is wrong the other way —
**+46 %** on `<u>` for the same species, for the three reasons listed under
"Pair emission" above. The factor two between Photos and that term was two
independent errors of comparable size in opposite directions; neither number was
right.

### Fit level

`fit_gen.py fit --suite postfsr`, 27.7 M gen events in 60–120 GeV, 5 Legendre
shape terms, `nm = 8192`, offsets from the generator's own
`m_Z` = 91.153510, `Gamma_Z` = 2.493202. Read banded against banded.

| kernel | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| **empirical, the sample's own, banded 2 GeV** | **+0.91 ± 0.54** | **+0.74 ± 1.13** |
| empirical, inclusive (`kern_incl_sc3.3e-4.npz`) | +0.15 ± 0.56 | +1.31 ± 1.14 |
| **`mc`: Photos standalone, sample cfg** | **+0.41 ± 0.56** | **−0.21 ± 1.14** |
| … ME correction applied to every event | +0.75 ± 0.56 | −0.02 ± 1.14 |
| … ME correction applied to no event | +0.48 ± 0.56 | −0.15 ± 1.14 |
| Photos, ME off, pairs off | +0.44 ± 0.55 | +3.19 ± 1.14 |
| Photos, ME on, pairs off | +0.57 ± 0.55 | +2.84 ± 1.14 |
| analytic exp. O(α), banded 2 GeV | +1.25 ± 0.54 | +6.34 ± 1.13 |
| analytic + O(α²)LL | +1.07 ± 0.54 | +2.67 ± 1.13 |
| analytic + O(α²)LL + NLL | +1.17 ± 0.54 | +2.34 ± 1.13 |
| analytic + O(α²)LL + NLL + `e`, `μ` pairs, eikonal (= Photos) | +1.69 ± 0.54 | −0.93 ± 1.13 |
| **`data`: analytic + O(α²)LL+NLL + exact pairs, all species** | **+1.38 ± 0.54** | **−1.04 ± 1.13** |
| … with the dispersive leading-log pair term instead | +2.35 ± 0.54 | −5.16 ± 1.13 |

The standalone kernel and the sample's own agree to **0.5 MeV on `m_Z` and
1.0 MeV on `Γ_Z`** — below the statistical error of the closure and below the
empirical kernel's own ±0.7 MeV `sigma_cap` quadrature bias. Applying the ME
correction to every event instead of the measured 39 % moves `m_Z` by 0.34 MeV,
so the mixture is a refinement inside the closure precision, not a requirement.
Exact pair emission is worth **−3.4 MeV on `Γ_Z`** and +0.21 MeV on `m_Z`
(against −7.5 / +1.18 for the dispersive leading-log term it replaces, and
−3.3 / +0.52 for the eikonal limit Photos generates); the ME correction, applied
to every event, at most 0.4 MeV on `Γ_Z` and 0.3 MeV on `m_Z`.

With 5 floating shape terms the fit does **not** respond to the pair term
through `<u>` alone. A uniform rescaling of the radiator by `ε` moves `m_Z` by
`ε × 57` MeV and `Γ_Z` by `ε × 337` MeV, and the old leading-log pair term —
implemented as `beta -> beta + beta_pair`, i.e. exactly such a rescaling —
follows that rule (2.5 % → +1.4 / −8.6 MeV predicted, +1.18 / −7.5 measured).
The exact term does not: it is 2.15 % of the radiator but moves the fit by only
+0.21 / −3.4 MeV, because its shape (a threshold at `u = 2 m_e/m`, no soft
singularity, a harder tail) is not a rescaling and the smooth `K(m)` absorbs
more of it. Most of the +4.1 MeV change on `Γ_Z` from the old `data` kernel to
the new one is therefore the *shape*, not the −16 % on the normalisation.

The +0.97 MeV between the `mc` and `data` kernels on `m_Z` and −0.8 MeV on `Γ_Z`
is the physics Photos leaves out: the O(α²) leading and next-to-leading logs,
`tau` and hadronic pairs, and the hard half of the leptonic pair spectrum.

### The two configurations

`fsr_config.py --config mc|data` builds both; the discretisation of each matches
the control it is read against (`sigma_cap` = 3.3e−4 for `mc`, as for the
empirical kernels; `var_budget` = 6e−10 for `data`, as for `fsr_analytic.py
kernel`), banded at 2 GeV over 50–200 GeV.

* **`mc`** — `data/kern_cfg_mc_sc3.3e-4.npz`. The sample's Photos physics at
  unlimited statistics: standalone Photos++ 3.61 with the sample's switches,
  mixed at the measured `f_ME(m)`. Use it to close the likelihood against this
  MC. It is **not** the best description of nature.
* **`data`** — `data/kern_cfg_data_vb6e-10.npz`. The analytic radiator of
  `fsr_analytic.py`: `DATA_VARIANT` = `exp2nll` (exponentiated exact O(α) plus
  the O(α²) leading log and its NLL term) convoluted with the exact O(α²) pair
  radiator for `DATA_PAIRS` = `("e", "mu", "tau", "had")`. First-principles
  throughout, exact mass dependence, and it supplies the three things Photos
  does not.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
cmssw-el7 --command-to-run photos_standalone/build.sh     # plain g++, no scram
python3 photos_standalone/prep_input.py                   # m_pre bands, boost sample
python3 photos_standalone/mcref.py --half                 # the sample, same binning
NPROC=240 photos_standalone/run.sh mcB  12 2200000 --me=1 --pairs=1 --fint=8
NPROC=240 photos_standalone/run.sh pair 12 2200000 --me=0 --pairs=1 --fint=8
python3 photos_standalone/mix.py --run data/photos/gen_mcB.npz \
        --run data/photos/gen_pair.npz --frac-file data/photos/f_me.json \
        -o data/photos/gen_mcMix.npz
NPROC=240 photos_standalone/run.sh paironly 12 2200000 --me=0 --pairs=1 --phot=0
photos_standalone/photos_pairdiag --n=1e9 --out=pd.bin   # pair kinematics
python3 fsr_config.py --config mc   -o data/kern_cfg_mc_sc3.3e-4.npz
python3 fsr_config.py --config data -o data/kern_cfg_data_vb6e-10.npz
./run_tf_z.sh python3 -u plot_pairs.py            # the pair-emission figures
./run_tf_z.sh python3 -u cmp_photos.py --runs "mcMix=Photos, sample cfg" ...
```

2.06e9 events take 80 s on 240 cores (Photos runs at 1.1e5 events/s/core); the
binary runs outside the container once built.

---

## Per-leg factorised kernel and the selection-conditional shape

`fsr_perleg.py` resolves the inclusive kernel leg by leg, which is what a
lepton `p_T` cut needs: the cut does not act on the pair variable `z`, it acts
on each muon separately. `cmp_perleg.py` is the validation against the
generator record; figures `~/public_html/ZMass/cvh/260915_fsr_perleg/`.

### The collinear factorisation

In the quasi-collinear limit each muon keeps its direction and its post-FSR
four-momentum is `p' = x p`.  In the massless limit that is exactly Lorentz
covariant, so `x = p_T'/p_T = E'/E` in any frame and `eta' = eta`,
`phi' = phi`.  The pair mass then factorises,

```
m'^2 = (x_+ p_+ + x_- p_-)^2 = x_+ x_- m^2 ,   z = x_+ x_- ,
u = -ln(m'/m) = (u_+ + u_-)/2 ,   u_q = -ln x_q ,
```

and in Mellin space the inclusive kernel is the **square** of the single-leg
time-like QED structure function, `K~(n) = D~(n)^2`.

### The single-leg radiator

`LegRadiator` is `fsr_analytic.FSRKernel` with the Mellin exponent halved.

With `a = alpha/pi`, `L = ln(m^2/m_mu^2)` and the inclusive exponent
`E(n) = a (L g0 + c1) + a^2 (L/2) g1`, `K~ = exp E`, the leg is `D~ = exp(E/2)`:

```
D(x) = C_D beta_D (1-x)^{beta_D-1} + h_D(x) + q2_D(x) + nll_D(x)

beta_D  = beta/2 = a (L - 1)
h_D(x)  = -(beta_D/2)(1+x) + (a/2) (1+x^2) ln x/(1-x)          = h(x)/2
q2_D(x) = (beta_D^2/8) [P (x) P]_reg(x)
nll_D(x)= (a^2 L/4) [ G_D(x) + g_delta,D delta(1-x) ]
G_D     = P1T + A - kappa (1+x) = G + P1T/2
g_delta,D = P1_delta + (3/2) kappa
C_D     = 1 - int h_D - int q2_D - int nll_D
```

`G`, `A`, `P1T`, `kappa`, `P1_delta` are exactly the objects of
`fsr_analytic`. Three of the four lines are the inclusive construction with
`beta -> beta/2` and `a -> a/2`; the fourth is not, and the extra `P1T/2` is
forced: the generic construction at coupling `a_hat` has Mellin exponent
`a_hat (L g0 + c1) + a_hat^2 (L/2) g1`, so `a_hat = a/2` undershoots the wanted
`(a^2 L/4) g1` by `(a^2 L/8) g1`, i.e. by `(a^2 L/8) P1T`. Adding it to
`(a^2 L/4) G` gives `(a^2 L/4) G_D`. The normalisation is untouched with no
numerical rescale, because `int P1T dx = -P1_delta`, `int A dx = 0` and
`int kappa (1+x) dx = (3/2) kappa`, so `int G_D dx + g_delta,D = 0` exactly.

**Pairs are emitted from one leg**, so the leg carries
`(1 - N/2) delta(1-x) + R_pair(x)/2` with the *same* exact `R_pair` the
inclusive kernel uses (`fsr_analytic.PairTerm`); its square is
`(1 - N) delta + R_pair + O(alpha^4)`, the inclusive pair factor. The leg's
support floor is `x_min = 2 m_mu/m` -- the muon cannot keep less than its own
mass, and `x_min^2` is exactly the pair's own threshold `4 m_mu^2/m^2`, so
`D (x) D` has support `z >= z_min` with nothing to trim.

### `D (x) D` against the inclusive kernel

`fsr_perleg.py check`. The Mellin square root is exact only through the order
the kernel is built to; what is left is the O(alpha^3) difference between the
two soft resummations, `[beta_D (1-x)^{beta_D-1}]^{(x)2}` for the leg against
`beta (1-z)^{beta-1}` for the pair.

| `n` | `exp1` | `exp2` | `exp2nll` | data cfg |
|---|---|---|---|---|
| 1 | -3.26e-5 | -3.36e-5 | -2.24e-5 | -2.14e-5 |
| 2 | +6.29e-5 | +3.45e-5 | -5.56e-5 | -5.45e-5 |
| 4 | -5.99e-5 | +3.90e-5 | -7.97e-5 | -7.82e-5 |
| 8 | -2.44e-4 | +4.11e-5 | -8.39e-5 | -8.21e-5 |

The order is read off the mass dependence: over `beta` = 0.038 to 0.080
(`m` = 10 to 1000 GeV) the `exp2nll` residual at `n = 8` scales as `beta^3`
(`res/beta^3` = -0.50 to -0.41) and the `exp1` one as `beta^2`
(`res/beta^2` = -0.058 to -0.076). `exp1` is second order because it *omits*
the O(alpha^2) leading log on purpose, and the leg's own exponentiation squared
supplies part of it; `exp2` and `exp2nll`, which carry the whole `L^2`
coefficient, are third order, i.e. beyond the accuracy of either form.

At the data configuration, density and moments:

| | `<u>` | `P(u>1e-3)` | `P(u>1e-2)` | `P(u>0.05)` | `P(u>0.2)` |
|---|---|---|---|---|---|
| `K`, `m` = 91.19 | 2.696648e-2 | 0.2733 | 0.1699 | 0.0935 | 0.0365 |
| `D (x) D` / `K` - 1 | -8.3e-4 | -1.9e-4 | -1.5e-4 | -2.5e-4 | +3.3e-4 |
| … at `m` = 60 | -2.4e-3 | +6.3e-5 | +3.1e-4 | +7.6e-4 | +2.2e-3 |
| … at `m` = 130 | +1.7e-4 | +1.1e-4 | +3.1e-4 | +9.5e-4 | +2.7e-3 |

`int D dx` = 0.99998893 rather than 1: the ladder truncates at
`x_min = 2 m_mu/m`, and the atoms are renormalised on their own support, which
is what the provider does anyway.

### The radiator, band by band

`beta_D = beta/2` and the leg's pair rate `N/2`, at the `data` configuration:

| `m` [GeV] | `beta` | `beta_D` | `N_pair` | `N_pair/2` | `<u>_D` | `<u>_K` |
|---|---|---|---|---|---|---|
| 60 | 0.05428 | 0.02714 | 2.9642e-3 | 1.4814e-3 | 2.486347e-2 | 2.492276e-2 |
| 80 | 0.05695 | 0.02848 | 3.3525e-3 | 1.6757e-3 | 2.629339e-2 | 2.632662e-2 |
| 91.19 | 0.05817 | 0.02908 | 3.5414e-3 | 1.7702e-3 | 2.694417e-2 | 2.696648e-2 |
| 110 | 0.05991 | 0.02996 | 3.8271e-3 | 1.9131e-3 | 2.787710e-2 | 2.788466e-2 |
| 130 | 0.06146 | 0.03073 | 4.0959e-3 | 2.0475e-3 | 2.870841e-2 | 2.870363e-2 |

`<u>_D` = `<u>_K` is not a coincidence and not a check: `u_pair = (u_+ + u_-)/2`
with two identical legs, so the two means are the same object; the small
difference is exactly the O(alpha^3) of `D (x) D` against `K` above.

### The selection-conditional shape

With `|eta| < eta_cut` FSR-independent in the collinear limit, the post-FSR
`p_T > p_T^cut` condition on a pre-FSR configuration `(p_T+, p_T-)` is
`x_q > a_q = p_T^cut/p_T q`, i.e. `u_q < b_q = -ln a_q`, so

```
K_sel(z | a_+, a_-) = int dx_+ D(x_+) D(z/x_+)/x_+ ,
                      x_+ in [max(a_+, z), min(1, z/a_-)]
K_sel(z | m)        = int da_+ da_- h(a_+, a_- | m) K_sel(z | a_+, a_-)
A(m)                = int dz K_sel(z | m) .
```

Doing the `a` integral **first** turns the double integral over `h` into its
2-D survival function evaluated on the leg's own ladder, which is what the code
computes:

```
K_sel(u | m) = int du_+ du_- D_u(u_+) D_u(u_-) G(u_+, u_- | m)
                                              delta(u - (u_+ + u_-)/2)
A(m)         = int du_+ du_- D_u(u_+) D_u(u_-) G(u_+, u_- | m)

G(u_+, u_- | m) = P( p_T+ > p_T^cut e^{u_+} , p_T- > p_T^cut e^{u_-} ,
                     |eta_q| < eta_cut  |  m )      on the PRE-FSR muons.
```

`G` is the 2-D reverse cumulative of `h` and is exact on the table's grid
edges; the `(a_+, a_-)` binning is therefore a discretisation of `G` only,
never of the kernel. `G` is the whole boson-kinematics input: production
(`p_T^Z`, `y_Z`), the decay angles with their angular coefficients and the PDFs
all sit inside it and nothing else does.

### The h table: the interface for SCETLib + DYTurbo

`fsr_perleg.py htable` measures `h` from a generator record; anything with the
same format can replace it, and nothing in the QED above has to change.

```
np.savez_compressed(out,
    h        = float32 (n_band, n_a+1, n_a+1),   # the table, symmetrised
    b_edges  = float64 (n_a+1,),                 # b = -ln a = ln(pT/pT_cut)
    bands    = float64 (n_band, 2),              # m_pre band edges [GeV]
    m_bar    = float64 (n_band,),                # weighted mean m_pre
    sumw     = float64 (n_band,),                # band total weight
    nev      = int64   (n_band,),
    provenance = json  {pt_cut, eta_cut, band_lo, band_hi, band_width, ...})
```

* `h[k, i, j]` is the weight of events in `m` band `k` whose two **pre-FSR**
  muons have `b_+ in [b_edges[i], b_edges[i+1])` and `b_- in [.., ..)` and pass
  `|eta| < eta_cut`, **divided by the band's total weight** including the events
  that fail. Index `n_a` is the overflow (`b` above the last edge). Events with
  `p_T < p_T^cut` (`a > 1`, `b < 0`) and `eta` failures are simply absent, so
  `sum_ij h[k] = P(both p_T > p_T^cut and both |eta| < eta_cut | m)` and
  `A(m) <= sum_ij h[k]`.
* The table is stored **symmetrised**, `(h + h^T)/2`: the radiator is the same
  on both legs, so only the symmetric part can ever enter, and the charge
  labelling of the two legs is then irrelevant (`p_T`-ordered muons are a valid
  labelling).
* `b_edges` is the grid in the leg's own variable; the default is `1e-3` up to
  `b = 0.08`, `5e-3` to `0.5` and `2e-2` to `3.0`, i.e. finest where the
  radiator has its weight (`u -> 0`). `p_T = p_T^cut e^b`.
* Everything about the boson is integrated out. A `p_T`- or `eta`-dependent
  efficiency instead of a step in `p_T` would need one more axis (the `eta`
  pair, or `eps(p_T, eta)` folded into the `h` fill); nothing else changes,
  because `G` is by construction *the probability that both legs survive their
  own threshold*, and a smooth efficiency only makes that probability smooth.

### What the collinear picture costs

Measured on 20.8 M events of the sample's own gen record with the two legs
**matched by charge** (`dump_gen_perleg.py`; `dump_gen_fsr.py` sorts the pre-
and post-FSR muons by `p_T` *independently*, which loses the correspondence in
6.3 % of the radiating events). `x` is taken in the Born `Z` rest frame.

**(a) the directions are conserved.** Per leg, `P(|d eta| > 1e-3) = 9.50e-2`,
`> 1e-2` 4.51e-2, `> 0.1` 1.31e-2, and the same for `phi` to three digits
(41 % of events radiate nothing at all, so `d eta = 0` exactly there). The tail
is real wide-angle emission, not a failure of the matching. It does not matter
for the `eta` cut, because it only flips a decision for a muon within `d eta` of
`|eta| = 2.4`: the `eta` decision taken on pre-FSR muons differs from the
post-FSR one in 1.68e-3 of events.

**(d) `z = x_+ x_-` holds where the weight is, and fails in the hard tail.**
`P(u > u0)` of the true `u = -ln(m'/m)` and of `(u_+ + u_-)/2` agree to 1e-5 up
to `u0 = 0.05`; the collinear product is 0.6 % low at `u0 = 0.1` and 7 % low at
`u0 = 0.5`. Broken down by the true `u`:

| `u` region | `P` | `<u>` true | `<u>` collinear | ratio-1 |
|---|---|---|---|---|
| `[0, 1e-3)` | 0.66498 | 5.99549e-5 | 5.99444e-5 | -1.8e-4 |
| `[1e-3, 1e-2)` | 0.10304 | 3.98460e-3 | 3.98292e-3 | -4.2e-4 |
| `[1e-2, 0.05)` | 0.07592 | 2.47672e-2 | 2.47109e-2 | -2.3e-3 |
| `[0.05, 0.2)` | 0.05643 | 1.04333e-1 | 1.03363e-1 | -9.3e-3 |
| `[0.2, 0.5)` | 0.02384 | 3.13528e-1 | 3.05379e-1 | -2.6e-2 |
| `[0.5, 10)` | 0.01275 | 8.98035e-1 | 8.13799e-1 | -9.4e-2 |

so `<u>` inclusively is 4.89 % low and, under `p_T > 25`, 1.96 % low: the
missing piece is the opening angle, which a wide-angle photon closes so the pair
mass falls further than `x_+ x_-` alone. It is entirely in `u > 0.05`, which the
`p_T` cut is already removing.

**(c) the legs are NOT independent.** In the peak band the joint tail
`P(u_+ > t, u_- > t)` is 2.2 (`t` = 1e-4) to 3.2 (`t` = 0.05) times the product
of the two marginals, because a wide-angle photon takes energy from **both**
muons: the recoil is shared. This is the `1/L` power correction to the collinear
picture, and it is the reason the measured per-leg `x` spectrum is *not* `D`:
`<u_leg>` = 2.581e-2 / 2.592e-2 against the analytic `D`'s 2.694e-2 at the same
mass, and the analytic `D` is the harder of the two by exactly the amount that
makes `D (x) D` reproduce the true (correlated) pair spectrum.

**(e) the acceptance decision closes to 2e-4.** Taking the collinear prediction
`x_q p_T^{pre} > 25` and the pre-FSR `eta` against the true post-FSR cut,

```
A(post-FSR cut)    = 0.364188
A(collinear x p_T) = 0.364257        ratio-1 = +1.90e-4
per-event mismatch = 1.51e-3   (eta only 1.68e-3, p_T only 2.32e-3)
```

The two per-event mismatches are twice the net, i.e. they very largely cancel:
the wide-angle events that the collinear rule wrongly keeps are balanced by the
ones it wrongly drops.

### The model against the MC, band by band

The construction is exact in the acceptance and a few per cent low in the mass
loss. Weighted by the band population, as the fit sees them:

| `m_pre` band | `A` (MC) | `A` model / MC | `<u>` (MC) | `<u>` model / MC |
|---|---|---|---|---|
| | | analytic / empirical / mc `D` | | analytic / empirical / mc |
| 60-70 | 0.19855 | 0.9921 / 0.9908 / 0.9914 | 5.699e-3 | 0.9385 / 1.0600 / 0.9379 |
| 70-80 | 0.29151 | 0.9997 / 0.9996 / 0.9991 | 7.868e-3 | 0.9459 / 1.0511 / 0.9445 |
| 80-86 | 0.34070 | 0.9994 / 1.0003 / 0.9991 | 9.379e-3 | 0.9441 / 1.0307 / 0.9412 |
| 86-90 | 0.36644 | 0.9987 / 0.9997 / 0.9983 | 10.146e-3 | 0.9510 / 1.0315 / 0.9497 |
| 90-92 | 0.37546 | 0.9986 / 0.9999 / 0.9984 | 10.539e-3 | 0.9489 / 1.0231 / 0.9463 |
| 92-96 | 0.38396 | 0.9991 / 1.0005 / 0.9988 | 10.866e-3 | 0.9493 / 1.0186 / 0.9470 |
| 96-105 | 0.40163 | 0.9972 / 0.9987 / 0.9969 | 11.762e-3 | 0.9444 / 1.0068 / 0.9423 |
| 105-120 | 0.42811 | 0.9921 / 0.9940 / 0.9915 | 13.288e-3 | 0.9475 / 0.9992 / 0.9454 |
| 120-140 | 0.45573 | 0.9922 / 0.9931 / 0.9912 | 15.513e-3 | 0.9353 / 0.9821 / 0.9362 |
| `A`-weighted | | 0.9965 / 0.9975 / 0.9960 | | 0.9451 / 1.0182 / 0.9435 |

`A(m)` closes to 0.4 %, and to 0.15 % over 70-105 GeV. The `mc` column is the
numerical square root of the standalone Photos kernel -- the *same* QED as the
MC -- and it reproduces the analytic `D` to 0.3 % in both columns: **the `<u>`
deficit is the collinear machinery, not the kernel physics** (see "The
machinery-only benchmark").

The `<u>` column is the price of the two collinear approximations, and they
pull in opposite directions:

* with the **empirical** per-leg `D` -- the sample's own `x` spectrum, so the
  QED content is exactly the MC's -- the only error left is **leg
  independence**, and it makes `<u>` 2-7 % **too large**: under independence a
  hard emission on one leg leaves the other untouched and the pair survives the
  cut, whereas in the generator a wide-angle photon pushes **both** legs down
  and the pair fails;
* with the **analytic** `D`, which is fixed by `D (x) D = K` rather than
  measured, `<u>` is 5-7 % **too small**, because `D` is the *harder* per-leg
  spectrum that compensates the missing opening-angle piece inclusively, and
  cutting on it is not the same as cutting on the physical `x`.

The conditional kernel's own mass dependence is reproduced: `<u>` of the
analytic model runs 6.58e-3 (60-80 GeV) to 14.59e-3 (110-150), against the
MC's 7.1e-3 to 14.9e-3 -- the factor 2.2 that makes the banded kernel
mandatory under this selection.

### Discretisation

Four knobs, all measured against the same reference (1 GeV bands, 290-edge
`(a_+, a_-)` grid, 3000-cell leg ladder, `var_budget = 6e-10`):

| knob | atoms | `<u>` (88-94 GeV) | `A` (88-94) |
|---|---|---|---|
| **1 GeV bands (default)** | 28 048 | 9.99526e-3 | 0.37492 |
| 0.5 GeV bands | 55 685 | 9.99521e-3 | 0.37493 |
| 2 GeV bands | 14 204 | 9.99593e-3 | 0.37498 |
| leg ladder 1500 (from 3000) | 28 582 | 9.99529e-3 | 0.37492 |
| `(a_+,a_-)` grid thinned 4x | 28 054 | 10.00647e-3 | 0.37493 |

The band width is free because the model's `m` dependence is analytic
(`beta(m)` and `G(.,.|m)`), not a measured table: 0.5 and 2 GeV agree with 1 GeV
to 7e-5 and 1.6e-4 relative. The leg ladder is converged at 1500 cells. Only the
`(a_+,a_-)` grid matters, and it matters through `G`, not through the kernel:
read off the default grid, `G(u_+,u_-)` reproduces a direct event count to
<= 5e-7 everywhere and exactly at the grid edges; thinned by 4 it is off by
2e-4 at `u = 1`, worth +1.1e-3 on `<u>`.

### The fit benchmark

`fit_gen.py fit --suite perleg`, 9.87 M selected gen events in 60-120 GeV, five
Legendre shape terms, `nm = 8192`, offsets from the generator's own
`m_Z` = 91.153510, `Gamma_Z` = 2.493202. Every row is the **same** run on the
**same** events, so the differences are same-run differences.

| model, fiducial `pT` > 25, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −1.01 ± 0.78 | +1.86 ± 1.50 |
| **MC-conditional kernel + `A(m)` (Bernstein 8)** | **−0.31 ± 0.87** | **+2.61 ± 1.75** |
| … `data/kern_fid_sc3.3e-4.npz`: `K_sel` measured on the selected events | | |
| … with the per-leg model's tabulated `A(m)` instead | −0.33 ± 0.87 | +2.12 ± 1.75 |
| **per-leg, analytic `D` (data cfg) + `h`** | **+1.16 ± 0.83** | **−2.89 ± 1.74** |
| per-leg, analytic `D`, exp. O(α) only | +1.25 ± 0.83 | −0.95 ± 1.74 |
| per-leg, analytic `D`, no pair emission | +1.35 ± 0.83 | +1.18 ± 1.74 |
| per-leg, analytic `D`, eikonal `e`,`μ` pairs (Photos' own) | +1.38 ± 0.83 | −2.32 ± 1.74 |
| **per-leg, empirical `D` + `h`** | **+28.69 ± 0.82** | **−57.06 ± 1.72** |
| inclusive empirical kernel + `A(m)` | −3.48 ± 0.88 | +3.39 ± 1.81 |
| inclusive analytic kernel (data cfg) + `A(m)` | −3.11 ± 0.87 | +1.72 ± 1.81 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| per-leg analytic − MC-conditional | **+1.47** | **−5.50** |
| per-leg empirical − MC-conditional | +29.00 | −59.67 |
| pair emission (data cfg − no pairs) | −0.19 | −4.07 |
| O(α²) LL + NLL (no pairs − exp1) | +0.11 | +2.13 |
| exact pairs − Photos' eikonal `e`,`μ` pairs | −0.22 | −0.57 |
| per-leg `A(m)` − Bernstein 8 of the MC's | −0.02 | −0.50 |

and the discretisation, all against the 1 GeV / default row:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| 0.5 GeV bands | +0.56 | +0.31 |
| 2 GeV bands | +0.16 | +0.67 |
| `(a_+,a_-)` grid thinned 4x | +0.02 | +0.10 |
| leg ladder 1500 | +0.25 | −0.08 |
| leg ladder 6000 | +0.04 | −0.02 |
| `var_budget` 6e-9 | +0.34 | −0.27 |
| `var_budget` 6e-11 | +0.24 | −0.14 |
| `A(m)` as a degree-8 Bernstein instead of the table | −0.45 | +0.69 |
| empirical `D` measured in 1 / 20 GeV windows | +0.17 / +0.04 | +0.80 / +0.17 |
| empirical `D` with 2 GeV kernel bands | −0.06 | +0.61 |

Every discretisation knob is below the ±0.83 / ±1.74 MeV statistical error, so
the construction is numerically converged and the two numbers that mean
something are the physics ones.

**The per-leg factorisation must use the effective `D`, not the measured one.**
`D` is *defined* by `D (x) D = K`; the physical per-leg energy-fraction spectrum
is a different object -- 55 % denser at `u_leg` = 1e-4 and
`P(u_leg > 1e-4)` = 0.27940 against `D`'s 0.21812 (figure `02_leg_x`) --
because the generator's two legs are correlated. Feeding the physical spectrum
in as an independent marginal gets the unradiated fraction wrong by 12 %
absolute -- `(1 - 0.27940)^2` = 0.51926 against the MC's own 0.63643 -- and that
is where the mass peak lives, so it costs **+29 / −60 MeV**. The same row is
reproduced to 0.2 MeV with the leg measured in 1, 10 or 20 GeV windows and with
1 or 2 GeV kernel bands, so it is the independence assumption and not statistics.
With the effective `D`, `D (x) D` gives 0.63571 against the same 0.63643, and
the fit closes to **+1.5 / −5.5 MeV** of the MC-conditional kernel. That
difference is the factorisation and not the QED: see the next section.

The `mc` → `data` step *can* be taken inside the per-leg construction --
`legsqrt` supplies the per-leg `D` of Photos' tabulated `K` -- and it is small
under this selection, −0.07 / −0.50 MeV against +0.97 / −0.83 inclusively. The
analytic proxies behave as they do inclusively: replacing the exact O(α²) pair
radiator by the **eikonal** limit Photos actually generates (`e`, `μ` only)
moves the per-leg fit by −0.22 / −0.57 MeV, against −0.31 / −0.11 for the same
replacement in the inclusive fit.

### The effective leg of a tabulated kernel

`fsr_perleg.py legsqrt`. `D` is *defined* by `D (x) D = K`, so for a kernel
that is a table rather than a closed form it is the **numerical convolution
square root**, and the `mc` configuration gets a per-leg radiator carrying
exactly the MC's own QED.

Everything is a measure on a uniform grid in `u` of step `du`; the leg's own
variable is `u_leg = 2 u`, because `u = (u_+ + u_-)/2`. The standalone Photos
histograms (`photos_standalone/`, 52.8 M events per 2 GeV band) become one atom
per filled bin at the bin's exact conditional mean, plus a genuine `delta(u)` of
weight `P(Photos emitted nothing)`, and are deposited on the grid by splitting
each atom linearly between its two neighbouring nodes -- **mass and mean exact**,
and the sum of two nodes is a node, so the convolution of two deposited measures
is their exact convolution.

With `g = g_0 e_0 + g_c` and `D = d_0 e_0 + D_c`, `D_c` lives on nodes `>= 1`
and `D_c (x) D_c` on nodes `>= 2`, so `d_0 = sqrt(g_0)` and

```
D_c = (g_c - D_c (x) D_c) / (2 sqrt(g_0))
```

is a contraction with factor `||D_c||/sqrt(g_0)` = 0.352. In Fourier space the
same equation is the scalar quadratic `d^2 + 2 sqrt(g_0) d - g_c = 0` whose
contracting root is `d = -sqrt(g_0) + sqrt(g^)`, i.e. the iteration is the
branch selection of `sqrt(g^)`; `conv_sqrt(..., method=)` runs either. In the
peak band the fixed point converges in **21 iterations** to a step of 7.5e-16
and agrees with the spectral root to **1.9e-16**. Over all 78 bands
`|g^| >= 0.4348` and `|arg g^| <= 0.115` rad, so the principal square root is
the right branch by a wide margin.

**The grid is the kernel's own resolution**, `du` = 2e-5 = the histogram's bin
width, and that is not a convenience: on a *finer* grid the tabulated kernel is
a comb -- isolated atoms with empty nodes between them -- and a comb is not
infinitely divisible, so its square root rings at the lattice scale. Peak band,
`legsqrt --check-band 20`:

| `du` | `\|g^\|` min | max `\|arg g^\|` | `D[0]` | negative mass | `D (x) D - g` | `W_1(g, atoms)` |
|---|---|---|---|---|---|---|
| 1e-4 | 0.5638 | 0.092 | 0.775225 | −5.9e-5 | 1.1e-16 | 1.20e-5 |
| 4e-5 | 0.5346 | 0.092 | 0.754876 | −7.1e-5 | 1.1e-16 | 5.24e-6 |
| **2e-5** | **0.5136** | **0.092** | **0.739863** | **−9.8e-5** | **4.4e-17** | **5.22e-6** |
| 1e-5 | 0.0563 | 0.512 | 0.718124 | −1.0e-1 | 1.1e-16 | 1.01e-6 |
| 5e-6 | 0.1232 | 1.205 | 0.672540 | −1.3e-1 | 1.1e-16 | 5.14e-7 |

`D (x) D` reproduces the deposited kernel to **4.4e-17** and `<u>` to 2.2e-16
(`<u>_leg / <u>_K - 1` is below 1.3e-13 in every band) -- the root is *exact*,
against the analytic `D`'s genuine O(α³) residual of ~1e-4. What the numerical
root costs instead is the grid representation of `K`: a Wasserstein-1 distance
of **5.2e-6** in `u`, 1.9e-4 of `<u>` = 2.73e-2 and 1.6 % of the reference
kernel's own atom width `sigma_cap` = 3.3e-4; in `P(u > u_0)` that is −1.4e-3
at `u_0` = 1.1e-3, +2.6e-4 at 1.07e-2 and +9.6e-5 at 5.3e-2 (figure
`11_dconvd_mc`). At fit level the whole of it is **+0.01 / +0.03 MeV**
(`du` 4e-5 against 2e-5).

The residual negative mass, −8.2e-5 (50 GeV) to −1.6e-4 (320 GeV and above), is
**all above `u_leg` = 4**, where the standalone's tail block is binned at 0.05
and *is* a comb on a 2e-5 grid: about 250 of the 3000 ladder cells come out
negative there. `merge_positive` folds each into its neighbours, exactly in the
mass and in the first two moments, so what the provider sees is a positive
measure on ~1710 cells with `d_0` = `sqrt(g_0)` and `<u_leg>` = `<u>_K` to
machine precision.

`D_mc` tracks the analytic `D` to **5 %** from `u_leg` = 4e-4 to the kinematic
limit, while the *physical* per-leg spectrum is 50 % denser at `u_leg` = 1e-3
and 10 % thinner above 0.5 (figure `10_leg_D_mc`): the effective leg is an
object of the kernel, not of the generator record.

### The machinery-only benchmark

The same `fit_gen.py fit --suite perleg` run, same events. With `D_mc` the QED
is the MC's on both sides, so the difference is the collinear factorisation and
nothing else. The `cond:` rows are the MC's **own** conditional kernel read in
the model's variables (`fsr_perleg.py condker`, measured on the 20.8 M-event
per-leg record, banded like the reference): `collinear mass` histograms
`u = (u_+ + u_-)/2` instead of `-ln(m'/m)`, `collinear selection` takes the
`p_T` decision on `x p_T^{pre}` and the `eta` decision on the pre-FSR muons.

| model, fiducial `pT` > 25, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −1.01 ± 0.78 | +1.86 ± 1.50 |
| MC-conditional, inclusive in `m_pre` + `A(m)` | −0.31 ± 0.87 | +2.61 ± 1.75 |
| **MC-conditional, banded + `A(m)`** | **−0.17 ± 0.84** | **+1.83 ± 1.74** |
| `cond`: true mass, true selection (per-leg record) | +0.29 ± 0.84 | +2.08 ± 1.74 |
| `cond`: collinear mass | +0.22 ± 0.83 | +0.91 ± 1.74 |
| `cond`: collinear selection | +0.42 ± 0.83 | +2.11 ± 1.74 |
| `cond`: collinear mass + selection | +0.10 ± 0.84 | +0.80 ± 1.74 |
| **per-leg, `mc` `D` (numerical root)** | **+1.23 ± 0.83** | **−2.39 ± 1.74** |
| per-leg, analytic `D` (data cfg) | +1.16 ± 0.83 | −2.89 ± 1.74 |
| per-leg, `mc` `D`, `du` = 4e-5 | +1.24 ± 0.83 | −2.36 ± 1.74 |
| per-leg, `mc` `D`, 2 GeV bands | +1.20 ± 0.83 | −1.69 ± 1.74 |
| inclusive `mc` standalone kernel + `A(m)` | −3.18 ± 0.88 | +2.19 ± 1.82 |
| inclusive empirical kernel + `A(m)` | −3.48 ± 0.88 | +3.39 ± 1.81 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| **the machinery: per-leg `mc` − `cond`: true** | **+0.95** | **−4.47** |
| … `z = x_+ x_-` (`cond`: collinear mass − `cond`: true) | −0.07 | −1.17 |
| … the acceptance decision (`cond`: collinear selection − true) | +0.13 | +0.03 |
| … both (`cond`: collinear mass + selection − true) | −0.19 | −1.28 |
| … **independent legs** (per-leg `mc` − `cond`: collinear both) | **+1.14** | **−3.19** |
| the kernel physics: per-leg `data` − per-leg `mc` | −0.07 | −0.50 |
| … the same difference inclusively | +0.97 | −0.83 |
| standalone against the sample's own `K` (inclusive rows) | +0.31 | −1.20 |
| the per-leg record against the full one (`cond`: true − MC-cond. banded) | +0.46 | +0.25 |
| banding the MC-conditional kernel (banded − inclusive) | +0.14 | −0.78 |
| the square-root grid (`du` 4e-5 − 2e-5) | +0.01 | +0.03 |
| the kernel band width (2 GeV − 1 GeV) | −0.03 | +0.71 |

The **independent-legs** row is the model's whole joint law against the
generator's: the model draws `u_+` and `u_-` independently from `D` *and*
independently of the pre-FSR `p_T` pair, integrating them against the aggregate
`G(u_+, u_-|m)`, whereas `cond`: collinear both keeps the generator's own joint
`(u_+, u_-, p_T^{pre})` event by event. It also carries the one floor the
`cond` rows do not, the standalone kernel against the sample's own, measured at
+0.31 / −1.20 by the two inclusive rows.

**Verdict.** The factorisation and the selection-conditional construction are
validated to **+0.95 MeV on `m_Z`** -- at the level of the two floors the test
carries, +0.46 from the per-leg record's own statistics and +0.31 from the
standalone kernel against the sample's, so of order +0.2 MeV once they are
taken out -- and are **not** validated on `Γ_Z`, where they cost **−4.5 MeV**
(−3.3 MeV with the standalone floor removed). Three quarters of that is
treating the two legs as independent; one quarter, −1.3 MeV, is `z = x_+ x_-`.
The acceptance *decision* costs nothing (+0.13 / +0.03), confirming the 1.9e-4
closure of `A` measured on the record. All of it is the same power correction:
a wide-angle photon takes energy from both muons at once, so it correlates the
legs *and* makes the pair mass fall further than `x_+ x_-` alone.

The selected `<u|m>` says the same thing without a fit: `D_mc` gives
0.938-0.950 of the MC's and the analytic `D` 0.935-0.951, the two agreeing to
0.3 % (table above, figure `08_mean_u`). The deficit does not move when the QED
is made identical to the MC's, so it is not the kernel.

What removes it is the next section: a two-leg law with the correct correlation,
taken from the exact O(α) matrix element.

## The correlated two-leg density

`D(x_+) D(x_-)` is the collinear limit, and under a lepton `p_T` cut its one
defect is that the two momentum fractions are **independent draws**. They are
not. At O(α), with one photon, in the pre-FSR rest frame,

```
m'^2 = (Q - k)^2 = m^2 (1 - 2 E_gamma/m) = z m^2        (exact, angle free)
E_+ + E_- + E_gamma = m   ->   x_+ + x_- = 1 + z        (exact)
```

so the two energy fractions sit on a **line**, not on the collinear hyperbola
`x_+ x_- = z`, and one number says where on it:

```
x_+ = 1 - (1-z) f ,   x_- = 1 - (1-z)(1-f) ,   f = (1 - beta_mu cos θ*)/2
```

with `cos θ*` the angle of the `mu-` to the photon in the `mu mu` rest frame --
the variable `fsr_analytic._T_rad`/`_T_pair` already use, in which the two
propagators are `2 p_-.k = s(1-x_+)` and `2 p_+.k = s(1-x_-)`. `f = 0` and
`f = 1` are the collinear end points, all on one leg, which is what `D (x) D`
gives at O(α); the ~`1/L` of the rate in between is the recoil, where the
photon takes energy from **both** muons.

### The sharing density in closed form

`T(s, z, c) (1 - beta_mu^2 c^2)^2` is a **quadratic in `c^2`** -- the two
propagators are linear in `c` and the numerators quadratic -- so three
evaluations of `_T_pair` per `z` give the matrix element exactly
(`fsr_analytic.share_coeffs`), verified against direct evaluation at the
**1e-14** level from `z` = 1 - 1e-6 down to the `2 m_mu` threshold, where
`beta_mu` = 0.855. In the massless limit it collapses to the textbook form

```
T ~ (x_+^2 + x_-^2)/((1-x_+)(1-x_-))  ~  (1 + zeta c^2)/(1 - c^2) ,
zeta = ((1-z)/(1+z))^2 ,
```

and `share_nodes` integrates it on the collinear-resolving substitution
`1 - beta_mu c = (1 - beta_mu) e^t`, i.e. `f = (1 - beta_mu) e^t/2`: the nodes
are uniform in `ln f` over the `L = ln(z m^2/m_mu^2)` e-folds between the mass
regulator `f_min = (1 - beta_mu)/2` = 1.3e-6 and `1/2`. Three properties:

* the **marginal identity** `int dc p_1(z, c) = R_1(z)` holds to **8e-12** at
  `z` = 0.999, 0.99, 0.9, 0.5, 0.05 (`00_corr.txt`). `r1_exact` for a whole
  array of `z` is the same closed form, `r1_fast`, 1e-11 over the full range
  and 200x faster; it is what the matched construction's hard spectrum uses;
* the density is **exactly symmetric** under `f -> 1 - f` (1e-16 in both the
  vector and the axial current): a neutral current is C even, so the sharing
  carries no charge asymmetry and only the half branch is tabulated;
* the vector/axial decomposition changes it by **<= 1e-5**, the same
  `m_mu^2/s` suppression the inclusive `R_A/R_V` check finds.

### The model

`z` is drawn from `K(z)` -- untouched -- and the sharing from `p_1(f|z)`:

```
K_sel(u | m) = K(u | m) Gbar(u | m) ,     u = -ln(m'/m) ,   z = e^{-2u}
Gbar(u | m)  = int df p_1(f|z) G(u_+(z,f), u_-(z,f) | m)
A(m)         = int du K(u|m) Gbar(u|m)
```

with the **same** `G` and the **same** `h` table. Two things are exact by
construction and neither is fitted: `int df p_1 = 1` at every `z`, so the `z`
marginal is the kernel that was handed in and the inclusive fit is unchanged;
and the mass is `z` itself, not `x_+ x_-`. The selected kernel is therefore the
inclusive kernel reweighted atom by atom, which is also why it is cheaper than
the per-leg form -- the double sum over the leg ladder is gone, and `Gbar` is
one 1-D ladder per band.

**The boson-kinematics interface does not change.** The pass decision stays
`x_q p_T,q^pre > p_T^cut`, i.e. `u_q < b_q`, so `h(a_+, a_-|m)` is the same
table with the same format. The exact three-body kinematics also rotate the two
muons, and that is *measured* to cost nothing: `A(post-FSR cut)/A(collinear
x p_T) - 1` = 1.9e-4 on the record (check (e) above) and +0.13 / +0.03 MeV at
fit level (`cond: collinear selection` minus `cond: true`). Nothing about the
`p_T^Z`, `y_Z` or decay-angle content of `G` has to be added. The other
independence the model assumes -- FSR ⊥ the boson kinematics at fixed `m` -- is
measured directly on the record by decorrelating the two blocks within a band:
`A` moves by <= 3e-4 and the selected `<u>` by <= 0.35 %, the size of the
permutation noise itself (0.2-0.5 % per draw).

### Against the generator

`00_corr.txt`, figures `12_share`, `13_share_u`, `14_joint`. Single-photon
events are tagged by `k^2 = (Q_pre - p'_+ - p'_-)^2 ~ 0` -- **63.3 %** of the
radiating events. On them the line constraint is exact:
`x_+ + x_- - (1+z)` is ±2.3e-7, the float32 storage of the record, against a
-8.0e-3 one-percent tail over all radiating events, which is the
`(sum k)^2/m^2` of the multi-photon configurations.

The sharing density itself, `P(0.01 < f < 0.99)` in the peak band:

| `u` range | MC, 1γ | exact O(α) | MC/exact | MC, all | all/1γ |
|---|---|---|---|---|---|
| 1e-3 - 1e-2 | 0.3476 | 0.3674 | 0.946 | 0.4607 | 1.326 |
| 1e-2 - 0.05 | 0.3440 | 0.3681 | 0.935 | 0.4618 | 1.342 |
| 0.05 - 0.2 | 0.3395 | 0.3710 | 0.915 | 0.4639 | 1.366 |
| 0.2 - 0.6 | 0.3407 | 0.3722 | 0.915 | 0.4651 | 1.365 |

so the exact matrix element reproduces the generator's own single-photon
sharing to 6-9 % -- that last 6-9 % is Photos against the exact ME, not the
construction -- and the generator's **all**-emission sharing is 35 % wider
still. `D (x) D` puts `P(0.01 < f < 0.99)` at **0.124**, so the exact sharing
recovers three quarters of the distance to the MC and the rest is the
multi-emission recoil (two photons, one on each leg).

The leg law that follows, at `m` = 91.105:

| `t` | `P(u_leg>t)` MC | model | `D` | joint/prod MC | model | `D (x) D` |
|---|---|---|---|---|---|---|
| 1e-5 | 0.35986 | 0.29942 | 0.26877 | 1.99 | 2.42 | 1 |
| 1e-4 | 0.27940 | 0.26299 | 0.21810 | 2.22 | 2.08 | 1 |
| 1e-3 | 0.19842 | 0.18758 | 0.16388 | 2.52 | 2.25 | 1 |
| 1e-2 | 0.11956 | 0.11489 | 0.10623 | 2.90 | 2.41 | 1 |
| 0.05 | 0.06821 | 0.06681 | 0.06465 | 3.19 | 2.39 | 1 |
| 0.2 | 0.03097 | 0.03073 | 0.03149 | 3.05 | 1.83 | 1 |

`<u_leg>` = 2.5734e-2 against the MC's 2.5861e-2 (0.5 %), where `D` -- which is
*defined* by `D (x) D = K` and is not a per-leg spectrum -- sits at 2.6940e-2,
4.2 % high. The model reproduces the **physical** per-leg marginal *and* most of
the joint correlation, which the product cannot do at the same time.

### The model against the MC, band by band

Weighted by the band population, as the fit sees them (`00_bands.txt`):

| `m_pre` band | `A` (MC) | `A` model / MC | `<u>` (MC) | `<u>` model / MC |
|---|---|---|---|---|
| | | corr `mc` / corr `data` / per-leg `mc` | | corr `mc` / corr `data` / per-leg `mc` |
| 60-70 | 0.19855 | 0.9927 / 0.9934 / 0.9914 | 5.699e-3 | 0.9711 / 0.9717 / 0.9379 |
| 70-80 | 0.29151 | 1.0006 / 1.0012 / 0.9991 | 7.868e-3 | 0.9846 / 0.9857 / 0.9445 |
| 80-86 | 0.34070 | 1.0006 / 1.0010 / 0.9991 | 9.379e-3 | 0.9841 / 0.9867 / 0.9412 |
| 86-90 | 0.36644 | 0.9999 / 1.0002 / 0.9983 | 10.146e-3 | 0.9946 / 0.9954 / 0.9497 |
| 90-92 | 0.37546 | 0.9999 / 1.0002 / 0.9984 | 10.539e-3 | 0.9915 / 0.9936 / 0.9463 |
| 92-96 | 0.38396 | 1.0003 / 1.0006 / 0.9988 | 10.866e-3 | 0.9925 / 0.9944 / 0.9470 |
| 96-105 | 0.40163 | 0.9983 / 0.9987 / 0.9969 | 11.762e-3 | 0.9890 / 0.9901 / 0.9423 |
| 105-120 | 0.42811 | 0.9930 / 0.9936 / 0.9915 | 13.288e-3 | 0.9945 / 0.9948 / 0.9454 |
| 120-140 | 0.45573 | 0.9927 / 0.9936 / 0.9912 | 15.513e-3 | 0.9863 / 0.9822 / 0.9362 |
| `A`-weighted | | 0.9975 / 0.9980 / 0.9960 | | **0.9887** / **0.9892** / 0.9435 |

The selected `<u|m>` goes from 0.944 of the MC's to **0.989**, and 0.992-0.995
in the four bands that carry the peak; `A(m)` from 0.996 to 0.998. The `mc` and
`data` columns agree to 0.2 %, so what is left is not the kernel.

### The matched construction, and the matching scale

The alternative composition splits the kernel at a scale `u_c` instead of
treating the whole loss as one photon:

```
K = D_< (x) D_< (x) H_> ,   H_> = (1 - N_>) delta + h_>(u) theta(u - u_c)
```

with `h_>` the **exact** O(α) emission (`r1_fast`) carrying its exact sharing
and `D_<` a soft remainder that is collinear-independent leg by leg. `D_<` is
not a free object: the deconvolution and the square root are one spectral
operation, `D^_< = sqrt(K^ / H^_>)`, on a uniform grid (`fsr_perleg.soft_leg`).
The **marginal identity closes to 1e-8** in `<u>` at every matching scale:

| `u_c` | `N_>` | `<u>` model / `<u>` K |
|---|---|---|
| 0.1 | 0.0618 | 0.99999999 |
| 0.03 | 0.1205 | 0.99999999 |
| 0.01 | 0.1808 | 0.99999999 |
| 3e-3 | 0.2495 | 0.99999999 |
| 1e-3 | 0.3129 | 0.99999999 |

`N_> < 1/2` keeps `|H^_>|` away from zero and is what limits how low `u_c` can
go; `u_c -> 0` is the single-photon model itself. The grid is `MATCH_DU` = 1e-4,
coarser than the square root's own 2e-5 on purpose: the *analytic* kernel is a
comb on a 2e-5 grid above `u` ~ 2e-3 (its cells are geometric) and a comb is not
infinitely divisible, so `kernel_grid` hands the square root the kernel's own
integral per grid cell instead of a deposited atom.

The selection weight is then the conditional expectation
`Gbar(u) = <G(u_+, u_-) | u>` over the configurations, numerator and denominator
binned the same way so the discretisation cancels; the inner double sum over the
soft ladder is `n_v^2` dense matrix products `M_j G M_k^T` (`shift_matrix`) and
not `n_leg^2` per hard configuration. Because it is a systematic variant and its
ratio to the single-photon model is smooth in `m`, it is built on a 12-mass grid
and read off band by band (`variant_ratio`). Discretisation: the matched /
single ratio of the selected `<u>` at the peak moves by 0.2 % between `n_u` =
1500 and 200 bins and by 0.1 % between `n_v` = 8 and 32.

### The fit benchmark

`fit_gen.py fit --suite perleg`, the same 9.87 M selected gen events in
60-120 GeV, five Legendre shape terms, `nm` = 8192. Every row is the **same**
run on the **same** events.

| model, fiducial `pT` > 25, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −1.01 ± 0.78 | +1.86 ± 1.50 |
| MC-conditional, banded + `A(m)` | −0.17 ± 0.84 | +1.83 ± 1.74 |
| `cond`: true mass, true selection | +0.29 ± 0.84 | +2.08 ± 1.74 |
| `cond`: collinear selection | +0.42 ± 0.83 | +2.11 ± 1.74 |
| `cond`: collinear mass + selection | +0.10 ± 0.84 | +0.80 ± 1.74 |
| **corr, `mc` `K`** | **+0.47 ± 0.83** | **−0.41 ± 1.75** |
| **corr, `data` `K`** | **+1.07 ± 0.83** | **−1.11 ± 1.74** |
| corr, `mc` `K`, matched `u_c` = 0.03 | +0.28 ± 0.83 | −0.33 ± 1.74 |
| corr, `mc` `K`, matched `u_c` = 0.01 | +0.21 ± 0.83 | −0.10 ± 1.74 |
| corr, `mc` `K`, matched `u_c` = 0.003 | +0.46 ± 0.83 | −0.38 ± 1.75 |
| per-leg, `mc` `D` (the collinear product) | +1.23 ± 0.83 | −2.39 ± 1.74 |
| per-leg, analytic `D` (data cfg) | +1.16 ± 0.83 | −2.89 ± 1.74 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| **the machinery: corr `mc` − `cond`: true** | **+0.18** | **−2.49** |
| … the same, against `cond`: collinear selection | +0.05 | −2.52 |
| … before, per-leg `mc` − `cond`: true | +0.95 | −4.47 |
| the sharing: corr `mc` − per-leg `mc` | −0.76 | +1.98 |
| the matching scale, `u_c` 0.003 → 0.03 | −0.19 | +0.06 |
| **the composition: matched (`u_c` 0.01) − single photon** | **−0.26** | **+0.31** |
| the kernel physics: corr `data` − corr `mc` | +0.60 | −0.71 |
| … the same difference inclusively | +0.97 | −0.83 |
| standalone against the sample's own `K` (the floor) | +0.31 | −1.20 |
| the per-leg record against the full one (the floor) | +0.46 | +0.25 |

**Verdict.** Replacing `D(x_+) D(x_-)` by the exact O(α) correlated two-leg
density at fixed `z` moves the machinery residual from +0.95 / −4.47 MeV to
**+0.18 / −2.49 MeV**, and with the standalone-kernel floor of +0.31 / −1.20
taken out, to **−0.13 / −1.29 MeV**. On `m_Z` the construction is now validated
at the level of its own floors. On `Γ_Z` the remaining −1.3 MeV is the
**multi-emission recoil**: the model shares the whole loss as if it were one
photon, and the generator's all-emission sharing is 35 % wider than its
single-photon sharing (table above) because a second photon on the other leg
does the same thing a wide-angle photon does. The two compositions of the
resummation bracket that ambiguity at **0.26 MeV on `m_Z` and 0.31 MeV on
`Γ_Z`**, and the matching scale itself at 0.19 / 0.06 MeV -- both far below the
±0.83 / ±1.74 MeV statistical error, so the answer is insensitive to how the
hard emission is matched to the soft ladder and the residual is not a
matching artefact.

`z = x_+ x_-` is gone by construction (the model carries `z`), the acceptance
decision costs nothing, and the leg independence -- three quarters of the old
residual -- is three quarters removed.

### What the detector level needs on top

The cut there is on the **reconstructed** `p_T`, so the resolution enters the
acceptance: the pass condition becomes `p_T^reco(x p_T) > p_T^cut` with
`p_T^reco` smeared, and `G` has to be built with the smearing folded in --
`G(u_+, u_-) = P(both reconstructed p_T above threshold | m)`, which is no
longer a pure generator quantity and no longer a step. The format above does
not change (it is still a joint survival function in the two thresholds); what
changes is who fills it. Also out of scope here: the `eta` cut is applied to
the reconstructed track, and the trigger/identification efficiency is a smooth
function of `(p_T, eta)` rather than a step.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
# 1. the per-leg gen dump (charge-matched legs; 400 MiniAOD files, ~20 min)
./run_perleg_dump.sh   0 199 submit50 14 &
./run_perleg_dump.sh 200 399 submit51 14 &
python3 merge_perleg.py -i "/ceph/submit/data/user/d/david_w/ZMass/zgen_perleg/pl_*.npz" \
        -o data/perleg_full.npz
# 2. D (x) D against K
python3 fsr_perleg.py check --n-leg 4000
# 3. h(a_+,a_-|m) and the selection-conditional kernels (numpy only, ~25 min)
./build_perleg.sh
# 4. the mc leg: the numerical square root of the standalone Photos kernel,
#    the per-leg kernels it feeds, and the MC's own conditional kernel read in
#    the model's variables (~3 min all together)
python3 fsr_perleg.py legsqrt --run data/photos/gen_mcMix.npz --check-band 20
./build_machinery.sh
# 5. the correlated two-leg kernels: the model for both configurations, then
#    the matching-scale scan (numpy only, ~40 min)
./build_corr.sh
# 6. the fit benchmarks (~4 min each) and the figures
./run_perleg_fit.sh physics
./run_perleg_fit.sh disc
./run_perleg_fit.sh machinery
./run_perleg_fit.sh corr
ssh submit51 "cd $Z && ./run_tf_z.sh --ceph python3 -u cmp_perleg.py \
     --kernel 'corr, mc K=data/kern_corr_mc_1gev.npz' \
     --kernel 'corr, data K=data/kern_corr_data_1gev.npz' \
     --kernel 'per-leg, mc D=data/kern_perleg_mc_1gev.npz' \
     --kernel 'per-leg, analytic D=data/kern_perleg_data_1gev.npz'"
# the Photos-like pair content, for the mc <-> data comparison
python3 fsr_perleg.py kernel --htable data/ht_pt25_1.0gev.npz \
     -o data/kern_perleg_paireik.npz --acceptance data/acc_perleg_paireik.json \
     --variant exp2nll --pair e mu --pair-table data/fsr/pairkern_eik.npz
```

`cmp_perleg.py` needs a node that can mount `/ceph` (the container's own hook
binds it whether or not `run_tf_z.sh` asks for it).

Figures: `01_angles` (the directions), `02_leg_x` (the measured per-leg `x`
against `D`), `03_leg_corr` (the legs are correlated), `04_z_vs_xx`
(`z = x_+ x_-`), `05_dconvd` (`D (x) D` against `K`), `06_ksel_peak`
(`K_sel(u|m)` in the peak band), `07_acceptance`, `08_mean_u`, `09_htable`,
`10_leg_D_mc` (the effective leg of `K_mc` against the analytic `D` and against
the physical per-leg spectrum), `11_dconvd_mc` (`D_mc (x) D_mc` against
`K_mc`), `12_share` (the exact O(α) sharing density against the generator's own
single-photon events), `13_share_u` (how often both legs lose, against the total
loss), `14_joint` (the leg correlation the sharing puts back), and
`00_perleg.txt`, `00_bands.txt`, `00_legsqrt.txt`, `00_corr.txt` with every
number above.

---

## Asymmetric cuts, and the resolution in the acceptance

The fiducial selection of a real Z channel is **not** the symmetric generator
cut of the sections above: it is a *leading* and a *trailing* muon threshold,
and it acts on the **reconstructed** `p_T`, not on the bare post-FSR one.  Both
generalisations live in one object, `fsr_perleg.PassRegion`, and nothing
downstream -- `ksel_band`, `gbar`, `matched_gbar`, the kernel builders -- knows
which selection it is serving: they take the array `PassRegion.grid(band)`
returns and read it exactly as they read the old survival function.

### The pass region

With `b_q = ln(p_T,q^pre / p_T^ref)` and the leg's loss `u_q`, the post-FSR
transverse momenta are `p_T^ref e^{b_q - u_q}` and the pass condition for
`leading > c_L`, `trailing > c_T` is

```
max(pT'_+, pT'_-) > c_L   and   min(pT'_+, pT'_-) > c_T ,
```

i.e. at fixed `(u_+, u_-)` the **union of two rectangles** in `(b_+, b_-)`,

```
A = {b_+ - u_+ > s_L , b_- - u_- > s_T} ,      s_c = ln(c / pT^ref)
B = {b_- - u_- > s_L , b_+ - u_+ > s_T} ,
A and B = {both over c_L}      (because c_L >= c_T)
```

so, with `S` the 2-D survival function of the `h` table and `P_q(c)` the
probability that leg `q` alone clears `c`,

```
G(u_+, u_-) = S(u_+ + s_L, u_- + s_T) + S(u_+ + s_T, u_- + s_L)
            - S(u_+ + s_L, u_- + s_L)
            = P_+(c_L) P_-(c_T) + P_+(c_T) P_-(c_L) - P_+(c_L) P_-(c_L) .
```

Three points about that form:

* **leading/trailing is decided after FSR** (and, at detector level, after the
  smearing).  The `max`/`min` form does that by construction; ordering the two
  legs *before* the losses would be a different -- and wrong -- selection;
* `c_L = c_T` collapses the three terms to one and the code returns the plain
  survival function, **bit for bit**, so every symmetric number of the sections
  above is reproduced unchanged;
* every term is a **product of two per-leg thresholds**.  That is what carries
  the construction over to a smooth pass probability: given each muon's own
  `(p_T, eta)` the two resolutions are independent, so the same three-term
  identity holds with `P_q(c)` a probability instead of an indicator.

`fsr_perleg.py gcheck` reads `G` off the table and compares it to a direct
weighted event count on the generator record.  At 25/10, peak band, on the grid
below, the two agree to **1e-9** -- the float32 storage of `h` -- at every
`(u_+, u_-)` from 0 to 0.5.

### One table per reference, not per cut

`p_T^ref` is the reference of the logarithmic axis and **not** a cut: a table
built at `p_T^ref = 10` serves 25/10, 25/25 and 10/10 alike, and which cuts are
applied is `PassRegion`'s business.  Two things follow.

* The table's floor `b_edges[0]` must sit below every threshold it will be
  asked about, and (for a resolution) below it by enough that an
  up-fluctuation cannot reach the cut: the default is `b_min = -0.3`, i.e.
  `p_T > 7.4` GeV for `p_T^ref = 10`, which is 11 core widths in the barrel and
  6 in the endcap.  The old tables cut at `b > 0` because an event under a
  *symmetric* step could never pass; that is exactly the line that changes.
* `G` has to be resolved where the leg radiator has its weight, which is at
  `u -> 0` **above each threshold**.  With two thresholds there are two such
  places, so `b_grid` replicates the fine part of the pattern at each
  `ln(c/p_T^ref)`.  The replication is exact -- `anchor + pattern` is the same
  expression the shifted reading evaluates -- so a fine node shifted by
  `s_L - s_T` lands on another node and the reading of `S` stays exact there
  rather than interpolating (164 of 164 nodes below `b = 0.5`).
  `b_grid((0,), 0, 3)` reproduces `B_EDGES` and the whole earlier chain
  unchanged.

The 25/25 kernel rebuilt on the `p_T^ref = 10` anchored table reproduces the
published `p_T^ref = 25` acceptance to **1e-8** in every band and the fit to
**+0.02 / -0.02 MeV**, so the generalisation is a no-op where it has to be.

### The 25/10 benchmark

`fit_gen.py fit --suite perleg --acc-pt 25 --acc-pt-trail 10`: 12.07 M selected
gen events (against 10.07 M at 25/25), 11.99 M in 60-120, five Legendre shape
terms, `nm = 8192`, offsets from the generator's own `m_Z` = 91.153510,
`Gamma_Z` = 2.493202.  Every row is the **same** run on the **same** events.
The reference is the MC's own conditional kernel, measured on the *same*
record and banded like the model (`fsr_perleg.py condker --gen
genmerged_full.npz --mode true`), so unlike the published `cond:` rows it
carries no per-leg-record floor.

| model, `pT` > 25/10, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −0.58 ± 0.70 | +1.18 ± 1.37 |
| MC-conditional, banded + `A(m)` | +0.60 ± 0.77 | +1.14 ± 1.60 |
| **corr, `mc` `K`** | **+0.47 ± 0.76** | **+0.35 ± 1.60** |
| **corr, `data` `K`** | **+1.02 ± 0.77** | **−0.19 ± 1.60** |
| corr, `mc` `K`, the **symmetric 25/25** kernel | +7.17 ± 0.76 | −5.95 ± 1.57 |

and the same suite at 25/25, on the same table, for reference:

| model, `pT` > 25, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −0.87 ± 0.78 | +1.12 ± 1.50 |
| MC-conditional, banded + `A(m)` | −0.24 ± 0.84 | +0.92 ± 1.74 |
| **corr, `mc` `K`** | **+0.49 ± 0.83** | **−0.43 ± 1.74** |
| **corr, `data` `K`** | **+1.07 ± 0.83** | **−1.11 ± 1.74** |
| corr, `mc` `K`, `p_T^ref` = 25 table (the published build) | +0.47 ± 0.83 | −0.41 ± 1.75 |
| `cond`: true, per-leg record (the published reference) | +0.29 ± 0.84 | +2.08 ± 1.74 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` | |
|---|---|---|---|
| **the machinery at 25/10** (corr `mc` − MC-conditional) | **−0.13** | **−0.79** | |
| … the same at 25/25 | +0.73 | −1.35 | |
| the kernel physics at 25/10 (corr `data` − corr `mc`) | +0.55 | −0.54 | |
| … the same at 25/25 | +0.58 | −0.68 | |
| **ignoring the asymmetry** (the 25/25 kernel on the 25/10 selection) | **+6.70** | **−6.30** | |
| the `p_T^ref` = 10 anchored table (25/25, against the published build) | +0.02 | −0.02 | |

**The interplay is weaker at 25/10, and the model is better there, not worse.**
The trailing cut at 10 GeV rejects **1.5 %** of the pairs whose leading muon is
already over 25 GeV (the symmetric cut rejects 18.0 % of the same pairs), so the
selection is very nearly a *single*-leg threshold and the two
defects the correlated model was built to repair -- leg independence and
`z = x_+ x_-` -- have much less to act on.  Quantitatively, band by band
(`00_bands_2510.txt`, figures `07_acceptance_2510`, `08_mean_u_2510`):

| | `A` model/MC (`A`-weighted) | selected `<u>` model/MC |
|---|---|---|
| 25/25, corr `mc` / `data` | 0.9975 / 0.9980 | 0.9887 / 0.9892 |
| **25/10, corr `mc` / `data`** | **0.9968 / 0.9971** | **0.9994 / 0.9936** |

`A(m)` closes to the same 0.3 %, and the selected `<u|m>` -- which at 25/25 was
1.1 % low and was the whole residual -- closes to **0.06 %** at 25/10, within
0.7 % in every band from 60 to 120 GeV.  The fit says the same thing: the
machinery residual is −0.13 / −0.79 MeV, i.e. consistent with zero on both
parameters at the ±0.76 / ±1.60 MeV statistical error of the test.

What is **not** small is getting the pass region wrong: feeding the symmetric
25/25 kernel to the 25/10 selection -- the same QED, the same `h` table, only
the wrong region -- moves the fit by **+6.7 / −6.3 MeV**.  That is the size of
the thing this section generalises.

### The reconstructed `p_T` resolution

`ptres.py measure` on the CVH two-track Z production
(`dymc_8p5M_260906_v2`, 3.67 M of 3.73 M candidates after the standard
selection -- finite `sigma_m > 0`, >= 8 valid hits on each leg, `chi2/ndof < 3`
-- and both legs gen-matched, i.e. 7.33 M muons).  The measured quantity is
**per muon**,

```
r = pT_reco / pT_gen - 1 ,
```

against the charge-matched **bare post-FSR** gen muon of the same candidate
(`Mu{plus,minus}gen_pt`, the status-1 muon the maker matches within dR < 0.1),
binned in **gen** `p_T` and `eta`: binning in reconstructed `p_T` would bin the
residual on a variable that contains it.  The production's only kinematic cut
is the 60-120 GeV window on the input-track pair mass; restricting the gen pair
mass to 85-95 moves the core width by **< 1 %** in every cell, so that window
does not censor the measurement.

The core is the iterated `+-2 sigma` truncated moment, started from the
interquartile range and corrected for the truncation.  `sigma_pT/pT` runs

| \|eta\| | `pT` = 9 | 28 | 42 | 145 GeV | 4-par. fit dev. (10-100 GeV) |
|---|---|---|---|---|---|
| 0.0-0.9 | 9.00e-3 | 9.94e-3 | 11.07e-3 | 20.8e-3 | 0.97 % |
| 0.9-1.6 | 14.80e-3 | 16.86e-3 | 17.99e-3 | 28.3e-3 | 1.26 % |
| 1.6-2.1 | 17.29e-3 | 19.66e-3 | 21.44e-3 | 37.3e-3 | 1.35 % |
| 2.1-2.4 | 26.16e-3 | 32.33e-3 | 37.73e-3 | 84.9e-3 | 3.41 % |

and the 4-parameter form
`(sigma/pT)^2 = a^2 + c^2 pT^2 + b^2/(1 + d^2/pT^2)`, fitted per `eta` band,
reproduces it to **1 % over 10-100 GeV** in the three inner bands and 3.4 % in
2.1-2.4; over the whole 8-200 GeV range the worst cell is 4.0 %, always one of
the two end cells, which carry a few thousand muons each.  That is inside the
5 % target, and the acceptance only ever uses `sigma` within a few per cent of
the two thresholds.

| \|eta\| | `a` [1e-3] | `b` [1e-3] | `c` [1e-5] | `d` |
|---|---|---|---|---|
| 0.0-0.9 | 8.983 | 7.814 | 11.868 | 71.65 |
| 0.9-1.6 | 15.153 | 9.686 | 15.460 | 37.53 |
| 1.6-2.1 | 17.110 | 10.916 | 22.810 | 30.90 |
| 2.1-2.4 | 24.072 | 17.347 | 57.150 | 18.85 |

**The law is not Gaussian and the acceptance sees the tail.**  In the
standardised variable `s = (r - mu)/sigma`, over 10-60 GeV,

| \|eta\| | `P(s < -3)` | `P(s > +3)` | `P(s < -5)` | `P(s > +5)` |
|---|---|---|---|---|
| 0.0-0.9 | 0.61-0.99 % | 0.78-1.06 % | 1.0-3.1e-3 | 1.1-4.0e-3 |
| 0.9-1.6 | 0.54-1.18 % | 0.76-1.14 % | 1.2-5.6e-3 | 1.0-5.2e-3 |
| 1.6-2.1 | 0.61-0.91 % | 0.87-1.08 % | 0.8-3.0e-3 | 1.2-2.0e-3 |
| 2.1-2.4 | 0.81-1.57 % | 1.29-2.71 % | 1.0-5.4e-3 | 3.3-11.7e-3 |

against a Gaussian's 0.135 % and 2.9e-7.  The model therefore carries two
renderings, and the benchmark runs both: `gauss` (the fitted core only) and
`shape` (`r = mu + sigma s` with `s` the measured standardised shape of the
same `eta` band, pooled over `p_T` cells).  The mean is a two-term fit
`mu = m0 + m1/pT`; it is below 3e-4 everywhere except the 2.1-2.4 band above
50 GeV, where the fit reaches -0.68e-3 at 90 GeV.

**The CVH covariance can almost be used directly.**  The per-leg pull
`(q/p - q/p_gen)/(sigma_rel |q/p|)` with `sigma_rel = Jpsi_sigmarel{plus,minus}`
-- the two-track fit's own reported relative momentum resolution -- has core
width **0.9558** and core mean -0.005 over 7.33 M legs, i.e. the covariance
**over**states the core by 4.4 %, by `eta` band 0.948 / 0.954 / 0.963 / 0.979.
That is the same statement the tails make: the reported variance includes the
non-Gaussian tail that the core estimator excludes.  For the acceptance it
means the covariance is usable as a first approximation but not as the core
width, and the shape has to come from a measurement like this one anyway.

### The pass probability, and the `h4` table

Per leg the step `x_q p_T,q > c` becomes a probability.  With
`v = ln(p_T^post / c) = b_q - u_q - ln(c/p_T^ref)` the distance to the
threshold in the leg's own logarithmic variable,

```
Pi(v; c, eta) = P( r > e^{-v} - 1 | p_T = c e^v, eta )
```

-- the resolution is evaluated at the **true** post-FSR `p_T`, which is what
the law of `r` is conditioned on -- and the union-of-rectangles identity above
goes through unchanged with `P_q(c) = Pi(v_q; c, eta_q)`.

`Pi` depends on `eta`, so the boson-kinematics table has to carry the `eta` of
each leg.  That is the **whole** extension the detector level forces:

```
h4[k, a, i, c, j]  =  weight of m band k with (b_+, eta_+) in cell (i, a) and
                      (b_-, eta_-) in cell (j, c), / the band's total weight,
b_mean[k, a, i]    =  that cell's own mean b, per leg marginal
```

one more axis per leg on a coarse `eta` grid, symmetrised under the
simultaneous swap of both axes of the two legs, and

```
G(u_+, u_-) = sum_terms sign * Pi_a^T h4[a, :, c, :] Pi_c  ,
Pi_a[i, p] = Pi(b_mean[a, i] - b_p - s ; c, eta_a)
```

is `n_eta^2` matrix triple products per band -- 0.7 s per band at
`n_eta = 4`, `n_b = 139`, reading onto the step table's own 494-node grid.
The cell's **mean** `b` rather than its midpoint removes the first-order
discretisation error and costs one number per `(eta, b)` cell.

`h4` is deliberately coarser in `b` than the step table (1e-2 near each
threshold, 4e-2 beyond, against 1e-3 / 5e-3 / 2e-2): once the step is smoothed
by `sigma ~ 1e-2`, `G` is a smooth `h` read through a sharp kernel, not a sharp
function, so it is `h`'s own smoothness that sets the grid.  The default table
is 152 bands x (4 x 139)^2 = 188 MB in memory, 3.0 MB on disk.

`gcheck` against a direct event count with the **same** resolution object on
both sides -- so what is measured is the table and nothing else -- at 25/10 in
the peak band:

| `(u_+, u_-)` | default `b` grid | fine `b` grid (5e-3 out to 0.5) |
|---|---|---|
| (0, 0) | -1.5e-4 | -7.8e-5 |
| (0.01, 0.01) | -7.9e-5 | -4.3e-5 |
| (0.2, 0.2) | +4.5e-4 | +2.5e-6 |
| (0.5, 0.2) | +5.1e-4 | +1.7e-4 |
| (0.5, 0.5) | +4.4e-4 | +1.5e-3 |

on `G ~ 0.31-0.45`.  The residual of the default grid is visible as a stripe
pattern in `20_passregion_2510sm` (third panel, `resolution - step`): above
`b = 0.2` the `h4` cells are 4e-2 wide and their midpoints alias against the
step table's own 1e-3 grid, at the 2e-3 level in `G` where the leg weight is
1e-2 of the total.  What that costs at fit level is the `fine b grid` row of
the toy benchmark below; the `8 eta bands` row is the same statement for the
`eta` grid, which is the resolution model's grid and not the boson's.

### The resolution toy

The acceptance effect is isolated from the mass resolution the way the real
likelihood separates them: the gen sample's **post-FSR muon `p_T`** are smeared
with the measured law, the selection is taken on the **smeared** `p_T`, and the
**fitted mass is left unsmeared** -- in the data likelihood the mass resolution
is the CF's job and the acceptance is this.  The draw is seeded
(`fit_gen.SMEAR_SEED`) and the two legs are drawn in a fixed order, so the fit,
the MC-conditional reference kernel and its `A(m)` see the *same* selection
event by event.

The model rows are the same correlated two-leg kernel with two different pass
regions: the **step** one (the current model, which ignores the resolution) and
the **smeared** one.  Their difference is the whole effect.

| model, smeared 25/25 selection, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −0.84 ± 0.78 | +1.07 ± 1.50 |
| MC-conditional, banded + `A(m)` (the smeared selection's own) | −0.21 ± 0.84 | +0.94 ± 1.75 |
| **corr, `mc` `K`, smeared acceptance** | **+0.40 ± 0.83** | **−0.37 ± 1.75** |
| corr, `mc` `K`, **step** acceptance | +0.52 ± 0.83 | −0.51 ± 1.75 |
| corr, `data` `K`, smeared acceptance | +1.08 ± 0.83 | −1.20 ± 1.75 |
| corr, `data` `K`, **step** acceptance | +1.10 ± 0.83 | −1.18 ± 1.75 |

| model, smeared 25/10 selection, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −0.58 ± 0.70 | +1.17 ± 1.37 |
| MC-conditional, banded + `A(m)` | +0.64 ± 0.77 | +1.06 ± 1.60 |
| **corr, `mc` `K`, smeared acceptance** | **+0.48 ± 0.76** | **+0.30 ± 1.60** |
| corr, `mc` `K`, **step** acceptance | +0.46 ± 0.76 | +0.33 ± 1.60 |
| corr, `data` `K`, smeared acceptance | +1.02 ± 0.77 | −0.25 ± 1.60 |
| corr, `data` `K`, **step** acceptance | +1.00 ± 0.77 | −0.21 ± 1.60 |
| corr, `mc` `K`, smeared, **Gaussian core only** | +0.47 ± 0.76 | +0.30 ± 1.60 |
| corr, `mc` `K`, smeared, **fine `b` grid** (5e-3 out to 0.5) | +0.48 ± 0.76 | +0.31 ± 1.60 |
| corr, `mc` `K`, smeared, **8 `eta` bands** | +0.48 ± 0.76 | +0.30 ± 1.60 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| **step → smeared acceptance, 25/25**, `mc` `K` / `data` `K` | **−0.12** / **−0.02** | **+0.14** / **−0.02** |
| **step → smeared acceptance, 25/10**, `mc` `K` / `data` `K` | **+0.02** / **+0.02** | **−0.03** / **−0.04** |
| the tail of `r`: `shape` − `gauss` pass probability (25/10) | +0.01 | +0.00 |
| the `h4` `b` grid: fine − default (25/10) | +0.00 | +0.01 |
| the `h4` `eta` grid: 8 bands − 4 bands (25/10) | +0.00 | +0.00 |
| the machinery under the smeared selection (corr `mc` − MC-cond.) at 25/10 | −0.16 | −0.76 |
| … the same under the step selection at 25/10 | −0.13 | −0.79 |
| … the same at 25/25 (smeared / step selection) | +0.61 / +0.73 | −1.31 / −1.45 |

and the same toy run with a **Gaussian** smearing instead of the measured shape
(so the tail is absent from the selection as well as from the model) gives
+0.40 / +0.41 / +0.39 for the `gauss` / `shape` / step acceptance -- the same
picture, 0.02 MeV wide.

**Verdict.** Ignoring the resolution in the acceptance -- using the step pass
region when the cut actually acts on the reconstructed `p_T` -- costs
**0.12 MeV on `m_Z` and 0.14 MeV on `Γ_Z` at 25/25**, and **0.02 / 0.04 MeV at
25/10**.  Both are well below the ±0.8 / ±1.6 MeV statistical error of this
test and below the machinery residual itself, but they are not zero and the
25/25 number is comparable to the `+0.18 / −2.49` closure the correlated
construction achieves, so the smeared pass region is the right default even
though the step one would not have broken anything.  It costs one `eta` axis on
the interface and 0.7 s per mass band.

Why 25/10 is six times smaller than 25/25: the resolution moves `A(m)` by
−0.56 % at 70 GeV to −0.10 % at 110 at the symmetric cut, a clear **slope**
across the window, and by −0.07 % to −0.02 % at the asymmetric one, which is
nearly flat -- and a flat multiplicative change of `A(m)` cannot move a mass.
At 25/25 both legs sit on the threshold; at 25/10 only the leading one does.

Three things the benchmark also settles:

* **the tail of `r` does not matter for the acceptance** (0.01 MeV between the
  measured shape and its Gaussian core), even though it is 10-1000x the
  Gaussian beyond 3 sigma.  What the acceptance integrates is the *pass
  probability* near the threshold, and the tail contributes a smooth, slowly
  varying few per mille there.  It is the mass resolution, not the acceptance,
  that will care about the tail;
* **the `h4` discretisation is free** at fit level: a `b` grid twice as fine
  (785 MB against 188) and an `eta` grid twice as fine both move the fit by
  < 0.01 MeV, even though the coarse `b` grid is visibly striped in `G` at the
  2e-3 level (figure `20_passregion_2510sm`);
* **the machinery residual does not change** when the selection is smeared
  (−0.16 against −0.13 MeV at 25/10, +0.61 against +0.73 at 25/25), i.e. the
  resolution in the acceptance and the FSR factorisation are independent
  questions, as the construction assumes.

### The interface for SCETLib + DYTurbo, with resolution

The two sides stay separable, and the split is exactly where it was:

**the boson-kinematics provider** hands over, per pre-FSR mass band,

```
h4(b_+, eta_+, b_-, eta_- | m)     the joint weight of the two muons'
                                   (ln pT/pT_ref, eta) at that mass,
                                   normalised to the band's TOTAL weight
b_mean(eta, b | m)                 the cell's own mean b, per leg marginal
```

on a `b` grid that reaches below the lowest threshold (`b_min = -0.3`) and a
coarse `|eta|` grid.  Everything about the boson -- `p_T^Z`, `y_Z`, the decay
angles with their angular coefficients, the PDFs -- is integrated into it and
nothing else is.  The 2-D `h(b_+, b_-|m)` of the earlier sections is its
`eta`-marginal and is what a step acceptance needs;

**the detector side** hands over

```
sigma_pT/pT(pT, eta)   and the shape of r = pT_reco/pT_gen - 1
```

as one `ptres.Resolution` object, which the model calls only through
`pass_prob(v, c, eta band)`.  It never sees `h4`, and the provider never sees
the resolution.

**The QED side does not change at all**: `K(z)`, the exact O(alpha) sharing
`p_1(f|z)`, `Gbar = int df p_1 G`, `A(m) = int du K Gbar` are the same objects
and the same code.

Three things the interface now *has* to say that it did not before:

* the `b` axis must extend **below** the lowest cut, because a muon under the
  threshold can be promoted over it.  A table that starts at the cut cannot be
  used with a resolution at all;
* the `eta` grid is the resolution's grid, not the boson's.  Its coarseness is
  a modelling choice of the detector side and is quantified at fit level below;
* `A(m)` is no longer a pure generator quantity.  It is still a single function
  of `m` in the likelihood, and it is still `int du K(u|m) Gbar(u|m)`; what
  changed is that `Gbar` carries a detector object.

### Is the acceptance correlated with the per-candidate `sigma`?

The likelihood conditions each candidate on its own mass resolution (the
v-form conditions on `k = sigma_m/m^p`).  If the acceptance and `k` are
correlated the population-level `A(m)` is not the conditional one.
`ptres.py sigcond` measures it on the reco MC (160 files, 381 k candidates),
comparing the **same** events selected on the reconstructed `p_T` and on the
true (bare post-FSR gen) `p_T`, with `u = -ln(m_gen,post / m_gen,pre)` from the
generator record:

| `k = sigma_m/m` quantile | `<k>` | `<u>` reco cut | `<u>` true cut | reco/true |
|---|---|---|---|---|
| 0-20 % | 0.00853 | 16.362e-3 | 16.365e-3 | 0.9998 |
| 20-40 % | 0.01074 | 14.818e-3 | 14.806e-3 | 1.0008 |
| 40-60 % | 0.01232 | 12.438e-3 | 12.434e-3 | 1.0003 |
| 60-80 % | 0.01479 | 12.829e-3 | 12.831e-3 | 0.9998 |
| 80-100 % | 0.03759 | 12.404e-3 | 12.407e-3 | 0.9997 |
| inclusive | 0.01679 | 13.869e-3 | 13.867e-3 | 1.0001 |

Two separate statements, and only one of them is a problem.

* **The promotion across the threshold is negligible.**  779 candidates of
  349 098 (0.22 %) are selected on the reconstructed `p_T` and not on the true
  one, and 724 the other way round; the promoted ones are biased -- `<k>` =
  0.0243 against 0.0168 inclusive and `<u>` = 16.9e-3 against 13.9e-3 -- but
  the reco and true columns agree to **<= 8e-4** in every `k` quantile and to
  1e-4 inclusively.  So the *resolution's own* contribution to a
  `k`-dependence of the acceptance is below the MeV level.
* **The selected FSR content depends strongly on `k` through kinematics.**
  `<u>` runs 16.4e-3 to 12.4e-3 across the `k` quintiles -- +-14 % about the
  mean -- and it does so identically under the true cut, so it is not a
  detector effect at all: a large-`sigma` candidate is a forward or soft one,
  which has a different acceptance and therefore a different selected FSR
  spectrum.  A population-level `K_sel(u|m)` inside a `k`-conditioned
  likelihood is therefore misspecified per candidate, even though it is right
  on average.

The fix does **not** need a new axis on the interface: `k` is predicted by the
two legs' `(p_T, eta)` -- the very variables `h4` is binned in -- to a median
ratio 1.063 with an 11.9 % 68 % spread and `corr(ln k, ln k_pred) = 0.875`, so
a `k`-conditioned `K_sel` is the same `h4` restricted to the cells of a `k`
class.  Quantifying the residual bias of the unconditioned form, and building
the conditioned one, is the next item; it is a property of the **conditioning**
and not of the asymmetric cuts or of the resolution in the acceptance, both of
which are closed here.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
# 1. the detector side: the CVH-refit muon pT resolution (numpy only, ~2 min)
python3 ptres.py measure --aux ../fullscale/runs/auxgen_dyv2.npz \
    --pairs ../fullscale/runs/zpairs_dyv2_full.npz \
    --seed ../fullscale/runs/auxseed_dyv2.npz -o data/ptres_dyv2.npz
# 2. the tables and every kernel of the four selections (numpy only, ~50 min)
./build_selection.sh
# 3. the pass region off the table against a direct event count
python3 fsr_perleg.py gcheck --gen data/genmerged_full.npz \
    --htable data/ht_ref10_1.0gev.npz --pt-cuts 25 10
python3 fsr_perleg.py gcheck --gen data/genmerged_full.npz \
    --htable data/ht_ref10_1.0gev.npz --h4 data/h4_ref10_1.0gev.npz \
    --resol data/ptres_dyv2.npz --pt-cuts 25 10
# 4. the fit benchmarks (~25 min each)
for S in 2525 2510 res2525 res2510 res2510g; do ./run_selection_fit.sh $S; done
# 5. is the acceptance correlated with the per-candidate sigma? (needs /ceph)
ssh submit50 "cd $Z && ./run_tf_z.sh --ceph python3 -u ptres.py sigcond \
    --files /ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2 \
    --max-tasks 40 --pt-cuts 25 10 --res data/ptres_dyv2.npz"
# 6. the figures (needs /ceph: the container's mount hook binds it)
F=~/public_html/ZMass/cvh/260916_fsr_selection
ssh submit50 "cd $Z && ./run_tf_z.sh python3 -u ptres.py figs \
    --res data/ptres_dyv2.npz --aux ../fullscale/runs/auxgen_dyv2.npz \
    --seed ../fullscale/runs/auxseed_dyv2.npz --outpath $F"
for S in '2510:--pt-cuts 25 10' '2510sm:--pt-cuts 25 10 --h4 data/h4_ref10_1.0gev.npz --resol data/ptres_dyv2.npz'; do
  ssh submit50 "cd $Z && ./run_tf_z.sh python3 -u cmp_perleg.py --selection-only \
     --htable data/ht_ref10_1.0gev.npz --outpath $F --suffix _${S%%:*} ${S#*:} \
     --kernel 'corr, mc K=data/kern_corr_mc_${S%%:*}.npz' \
     --kernel 'corr, data K=data/kern_corr_data_${S%%:*}.npz'"
done
```

Figures in `~/public_html/ZMass/cvh/260916_fsr_selection/`: `06_ksel_peak_*`,
`07_acceptance_*`, `08_mean_u_*` (model against MC, per selection),
`20_passregion_*` (the pass region, step and smeared), `21_passprob_*` (the
per-leg pass probability against the step), `22_fitshifts` (every fit row),
`30_ptres_sigma` (the resolution and its 4-parameter form), `31_ptres_shape`
(the standardised shape against a Gaussian), `32_ptres_tails`,
`33_ptres_pull` (the CVH covariance pull), and `00_bands_*.txt` with the
band-by-band tables.

---

## Fold matrix from the kernel table

The provider folds FSR with one constant matrix, `p_post = F p_born`, built at
construction and applied as a single `matvec` per likelihood evaluation. `F` is
built from either of two representations of the kernel, dispatched on the keys
of `fsr=` (`rabbit/lineshapes/zgamma.py`):

| | atoms `(r, w, m_lo, m_hi)` | table `(m_nodes, u_edges, K, u_mean, p0)` |
|---|---|---|
| in `m_pre` | piecewise constant: one atom set per band | interpolated at every Born grid point |
| in `u` | point masses, cells merged under `var_budget` / `sigma_cap` | cell-integrated, nothing merged |
| its discretisations | band width, `sigma_cap`/`var_budget` | node spacing, cell count — both convergent |

### The format

`fsr_table.py` writes an npz with

| array | shape | |
|---|---|---|
| `m_nodes` | `(n_m,)` | pre-FSR mass nodes [GeV], increasing |
| `u_edges` | `(n_c + 1,)` | cell edges in `u = -ln(m_post/m_pre)`, increasing, `u_edges[0] = 0` |
| `K` | `(n_m, n_c)` | probability **mass** in each cell |
| `u_mean` | `(n_m, n_c)` | its first moment `<u>`, inside its own cell |
| `p0` | `(n_m,)` | point mass at `u = 0` |

and `sum_c K + p0 = 1` per row (the provider renormalises the rows anyway, as
the atom form renormalises each band). `p0` is a genuine `delta`: the fraction
of events the generator left untouched in the `mc` configuration, zero in the
analytic one, whose soft end is an integrable singularity and not a delta.

The default `u` grid is one cell below `u = 1e-9` and 2000 geometric cells to
`u = 7`; the mass nodes run 50-200 GeV — the Born grid of the fold reaches the
luminosity table's upper edge — every 1 GeV. That is 151 x 2000 float64,
**4.9 MB** on disk, against 0.99 MB for the 30 835 atoms of the same kernel.
The first cell carries 31 % of the analytic kernel's mass and all of it sits at
`m' = m` to 1e-8 of a grid spacing, which is why one cell there is enough.

In the card the table is serialised **by reference** when it came from an npz
that is still on disk (591 bytes of JSON) and inline otherwise, as zlib'd
base64 of the float64 arrays — 0.07 MB of JSON for 3800 cells, so ~11 MB for a
151 x 2000 table. The arrays come back bit-identical.

### Producers

```bash
python3 fsr_table.py analytic -o data/ktab_data_dm10_c2000.npz --dm-node 1.0
python3 fsr_table.py mc       -o data/ktab_mc_dm10.npz         --dm-node 1.0
python3 fsr_table.py corr --htable data/ht_ref10_1.0gev.npz --pt-cuts 25 25 \
        -o data/ktab_corr_data_2525.npz -a data/atab_corr_data_2525.json
python3 fsr_table.py cond --gen data/genmerged_full.npz --pt-cuts 25 25 \
        -o data/ktab_cond_2525.npz -a data/atab_cond_2525.json
python3 fsr_table.py check -i data/ktab_*.npz
python3 fsr_table.py atoms -i <atom file> -o <table>      # the identity test
```

* **`analytic`** — `fsr_analytic.FSRKernel` at each node. The photonic part is
  integrated **cell by cell** in `t = (1-z)^beta`, the substitution that removes
  the `C beta (1-z)^{beta-1}` endpoint exactly: the singular piece is
  `C [(1-z_lo)^beta - (1-z_hi)^beta]` per cell analytically (a constant
  integrand in `t`, on which Gauss-Legendre is exact) and the regular remainder
  is the same quadrature the atom form uses. The pair branch
  `K = K_phot (x) [(1-N) delta + R_pair]` is folded in by depositing the
  (coarse photonic atom) x (pair atom) products into the cells at their own
  `u`, mass and first moment exact. **1 s** for 151 x 2000, against 1.9 s for
  the 75-band atom set.
* **`mc`** — the standalone Photos histograms are already cells, so the table
  is a *coarsening* of them (merging bins adds masses and first moments, which
  is exact) and the 78 generated 2 GeV bands are interpolated onto the nodes.
  The default `u` grid is therefore built from the run's own edges: 1268 cells.
  **1 s**, against **53 s** for the 143 245-atom set, whose per-band merge is a
  Python loop over 100 k histogram bins.
* **`corr`** — the selection-conditional kernel `K_sel(u|m) = K(u|m) Gbar(u|m)`
  of `fsr_perleg`: the inclusive cells above, multiplied per cell by `Gbar` at
  the cell's own mean. `Gbar` is a property of the `h` table and is computed on
  its bands and interpolated in `m`; `A(m) = int K Gbar` comes out of the same
  integral and is written as the tabulated acceptance on the same nodes.
  **82 s**, 81 of which is `Gbar` on 152 bands.
* **`cond`** — the **measured** kernel of a generator sample: the events' own
  `u = -ln(m_post/m_pre)` deposited into the cells with mass and first moment,
  one row per `m_pre` band at the band's weighted mean `m`, `p0` the record's
  unradiated fraction (`u < 1e-5`, the atom form's own floor), and the
  tabulated `A(m) = P(pass|m)` measured on the same events.  `--pt-cuts` makes
  it the selection-conditional kernel `K_sel(u|m)`, `--inclusive` the sample's
  own `K(u|m)`, `--half` one half of the events (the statistical floor).
  **7 s** for 11 bands x 2000 cells on 29.3 M events.
* **`corr --sample`** — the same measured inclusive `K`, on the `h` table's
  bands, fed to the `corr` producer in place of the standalone Photos run, so
  that the model carries the sample's own QED.  `--mode`/`--rho` select the
  two-leg law of `fsr_perleg corr`.
* **`atoms`** — an atom file as a table of point-like cells `[u-eps, u+eps]`;
  the identity test of the two paths.

### How a column is built

Column `b` is what a unit of Born probability at `m_b` becomes on the output
grid, so it is built from the kernel **at that mass** — the rows interpolated
to `m_b`, linearly or (`fsr_minterp="cubic"`) by a natural cubic spline, both
of which reproduce a constant exactly and so preserve `sum_c K + p0 = 1`. Each
cell is then deposited by the rule that is exact for its own width in `m'`,
`m_b (e^{-u_lo} - e^{-u_hi})`:

* **narrower than the grid spacing** — a point mass at the cell's own first
  moment, which is what an atom is, except that the cell is *integrated* and
  not *merged*, so what is lost is the cell's second moment alone and no
  `sigma_cap` enters. `p0` is the same deposit at `u = 0`, i.e. on the
  diagonal. `fsr_deposit` picks how it is spread onto the two neighbouring
  nodes: `"interp"` (default) a tent of half width `r dm` and height `1/r`,
  which makes the column the exact fold of the piecewise-linear Born density —
  *identical to the atom form, matrix element by matrix element*; `"mass"` the
  unit-width tent, which splits the mass linearly and conserves it exactly.
* **wider** — the tail, where the output grid resolves the kernel: the cell's
  mass spread over its own width and projected onto the hat basis exactly,
  `int rho lambda_i`, which twice-integrating by parts is the second difference
  at the nodes of the twice-integrated density. Two cumulative sums and one
  `searchsorted` per column, no per-cell loop, and mass *and* position are
  exact however the cell edges fall between the nodes.

The two branches meet where a cell is one grid spacing wide, where they agree.
Both choices in the narrow branch matter, and they trade off:

| against the exact fold `sum_j (w_j/r_j) p_born(m/r_j)` of a 40-atom kernel | `nm` = 1024 | 2048 | 4096 |
|---|---|---|---|
| `"interp"` (and the atom form), max rel dev on the pdf | 1.9e-4 | 4.7e-5 | 1.4e-5 |
| `"mass"` | 2.3e-3 | 2.7e-3 | 3.4e-3 |
| largest column excess, `"interp"` | +3.2e-3 | +3.8e-3 | +4.6e-3 |
| largest column excess, `"mass"` | 4e-16 | 7e-16 | 7e-16 |

`"mass"` conserves probability exactly but sampling a unit-width tent on a grid
of spacing `r dm` is **not a partition of unity**, and the O(1-r) ripple that
leaves in the output density does not shrink with `dm`. `"interp"` puts the
same ripple in the column *mass* instead, where the smooth Born density
averages it away — which is why it converges as `dm^2` and is the default. The
atom form is `"interp"` by construction and carries the same column ripple:
up to +0.5 % on a 40-atom kernel, against 4e-16 for `"mass"`.

Projecting the tail rather than reading its density off at the nodes is what
makes the cell count converge: a nodal read-off misplaces a cell one grid
spacing wide by up to `dm/2`, which on its own is worth 0.22 MeV of `m_Z`
between 1000 and 4000 cells.

### What it costs

Build of the `(nm, n_born)` matrix, 50-130 GeV window with the Born grid
extended to 200 GeV, on an idle `submit81` core:

| | `nm` = 8192 (0.94 GiB matrix) | `nm` = 32768 (15.0 GiB) |
|---|---|---|
| atoms, data (30 835 atoms, 75 bands) | 2.1 s | 14.1 s |
| atoms, mc (143 245 atoms, 78 bands) | 6.7 s | 27.1 s |
| table, data (151 x 2000) | 4.7 s | 61.8 s |
| table, data (151 x 4000) | 5.6 s | 66.5 s |
| table, mc (151 x 1268) | 4.2 s | 58.2 s |
| peak RSS, atoms / table | 3.4 / 3.7 GiB | 42.8 / 47.3 GiB |

Both are quadratic in `nm` (the atom cost is `n_atoms x nm`, the table's is
`n_born x (n_c + nm)` with `n_born ~ nm`), and the matrix — hence the
**per-evaluation cost** — is identical. The table is 2.2x the atom build at
`nm` = 32768 and below it for a kernel with many atoms; the peak RSS is 10 %
above the atom path, both dominated by the `numpy` matrix and its `tf.constant`
copy.

### What it is worth

`fsr_table.py fit` runs `fit_gen`'s closure fit on an explicit list of kernels,
so an atom row and its table are the same events, the same window and the same
five `K(m)` terms. 27.7 M gen events in 60-120 GeV, `nm` = 8192; offsets from
the generator's own `m_Z`, `Gamma_Z`.

| inclusive, `data` configuration (exp. O(a) + O(a^2)LL + NLL + all pairs) | d`m_Z` [MeV] | d`Gamma_Z` [MeV] |
|---|---|---|
| **atoms, 2 GeV bands, `var_budget` 6e-10** (the published row) | **+1.381 ± 0.543** | **−1.036 ± 1.132** |
| table, 4 GeV nodes, 2000 cells | +1.139 ± 0.594 | −1.226 ± 1.140 |
| table, 2 GeV nodes, 2000 cells | +1.131 ± 0.589 | −1.238 ± 1.139 |
| **table, 1 GeV nodes, 2000 cells** | **+1.105 ± 0.592** | **−1.221 ± 1.139** |
| table, 0.5 GeV nodes, 2000 cells | +1.102 ± 0.592 | −1.230 ± 1.139 |
| table, 1 GeV nodes, 1000 cells | +0.944 ± 0.647 | −1.232 ± 1.141 |
| table, 1 GeV nodes, 4000 cells | +1.060 ± 0.587 | −1.221 ± 1.138 |
| table, 1 GeV nodes, 8000 cells | +1.105 ± 0.578 | −1.200 ± 1.137 |
| … 2000 cells, `fsr_deposit="mass"` | +1.105 ± 0.592 | −1.221 ± 1.139 |
| … 4000 cells, `fsr_deposit="mass"` | +1.060 ± 0.587 | −1.221 ± 1.138 |

| inclusive, `mc` configuration (standalone Photos) | d`m_Z` [MeV] | d`Gamma_Z` [MeV] |
|---|---|---|
| **atoms, 2 GeV bands, `sigma_cap` 3.3e-4** | **+0.412 ± 0.561** | **−0.212 ± 1.136** |
| table, 4 GeV nodes | −0.263 ± 0.588 | −1.039 ± 1.140 |
| table, 2 GeV nodes | −0.253 ± 0.589 | −1.051 ± 1.140 |
| **table, 1 GeV nodes** | **−0.249 ± 0.589** | **−1.106 ± 1.141** |
| … `fsr_deposit="mass"` | −0.249 ± 0.589 | −1.106 ± 1.141 |

| fiducial 25/25, `corr` kernel, `data` configuration | d`m_Z` [MeV] | d`Gamma_Z` [MeV] |
|---|---|---|
| **atoms, 1 GeV bands** (the published row) | **+1.070 ± 0.829** | **−1.113 ± 1.744** |
| table, 2 GeV nodes | +0.462 ± 0.903 | +0.786 ± 1.760 |
| **table, 1 GeV nodes** | **+0.552 ± 0.905** | **−0.133 ± 1.758** |
| table, 0.5 GeV nodes | +0.546 ± 0.905 | −0.451 ± 1.758 |

**The atom minus table difference is the discretisation the atom form carries**
(same events, so it is not a statistical difference):

| | d`m_Z` [MeV] | d`Gamma_Z` [MeV] |
|---|---|---|
| inclusive `data` (2 GeV bands + `var_budget` 6e-10) | **+0.28** | **+0.19** |
| inclusive `mc` (2 GeV bands + `sigma_cap` 3.3e-4) | **+0.66** | **+0.89** |
| fiducial 25/25 `corr` (1 GeV bands + `var_budget` 6e-10) | **+0.52** | **−0.98** |

For the `mc` configuration that is the `sigma_cap` systematic the atom scan
already showed (3.3e-4 against 1e-4 was worth 0.7 / 0.6 MeV) and it is now
simply gone. The table's own knobs are convergent and an order of magnitude
smaller: **the node spacing is worth 0.04 MeV on `m_Z` between 4 and 0.5 GeV**
(0.014 MeV for `mc`), and the cell count **±0.05 MeV** — 2000 / 4000 / 8000
cells give +1.105 / +1.060 / +1.105, an oscillation and not a trend, set by
where the narrow/wide crossover falls between the cells; 1000 is visibly too
coarse and 2000 is the default. `Gamma_Z` is stable to 0.02 MeV across both. For `corr` the node spacing also refines
`A(m)`, which is steep at the low edge, so the two cannot be separated there;
`m_Z` converges to +0.55 MeV.

The **deposit rule is invisible at this scale**: on a real table every rule
agrees to 0.4 keV on `m_Z` and 0.1 keV on `Gamma_Z`, with the same error and
the same correlations, because the cells near `u = 0` are far finer than the
grid and their ripples average away. It only shows up on a table of isolated
point masses, i.e. on an atom set.

One thing does change and it is not a bias: with a kernel that is **continuous
in `m_pre`**, `m_Z` becomes correlated with the floated smooth `K(m)`
(`rho` = +0.39 against −0.02 for the bands) and its error grows by 8-9 %
(0.592 against 0.543). That is the band staircase and not the atom fold's
column ripple — the exactly mass-conserving `"mass"` rule has no ripple at all
and gives the same error and the same correlations. A staircase in `m_pre` is
structure the kernel does not have, and a fit that can see it separates the
kernel from `K(m)` on information that is not there; the table's error is the
honest one.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
for S in "1.0 2000" "4.0 2000" "2.0 2000" "0.5 2000" "1.0 1000" "1.0 4000"; do
  set -- $S
  python3 -W ignore fsr_table.py analytic --dm-node $1 --n-cell $2 \
     -o data/ktab_data_dm$(echo $1 | tr -d .)_c$2.npz
done
python3 -W ignore fsr_table.py mc -o data/ktab_mc_dm10.npz --dm-node 1.0
python3 -W ignore fsr_table.py corr --htable data/ht_ref10_1.0gev.npz \
   --pt-cuts 25 25 -o data/ktab_corr_data_2525.npz \
   -a data/atab_corr_data_2525.json --dm-node 1.0
./run_tf_z.sh python3 -u fsr_table.py fit --gen data/genmerged_full.npz \
   --nm 8192 --shape 5 \
   --row "atoms data, 2 GeV bands=data/kern_cfg_data_vb6e-10.npz" \
   --row "table data, 1 GeV nodes, 2000 cells=data/ktab_data_dm10_c2000.npz" \
   -o data/fit_ktab_postfsr.json
./run_tf_z.sh python3 -u fsr_table.py fit --gen data/genmerged_full.npz \
   --nm 8192 --shape 5 --acc-pt 25 \
   --row "corr atoms=data/kern_corr_data_2525.npz:data/acc_corr_data_2525.json" \
   --row "corr table=data/ktab_corr_data_2525.npz:data/atab_corr_data_2525.json" \
   -o data/fit_ktab_corr2525.json
./run_tf_z.sh python3 -u /work/submit/david_w/ZMass/rabbit-vmass/tests/\
test_zgamma_kernel.py --skip 1 2 3 4 5 6 7      # the table tests
```

Nothing downstream has to change to use a table: `fit_gen.py --kernel` /
`--kernel-alt` and `fullscale/make_card.py --fsr` all pass a path straight to
`ZGammaLineshape(fsr=)`, which dispatches on the npz keys.

---

## The multi-emission two-leg density

The correlated model of the previous section shares the whole loss as if it
were **one** photon, and the collinear product `D (x) D` shares it as if every
photon were collinear.  Both are limits of one object, and the object is fixed
by the fact that **photon energies add**: in the pre-FSR rest frame

```
eps_q = 1 - x_q ,      eps_+ + eps_- = eps = 2 sum_i E_i / m       (exact)
```

so the pair `(eps_+, eps_-)` is a sum over photons and its law is the 2-D
compound Poisson whose Levy measure is the exact O(alpha) emission density
carrying its exact sharing,

```
nu_2(eps_+, eps_-) = nu(delta) p_1(f | 1 - delta) ,
                     eps_+ = delta f ,   eps_- = delta (1 - f) ,
log P^(s_+, s_-)   = nu_2^(s_+, s_-) - N ,      N = int nu_2 ,
```

with `nu` the Levy measure of the inclusive kernel in `eps` and `p_1` the same
`fsr_analytic.share_nodes` density the single-photon model uses.
`fsr_perleg.py corr --mode multi`; `--mode coll` and `--mode lin` are the two
limits below.

### Why this is the object, and not a matching

Three properties, none fitted and none carrying a matching scale:

* **the marginal identity is exact.** On the diagonal `s_+ = s_-` the sharing
  integrates to one at every `delta`, so `log P^(s,s) = nu^(s) - N = log K^(s)`:
  the total-loss law is the kernel that was handed in and the inclusive fit is
  untouched;
* **at O(alpha) the law IS `nu_2`**, the exact matrix element's angular
  distribution -- no collinear approximation in the recoil;
* **in the collinear limit it collapses exactly to `D (x) D`.** With
  `p_1 -> [delta(f) + delta(f-1)]/2`,
  `log P^ = [nu^(s_+) - N]/2 + [nu^(s_-) - N]/2`, and `K^{1/2}` is the leg
  radiator.

Factorising the exponential the other way is the **first-order matching**:

```
P = exp_*(nu_2^coll - N) (x) exp_*(dnu_2) = [D (x) D] (x) exp_*(dnu_2) ,
dnu_2 = nu_2^exact - nu_2^collinear ,
```

and `dnu_2` integrates to zero over `f` at every `delta`, so it changes no `z`
marginal at any order.  Keeping its first term, `P = D (x) D + [D (x) D] (x)
dnu_2`, is `--mode lin`: the collinear ladder plus **one** wide-angle emission.
`multi` resums it; the two bracket the resummation of the sharing at
**0.12 / 0.16 MeV**.

### The Levy measure

`nu` is read off the kernel by the same spectral operation the convolution
square root is -- `log` in place of `sqrt` -- on a uniform grid in `eps`:
`nu = F^-1 log F K`, `N = -log K[0]`.  `nu[0] = 0` and `N` come out of it rather
than being imposed, because the `n`-photon term of `exp_*` starts at node `n`.
The same object obeys the causal recursion `k g_k = sum_j j nu_j g_{k-j}`, which
makes `nu` below any `eps_max` independent of everything above it -- what lets a
**fine short grid** resolve the sharing at small `u`.  The two branches agree to
**1e-14**.

Peak band, `mc` kernel:

| | `h` = 1e-3 | 5e-4 | 2.5e-4 |
|---|---|---|---|
| `N` | 0.4157 | 0.4560 | 0.4963 |
| `nu < 0` | 0 | -3.8e-5 | -1.2e-4 |
| spectral vs causal recursion | 2.7e-14 | 1.4e-14 | 6.8e-15 |
| **marginal identity, `multi`** | **1.2e-9** | **6.0e-10** | **3.1e-10** |
| marginal identity, `coll` | 2.7e-9 | 1.4e-9 | 7.5e-10 |
| weight above `eps` = 1 | -2.6e-7 | -2.6e-7 | -2.6e-7 |
| `min P`, `multi` | 0 | -2.1e-7 | -1.3e-7 |

The marginal identity -- `int P` along the line `eps_+ + eps_- = eps` against the
kernel -- holds to **3e-10** because the 2-D atoms are deposited **along their
own anti-diagonal** (`deposit_diag`), which makes the projection of `nu_2` equal
`nu` to machine precision.  `N` grows with the grid because the soft end of `nu`
is logarithmic; nothing observable depends on it.

`coll` is the closure test: its joint leg tail over the product of its marginals
is **1.000** at every threshold -- the definition of independent legs -- and its
leg marginal reproduces `conv_sqrt(K_eps)` to 1.7e-4, with `<u_leg>` agreeing to
2e-4.

### Against the generator

Peak band, `00_multi.txt`.  The `f` resolution of the model is `h/eps`, so the
soft slices are read off the fine short grid (`eps < 0.021`, `h` = 1e-5) and the
hard ones off the full one (`h` = 2.5e-4).  `P(0.01 < f < 0.99)`:

| `u` slice | MC, all | `single` | **`multi`** | `coll` | exact O(α) |
|---|---|---|---|---|---|
| 1e-3 - 1e-2 | 0.4604 | 0.3662 | **0.4568** | 0.1216 | 0.3676 |
| 1e-2 - 0.05 | 0.4625 | 0.3765 | **0.4693** | 0.1266 | 0.3686 |
| 0.05 - 0.2 | 0.4642 | 0.3717 | **0.4640** | 0.1246 | 0.3719 |
| 0.2 - 0.6 | 0.4667 | 0.3724 | **0.4643** | 0.1241 | 0.3724 |

`single` reproduces the exact O(α) column, `coll` reproduces `D (x) D`'s own
0.124, and **`multi` reproduces the generator's all-emission sharing** to
-0.8 %, +1.5 %, 0.0 % and -0.5 %.  The 35 % gap between one-photon and
all-emission sharing is closed by the exponentiation and by nothing else.

The leg law, full grid at `h` = 2.5e-4, against the standalone at 2.5e9 events
per setting:

| `t` | `P(u_leg>t)` MC | `single` | `multi` | joint/prod MC | `single` | `multi` | `coll` |
|---|---|---|---|---|---|---|---|
| 1e-3 | 0.19887 | 0.19371 | 0.20375 | 2.516 | 2.21 | 2.49 | 1.000 |
| 1e-2 | 0.11986 | 0.11564 | 0.12017 | 2.903 | 2.39 | 2.88 | 1.000 |
| 0.05 | 0.06845 | 0.06676 | 0.06842 | 3.185 | 2.41 | 3.11 | 1.000 |
| 0.2 | 0.03109 | 0.03086 | 0.03116 | 3.048 | 1.84 | 2.84 | 1.000 |

`multi` reproduces the per-leg marginal to **0.3 %** above `u_leg` = 1e-2 (2.5 %
at 1e-3, the grid's own reach) and the joint tail ratio to **1-7 %**, where the
single-photon model is 12-40 % low on the joint tail.

### Photos' own angle is exact; the 6-9 % "deficit" was the tag

`photos_standalone/photos_share.cc` accumulates the per-leg law directly, with
`photos_gen.cc`'s event generation unchanged: `x_q = 2 (p'_q . Q_pre)/m^2`, an
invariant, so `f` and `u_q` are built in the pre-FSR Z rest frame with or
without `--kinboost` (checked boost-invariant to 8e-4).  2.5e9 events per
setting at the peak band, three classes: **exactly one photon and no pair**,
**all events**, and **`|k^2|/m^2 < 1e-8`** -- which is the tag
`cmp_perleg.one_photon` uses on the sample's gen record.  `share_merge.py`
combines and mixes them at the same `f_ME` the kernel mixture uses;
`data/photos/share_peak.npz`.

`P(0.01 < f < 0.99)`, exactly-one-photon class, against the exact O(α) density
at the slice's own `(m, z)`:

| `u` slice | exact | ME on 100 % | ME off | sample-like mix |
|---|---|---|---|---|
| 1e-4 - 1e-3 | 0.36704 | 0.36708 (1.0001) | 0.36713 | 0.36711 |
| 1e-3 - 1e-2 | 0.36725 | 0.36735 (1.0003) | 0.36736 | 0.36736 |
| 1e-2 - 0.05 | 0.36837 | 0.36847 (1.0003) | 0.36856 | 0.36853 |
| 0.05 - 0.2 | 0.37144 | 0.37105 (0.9989) | 0.37242 | 0.37189 |
| 0.2 - 0.6 | 0.37223 | 0.37006 (0.9941) | 0.38178 | 0.37728 |

**Photos has no angular deficiency worth modelling.** With the exact Z
matrix-element correction on for 100 % of events its single-photon sharing
agrees with the exact O(α) density to 0.03 % below `u` = 0.05, 0.11 % at
0.05-0.2 and 0.6 % at 0.2-0.6, and the `f` *shape* is flat at 1.000 ± 0.0005
over four decades in `f`.  With the correction **off** it agrees to 0.05 % below
`u` = 0.05 and breaks only at hard emission (+2.6 % at `u` = 0.2-0.6, entirely
in the wide-angle core `f` in [0.1, 0.5]) -- which is what the correction is
for.

The 5-9 % by which the sample's "1γ" column sat below the exact O(α) density is
therefore **not** Photos' angular approximation: it is the `k^2 ~ 0` **tag**.
Two photons collinear *with each other* also give `k^2 ~ 0`; the tag keeps
62.1 % of radiating events in the standalone (63.3 % on the sample) while only
53.9 % of them are genuinely one photon and no pair.  Applying the tag to the
standalone, with the physics untouched, moves `P(0.01 < f < 0.99)` from 0.3674
to 0.3486 -- reproducing the sample's 0.3476 / 0.3440.  A `--ktag` scan over
1e-9 to 1e-5 leaves the tagged value at 0.348-0.352: it is the class, not the
threshold.  The standalone's **all-emission** column reproduces the sample to
0.1-0.2 % and its joint tail ratios to 0.2-0.4 %, so the standalone setup is the
sample.

### The mass-leg relation, and what the collinear product was really costing

With `n` photons, energy conservation in the pre-FSR rest frame is exact,
`eps_+ + eps_- = eps`, and the mass is `z = 1 - eps + (sum k)^2/m^2`.  Decompose
each photon along the two muon light-cone directions,
`k_i = eps_+^i p_+ + eps_-^i p_- + k_T`; averaged over the relative azimuth

```
(sum k)^2/m^2 = eps_+ eps_- - sum_i eps_+^i eps_-^i ,
x_+ x_- - z   = sum_i eps_+^i eps_-^i = sum_i delta_i^2 f_i (1 - f_i) ,
R             = <x_+ x_- - z> / <eps_+ eps_-> .
```

`R = 1` is `z = 1 - eps`, which is **exact for one photon at any angle** and is
the convention of the whole `corr` family; `R = 0` is `z = x_+ x_-`, which is
**exact when every photon is collinear to one leg** and is the convention of the
collinear product.  `R` is measurable on the generator record from `x_+`, `x_-`
and `m_post` alone, and on the model from the compound Poisson's first moment
(`fsr_perleg.multi_R`, Mecke's formula):

| `u` | 1e-3 | 1e-2 | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|---|
| generator | 0.737 | 0.733 | 0.726 | 0.720 | 0.706 | 0.686 | 0.617 |
| model, `multi` | 0.752 | 0.734 | 0.731 | 0.737 | 0.735 | 0.739 | 0.731 |
| model, `single` | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| model, `coll` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

so the generator is **73 %** of the way to the one-photon convention, and the
exponentiated model reproduces that composition to 1-2 % -- which nothing in the
construction was arranged to do.  It also settles what the old "independent
legs" row was measuring: `coll` and the per-leg product have the **same** sharing
and differ only in this convention, and at fit level that difference is
**+1.65 MeV on `Gamma_Z`**, against +0.24 MeV for the sharing itself.  Three
quarters of the collinear product's old -4.5 MeV was the mass-leg relation, not
the leg correlation.

`coll` is therefore not a physics variant: its photons are collinear (`R` = 0)
while its mass rule is `z = 1 - eps` (`R` = 1), so it is internally
inconsistent, and it is kept only as the closure test of the 2-D law.  `single`
is exactly self-consistent (one photon, `R` = 1 both ways).  `multi` carries the
generator's own `R` = 0.73 in its configurations while still assigning
`z = 1 - eps`: a residual inconsistency of 0.27 of `eps_+ eps_-`, which for its
own configurations (one hard photon plus a soft ladder, `<eps_+ eps_->` = 6 % of
`1 - z` at `u` = 0.5) is worth of order 0.1 MeV.

### The fit benchmark

`fit_gen.py fit --suite perleg`, the same 9.87 M selected gen events in
60-120 GeV, five Legendre shape terms, `nm` = 8192, **atoms** throughout (the
cell-integrated table representation of `fsr_table.py` carries its own
+0.52/−0.98 MeV offset against atoms and is not mixed in here; the whole
benchmark in that representation is "The machinery test in the table
representation" below).  Every row of a
table is the same run on the same events.

| model, fiducial `pT` > 25/25, \|η\| < 2.4, window 60-120 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| pre-FSR + `A(m)` control | −1.01 ± 0.78 | +1.86 ± 1.50 |
| MC-conditional, banded + `A(m)` | −0.31 ± 0.87 | +2.61 ± 1.75 |
| `cond`: true mass, true selection | +0.29 ± 0.84 | +2.08 ± 1.74 |
| `cond`: collinear selection | +0.42 ± 0.83 | +2.11 ± 1.74 |
| `cond`: collinear mass + selection | +0.10 ± 0.84 | +0.80 ± 1.74 |
| **corr, `mc` `K` (one photon)** | **+0.47 ± 0.83** | **−0.41 ± 1.75** |
| corr, `mc` `K`, matched `u_c` = 0.01 | +0.21 ± 0.83 | −0.10 ± 1.74 |
| **multi, `mc` `K`** | **+0.44 ± 0.83** | **−0.36 ± 1.75** |
| multi, `data` `K` | +1.04 ± 0.83 | −1.02 ± 1.74 |
| lin, `mc` `K` (first order only) | +0.43 ± 0.83 | −0.34 ± 1.75 |
| coll, `mc` `K` (the collinear limit) | +0.51 ± 0.83 | −0.39 ± 1.74 |
| per-leg, `mc` `D` (the collinear product) | +1.23 ± 0.83 | −2.39 ± 1.74 |
| inclusive `mc` standalone + `A(m)` | −3.18 ± 0.88 | +2.19 ± 1.82 |
| inclusive empirical kernel + `A(m)` | −3.48 ± 0.88 | +3.39 ± 1.81 |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| **the multi-emission sharing: multi − corr** | **−0.03** | **+0.05** |
| its first-order form: lin − corr | −0.04 | +0.07 |
| the sharing law altogether: corr − coll | −0.04 | −0.02 |
| **the mass-leg relation: coll − per-leg** | **−0.72** | **+2.01** |
| the machinery: multi − `cond`: true | +0.16 | −2.44 |
| … before, corr − `cond`: true | +0.19 | −2.49 |
| … per-leg − `cond`: true | +0.95 | −4.47 |
| the kernel physics: multi `data` − multi `mc` | +0.59 | −0.67 |
| standalone against the sample's own `K` (the floor) | +0.31 | −1.20 |

and the discretisation, all against the default row:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| `eps` grid `h` 2.5e-4 (from 5e-4) | −0.01 | +0.01 |
| sharing quadrature 32x8 (from 16x4) | 0.00 | 0.00 |
| `share_floor` 1e-4 (from 1e-6) | 0.00 | 0.00 |

Under the **asymmetric 25/10 cut**, on the `pT_ref` = 10 table (a different
selection and a different event set -- not comparable row by row with the table
above):

| model, `pT` > 25/10 | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| `cond`: 25/10, true mass and selection | +0.60 ± 0.77 | +1.14 ± 1.60 |
| corr, `mc` `K` | +0.47 ± 0.77 | +0.35 ± 1.60 |
| **multi, `mc` `K`** | **+0.49 ± 0.77** | **+0.35 ± 1.60** |
| corr, `data` `K` | +1.02 ± 0.77 | −0.19 ± 1.60 |
| multi, `data` `K` | +1.02 ± 0.77 | −0.20 ± 1.60 |
| coll, `mc` `K` | +0.46 ± 0.77 | +0.35 ± 1.60 |

multi − corr is **+0.02 / +0.00**, and the machinery residual is
**−0.11 / −0.78**, the same as the single-photon model's −0.13 / −0.79.  The
correction vanishes here because the trailing threshold at 10 GeV leaves the
softer muon so much headroom that how the loss is shared no longer decides the
event: `rho(u = 0.5)` is 0.9997 against 1.0286 at 25/25.

### Verdict

**The multi-emission sharing is not the residual.**  The construction is
validated -- it reproduces the generator's all-emission sharing to
0.8 / 1.5 / 0.0 / 0.5 %, its per-leg marginal to 0.3 % and its joint tail ratio
to 1-7 %, where the single-photon model is 12-40 % low on the joint tail -- and
at fit level it is worth **−0.03 / +0.05 MeV** at 25/25 and **+0.02 / +0.00**
at 25/10.  The machinery residual on `Γ_Z` is −0.62 ± 0.43 MeV once both
benchmark rows are cell-integrated and the model carries the MC's own inclusive
`K` (−2.44 with the atom rows and the standalone kernel); see "The machinery
test in the table representation".

**What the collinear product was really costing is the mass-leg relation, not
the leg correlation.**  `coll` -- the same collinear sharing as `D (x) D`, in
the `corr` family's `z = 1 - eps` convention -- sits within 0.04 MeV of the
single-photon model, while the per-leg product, which differs from it *only* in
using `z = x_+ x_-`, is 0.72 / 2.01 MeV away.  The old attribution of that
+1.98 MeV to "independent legs" was wrong: the legs' correlation is worth
nothing under this selection, and the whole of it is the mass-leg convention.

**Photos contributes nothing either.**  With the exact-ME correction on,
Photos' single-photon sharing is the exact O(α) density to 0.05 / 0.03 / 0.24 /
0.6 % over the four slices, and the sample's apparent 5-9 % deficit is the
`k^2 ~ 0` tag rather than the generator.  There is no Photos angular
approximation left to fold into the model, and no part of the residual belongs
to one.

What is left, with the sharing, the acceptance decision, the Photos angle and
the exact `z` all excluded, is the discretisation of the benchmark rows
themselves -- the model side's atom form is worth +0.88 / −1.16 MeV against its
own cell-integrated table and the `cond` side's −0.11 / +0.32, and the two do
not cancel.  With both rows tabulated the residual is +0.65 / −0.62 MeV at the
tables' own statistical floor of 0.30 / 0.43.  The residual mass-leg
inconsistency of
`multi` -- its own configurations carry `R` = 0.73 while it assigns `z = 1 - eps`
(`R` = 1) -- is the one identified effect not yet removed, and it is bounded at
~0.1 MeV by `<eps_+ eps_->` being 6 % of `1 - z` where the model's hard
emissions live.  Closing it needs the third additive coordinate
`sum_i delta_i^2 f_i (1 - f_i)`, i.e. a 3-D compound Poisson, and is not worth
it at this level.

### The machinery test in the table representation

The benchmark above is **atoms on both sides, and the two atom sets are
different objects**: the model's is a `var_budget` merge of the kernel ladder
per `h` band with its `A(m)` on the 152 band means, the MC-conditional
reference's is a `sigma_cap` merge of a fine `u` histogram per `m` band with
its `A(m)` on 1 GeV bins.  Neither discretisation cancels in the difference.
Both sides become cell-integrated tables:

* **`fsr_table.py cond`** measures the reference as a table -- the selected
  events' own `u` histogram per `m` band, the cell's mass **and** its first
  moment, `p0` the record's unradiated fraction (`u < 1e-5`, the atom form's
  own floor), one node at each band's weighted mean `m`, the provider
  interpolating linearly between them.  A cell with a negative net weight (3
  in 22 000, all above `u = 0.6`, carrying 1e-5 of a row: the sample has
  negative MiNNLO weights) is merged with its neighbours until the group is
  positive and deposited at the group's own mean -- mass and first moment
  exact, the group's second moment lost, which is what one atom carries.
* **`fsr_table.py corr`** writes the model on the same format, with `--mode`
  for the two-leg law and `--rho` for its correction table.
* **`fsr_table.py corr --sample`** additionally builds the model's inclusive
  `K` from the **sample's own** `(m_pre, m_post)` on the `h` table's bands
  instead of the standalone Photos run.  The model then carries the MC's own
  QED, the MC's own `h` table and the MC's own events, so the
  standalone-against-sample kernel floor is gone from the residual and only
  the two-leg construction is left in it.

`fsr_table.py fit` reproduces `fit_gen.py fit --suite perleg` row for row
(`cond`: true +0.287/+2.080, per-leg product +1.234/−2.392, `multi` atoms
+0.443/−0.357, MC-conditional banded −0.172/+1.830), so the rows below are the
same events, the same window and the same five `K(m)` terms as the tables
above.

| fiducial `pT` > 25/25, one run, 9.87 M events | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| `cond`: true, atoms, per-leg record | +0.29 | +2.08 |
| **`cond`: true, atoms**, the fit's own events | **−0.24** | **+0.92** |
| **`cond`: true, table**, the same bands | **−0.13** | **+0.61** |
| `cond`: true, table, 2 GeV bands, 8000 cells | +0.54 | +0.72 |
| **model `multi`, atoms**, standalone `K` | **+0.44** | **−0.36** |
| **model `multi`, table**, standalone `K`, 1 GeV nodes | **−0.44** | **+0.80** |
| model `multi`, table, standalone `K`, 0.5 GeV nodes | −0.46 | +0.51 |
| model `multi`, table, standalone `K`, atom `A(m)` | −1.17 | +0.63 |
| model `multi`, atoms, standalone `K`, table `A(m)` | +1.15 | −0.19 |
| model `multi`, table, **sample's own `K`**, 1 GeV nodes | +1.18 | +0.27 |
| **model `multi`, table, sample's own `K`, 0.5 GeV nodes** | **+1.19** | **+0.10** |

Same-run differences:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| the machinery, atoms, against the per-leg record | +0.16 | **−2.44** |
| the machinery, atoms, same events | +0.68 | −1.28 |
| **the machinery, converged tables, the MC's own `K`** | **+0.65** | **−0.62** |
| the **atom bias of the model side** | **+0.88** | **−1.16** |
| … of which the `A(m)` grid alone | −0.74 | −0.17 |
| the **atom bias of the `cond` side** | **−0.11** | **+0.32** |
| the per-leg record against the fit's own events | +0.53 | +1.16 |
| the kernel floor, fiducial (standalone − sample) | −1.62 | +0.53 |
| the kernel floor, inclusive tables | −1.21 | −1.42 |
| the kernel floor, inclusive atoms | +0.27 | −1.52 |
| the sharing (`multi` − `single`), sample `K` | −0.06 | +0.06 |
| `rho` driven by the sample's own `K` | −0.00 | +0.00 |

**The −2.44 MeV on `Γ_Z` is the two atom sets and the two floors, not the
machinery.**  Term by term, from the atom row to the converged table row:

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| the residual as published, atoms, per-leg-record reference | +0.16 | −2.44 |
| the per-leg record replaced by the fit's own events | +0.53 | +1.16 |
| the `cond` side cell-integrated and converged | −0.78 | +0.21 |
| the model side cell-integrated and converged | −0.91 | +0.86 |
| the standalone `K` replaced by the sample's own | +1.65 | −0.41 |
| **what is left** | **+0.65** | **−0.62** |

The model side's atom bias is the larger of the two by an order of magnitude on
`Γ_Z` (−1.16 against +0.32) and it is **the kernel**, not the acceptance: the
`A(m)` grid is worth −0.17 of it, the remaining −0.99 is the `var_budget` merge
and the `h`-band staircase in `m_pre`.  On `m_Z` the split is the other way
round, +0.74 of the +0.88 being the `A(m)` grid, which is steep at the low edge
of the window and is resolved by the nodes and not by the bands.

Discretisation, and the test's own statistical floor from a half-sample split
of each table (`σ = |A − B| / 2`, two independent halves of the same events, the
fit run on the full sample in both cases):

| | Δ`m_Z` | Δ`Γ_Z` |
|---|---|---|
| `cond` table, cells 1000 / 2000 / 4000 / 8000 | +0.15 / +0.35 / +0.48 / **+0.54** | +0.41 / +0.56 / +0.68 / **+0.72** |
| … the same scan on one half of the sample | +0.79 / +0.89 / +1.06 / +1.09 | +1.09 / +1.12 / +1.19 / +1.19 |
| `cond` table, bands 4 / 2 / 1 GeV | −0.02 / +0.35 / +0.17 | +0.85 / +0.56 / +0.56 |
| model table, nodes 2 / 1 / 0.5 / 0.25 GeV, standalone `K` | −0.54 / −0.44 / −0.46 / −0.46 | +1.83 / +0.80 / +0.51 / +0.51 |
| model table, nodes 1 / 0.5 / 0.25 GeV, sample `K` | +1.18 / +1.19 / +1.19 | +0.27 / +0.10 / +0.10 |
| model table, cells 1000 / 2000 / 4000 | −0.42 / −0.44 / −0.45 | +0.80 / +0.80 / +0.80 |
| `σ`(`cond` table) | 0.29 | 0.40 |
| `σ`(model table) | 0.02 | 0.14 |
| **`σ`(the difference)** | **0.30** | **0.43** |

The `cond` table's cell count converges geometrically -- the increments are
+0.20, +0.13, +0.06 on `m_Z` and +0.16, +0.12, +0.03 on `Γ_Z` -- and the same
increments appear on one half of the sample, so it is a representation effect
and not the noise; 8000 cells is the converged row and 2000 is 0.19/0.15 short
of it.  The model's node spacing is converged at 0.5 GeV (0.25 GeV repeats it
to 0.00/0.00) and its cell count at 2000 (±0.02/±0.00).  The `cond` band width
is not resolved beyond the statistical floor: 4 / 2 / 1 GeV scatter by
0.37/0.30 against `σ` = 0.29/0.40.

What the test still carries, measured on the record (`cmp_machinery.py
--record`, `01_floors.txt`):

| | |
|---|---|
| the `eta` decision, pre-FSR in the model against post-FSR in the record | differs on 1.7e-3 of the sample, 3.3e-4 inside the `pT` cut; `A`(60-120) moves by −9.6e-5 relative |
| the soft floor `u < 1e-5`, a `delta` in the measured kernel and resolved in the standalone one | `P` = 0.364 selected, `<u>` contribution 3.2e-7, i.e. **0.03 MeV** of `m_Z` |
| the weight clipping | the same on both sides: 41 events of 29.27 M, `Neff/N` = 0.680 |
| the mass-leg inconsistency of `multi`, `R` = 0.73 against its own `z = 1 - eps` | bounded at ~0.1 MeV |

**Verdict.**  With every row cell-integrated, the reference measured on the
fit's own events and the model carrying the MC's own inclusive `K`, the
two-leg machinery is validated to **+0.65 ± 0.30 MeV on `m_Z`** and
**−0.62 ± 0.43 MeV on `Γ_Z`**, the errors being the tables' own statistical
floors; the `cond` band width adds another ±0.3 on each, and no identified
effect above 0.1 MeV is left unremoved.  The −1.2 MeV on `Γ_Z` that survived
the kernel floor in the atom representation was the atom discretisation of the
two benchmark rows, which do not cancel because the two atom sets are built by
different rules.

Figures `~/public_html/ZMass/cvh/260917_fsr_machinery/`: `50_ksel_band` (the
MC's own `P(u > u_0 | m)` against each model's, band by band, with the ratio),
`51_moments` (`p_0(m)` and `<u|m>` with the ratio), `52_fitshifts`,
`53_converge` (the cell and node scans with the half-sample scan beside them),
`00_machinery.txt` and `01_floors.txt` with every number above.

### Reproducing the table representation

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
# 1. every row as a table: the MC's own conditional kernel, the model on the
#    standalone K and on the sample's own K, and the discretisation variants
#    (numpy only, ~30 min)
./build_machinery_table.sh
# 2. the fit benchmarks -- each launches detached on $NODE, poll data/00_fit_*.log
NODE=submit50 ./run_machinery_table.sh main    # the four numbers + the floors
NODE=submit81 ./run_machinery_table.sh conv    # the table knobs
NODE=submit81 ./run_machinery_table.sh conv2   # the knobs pushed one step
NODE=submit50 ./run_machinery_table.sh noise   # the half-sample splits
NODE=submit52 ./run_machinery_table.sh incl    # the kernel floor, inclusive
# 3. the summary and the figures
./run_machinery_figs.sh
# any row list, finished or still running: python3 show_fit.py <json|log> ...
```

### The data configuration

`fsr_config.SHARE_MODE` = `"multi"`.  It is the construction that is right at
O(α) in the angle **and** carries the collinear ladder exactly, it costs
nothing to build on top of the single-photon model (one `rho` table per
selection), and its fit difference from `corr` is below the test's resolution --
so the choice is made on correctness, not on a measured gain.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
# 1. the standalone Photos sharing: ME on / off / the sample-like mixture,
#    2.5e9 events per setting on the peak band (~5 min each on 200 cores)
photos_standalone/build.sh                       # OUTBIN=photos_share too
./photos_standalone/run_share.sh A   20 --me=1 --pairs=1 --fint=8
./photos_standalone/run_share.sh B   20 --me=0 --pairs=1 --fint=8
python3 photos_standalone/share_merge.py --combine ... -o data/photos/share_peak.npz
# 2. the construction's own closure: the Levy measure, the marginal identity,
#    the leg law, the sharing and the mass-leg ratio (numpy only, ~10 min)
python3 fsr_perleg.py multicheck --htable data/ht_pt25_1.0gev.npz \
        --run data/photos/gen_mcMix.npz --h-scan 2.5e-4 5e-4 1e-3 \
        -o data/multi_check_mc.npz
python3 fsr_perleg.py multicheck --htable data/ht_pt25_1.0gev.npz \
        --pair e mu tau had --eps-max 0.021 --h-scan 1e-5 5e-6 2e-5 \
        -o data/multi_check_data_fine.npz
# 3. the kernels, both configurations and both selections (~40 min)
./build_corr.sh
# 4. the fit benchmarks (~4 min each) and the figures
./run_perleg_fit.sh multi
./run_perleg_fit.sh multi2510
ssh submit51 "cd $Z && ./run_tf_z.sh --ceph python3 -u cmp_multi.py \
   --check data/multi_check_mc.npz --check-fine data/multi_check_data_fine.npz \
   --check-alt 'data cfg=data/multi_check_data.npz' \
   --photos-share data/photos/share_peak.npz \
   --fit '25/25=data/fit_perleg_multi.json' \
   --fit '25/10=data/fit_perleg_multi2510.json'"
```

Figures `~/public_html/ZMass/cvh/260916_fsr_multiemission/`: `40_share_f` (the
sharing density at `u` in [0.05, 0.2), model against the generator and against
the standalone's three classes), `40b_share_f_soft` (the same at
`u` in [1e-3, 1e-2) off the fine grid), `41_share_u`
(`P(0.01 < f < 0.99)` against the total loss), `42_joint` (the joint leg tail
over the product), `43_gbar` (the selection weight of each construction over the
single-photon model's), `44_fitshifts`, `45_massconv` (the mass-leg ratio `R`),
and `00_multi.txt` with every number above.

---

## Resolution classes: the kernel under the likelihood's own conditioning

The likelihood conditions every candidate on its own mass resolution -- the
v-form conditions on `k = sigma_m/m^p` -- and the selected radiation depends on
`k`: `<u>` runs 16.1e-3 to 12.5e-3 across the `k` quintiles of the reco MC, and
does so identically under a cut on the *true* `p_T`, so it is a kinematic
correlation and not threshold promotion (previous section).  A population-level
`K_sel(u|m)` inside a `k`-conditioned likelihood is therefore misspecified per
candidate.  This section builds the conditioned kernel and measures what the
unconditioned one costs.

### The resolution is a function of the table's own axes

The pair mass is `m^2 = 2 pT_+ pT_- (cosh dEta - cos dPhi)`, so

```
d ln m = 1/2 (d ln pT_+ + d ln pT_-) + 1/2 d ln(2(cosh dEta - cos dPhi))
```

and with the two legs' momentum errors independent and the angular term
resolution-free,

```
k = sigma_m/m = 1/2 sqrt( s_+^2 + s_-^2 ) ,   s_q = sigma_pT/pT (pT_q, eta_q) .
```

On the reco MC (3.35 M candidates selected at 25/10, `fsr_kclass.py check`):

| `s_q` from | `k / k_pred` median | 68 % spread | `corr(ln k, ln k_pred)` |
|---|---|---|---|
| the two-track fit's own per-leg `sigma_rel` | 1.0005 | **0.33 %** | 0.9989 |
| the measured `sigma_pT/pT(pT, eta)` map, reco `(pT, eta)` | 1.0640 | 11.93 % | 0.8693 |
| the same map at the gen `(pT, eta)` | 1.0641 | 11.94 % | 0.8592 |

**The formula is exact.**  Fed the fit's own per-leg relative momentum
resolutions it reproduces the two-track covariance's `sigma_m` to 0.33 %, so
the opening angle carries no measurable resolution and the leg-leg correlation
of the covariance is negligible.  What the *map* does not know is the rest: the
6.4 % offset is the map being a core width where the covariance carries the
tail (the `0.9558` pull of the previous section), and the 11.9 % spread is the
per-track hit pattern, material and alignment.

`k_pred` is therefore a function of `(b_+, eta_+, b_-, eta_-)` -- exactly the
axes of the `h4` table the boson-kinematics provider already hands over -- and
a **resolution class** is a bin of it.  Nothing new is asked of SCETLib or
DYTurbo.

### What the classes separate

Class edges are `k_pred` quantiles of the selected population
(`fsr_kclass.py classes`, 25/10, `data/kcl10_2510.json`); a coarser class count
is a contiguous grouping of the deciles, so one build serves the whole scan.
Selected `<u>` on the generator record at 25/10 (population `18.69e-3`):

| classes | selected `<u>` per class [1e-3] |
|---|---|
| 3 | 25.16 / 14.95 / 15.97 |
| 5 | 20.96 / 25.08 / 12.88 / 18.56 / 16.00 |
| 10 | 36.46 / 5.46 / 35.66 / 14.50 / 11.71 / 14.05 / 18.65 / 18.46 / 17.45 / 14.55 |

**The classes separate the radiation far more sharply than `k` itself does** --
a factor 7 between adjacent deciles, against the 1.3 of the `k` quintiles --
because `k` is `k_pred` smeared by the 11.9 % the map does not know.  The
mechanism is that all three terms of the 4-parameter width model increase with
`p_T`, so `sigma_pT/pT` is monotonically increasing in `p_T`; inside one `eta`
configuration the low-`k_pred` end is therefore the low-`p_T` end, and a
candidate is at low post-FSR `p_T` because **it radiated**.  Within an `eta`
configuration the class variable *is* a radiation variable, which is why the
run is not monotone in `k_pred`: the ordering by `k_pred` mixes the `eta`
configurations, and `<u>` is not a function of `k_pred` alone.

### Is a class enough to condition on `k`?

The likelihood conditions on `k`, not on the class, so the class is sufficient
only if `k` carries no further information about `u` once the class is fixed.
It does, at the level of `d<u>/d ln k = -15 to +11 e-3` inside a decile --
`+-1.7e-3` on `<u>` over the `+-1 sd` of `ln k` in the class.  But that is the
**class's own kinematic width**, not a detector-FSR correlation: with the muon
kinematics controlled cell by cell and only the *residual* of `k` left,

| kinematic cells | `d<u>/d ln k` [1e-3] | over `+-1 sd` of the residual |
|---|---|---|
| `\|eta\|`/0.16, 10 `pT` bins per leg (7 067) | −4.43 ± 0.13 | −0.69e-3 |
| `\|eta\|`/0.10, 16 `pT` bins per leg (14 451) | −1.38 ± 0.12 | −0.19e-3 |
| `\|eta\|`/0.05, 16 `pT` bins per leg (15 488) | **+0.07 ± 0.11** | **+0.01e-3** |

**Given the muon kinematics, the residual of the per-track resolution -- hit
pattern, material, alignment -- knows nothing about the radiation**, as it must
not.  What a coarse class leaves behind is the kinematic variation the class
does not resolve, and that is reducible by refining the class or the `sigma`
map, not an irreducible correlation.  (The same number with `k` taken from the
two-track fit's per-leg `sigma_rel` instead of `sigma_m/m` is
−4.85 / −1.57 / −0.02 e-3, i.e. the conclusion does not depend on which of the
two the class is conditioned on.)

### A class is part of the pass region, not a partition of the table

The class of a real candidate is read off its **reconstructed** muons, i.e.
after FSR -- and a candidate lands in a low-`k_pred` class *because* it
radiated.  A class is therefore a region in the **post-FSR** muon kinematics,
exactly as the two `p_T` thresholds are, and it belongs where they belong: in
`G(u_+, u_-)`, not in the `h` table, whose axes are the pre-FSR `(b, eta)`.
Restricting the table to the class instead makes class membership a property of
the Born kinematics and is simply wrong -- figure `17_meanu_class_restricted`:
the model's `<u|m, class>` is 0.6 to 1.5 times the MC's, and the classes'
spread collapses to almost nothing.

The correct construction is cheap.  With `t = s^2` increasing in `p_T` at fixed
`eta`, `4 k_pred^2 = t_+ + t_-` is increasing in both post-FSR momenta, so

```
U(T) = { t_a(v_+) + t_c(v_-) >= T } ,   v_q = b_q - u_q
```

is an upper-right set whose boundary `v_- = g(v_+)` is decreasing -- a
**staircase** -- and a decreasing staircase is a signed sum of quadrants,

```
P( U_k Q_k ) = sum_k S(x_k, y_k) - sum_k S(x_{k+1}, y_k) ,   Q_k = {v_+ > x_k, v_- > y_k}
```

(the pairwise intersections are nested, so inclusion-exclusion truncates).  A
class is `U(T_lo) \ U(T_hi)`, and intersecting a quadrant with the lepton
thresholds only raises its shifts, so **everything stays a quadrant and the
whole region is read off the same survival function of the same `h4` table**.
`fsr_kclass.ClassPassRegion` duck-types `fsr_perleg.PassRegion`;
`build_corr_kernel` never learns that a class is involved.

`sum_C G_C = G` holds exactly (the decomposition telescopes).  Against a direct
weighted event count at 25/10 in the peak band, 90/300 staircase knots:

| `(u_+, u_-)` | worst class residual, 90 knots | 300 knots |
|---|---|---|
| (0, 0) | 4.4e-3 | 2.7e-3 |
| (0.2, 0.2) | 3.5e-3 | 2.4e-3 |

on `G_C ~ 0.09`.  At fit level what matters is the *shape*, and figure
`14_meanu_class` has it: the model reproduces the MC's own class-conditional
`<u|m>` to **±5 % in every class over the whole window**, including class 1,
whose `<u>` climbs from 18e-3 at 75 GeV to 44e-3 at 110 GeV while class 2's
stays flat at 13e-3.

### The fit benchmark

`fit_gen.py fit --suite kclass` partitions the *same* selected events into the
`k_pred` classes -- assigned from the **post-FSR** muons, which is what a real
candidate's reconstructed muons give -- and fits **all classes simultaneously**
with common `m_Z`, `Gamma_Z` and five Legendre shape terms
(`fit_gen.MultiFit`): every class carries its own `K_sel(u|m, class)`, its own
`A(m|class)` and its own window normalisation.  `nm = 8192 -> 4096` for cost;
the inclusive rows reproduce the published ones of the previous section to
0.5 MeV.

**Without a resolution the conditioning is a no-op, exactly.**  With the same
density in every class the per-class normalisations are the same and the sum of
the class likelihoods *is* the pooled one, so the misspecified row is bit for
bit the inclusive fit.  The generator-level table says so:

| 25/10, no resolution | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| inclusive, MC-conditional + `A(m)` | +0.76 ± 0.77 | +1.04 ± 1.60 |
| classes, MC-conditional **per class** | +0.60 | +1.09 |
| classes, MC-conditional **population** | +0.76 | +1.04 |
| classes, corr `mc` `K` **per class** | +0.45 | +0.49 |
| classes, corr `mc` `K` **population** | +0.45 | +0.25 |
| classes, corr `data` `K` **per class** | +0.93 | −0.25 |
| classes, corr `data` `K` **population** | +1.21 | −0.20 |
| classes, corr `mc` `K` per class, **restricted table** | +1.34 | +0.28 |

-- every difference below 0.3 MeV, and the class-conditional model closes to
the inclusive one, which is the first check that the staircase construction is
right.

**With the per-candidate resolution it is not.**  Each class's mass is smeared
with the class's own relative Gaussian (`k` = 8.41 / 10.83 / 11.88 / 14.09 /
20.87 e-3 at 25/10, from `k_pred` times the measured `k/k_pred` = 1.064) and
the model carries the same one, applied to its own mass grid after the FSR fold
-- it cannot go into the multiplicative kernel, which is `r <= 1` by
construction while a resolution fluctuates both ways.  The only thing that
differs between the rows is then the FSR kernel and its acceptance:

| 25/10, with the resolution | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| MC-conditional **per class** | **+0.48 ± 1.13** | **+1.02 ± 2.31** |
| MC-conditional **population** | −5.35 ± 1.30 | −2.15 ± 2.36 |
| corr `mc` `K` **per class** | **+0.82 ± 1.13** | **−0.28 ± 2.31** |
| corr `mc` `K` **population** | −5.18 ± 1.28 | −1.42 ± 2.34 |
| corr `data` `K` **per class** | +1.19 ± 1.13 | −1.19 ± 2.31 |
| corr `data` `K` **population** | −3.91 ± 1.32 | −1.22 ± 2.34 |
| corr `mc` `K` per class, **restricted table** | +2.75 ± 1.13 | +1.74 ± 2.29 |
| corr `mc` `K` population, **one average resolution** | −7.26 ± 1.34 | +30.57 ± 2.45 |

| 25/25, with the resolution | Δ`m_Z` [MeV] | Δ`Γ_Z` [MeV] |
|---|---|---|
| MC-conditional **per class** | +1.53 ± 1.23 | −0.06 ± 2.51 |
| MC-conditional **population** | −2.54 ± 1.44 | −0.32 ± 2.57 |
| corr `mc` `K` **per class** | +2.48 ± 1.22 | −3.02 ± 2.50 |
| corr `mc` `K` **population** | −3.99 ± 1.37 | −0.05 ± 2.54 |
| corr `data` `K` **per class** | +2.64 ± 1.22 | −3.66 ± 2.51 |
| corr `data` `K` **population** | −3.98 ± 1.40 | −0.63 ± 2.54 |
| corr `mc` `K` per class, **restricted table** | +4.77 ± 1.22 | +0.29 ± 2.49 |
| corr `mc` `K` population, **one average resolution** | −5.07 ± 1.51 | +27.93 ± 2.65 |

Same-run differences:

| | 25/10 Δ`m_Z` / Δ`Γ_Z` | 25/25 Δ`m_Z` / Δ`Γ_Z` |
|---|---|---|
| **the conditioning, MC-conditional `K`** (population − per class) | **−5.83 / −3.18** | **−4.07 / −0.26** |
| **the conditioning, corr `mc` `K`** | **−6.00 / −1.14** | **−6.47 / +2.97** |
| **the conditioning, corr `data` `K`** | **−5.10 / −0.03** | **−6.62 / +3.03** |
| … the same three without the resolution | +0.16 / −0.05, +0.01 / −0.24, +0.28 / +0.05 | −0.30 / −0.48, 0.00 / −0.26, −0.19 / −0.43 |
| the class as a restricted table (− the pass region), corr `mc`, with / without the resolution | +1.93 / +2.01 and +0.89 / −0.21 | +2.29 / +3.31 and +1.32 / +0.60 |
| dropping the conditioning altogether (one average resolution − per class) | −8.08 / +30.85 | −7.55 / +30.95 |

**Verdict.**  A population-level `K_sel` and `A(m)` inside a likelihood that
conditions each candidate on its own resolution costs **−4 to −6.6 MeV on
`m_Z`** and up to −3 MeV on `Γ_Z`, at both selections and with either QED
configuration, while the class-conditional model closes at
**+0.5 to +2.6 MeV** -- inside the ±1.2 MeV statistical error of the test on
three of the four rows.  The bias is a *conditioning* effect and nothing else:
the same comparison without the resolution is identically zero.  Two more
things the table settles:

* **the class must go in the pass region.**  The same class built as a
  restriction of the `h` table moves the fit by +1.9 / +2.3 MeV on `m_Z`
  relative to the correct construction, i.e. half the effect it is there to
  remove, and in the wrong direction;
* **not conditioning at all is worse on `Γ_Z` than on `m_Z`.**  Using one
  average resolution for every candidate moves `Γ_Z` by **+31 MeV** -- the
  resolution-mass pairing of the v-form, which is the reason the likelihood
  conditions in the first place -- and `m_Z` by −7.3 MeV.  Conditioning is not
  optional; the point of this section is that once you condition, the FSR
  kernel has to follow.

### How many classes

The class count is **not** an accuracy knob of the FSR model: the model has to
match whatever partition the likelihood conditions on, and at every count the
class-conditional model closes while the population one does not.  At 25/10,
with the resolution, `mc` `K` (the `data` `K` rows exist only at 5 classes):

| classes | per class | population | population − per class |
|---|---|---|---|
| 3 | +2.64 / −2.17 | −0.68 / −7.55 | **−3.32 / −5.38** |
| 5 | +0.82 / −0.28 | −5.18 / −1.42 | **−6.00 / −1.14** |
| 10 | N10PER | N10POP | **N10DIFF** |

The three rows are not the same experiment -- each conditions the *toy* on its
own class resolutions as well as the model, so the truth moves with the count
-- and that is the point: what the count changes is how finely the likelihood
conditions, and the FSR kernel has to follow it, whatever it is.  What the
count does have to be fine enough for is the residual of the previous section:
`k` must carry no information about `u` once the class is fixed, and at a
decile that residual is `±1.7e-3` on `<u>` over the class's own `ln k` spread,
against a `36 -> 5e-3` swing *between* deciles.  Refining the `sigma` map is
the cheaper lever there than refining the class -- with `|eta|` cells of 0.05
the residual is `+0.01e-3`.

### The interface, and what it costs

**Nothing new crosses the boson-kinematics interface.**  The provider hands
over `h4(b_+, eta_+, b_-, eta_- | m)` and `b_mean` exactly as in the previous
section; the class is computed *inside* the kernel machinery, in the pass
region, from the same `sigma_pT/pT(pT, eta)` map the smooth acceptance already
uses.  SCETLib and DYTurbo supply what they already supply.

**What the detector level has to carry** is one `(K_sel, A)` pair per class and
a class label per candidate.  The label costs nothing: a candidate computes
`k_pred = 1/2 sqrt(s(pT_+, eta_+)^2 + s(pT_-, eta_-)^2)` from its own two
reconstructed muons through the same map, and the class edges are four numbers.
In `rabbit` that is one `MassCFTerm` per class, each with its own
`TabulatedLineshapeKernel(provider=...)` and the candidates of that class;
`m_Z`, `Gamma_Z` and the shape terms are shared parameters, so the fit is
unchanged.

The card and graph cost is `n_class` times one kernel:

| | population | 5 classes | ratio |
|---|---|---|---|
| atoms (`r, w, m_lo, m_hi`) | 43 221 | 239 177 (39-67 k each) | 5.5x |
| on disk, atoms | 1.5 MB | 8.1 MB | 5.4x |
| cell-integrated table (`fsr_table.py`, `dm` = 1 GeV, 2000 cells) | 4.85 MB | 24 MB | 5x |
| `A(m)` grid | 151 pairs | 5 x 151 pairs | 5x |
| the fold matrix in the graph (`nm` = 4096) | 250 MB | 1.25 GB | 5x |

The per-*candidate* cost is unchanged -- the classes are disjoint sets of
candidates -- so what grows is the constant tabulation, linearly, and the fold
matrices are the only thing that makes 10 classes uncomfortable rather than 5.
A continuous `k` dependence (interpolating the kernel between class nodes)
would trade that for an interpolation at every Born grid point; at five classes
it is not needed.

Building the kernels costs `n_class` x one `corr` build, plus the staircase:
`ClassPassRegion` carries 542 to 8 334 signed quadrant terms per band against
the plain pass region's 48, i.e. 100-900 s per class against ~100 s, and it is
embarrassingly parallel over `(class, cut, configuration)`.

### Reproducing

```bash
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
cd $Z
# 1. is k a function of the two legs' (pT, eta), and is its residual
#    independent of the radiation?  (the reco MC caches, numpy only, ~3 min)
python3 -u fsr_kclass.py check --aux ../fullscale/runs/auxgen_dyv2.npz \
    --pairs ../fullscale/runs/zpairs_dyv2_full.npz --res data/ptres_dyv2.npz \
    --pt-cuts 25 10 -o data/kclass_check_2510.npz
# 2. the class edges: deciles of k_pred on the selected gen sample
python3 -u fsr_kclass.py classes --gen data/genmerged_full.npz \
    --res data/ptres_dyv2.npz --nclass 10 --pt-cuts 25 10 -o data/kcl10_2510.json
# 3. the kernels.  `buildpr` is the model -- the class inside the pass region;
#    `build` is the control -- the class as a restriction of the h table, which
#    also writes the MC's own conditional kernel per class (~1 h each)
python3 -u fsr_kclass.py buildpr --classes data/kcl10_2510.json --tag kpr \
    --ngroups 5 --cut-sets 25,25 25,10 --nknot 300 --nproc 10
python3 -u fsr_kclass.py buildpr --classes data/kcl10_2510.json --tag kpr \
    --ngroups 3 10 --cut-sets 25,10 --configs mc --nknot 300 --nproc 13
python3 -u fsr_kclass.py build --gen data/genmerged_full.npz \
    --classes data/kcl10_2510.json --tag kcl --ngroups 3 5 10 --nproc 12
# 4. G of `pass AND class` off the table against a direct event count
python3 -u fsr_kclass.py ccheck --gen data/genmerged_full.npz \
    --htable data/ht_ref10_1.0gev.npz --h4 data/h4_ref10_1.0gev.npz \
    --classes data/kcl10_2510.json --ngroup 5 --pt-cuts 25 10 --nknot 300
# 5. the fit benchmarks (~15-20 min each)
./run_kclass_all.sh "2510 5" "2525 5" "2510 3" "2510 10" "2510 5 single"
# 6. the figures
ssh submit51 "cd $Z && ./run_tf_z.sh python3 -u fsr_kclass.py figs \
    --check data/kclass_check_2510.npz --classes data/kcl10_2510.json \
    --tag kpr --control kcl --cuts 2510 --ngroup 5 \
    --fits data/fit_kclass_2510_n5.json data/fit_kclass_2525_n5.json \
    --outpath ~/public_html/ZMass/cvh/260916_fsr_kclass"
```

Figures in `~/public_html/ZMass/cvh/260916_fsr_kclass/`: `10_kpred_vs_k` (the
formula against the two-track covariance), `11_u_vs_k` (the selected radiation
against the per-candidate resolution, reco and true cut), `12_u_vs_k_inclass`
(what is left of that inside a class), `13_slope_convergence` (and that it goes
away when the kinematics are controlled), `14_meanu_class` (the class-
conditional `<u|m>`, model against MC), `15_acceptance_class` (`A(m|class)`),
`16_fitshifts_*` (every fit row), `17_meanu_class_restricted` (the same as
`14` with the class built as a restriction of the `h` table -- the control that
shows it is not a partition).

---

## What is still missing for a *data* Z channel

* **The LO→MiNNLO `K(m)`, and its truncation.** The card must float a smooth
  multiplicative shape; without one the model is wrong by **+76 MeV on `Γ_Z`**.
  Five Legendre terms close at generator level and cost 1.2× on the errors, and
  they make the fit independent of the PDF set, the PDF order and μ_F. At full
  statistics the truncation order is the one item still open — see
  `../fullscale/SUMMARY.md`.
* **The selection-conditional part of the FSR kernel.** The QED content is
  settled by `fsr_analytic.py` and the selection-conditional shape by
  `fsr_perleg.py` (see above): `K_sel` and `A(m)` follow from the same radiator
  plus one boson-kinematics table `h(a_+, a_- | m)`, and close against this MC
  to +1.5 / −5.5 MeV. Run with the MC's own per-leg radiator the closure is
  +0.95 / −4.5 MeV, so that number is the **collinear factorisation** and not
  the QED; −3.2 MeV of the `Γ_Z` part is the independence of the two legs, and
  the fix is the correlated two-leg density of the exact O(α) 3-body matrix
  element. `A(m)` itself is not the problem (+0.13 / +0.03). The **asymmetric**
  cut of a real selection and the **resolution** in the acceptance are both
  built and benchmarked (see "Asymmetric cuts, and the resolution in the
  acceptance"); what is left there is that the likelihood conditions each
  candidate on its own `sigma` while `K_sel(u|m)` does not.
  `kern_from_selected.py` remains the MC-measured alternative, which needs a
  selected gen record at every calibration point.
* **Background.** `UniformBackground` / `BernsteinBackground` are wired up with
  a fixed or floating fraction, but the shape and normalisation of the real
  background (Z→ττ, top, QCD) are not measured.
* **Theory nuisances.** EW loop corrections and running α / sin²θ_W remain. PDF
  and μ_F are *covered* by the floating `K(m)`; the fixed-vs-running width
  convention is **settled** (POWHEG converts to the constant-width scheme
  unconditionally and the provider's default is exactly that); floating
  `sin²θ_W` as a shape nuisance does *not* work.
* **`m_Z` and `alpha` are exactly degenerate** in a single-resonance fit. The Z
  channel measures `m_Z` only jointly with a channel that pins the momentum
  scale — which is the whole point of the unified likelihood.
* **The τ-grid cost at scale.** Integrating on a 16× grid multiplies the
  per-chunk graph tensors by 16; the Hessian tape already peaks at ~88 GB for
  200 k candidates at `nt = 256`, so `rabbit_fit` on an H200 needs `--chunk`
  tuning, or `trust-krylov` with Hessian-vector products instead of a
  materialised Hessian.
* **The selection variable is not the fitted observable.** The maker cuts on
  `Jpsitrk_mass` (the input-track dimuon mass), not on the CVH-refit
  `Jpsi_mass`; 459/459 prototype candidates pass on the former, 458/459 on the
  latter. The truncation normalisation assumes the cut is on the observable.

Note for anything that needs gen *flags*: the maker's gen block is filled from
an `edm::View<reco::Candidate>`, which does **not** expose `GenStatusFlags`.
`status()` and `pdgId()` are enough for the Z; anything needing the flags has to
change the token type to `edm::View<reco::GenParticle>`.

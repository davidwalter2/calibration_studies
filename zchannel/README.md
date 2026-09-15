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
| | | analytic `D` / empirical `D` | | analytic / empirical |
| 60-70 | 0.19855 | 0.9995 / 0.9978 | 5.699e-3 | 0.9492 / 1.0735 |
| 70-80 | 0.29151 | 0.9988 / 0.9987 | 7.867e-3 | 0.9519 / 1.0573 |
| 80-86 | 0.34070 | 0.9986 / 0.9994 | 9.379e-3 | 0.9449 / 1.0338 |
| 86-90 | 0.36644 | 0.9983 / 0.9994 | 10.146e-3 | 0.9511 / 1.0315 |
| 90-92 | 0.37546 | 0.9986 / 0.9999 | 10.539e-3 | 0.9490 / 1.0232 |
| 92-96 | 0.38396 | 0.9984 / 0.9998 | 10.866e-3 | 0.9495 / 1.0186 |
| 96-105 | 0.40163 | 0.9987 / 1.0002 | 11.762e-3 | 0.9453 / 1.0071 |
| 105-120 | 0.42811 | 0.9983 / 1.0000 | 13.288e-3 | 0.9468 / 1.0010 |
| 120-140 | 0.45573 | 0.9985 / 0.9998 | 15.513e-3 | 0.9305 / 0.9746 |

`A(m)` closes to 0.15 %. The `<u>` column is the price of the two collinear
approximations, and they pull in opposite directions:

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
the fit closes to **+1.5 / −5.5 MeV** of the MC-conditional kernel --
of which the inclusive `mc` → `data` QED difference is already +0.97 / −0.80
(see "Fit level" above), leaving about +0.5 / −4.7 MeV for the factorisation
itself.

The full `mc` → `data` step cannot be taken inside the per-leg construction --
there is no per-leg `D` for Photos, because `D` is defined by `D (x) D = K` and
Photos' `K` is a table, not a closed form. What *can* be compared are the
analytic proxies, and they behave as they do inclusively: replacing the exact
O(α²) pair radiator by the **eikonal** limit Photos actually generates
(`e`, `μ` only) moves the per-leg fit by −0.22 / −0.57 MeV, against −0.31 /
−0.11 for the same replacement in the inclusive fit.

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
# 4. the fit benchmark (~5 min each) and the figures
./run_perleg_fit.sh physics
./run_perleg_fit.sh disc
./run_tf_z.sh python3 -u cmp_perleg.py \
     --kernel "per-leg, analytic D (data cfg)=data/kern_perleg_data_1gev.npz" \
     --kernel "per-leg, empirical D=data/kern_perleg_emp_1gev.npz" \
     --kernel "MC-conditional=data/kern_fid_sc3.3e-4.npz"
# the Photos-like pair content, for the mc <-> data comparison
python3 fsr_perleg.py kernel --htable data/ht_pt25_1.0gev.npz \
     -o data/kern_perleg_paireik.npz --acceptance data/acc_perleg_paireik.json \
     --variant exp2nll --pair e mu --pair-table data/fsr/pairkern_eik.npz
```

Figures: `01_angles` (the directions), `02_leg_x` (the measured per-leg `x`
against `D`), `03_leg_corr` (the legs are correlated), `04_z_vs_xx`
(`z = x_+ x_-`), `05_dconvd` (`D (x) D` against `K`), `06_ksel_peak`
(`K_sel(u|m)` in the peak band), `07_acceptance`, `08_mean_u`, `09_htable`,
and `00_perleg.txt` with every number above.

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
  to +1.5 / −5.5 MeV. What is *not* settled is the detector-level version of
  `h`, whose thresholds act on the reconstructed `p_T` and therefore carry the
  resolution. `kern_from_selected.py` remains the MC-measured alternative,
  which needs a selected gen record at every calibration point.
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

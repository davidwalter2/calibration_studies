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
| `fsr_analytic.py` | the **analytic** QED FSR kernel: exact O(α) + exponentiation + O(α²)LL + pair emission, and the exact matrix element it is validated against |
| `cmp_fsr.py` | the analytic kernel against the Photos++ generator record (figures + moment tables) |
| `fit_gen.py` | **generator-level closure**: FSR kernel, acceptance, and the fit |
| `kern_from_selected.py` | rebuild the kernel *and* `A(m)` from the gen record of the SELECTED reconstructed candidates |
| `fit_gensel.py` | the same closure on the selected candidates' own gen masses, so the fold is isolated from the detector |
| `make_z_card.py` | pairs cache + kernel → a rabbit datacard with one `MassCFTerm` |
| `fit_z.py` | read the card back, fit, project the covariance to full statistics |
| `z_variants.py` | rebuild the term under each modelling choice and report the bias |
| `check_tgrid.py` | is the in-maker's 64-point τ grid fine enough? (no — see below) |
| `make_lumi_scale.py` | parton-luminosity tables at μ_F = k Q (scale systematic) |
| `plot_gen.py` | the closure figures |
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
kernel treats separately through `beta_pair` at leading log — nothing is double
counted, and the pair sector's own O(alpha^2 L) terms stay inside the ~1/L = 8 %
uncertainty `beta_pair` already carries (8 % of 1.4 % of the radiator).

**`oalpha`**, fixed-order O(alpha) with a soft cutoff `x_cut`: a delta at `z = 1`
carrying `1 - P(x > x_cut)` plus (1) above it. Not a model — it measures the
size of the exponentiation.

**Pair emission.** A virtual photon of mass² `q^2` radiated off the muon
converts to a pair; the pair removes the same energy a photon would, so at
leading log the `z` dependence is the photon one and only the coefficient
changes, with the collinear log cut off at `q^2` instead of `m_mu^2`:

```
beta_pair = (2 alpha/pi) int_{q2_thr}^{s} (dq^2/q^2) rho(q^2) [ln(s/q^2) - 1] (4)
rho_lepton = (alpha/3pi)(1 + 2 m_l^2/q^2) sqrt(1 - 4 m_l^2/q^2)
rho_had    = (alpha/3pi) R(q^2),  R ~ 2 above 1 GeV^2
```

At `m = 91.19` GeV this gives `beta_pair` = 8.31e−4 (e⁺e⁻), 1.32e−3 (e, mu, tau
and hadrons together), i.e. **1.43 % and 2.26 % of the photonic `beta`**. The
soft part is exponentiated with the photon (`beta -> beta + beta_pair`) and the
hard remainder `-(beta_pair/2)(1+z)` is added to `h`; virtual pairs cancel the
soft part of (4) and otherwise change only the overall rate, so they do not
enter a normalised kernel. (4) is LL; the `-1` and the upper limit carry ~1/L =
8 % on `beta_pair`.

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
| … + `e⁺e⁻` pairs | +1.72 ± 0.54 | −2.02 ± 1.13 |
| … + `e`, `μ`, `τ`, hadron pairs | +2.28 ± 0.54 | −4.86 ± 1.13 |
| analytic exp. O(α) + O(α²)LL **+ O(α²)NLL**, banded | +1.17 ± 0.54 | +2.34 ± 1.13 |
| **… + `e`, `μ`, `τ`, hadron pairs** | **+2.35 ± 0.54** | **−5.16 ± 1.13** |
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
| `e⁺e⁻` pairs | 0.65 | −4.7 |
| all pairs (`e`, `μ`, `τ`, hadrons) | 1.21 | −7.5 |
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
python3 fsr_analytic.py kernel -o data/fsr/kan_exp2nll_pair_all.npz \
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
  i.e. ≲0.1 MeV on `m_Z`), the pair term's own 8 % LL uncertainty (0.1 MeV), and
  the additive-vs-exponentiated O(α³) ambiguity of the NLL term (0.004 MeV). The
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

### Pair emission against the analytic pair term

Photos 3.61 emits **`e+e-` and `mu+mu-` pairs only** — `PHOPAR(..., 11,
0.000511, ...)` and `PHOPAR(..., 13, 0.1057, ...)` in `photosC.cxx`, called once
before and once after the photons with `STRENG = 0.5`. No `tau`, no hadrons. The
matching analytic species set is therefore `("e", "mu")`,
`beta_pair` = 1.041e−3 at the Z (against 8.31e−4 for `e` alone and 1.317e−3 for
all species).

Running Photos with `setPhotonEmission(false)` isolates the pair kernel
(`04_pair_only`). In the peak band:

| | Photos | analytic `e`+`mu` | ratio |
|---|---|---|---|
| `<u>` | 2.518e−4 | 5.311e−4 | **0.474** |
| `P(u > 1e-2)` | 1.820e−3 | 3.319e−3 | 0.549 |
| `P(u > 5e-2)` | 1.082e−3 | 1.764e−3 | 0.613 |
| `P(u > 0.5)` | 8.68e−5 | 2.511e−4 | 0.346 |
| `P(emitted nothing)` | 0.997273 | 0.984124 | |

The rates are not comparable at small `u` and the shapes differ by construction:
the analytic term *exponentiates* `beta_pair`, so it has a soft singularity at
`u -> 0`, while Photos generates **real** pairs above `2 m_l` with a
triple-log crude probability and an ME rejection — its spectrum turns over below
`u ~ 1e-4` and its total rate is 2.73e−3. The IR-safe statement is the mean mass
loss: **Photos's pair emission removes 47 % of what the dispersive leading-log
estimate gives for the same two species, and 38 % of the all-species value.**
Pairs are therefore the one place where Photos is not merely incomplete in
species but also soft in shape, and the `data` configuration takes the analytic
term instead.

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
| analytic + O(α²)LL + all pairs | +2.28 ± 0.54 | −4.86 ± 1.13 |
| **`data`: analytic + O(α²)LL+NLL + all pairs** | **+2.35 ± 0.54** | **−5.16 ± 1.13** |

The standalone kernel and the sample's own agree to **0.5 MeV on `m_Z` and
1.0 MeV on `Γ_Z`** — below the statistical error of the closure and below the
empirical kernel's own ±0.7 MeV `sigma_cap` quadrature bias. Applying the ME
correction to every event instead of the measured 39 % moves `m_Z` by 0.34 MeV,
so the mixture is a refinement inside the closure precision, not a requirement.
Pair emission is worth **−2.9 to −3.3 MeV on `Γ_Z`** and +0.04 to +0.18 MeV on
`m_Z`; the ME correction, applied to every event, at most 0.4 MeV on `Γ_Z` and
0.3 MeV on `m_Z`.

The +1.9 MeV between the `mc` and `data` kernels on `m_Z` and −5.0 MeV on `Γ_Z`
is the physics Photos leaves out: the O(α²) leading and next-to-leading logs,
`tau` and hadronic pairs, and the factor two on the leptonic pair mass loss.

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
  the O(α²) leading log and its NLL term) with `DATA_PAIRS` =
  `("e", "mu", "tau", "had")`. First-principles throughout, exact mass
  dependence, and it supplies the three things Photos does not.

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
python3 fsr_config.py --config mc   -o data/kern_cfg_mc_sc3.3e-4.npz
python3 fsr_config.py --config data -o data/kern_cfg_data_vb6e-10.npz
./run_tf_z.sh python3 -u cmp_photos.py --runs "mcMix=Photos, sample cfg" ...
```

2.06e9 events take 80 s on 240 cores (Photos runs at 1.1e5 events/s/core); the
binary runs outside the container once built.

---

## What is still missing for a *data* Z channel

* **The LO→MiNNLO `K(m)`, and its truncation.** The card must float a smooth
  multiplicative shape; without one the model is wrong by **+76 MeV on `Γ_Z`**.
  Five Legendre terms close at generator level and cost 1.2× on the errors, and
  they make the fit independent of the PDF set, the PDF order and μ_F. At full
  statistics the truncation order is the one item still open — see
  `../fullscale/SUMMARY.md`.
* **The selection-conditional part of the FSR kernel.** The QED content is
  settled by `fsr_analytic.py` (see above) and no longer has to be measured; what
  still has to come from MC is the *ratio* `K_sel/K` that a lepton `p_T` cut
  imposes on the kernel at fixed `m_pre`, together with `A(m)`.
  `kern_from_selected.py` builds both from the selected candidates' own gen
  record; the gen-fiducial kernel describes a sample radiating 1.66× more than
  the selected one.
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

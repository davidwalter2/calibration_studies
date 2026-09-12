# Z → μμ channel of the unbinned CVH mass likelihood

The J/ψ channel of `rabbit.unbinned.MassCFTerm` fits a momentum scale `alpha`
against a resonance of known, essentially zero-width mass. The Z channel fits
`m_Z` and `Γ_Z` themselves, against a lineshape that has to be computed. This
directory is that channel: the FSR kernel, the acceptance, the datacard builder,
the fit driver, a systematics scan and the generator-level closure.

Everything downstream of `MassCFTerm` lives on the rabbit branch
`z-lineshape-kernel` (worktree `/work/submit/david_w/ZMass/rabbit-zlineshape`):
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
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/rabbit-zlineshape:\
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

## What is still missing for a *data* Z channel

* **The LO→MiNNLO `K(m)`, and its truncation.** The card must float a smooth
  multiplicative shape; without one the model is wrong by **+76 MeV on `Γ_Z`**.
  Five Legendre terms close at generator level and cost 1.2× on the errors, and
  they make the fit independent of the PDF set, the PDF order and μ_F. At full
  statistics the truncation order is the one item still open — see
  `../fullscale/SUMMARY.md`.
* **The kernel and `A(m)` must be rebuilt with the analysis selection**, with
  `sigma_cap ≤ 3.3e-4` and banded in `m_pre`. `kern_from_selected.py` does this
  from the selected candidates' own gen record; the gen-fiducial kernel
  describes a sample radiating 1.66× more than the selected one.
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

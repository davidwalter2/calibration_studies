# Generator settings — `DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos`

RunIISummer20UL16MiniAODv2, `106X_mcRun2_asymptotic_v17-v2`, 1731 files / 96.4 M events.
This is the sample the detector-level Z channel of the unbinned CVH mass likelihood
will be fitted on, so these are the numbers the `ZGammaLineshape` kernel has to
reproduce at generator level.

Provenance chain (all commands and raw outputs under `data/prov/`):

```
edmProvDump <MiniAOD>                       -> provdump_full.txt
  externalLHEProducer.args                  -> the gridpack tarball on /cvmfs
  tar xzf <gridpack> powheg.input pwhg_main -> powheg.input, pwhg_main
  ./pwhg_main  (el7 container, LHAPDF 6.4.0) -> pwhg_main_init.log   <- the authoritative EW numbers
```

Running `pwhg_main` itself is what settles the electroweak scheme: `powheg.input`
sets **no** EW inputs, so every one of them is a POWHEG default, and POWHEG prints
both the inputs and the *converted* values it actually uses.

---

## 1. Matrix element — POWHEG-BOX-V2 `Zj` + MiNNLO

Gridpack (`externalLHEProducer.args`):

```
/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc7_amd64_gcc10/13TeV/powheg/Vj_MiNNLO/
Zj_slc7_amd64_gcc10_CMSSW_12_3_1_ZJToMuMu-suggested-nnpdf31-ncalls-doublefsr-q139-
powheg-MiNNLO31-svn3900-ew-rwl6-j200-st2fix-ana-hoppetweights-ymax20.tgz
```
built 2022-04-21 by K. Long; source tree `POWHEG-BOX/Zj` + `POWHEG-BOX/MiNNLOStuff`
(svn 3900), binary `pwhg_main` shipped inside the tarball.

| `powheg.input` key | value | meaning |
|---|---|---|
| `vdecaymode` | 2 | Z → μμ |
| `min_Z_mass` | **50 GeV** | the hard `m_ll` cut of the sample (observed `min m_pre` = 50.00018) |
| `max_Z_mass` | 13000 GeV | inactive |
| `lhans1` / `lhans2` | **306000** | `NNPDF31_nnlo_as_0118` member 0 |
| `alphas_from_pdf` | 1 | α_s from the PDF evolution |
| `minlo` / `minnlo` | 1 / 1 | MiNNLO' |
| `modlog_p` | 6 | modified logarithms |
| `Q0` / `npow` / `profiledscales` | 1.390985 / 1 / 1 | profiled scales |
| `alphas_cutoff_fact` / `pdf_cutoff_fact` | 0.0 / 1.1 | (PDF freeze at 1.815 GeV) |
| `bornktmin` | 0.26 | |
| `withdamp` / `doublefsr` | 1 / 1 | |
| `ebeam1` / `ebeam2` | 6500 / 6500 | √s = 13 TeV |
| `use_NNLOPS_pdfs`, `distribute_by_ub(_AP)`, `sudscalevar`, `inc_delta_terms` | 1,1,1,1,0 | MiNNLO details |
| `runningwidth` | **absent** — see below | |

Scales: `mur = muf = Z mass` for the Bbar function (POWHEG banner).

### Electroweak inputs — the width convention, settled

`pwhg_main` prints (verbatim from `data/prov/pwhg_main_init.log`):

```
 powheginput keyword Zmass   absent;  ...  sthw2 absent; gmu absent; psZmass absent; Zwidth absent
 *************************************
 Using GF,mZ,sthw2 input scheme
 input Z mass        =  91.187600000000003
 input Z mass (phsp) =  91.187600000000003
 Z width             =   2.4941343245745466
 input sthw2         =   0.23153999447822571
 input gmu           =   1.1663786999999999E-005
 *************************************
 Z mass              =  91.153509740726733
 Z mass (phsp)       =  91.153509740726733
 Z width             =   2.4932018986110700
 Z width (phsp)      =   2.4932018986110700
 W mass              =  79.906853549493746
 1/alphaem           = 128.82531590804655
 alphaem             =   7.7624494296896114E-003
 sthw2               =   0.23153999447822571
 (unit_e)^2          =   9.7545816408700650E-002
 (g_w)^2             =   0.42129143446046846
 *************************************
    50.000000000000000      < M_Z <   13000.000000000000
```

The two blocks are the **input** (PDG, running-width) parameters and the
**used** (constant-width) ones. The conversion is exactly

```
m_fixed = m_run / sqrt(1 + (Gamma_run/m_run)^2)     91.1876 / 1.000374  = 91.153509740726733
G_fixed = G_run / sqrt(1 + (Gamma_run/m_run)^2)      2.494134 / 1.000374 =  2.4932018986110700
```
(reproduced to 3e-15 GeV), i.e. **POWHEG runs a fixed (constant) width
Breit–Wigner** with `m_Z = 91.153509740726733`, `Γ_Z = 2.4932018986110700`.
There is no `runningwidth` flag in this process — the conversion is
unconditional. **The `width_scheme="fixed"` default of `ZGammaLineshape` is the
generator's own convention**, and the 34 MeV fixed-vs-running ambiguity flagged
in the earlier note is therefore resolved, not merely bounded.

Comparison with `rabbit.lineshapes.zgamma` (and
`calibration_studies/lineshape/constants.py`, whence it was ported):

| quantity | POWHEG | provider | difference |
|---|---|---|---|
| `G_F` | 1.1663787e-5 | 1.1663787e-5 | 0 |
| `m_W` | 79.906853549493746 | `MW` = 79.906853549493746 | 0 |
| `m_Z` (fixed) | 91.153509740726733 | `MZ_FIXED` = 91.153509740726733 | 0 |
| `Γ_Z` (fixed) | 2.4932018986110700 | `GZ_FIXED` = **2.4932** | **−1.9 keV** |
| `sin²θ_W` | 0.23153999447822571 | `1 − MW²/MZ_FIXED²` = 0.2315399944782256 | 1.1e-16 |
| `1/α(m_Z)` | 128.82531590804655 | `√2 G_F m_W² s²/π` → 128.82531590804655 | 0 |
| Γ_Z used as the "truth" of the closure | 2.4932018986110700 | | |

`Γ_Z` is a *derived* quantity in POWHEG (`Zwidth` absent → LO width from
G_F, m_Z, sin²θ_W = 2.4941343245745466 running / 2.4932018986110700 fixed).
The provider's rounded `GZ_FIXED = 2.4932` is 1.9 keV below it — irrelevant at
the 2 MeV target, but the closure below quotes the exact value as truth.

**Not modelled by the provider but present in the sample:** MiNNLO' QCD
corrections (the provider's parton luminosity is LO with μ_F = Q and the hard ME
is LO), Z→μμ EW loop corrections (POWHEG `Zj` is LO-EW as well, so this is a
*common* omission), and the LHE-level extra-parton kinematics (which do not
change `m_ll`).

## 2. Parton shower / hadronisation — Pythia8, CP5

`Pythia8HadronizerFilter`, `comEnergy = 13000`, parameter sets
`pythia8CommonSettings`, `pythia8CP5Settings`, `pythia8PSweightsSettings`,
`processParameters`. CP5 uses `PDF:pSet = LHAPDF6:NNPDF31_nnlo_as_0118`.

`processParameters` (the ones that matter for the mass):

```
SpaceShower:pTmaxMatch = 1
TimeShower:pTmaxMatch  = 1
ParticleDecays:allowPhotonRadiation = on
TimeShower:QEDshowerByL     = off     <-- no Pythia QED FSR off the leptons
TimeShower:QEDshowerByOther = off
BeamRemnants:hardKTOnlyLHE = on
BeamRemnants:primordialKThard = 2.225001
SpaceShower:dipoleRecoil = 1
```

## 3. QED final-state radiation — Photos++

```
ExternalDecays = PSet( Photospp = untracked PSet( parameterSets = vstring() ),
                       parameterSets = vstring('Photospp') )
```

The provenance is **silent**, not empty: the `Photospp` PSet is a
`cms.untracked.PSet` (`ExternalDecayDriver` fetches it with
`getUntrackedParameter`) and untracked parameters are not stored in EDM
provenance. The configuration is the GEN request's fragment
(`SMP-RunIISummer20UL16wmLHEGEN-00496`, CMSSW_10_6_30_patch1, which
`edmProvDump`'s processing history on this file matches), and every non-default
switch in it is confirmed in the generated events:

| switch | value | Photos 3.61 default | evidence in the gen record |
|---|---|---|---|
| `setExponentiation` | True | True | photon multiplicity under the Z reaches 7; 5.9 % of events have ≥ 3 |
| `setInfraredCutOff` | **1e-7** | 0.01 | minimum photon energy in the Z rest frame 2.6e−6 GeV; 75.4 % of photons below the default cutoff |
| `setMeCorrectionWtForW` | **True** | False | (the W samples carry a byte-identical block) |
| `setMeCorrectionWtForZ` | **True** | False | standalone closure, `../README.md` |
| `setMomentumConservationThreshold` | 0.1 | 0.1 | |
| `setPairEmission` | **True** | False | 2.430e−3 of events carry exactly two status-1 `e±` under the Z (never one) and 3.066e−4 exactly four muons (never three), with masses starting at 1.028 and 216.5 MeV against `2m_e` = 1.022 and `2m_mu` = 211.3 |
| `setPhotonEmission` | True | True | |
| `setStopAtCriticalError` | False | True | |
| `suppressAll` + `forceBremForDecay(23, ±24)` | on | off | 100.00 % of the status-746 Photos history entries sit under the Z branch |

Combined with `TimeShower:QEDshowerByL = off`, **all** muon FSR in this sample
comes from Photos++. The interface keeps the un-radiated muon copies at
`status == 746`, and writes them whenever Photos touched the muons — by a
photon *or* by a pair — so `npre == 0` is "Photos emitted nothing at all".

Two consequences for the FSR kernel, both quantified in
`../README.md`, "Photos++ standalone and the two kernel configurations":

* the exact Z matrix-element correction is switched on but only **fires** on the
  38.9 % of events whose Z has two opposite-sign fermion mothers — the
  gluon-initiated Born of POWHEG `Zj` + MiNNLO kills it on the rest;
* `dump_gen_fsr.py` requires exactly two hard-process status-1 muons, so the
  3.07e−4 of events in which Photos emitted a `mu+ mu-` pair are dropped. The
  loss is biased against the largest non-photon energy loss.

## 4. Event weights

`GenEventInfoProduct.weight()` is **not** ±1. On one file (88 443 events):

| | |
|---|---|
| dominant values | +2374.19 (91.92 %), −2374.19 (7.79 %) |
| tail | 0.29 % of events with other values, up to \|w\| = 2.7e5 |
| ⟨w⟩ | 2004.63 |
| N_eff = (Σw)²/Σw² | 53 219 for N = 88 443 → **N_eff/N = 0.602** |

The negative weights are MiNNLO's; the fit must be **weighted**, and its
covariance must use the sandwich form `H⁻¹ J H⁻¹` with `J = Σ w² gg^T`.

## 5. Generator-level mass definitions in `prunedGenParticles`

| variable | definition | agreement with `m_pre` |
|---|---|---|
| `m_pre` | mass of the hard-process Z, `status == 62` | — (the reference) |
| `m_pre22` | same Z, `status == 22` (first copy) | max \|Δ\| = 7.6e-6 GeV (float32 storage) |
| `m_pre746` | the pre-Photos μ⁺μ⁻ pair, `status == 746`; exists only in the 58.7 % of events that radiated | rms 4.0e-6 GeV |
| `m_prelep` | the 746 pair when it exists, else the `status == 1` pair (which *is* the pre-FSR pair then) | rms 4.0e-6 GeV, max 4.8e-5 GeV |
| `m_post` | the two prompt `status == 1` hard-process muons — what the tracker sees | the FSR kernel |
| `m_dress` | `m_post` with prompt photons inside ΔR < 0.1 added back | reference only |

**Both pre-FSR definitions coincide to the MiniAOD float precision (4 keV)**, so
the choice is immaterial; the fits below use `m_pre` (defined in every event).
`min(m_pre) = 50.00018 GeV` confirms the `min_Z_mass 50` hard cut, which is the
natural lower edge of the provider's Born window.

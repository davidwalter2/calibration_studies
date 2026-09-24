# K_S -> pi+ pi- displaced-vertex momentum-scale closure

CVH two-track refit of K_S -> pi+ pi- from the inclusive B -> J/psi + X MC,
vertex constraint ON, beam-spot constraint OFF, pion mass hypothesis on both
legs; the momentum scale `alpha` is measured on the K_S mass with the same
unbinned mass-CF likelihood and the same two first-principles corrections
(sigma-artefact, Jensen) as the J/psi closure.

**Code: CMSSW `CMSSW_15_0_19_patch2_dev2` at commit `3d4c926ff461`**, which
carries the multiple-scattering within-step correlation sign
`res(1,4) = +S3` (defect (d) of `resolution/DEFECTS_STATE.md`) and the
reference-EDM fix under `doVtxConstraint` (defect (a)).  Every number in
sections 4b and 8-10 is from this code.  The same refit on the pre-fix code
is kept only as the before-fix comparison of section 11.

## 1. The sample

`/ceph/submit/data/group/cms/store/mc/inclusive_btojpsix_2016postvfp_v3/`
125999 ROOT files, ~53.5 events/file (~6.7 M events), ALCARECOTkAlJpsiX,
split-1, SimTracks + SimVertices kept.  Produced with the official UL16 chain
(step1/step3 `CMSSW_10_6_20_patch1`, step2 HLT `CMSSW_8_0_36_UL_patch1`).

### Track and candidate content (verified with `edmDumpEventContent` + FWLite)

* `vector<reco::Track> ALCARECOTkAlJpsiX` -- **18.26 tracks/event**, i.e. NOT
  muon-only: the pion tracks are there, with `TrackExtra` and the pixel/strip
  cluster collections, so a two-track refit has all the hits it needs.
* `vector<reco::VertexCompositeCandidate> ALCARECOTkAlJpsiXB0KsResonances`
  -- B0 -> [J/psi -> mu mu] [K_S -> pi pi] pairings.  `daughter(0)` is the
  J/psi (pdgId 443) with two muon leaves, `daughter(1)` is the K_S (pdgId 310)
  with two charged leaves of mass 0.13957.  **0.185 candidates/event.**
  This is what makes the test possible with NO new C++: the generic
  `VertexCompositeCandidate` decomposition in
  `ResidualGlobalCorrectionMakerTwoTrackG4e.cc` takes `subsystemDaughter = 1`
  and fits the K_S subsystem.
* also present: `B0Kstar`, `BPlus`, `Bc`, `BsPhi`, `JpsiOnly`, `Lambdab`,
  `Psi2S` resonance collections; `SimTrack`/`SimVertex`; `genParticles` +
  barcodes; `offlinePrimaryVertices`; the three dE/dx ValueMaps.
* (run, lumi, event) is UNIQUE: 11229/11229 over a 210-file scan.

### How the K_S decays -- and why no FSR kernel is needed

`edmProvDump`: the generator is `Pythia8GeneratorFilter` with
`ExternalDecays = {EvtGen130}`; `pythia8CommonSettings` carries
`ParticleDecays:limitTau0 = on`, `ParticleDecays:tau0Max = 10` (mm) and
`ParticleDecays:allowPhotonRadiation = on`.

* c*tau(K_S) = 26.8 mm > 10 mm, so **neither Pythia nor EvtGen decays the
  K_S**: measured on 16110 events, 181290 gen K_S, **every one of them has
  zero daughters in the gen record and zero photons.**  Geant4 decays them.
* The Geant4 decay is pure two-body phase space with **no radiation**.  The
  simulated pi+pi- invariant mass at the decay vertex is
  **0.4976144 GeV, spread 7e-6 GeV (SimTrack float rounding only)** -- i.e. a
  delta.  The Geant4 K_S mass is 3.4e-6 GeV (6.8e-6 relative) above the PDG
  497.611 +- 0.013 MeV, so the closure uses the PER-CANDIDATE SIMULATED mass,
  not a PDG number, and no FSR kernel is required.
* Decay modes of the simulated K_S (80682 of them in an 11229-event scan):
  pi+pi- 69.2 %, pi0pi0 31.2 % of the decays that happen; the rest are nuclear
  interactions and single-prong endings.

### Pileup

**Premixed.**  `mixData` is a `PreMixingModule` with an `EmbeddedRootSource`
secondary source, `addPileupInfo` has `isPreMixed = true`, and the path list
contains a `datamixing_step`.  Consequence, verified directly: every SimVertex
in the sample carries `EncodedEventId` **bx = 0, event = 0** (7.78 M vertices
scanned, one population) -- the premix library contributes no simulation
truth.  So a sim-matched K_S is necessarily from the signal interaction, and a
pileup K_S can only appear as an UNMATCHED candidate.  The unmatched
population shows no K_S mass peak (pre-refit mass q05/med/q95 =
0.4357/0.4974/0.5588, flat across the selection window, against
0.4842/0.4988/0.5137 for the matched ones), so genuine pileup K_S are a
negligible part of it; it is combinatorics.

### Charm FSR in this sample (asked for alongside the K_S facts)

Decayer: **EvtGen130** for everything below tau0Max, Pythia8 above it,
Geant4 for the long-lived.  Measured on 16110 events:

| decay | N | with >=1 gen photon |
|---|---|---|
| D0 -> K-pi+ (two-body) | 785 | **131 (16.7 %)** |
| D*+ -> D0 pi+ (two-body) | 5017 | **5 (0.10 %)** |
| J/psi -> mu mu | 16358 | 4850 (29.6 %) |
| K_S (any) | 181290 | **0** |

So FSR **is** modelled for D0 -> K pi (PHOTOS through EvtGen: 16.7 % photon
rate, and the radiative ones pull the K pi mass below M_D0 -- the non-radiative
ones sit exactly at 1.864840 GeV = M_D0, the radiative tail reaches down to
1.555), and is **effectively absent for D*+ -> D0 pi+** (0.1 %), as the 5.9 MeV
Q value requires.  A D0 -> K pi closure would therefore need a radiative
kernel; the K_S one does not.

## 2. Yield (210-file scan, 11229 events)

| | per event | comment |
|---|---|---|
| gen K_S (status 1, undecayed) | 11.25 | full inelastic event |
| simulated K_S -> pi+pi- | 7.19 | most soft/forward |
| ... with BOTH daughters on an ALCARECO track | **0.0239** | the reconstructable ceiling of this ALCARECO |
| B0Ks candidates | 0.185 | what the refit is run on |
| ... truth-matched to a sim K_S | **0.0128** (6.9 % purity) | 54 % of the ceiling |

Reconstructable sim K_S: decay radius q10/med/q90 = 0.43/1.91/12.15 cm
(77.6 % below 4 cm, 94.4 % below 20 cm, 100 % below 60 cm), |z| median 3.3 cm,
K_S pT median 1.71 GeV, daughter pT median 0.53/1.14 GeV and |p| 0.90/2.01 GeV;
55.2 % come from a B hadron (44 % of the matched reco ones from a B0).
64.2 % have >= 8 valid hits on BOTH legs -- the `minLegHits = 8` acceptance.

Projected over the full sample: ~86 k truth-matched K_S candidates before the
downstream selection, ~55 k after it.

## 3. Refit configuration

Driver `resolution/ksclosure/runCvhKs.py` (a cmsRun cfg in this repo -- the
CMSSW area is used exactly as released, no code change, no rebuild).
Full flag list and provenance in
`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed/PROVENANCE.txt`.

The one thing to know: the maker's own gen matching is muon-specific
(`|pdgId| == 13`, `ResidualGlobalCorrectionMakerTwoTrackG4e.cc:6116`), so it
is run with `doGen = False, requireGen = False` -- with `requireGen = True` a
pion channel would be silently emptied -- and the truth match is done OFFLINE
by (run, lumi, event) + decay vertex + both daughter momenta, against
`ks_truth_dump.py`.

## 4. Files

(the sigma-artefact study of section 9 adds eight more; they are listed in
section 9.8)

| | |
|---|---|
| `runCvhKs.py` | the cmsRun driver |
| `ks_truth_dump.py` | Geant4 K_S -> pipi truth, one row per decay, keyed by (run,lumi,event) |
| `ks_pairs.py` | the mass-CF pairs cache (cf_inmaker layout) + the truth join |
| `ks_yield.py` | yield / purity / reconstructable-ceiling study |
| `gen_decays.py` | decayer + FSR bookkeeping (K_S, D0, D*, B0, J/psi) |
| `inspect_sample.py`, `dump_b0ks.py` | event-content and candidate-structure dumps |
| `prod/` | slurm arrays for the refit and the truth dump |
| `run_ks_fixed.sh` | the whole chain for one production: pairs cache, fits, systematics, a_res forms, tables, figures |
| `ks_compare_closure.py` | the closure table of two productions side by side |
| `ks_fix_plots.py` | the two-production figures (section 11) |
| `smoke_compare.py` | candidate-by-candidate comparison of two refits of the same input |

Production output:
`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed/`
(`task_NNNN/` refit output, `truth/` truth, `chunks/` input lists).  The truth
dump is FWLite over the generator and Geant4 records and does not touch the
CVH code; `truth/` is a link to the one of the before-fix production.
Working directory for the caches, cards and fits: `runs/fixed/`.

## 4b. Production (slurm, 1000 tasks of 126 files)

`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed/`,
`prod/submit_ks_half.sh`, gaps with `prod/resubmit_half.sh`, per-chunk seal
with `prod/seal_halves.sh`, summary with `prod/prod_summary.sh`.  Each
252-file chunk of the input list runs as two 126-file halves writing into the
same `task_NNNN/` (`globalcor_ks_{a,b}_*.root`); `ks_pairs.py` globs the task
directory, so the truth join sees the 500 chunks unchanged.  The halves exist
because the `submit` partition backfills a 12 min request and not a 15 min one.

```
attempted 349 766   succeeded 347 583   failed 2 183 (0.624 %)
1000 tasks: median 321 s, max 656 s, total 90.6 core-hours
```

The failures are propagation failures of the combinatorial candidates; the
truth-matched set is unaffected by them.  Reference iterations are no longer
capped at 10 (mean 5.5 on the smoke chunk): the reference EDM is now finite
under the vertex constraint and the fit stops when it converges.

## 5. The chain, validated end to end

A 240-file local pass (6 x 40 files, 10 795 events) exercised every step:

| step | result |
|---|---|
| refit | 703 candidates attempted, **698 succeeded, 5 failed (0.71 %)** |
| truth join | 92 of 698 truth-matched (**13.2 %** -- the pre-fit `minLegHits`/`minNdof` cuts raise the purity from the 6.9 % of the raw candidate collection) |
| pull `(m_reco-m_gen)/sigma_m` | mean -0.079, median -0.026, std 0.894, robust width 0.713 |
| residual `m_reco - m_gen` | median **-0.18 MeV**, RMS 6.39 MeV (sigma_m/m ~ 1.3e-2) |
| card | `make_card.py --residual-mode --mref 0.497611`, delta kernel, alpha floating, `--ares on --jensen exact --corr-form fluctuation` |
| fit | `rabbit_fit.py --paramModel UnbinnedParams --minimizerMethod trust-exact --freezeParameters k_hit k_ms k_ioni k_rad`, **EDM 2.9e-28** |
| alpha (91 candidates) | -0.036 +- 0.948 (1e-3) -- statistics only, the chain test |

Projected from that error: sigma(alpha) = 0.948 x sqrt(91/N), i.e. **~0.04e-3 at
the ~57 k truth-matched candidates the full sample gives** -- about 1.6x the
J/psi v3 statistical error (0.025e-3).

### Production candidates (first 8 chunks, 887 truth-matched)

| | K_S -> pi pi | J/psi -> mu mu (same MC) |
|---|---|---|
| sigma_m | 6.07 MeV | 31 MeV |
| sigma_m/m | 0.0122 | 0.0101 |
| pull robust width | 0.965 | 0.977 |
| pull std | 1.176 | 1.032 |
| chi2/ndof median (ndof median) | 0.864 (23) | -- |
| chi2/ndof > 3 | 0.11 % | -- |
| `vgf` (Gaussian hit share) | 0.043 | 0.100 |
| `f_ang` | 0.708 | 0.086 (gun) |
| CF families at tau = 2 | hit 4.5 %, **MS 91.9 %**, ioni 2.0 %, rad 0.0 % | hit 10.8 %, MS 87.4 %, ioni 1.7 %, rad 0.0 % |
| momentum lever arm `f` | 0.661 (median), 0.634 (1/sigma^2-weighted) | 0.9953 |

`S_rad` is identically zero for pions: `Geant4ePropagator::fillRadiativeSpectrum`
returns early for non-muons.  There is also NO nuclear-elastic family in the
in-maker CF, while a pion crossing the tracker takes ~0.05 elastic nuclear
collisions of 25-35 mrad -- a few per cent of candidates carry one unmodelled
angular kick on one leg.  That is a tail, not a width, and it is the leading
known missing resolution effect for a hadron channel; the log-scale
`ks_pull_model_tails` figure is where it would show.

### Two things that are genuinely different from the J/psi

* **`f_ang` median 0.74** (J/psi gun 0.086, data 0.106, DY 0.0003).  For
  K_S -> pi pi three quarters of sigma_m^2 comes from the OPENING ANGLE, not
  from the two momenta -- soft, widely-separated daughters.  The Jensen term
  scales by (1.5 - f_ang)/1.5 = 0.51, so it is half of what the naive
  1.5 (sigma_m/m)^2 would give: median 1.5 s^2 = 1.30e-4 relative = 0.06 MeV.
  `Jpsi_fang` is per-candidate and truth-free, so this needs no new input --
  but it does mean the K_S carries less momentum-scale information per unit of
  mass resolution than a J/psi, which is why its statistical error per
  candidate is larger than sigma_m/m alone would suggest.
* **`a_res` median 0.0133** (the sigma-artefact prefactor (1+vgf) sigma/m),
  against ~0.011 at the J/psi -- the correction is of the same size.

### One shared-code fix this needed

`fullscale/make_card.py` built the `ZGammaLineshape` provider unconditionally,
including in `--residual-mode` where the kernel is a delta and the provider is
discarded.  Its `check_tau_range` then aborted the card: the K_S needs
tau ~ 4.6e3 1/GeV (max(tgrid)/sigma_min = 7.89/0.0017) against the 40 the Z
lineshape tabulates.  The provider is now skipped in residual mode (and
`provider_config` records `delta (residual mode)`).  Nothing else changes: the
kernel and the parameter declarations never read it there.

## 6. The momentum-scale lever arm: alpha on the mass is NOT the momentum scale

`MassCFTerm`'s `alpha` is the relative shift of the MASS:
`delta_i = mobs_i - m_ref * alpha * 1e-3`.  A MOMENTUM scale `eps`
(p -> (1+eps) p on both legs) shifts the mass of candidate i by `m_i f_i eps`,
with

```
m^2 = 2 m_pi^2 + 2 (E1 E2 - p1.p2)
f   = d ln m / d ln p = [ m^2 - m_pi^2 (2 + E1/E2 + E2/E1) ] / m^2
```

which is 1 only in the ultra-relativistic limit.

| channel | f (symmetric decay) |
|---|---|
| Z -> mu mu | 1 - 1.3e-6 |
| J/psi -> mu mu | **0.99534** |
| K_S -> pi pi | **0.6853** |

Because `2 m_pi / m_KS = 0.561`, a third of the K_S mass is rest mass that
does not move with the momentum scale.  Measured on this sample: f has median
0.665, q10 0.517, and a 1/sigma^2-weighted mean **0.642** (asymmetric decays
have E1/E2 + E2/E1 > 2 and sit lower; the minimum seen is 0.235).

The maximum-likelihood estimate of a location shift weights candidates by their
Fisher information (~1/sigma^2), so

```
eps = alpha_mass / <f>_{1/sigma^2}
```

and the statistical error transforms with it.  Ignoring this would understate
the K_S momentum scale by a factor 1.56.  The J/psi closure never had to
distinguish the two (f = 0.9953), which is why `make_card.py`'s `alpha` is
defined on the mass.  **This is not specific to the K_S**, and for other two-body channels it is
much more severe.  Scanning the decay angle at a representative parent pT:

| channel | f at cos(theta*) = 0 | f averaged over cos(theta*) | f min |
|---|---|---|---|
| Z -> mu mu (pT 20) | 1.0000 | 1.0000 | 1.0000 |
| J/psi -> mu mu (pT 10) | 0.9953 | 0.9909 | 0.949 |
| D0 -> K pi (pT 8) | 0.8564 | 0.7824 | 0.145 |
| K_S -> pi pi (pT 1.7) | 0.6853 | 0.5706 | 0.147 |
| **Lambda -> p pi (pT 3)** | **0.0623** | **0.0470** | 0.005 |

`Lambda -> p pi` is essentially BLIND to the momentum scale: m_Lambda exceeds
m_p + m_pi by only 38 MeV, so 97 % of the Lambda mass is rest mass that does
not move when the momenta are scaled.  A Lambda mass measured to 1 MeV
constrains the momentum scale no better than a K_S mass measured to 13 MeV.
This is a real reduction in momentum-scale information per unit of mass
resolution, not a bookkeeping convention, and it is the first thing to check
before proposing any hadronic two-body channel as a calibration probe.

Both numbers are printed by `ks_table.py`; `eps` is the one that compares with
the J/psi closure.

## 7. The sigma-artefact prefactor and the angular share

`a_i = (1 + f_hit) sigma_i/m_i` (implemented as `(1 + vgf) sigma/|m|`) comes
from `sigma_m^2 = A m^4 + B m^2 + C`, i.e. from a mass whose resolution is
carried by the two MOMENTA: the hit term gives `sigma_p/p ~ p` hence
`sigma_m ~ m^2`, the MS term `sigma_p/p ~ const` hence `sigma_m ~ m`, and
`d ln sigma_m / d ln m = 1 + f_hit`.

For K_S -> pi pi that premise looks broken: **`f_ang` has median 0.70**, so
70 % of `sigma_m^2` comes from the OPENING ANGLE, and a suppression
`a_eff/a_used = (1 - f_ang)[(1 + vgf)(1 - f_ang) + f_ang]/(1 + vgf)` (0.28
median / 0.47 weighted) follows if an angular fluctuation moves `m` without
moving `sigma_m`.

**It does move `sigma_m`, and the suppression is wrong**: `sigma_m^2 =
J^T Sigma J` and the mass Jacobian `J` is itself a function of the opening
angle, so `sigma_m` responds to an angular fluctuation even at perfectly fixed
`Sigma`.  Section 10 measures the coefficient from truth on this sample and on
the J/psi, excludes the suppressed form at 6.8 and 10 sigma, and confirms the
closed form to 0.34 +- 0.68 % at the J/psi.  **Use the closed form.  The 0.28 /
0.47 numbers above, and the `--a-scale 0.470` card, are superseded.**
`make_card.py`'s scalar `--a-scale` (default 1) and per-candidate
`--a-res-key` (default absent) are what let this be measured rather than
assumed; both leave every existing card bit-identical.

## 8. Result (full sample, 52 829 truth-matched candidates)

`alpha_mass` is what `MassCFTerm` fits; `eps = alpha_mass / <f>` with
`<f> = 0.630` is the momentum scale (section 6).  Every fit converges with
EDM < 3e-16, far inside the 1e-3 tolerance.

347 583 candidates were fit, 52 829 of them truth-matched (15.20 %).
Pull `(m_reco - m_gen)/sigma_m`: mean +0.022, median +0.038, std 1.223, robust
width 0.954.  Residual `m_reco - m_gen`: median +0.218 MeV, RMS 8.90 MeV.
chi2/ndof median 0.862 (ndof median 22), 0.75 % above 3.

### The correction ladder

| | alpha_mass [1e-3] | eps [1e-3] |
|---|---|---|
| naive (no corrections) | +0.181 +- 0.042 | +0.288 +- 0.067 |
| Jensen exact only (a_res off) | +0.093 +- 0.042 | +0.148 +- 0.067 |
| a_res (J/psi closed form) + Jensen exact | +0.261 +- 0.042 | +0.414 +- 0.067 |
| **a_res MEASURED from truth + Jensen exact** | **+0.258 +- 0.042** | **+0.410 +- 0.067** |
| a_res (J/psi closed form), no Jensen | +0.349 +- 0.042 | +0.553 +- 0.067 |

**The displaced K_S momentum-scale closure is**

```
eps = +0.41 +- 0.07 (stat) +- 0.03 (a_res) +- 0.04 (other)  x 1e-3
```

against the J/psi -> mu mu closure of **+0.006 +- 0.025 x 1e-3** on the same
MC.  The central value is the fit with the sigma-artefact slope MEASURED from
truth (section 9; the shipped closed form gives +0.414), and the a_res
uncertainty is that measurement's own, 11.2 % statistical plus 3.8 % estimator
closure on a total a_res correction of 0.266e-3.  Statistics and the unmodelled nuclear-elastic
tail are now the limiting systematics, not a_res.

### Systematics (nominal = a_res in the J/psi form + Jensen, +0.414)

| variation | eps [1e-3] | shift |
|---|---|---|
| residual window 3 sigma (from 9) | +0.372 | -0.042 |
| residual window 5 sigma | +0.410 | -0.005 |
| floating uniform background | +0.422 | +0.007 |
| chi2/ndof < 1.5 (from 3) | +0.410 | -0.004 |
| truth match tight (0.05/0.05/0.20/1 cm), n 50 558 | +0.387 | -0.027 |
| truth match loose (0.60/0.60/0.90/5 cm), n 56 594 | +0.397 | -0.018 |

Everything except the 3-sigma residual window is a no-op at the 0.03e-3 level.
The 3-sigma window shift, -0.042e-3, IS the unmodelled tail.

### Populations and bins (nominal correction set)

| sample | n | eps [1e-3] |
|---|---|---|
| all | 52829 | +0.414 +- 0.068 |
| from a B hadron | 31955 | +0.557 +- 0.088 |
| from a B0 | 27720 | +0.487 +- 0.094 |
| prompt / fragmentation | 20874 | +0.198 +- 0.105 |
| decay radius < 2 cm | 12392 | +0.483 +- 0.136 |
| 2 - 4 cm | 12992 | +0.079 +- 0.135 |
| 4 - 10 cm | 13508 | +0.680 +- 0.127 |
| > 10 cm | 13937 | +0.380 +- 0.145 |
| min-leg p < 0.8 GeV | 11686 | +0.265 +- 0.130 |
| 0.8 - 1.5 GeV | 17593 | +0.365 +- 0.108 |
| > 1.5 GeV | 23550 | +0.595 +- 0.116 |

* the four decay-radius bins scatter by **chi2 = 10.9/3 (p = 0.012)** around
  their mean -- the first evidence of a radius dependence, and it is not
  monotonic.  The model-free median residual (`ks_resid_vs_radius`) shows the
  same shape in finer bins: **+0.58 +- 0.17 MeV below 1 cm**, a dip to
  +0.03 to +0.19 MeV between 1 and 3 cm, and a +0.22 to +0.30 MeV plateau
  beyond 3 cm.  Those radii are the beam pipe (2.2 cm) and BPix L1 (2.9 cm):
  a displaced closure resolves structure on the scale of the innermost
  material, which is exactly what it is for.
* momentum: the three bins scatter by chi2 = 4.0/2, and the highest-minus-
  lowest difference is **+0.33 +- 0.17e-3** -- a hint that the non-closure
  grows with the pion momentum, i.e. a curvature-like rather than an
  energy-loss-like term (energy loss would fall as 1/p).
* B-daughter minus prompt = **+0.36 +- 0.14e-3**, but the two differ in
  momentum as well as in production point, so this is not an independent
  statement.

### The unmodelled tail

`|z| >= 3` holds **1.62 %** of candidates against 0.27 % for a Gaussian.  The
log-scale `ks_pull_model_tails` figure shows the per-candidate CF model
describing the core to a few per cent, the data running ~20 % above it at
z = +2 to +3, and ~1 candidate per bin sitting flat out to |z| = 10.  The
in-maker CF has **no nuclear-elastic family** and `S_rad` is identically zero
for pions.  The excess is SYMMETRIC (0.78 % at z >= +3, 0.83 % at z <= -3).
Section 12 adds the nuclear-elastic family: the whole channel over-predicts
this tail 3.6x, the part of it that the reconstruction cannot reject (the
material from the decay vertex to the first measured module) describes it.

### What the K_S adds beyond the J/psi

* **species**: charged pions, spin 0, where the J/psi gives muons.  The CF
  model self-selects the spin-0 knock-on branch and suppresses Kokoulin, and
  `ReferenceSpeciesDedx` corrects the +5.24e-3 dE/dx defect of serving pions
  from the proton table -- none of which the muon channel exercises.
* **displacement**: decay radius median 4.2 cm, q90 22 cm, up to 60 cm.  The
  reference trajectory starts at the decay vertex, so the material integral,
  the number of layers and the lever arm all change along the flight path --
  a test the prompt channels cannot do at all, and the radius bins already
  show structure at p = 0.012.
* **momentum**: pion |p| 1.3 (weaker leg) and 2.6 GeV (stronger), against
  3-10 GeV for J/psi muons -- the regime where MS dominates (92 % of the CF
  against 87 %) and where the CVH was "never validated" below ~0.9 GeV.
* **an unmodelled resolution channel made visible**: the 1.6 % beyond 3 sigma
  is a direct, quantitative handle on the missing nuclear-elastic term, which
  the muon channels cannot see because a muon has none.
* what it does NOT add is precision: at 2 m_pi / m_KS = 0.56 the momentum
  lever arm is 0.63, so 53 k K_S buy sigma(eps) = 0.067e-3 where 129 k J/psi
  buy 0.025e-3.

### A caveat on the J/psi comparison

The reference J/psi -> mu mu number (+0.006 +- 0.025e-3) was measured on the
`btojpsix_v3_260904f_m0` refit, which ran with the ALIGNED geometry from the
GT and before `3d4c926ff461`; this K_S production runs with
`useIdealGeometry=True` on the fixed code.  The code difference does not
matter for the J/psi: the correlation-sign fix moves the J/psi scale by
+0.003e-3 (`DEFECTS_STATE.md`).  The geometry difference does: the
like-for-like comparison is the ideal-geometry J/psi closure (pairs cache
`fullscale/runs/jpairs_ideal_n600.npz`), which has not been fit yet.


## 9. The sigma-artefact slope and the track angles

`a_res` is a REGRESSION SLOPE, not a formula: `sigma_i = sigma_bar_i +
a_i (m_i - mu_i)` defines it, so

```
a = Cov(dm, d sigma_m)/Var(dm) = (J^T Sigma G)/(J^T Sigma J),
J = grad_u m,  G = grad_u sigma_m,  u = (plus, minus) x (q/p, lambda, phi)
```

over the six reference parameters.  Everything below is the dimensionless
prefactor `A = a m/sigma_m`, which the shipped closed form predicts to be
`1 + f_hit`.  Section 7 argued that `A` should be suppressed by the angular
share of the mass variance.  **It is not.  That suppression is wrong, at both
the K_S and the J/psi, and section 7's `a_scale` numbers (0.28 / 0.47) must not
be used.**

### 9.1 What the exports already carry, and the gates

`Jpsi_covrefmom` (the 21-float upper triangle of `Sigma` in (plus, minus) x
(q/p, lambda, phi)), `Jpsi_jacrefmom` (`dm/du` in the same order) and
`Jpsi_qoprefplus/minus` are in BOTH productions.  Nothing else was needed --
**no new maker export**.  `ks_cov_extract.py` re-reads them with `ks_pairs.py`'s
own truth join, so its rows are position-identical to `runs/kspairs_all.npz`
(asserted on run/lumi/event and `sigma`); `ares_jpsi_extract.py` does the same
for a J/psi production, whose gen matching is in the file.

| gate | K_S | J/psi |
|---|---|---|
| analytic `dm/du` (pion / muon hypothesis) vs `Jpsi_jacrefmom` | 2.1e-7 | 7.3e-8 |
| `sqrt(J Sigma J^T)/Jpsi_sigmamass`, q01 - q99 | 1.000000 | 1.000000 |
| `f_ang` rebuilt from the 6x6 vs `Jpsi_fang`, max diff | 2.0e-7 | 5.5e-7 |
| mass from the two-leg state vs `Jpsi_mass`, median rel. | 1.1e-7 | 8.2e-8 |

### 9.2 Why `(1 - f_ang)` is wrong: `J` itself depends on the angles

```
d(sigma_m^2)/du_i = 2 (dJ/du_i)^T Sigma J  +  J^T (dSigma/du_i) J
                    \------ KINEMATIC -----/  \----- RESOLUTION -----/
```

The first term is EXACT -- `dJ/du` is the mass Hessian, `ares_kin.jac_hess` --
and it is non-zero in the ANGULAR components whatever the detector does,
because `sigma_m^2 = J^T Sigma J` and `J` is a function of the opening angle.
Form (ii) assumed that an angular fluctuation moves `m` but not `sigma_m`;
that is false at the leading, model-free order.

The kinematic term alone is not the answer either: freezing `Sigma` freezes
`sigma_(q/p)`, i.e. `sigma_p/p ~ p`, which is the PURE HIT limit `f_hit = 1`.
Closed form for a symmetric ultra-relativistic two-body pair: `A = 2` exactly.
Measured on the J/psi sample: `A_kin` median **1.9813**.  That identity is the
check that the Hessian machinery is right.

The resolution term is modelled as `Sigma_ab = s_a s_b rho_ab` with `rho`
locally constant, which gives `R_i = sum_a (d ln Sigma_aa/du_i) w_a` with
`w_a = J_a (Sigma J)_a` the component's share of the mass variance.  Two
sources for the six log-derivatives:

* **path length** (`msmodel`): material `~1/cos lambda` and `sigma_p/p` flat,
  so `d ln Sigma_aa/d lambda = tan lambda` and `d ln Sigma_aa/d ln p = -2`.
* **population** (`ares_grad.local_linear`): a local linear fit of
  `ln Sigma_aa` on `(ln p_l, lambda_l)` per leg, detector controls partialled
  out, the `lambda` slope fitted to `c tan(lambda)` through the origin so the
  detector's z-symmetry is enforced.

| `d ln Sigma_aa/...` | `d ln p` (K_S) | `/tan(lam)` (K_S) | `d ln p` (J/psi) | `/tan(lam)` (J/psi) |
|---|---|---|---|---|
| `Sigma_(q/p)` | -1.938 | 1.211 | -1.791 | 1.008 |
| `Sigma_lambda` | -1.634 | 0.424 | -1.202 | -0.325 |
| `Sigma_phi` | -1.587 | 2.416 | -1.427 | 2.098 |

The J/psi `q/p` row is the path-length model to 10 % on both entries.
**Azimuthal symmetry is confirmed**: a `cos(phi)`/`sin(phi)` term in the same
fit has |coefficient| q95 <= 0.11 against derivatives of order 1-2, so
`d ln Sigma/d phi = 0` and the `phi` components of `G` are purely kinematic.

### 9.3 Measuring `A` from truth -- and the cut that must not be used

`a (m - mu)/sigma = A (m - m_gen)/m_gen`, so the regressor is the RELATIVE MASS
RESIDUAL and carries no reported width: there is no `sigma`-on-`sigma`
correlation to worry about.  `ln sigma_bar` is removed by Frisch-Waugh-Lovell
inside 8x8 cells of the TRUE `(ln p_plus, ln p_minus)` with 29 truth and
detector controls (true momenta, true lambdas, decay vertex, valid and pixel
hit counts and their products).  Pooling the within-cell residuals is
algebraically one joint fit with per-cell controls and a single shared slope.

**The tail control must be an ABSOLUTE residual window, never a pull.**  A cut
`|m - m_gen| < k sigma` uses the candidate's own width as the threshold, and
that width is the signal: a candidate whose mass fluctuated up has a larger
`sigma`, a looser threshold, and survives where its mirror image does not.
Estimator closure on a toy built from this sample (`ares_angles.py --toy`,
`sigma_bar` = the real width times a log-normal pattern factor 0.35,
`A_true = 1.05`, 20 replicas):

| tail control | measured `A` | bias |
|---|---|---|
| none | +1.0200 +- 0.0365 | -0.030 |
| `\|m - m_gen\| < 60 MeV` (the card's own window, used here) | +1.0676 +- 0.0384 | **+0.018** |
| `\|m - m_gen\| < 3 sigma_bar` (truth width) | +1.0466 +- 0.0359 | -0.003 |
| `\|m - m_gen\| < 3 sigma` (PULL cut) | +1.1958 +- 0.0483 | **+0.146** |

so the estimator is unbiased to +-0.04 as used, and a pull cut inflates `A` by
14 %.  `fullscale/measure_a.py` uses exactly that pull cut (`--zmax` on `z`)
and bins in cells of `sigma/m` and `asym`, both built from the reported widths;
its `--zmax` help now carries the warning.  Its published Z-leg number (1.2110
against `1 + vgf` = 1.2625) is biased in the HIGH direction by both, so the
true Z deficit is larger than the 4 % quoted in `MASSCFTERM_SPEC` section 2 --
by how much is not measured here, the toy above is K_S kinematics.

### 9.4 Result

`A_pred` is each form's population slope formed with the SAME weights and the
SAME realised fluctuations as the measurement, so the columns are directly
comparable.

| | K_S -> pi pi (52 763 cand.) | J/psi -> mu mu (1 973 266 cand.) |
|---|---|---|
| `sigma_m/m` median | 0.01211 | 0.01087 |
| `f_hit` median | 0.0469 | 0.0856 |
| `f_ang` median | **0.7047** | **0.0630** |
| **`A` MEASURED** | **+1.0712 +- 0.1196** | **+1.1026 +- 0.0075** |

| form | `A_pred` (K_S) | pull | `A_pred` (J/psi) | pull |
|---|---|---|---|---|
| (i) `1 + f_hit` -- **as shipped** | 1.1056 | **-0.3** | 1.1064 | **-0.5** |
| (ii) `(1-f_ang)[(1+f_hit)(1-f_ang)+f_ang]` | 0.2582 | **+6.8** | 1.0268 | **+10.1** |
| (ii') `(1-f_ang)(1+f_hit)` | 0.2701 | +6.7 | 1.0325 | +9.3 |
| (iii) full, momentum sector of `G` only | 0.3047 | +6.4 | 0.8994 | +27.0 |
| (iii) full, path-length `Sigma` | 1.2475 | -1.5 | 0.9520 | **+20.1** |
| (iii) full, path-length `Sigma`, `dSigma/dlambda = 0` | 1.2516 | -1.5 | 0.9513 | +20.1 |
| (iii) full, population `Sigma` | 1.2841 | -1.8 | 1.0775 | +3.4 |
| kinematic only (`Sigma` frozen, = `f_hit` = 1) | 1.6586 | -4.9 | 1.9850 | -117 |

In bins of TRUE kinematics the same ordering holds.  chi2 of the measurement
against each form, K_S:

| bins (chi2 / n) | (i) `mom` | (ii) `ang` | (iii) path-length | (iii) population |
|---|---|---|---|---|
| K_S, `f_ang` projected on truth, 7 | **9.1** | 71.1 | 10.3 | 11.4 |
| K_S, softer pion's true \|p\|, 6 | **7.3** | 65.7 | 13.7 | 14.0 |
| K_S, max true \|lambda\|, 5 | **11.3** | 68.3 | 13.7 | 14.9 |
| J/psi, `f_ang` projected on truth, 7 | **12.4** | 122.0 | 467.3 | 25.1 |
| J/psi, softer muon's true \|p\|, 6 | **9.8** | 117.1 | 432.8 | 19.4 |
| J/psi, max true \|lambda\|, 5 | 24.7 | 113.7 | 469.9 | **19.1** |

Over a factor 5 in the softer muon's true momentum the J/psi `A_meas` runs
1.045 - 1.129 against (i) 1.089 - 1.133, (ii) 1.00 - 1.06 and path-length (iii)
0.93 - 0.96 -- (i) tracks it, (ii) and (iii) do not.

**Binning on `f_ang` is a trap** and is kept in the output as the
demonstration.  `f_ang` is built from the FITTED state and the FITTED
covariance, so it moves with the fluctuation being measured.  Binned on the
reconstructed `f_ang` the K_S septiles read 1.80, 1.99, 2.04, 1.98, 2.07, 1.88,
0.93, far above the inclusive 1.07 on the same candidates, and every form is
rejected (chi2 299 - 1447 / 7).  Binned on `f_ang` PROJECTED ON THE TRUE
KINEMATICS (the same variable with the fluctuation regressed out; correlation
0.48) the septiles are 1.35, 1.33, 1.04, 0.55, 0.72, 1.37, 1.16, flat within
their errors and consistent with the inclusive.  The structure was entirely the binning.

### 9.5 What it costs on the momentum scale

Same candidates, same window, same Jensen term, same minimiser; only `a_res`
changes.  `make_card.py` gained `--a-res-key`, which takes `a_res` per
candidate from a column of the pairs cache (default unchanged;
`--a-res-key ares_mom` reproduces the shipped card's `alpha` to 1e-15).
`eps = alpha/0.6300`, every fit EDM < 6e-16.  The rows for the shipped form
and the measured slope are on the fixed code; the other forms were fit on the
pre-fix refit of the same candidates, where the shipped form gave +0.4130
(the correlation-sign fix moves each row by ~1e-6, section 11, far below the
form differences the table is about).

| `a_res` form | median `a_res` | `alpha` [1e-3] | `eps` [1e-3] | `eps - eps(i)` |
|---|---|---|---|---|
| (ii) `ang` | 0.00349 | +0.1506 | +0.2391 | **-0.1740** |
| (ii') `ang_simple` | 0.00367 | +0.1526 | +0.2422 | -0.1708 |
| (iii) momentum sector of `G` only | 0.00371 | +0.1583 | +0.2513 | -0.1617 |
| **(i) `mom`, as shipped (fixed code)** | 0.01290 | +0.2609 | **+0.4142** | 0 |
| measured slope, `A` = 1.0712 (fixed code) | 0.01294 | +0.2584 | **+0.4102** | -0.0040 |
| (iii) path-length `Sigma` | 0.01387 | +0.2869 | +0.4554 | +0.0423 |
| (iii) path-length, `dSigma/dlambda = 0` | 0.01392 | +0.2871 | +0.4558 | +0.0428 |
| (iii) population `Sigma` | 0.01429 | +0.2930 | +0.4650 | +0.0520 |
| kinematic only | 0.01963 | +0.3629 | +0.5761 | +0.1630 |

statistical error on `eps`: 0.0673e-3.

Two sizes have to be kept apart.

**The choice of FORM is worth 0.216e-3 on `eps`** -- `(iii) - (ii)`, 3.2x the
statistical error and 20x the 1e-5 target.  It is the largest single decision
in this closure, and the truth measurement settles it (9.4).

**The angular dependence OF THE COVARIANCE is worth 4e-7.**  That is the
`1/cos lambda` path-length term, isolated as `full_ms - full_ms_nolam` =
-0.0004e-3 on `eps`, 25x below the target and 0.3 % of the slope.  The angular
SECTOR of `G` carries 67 % of `A` at the K_S, but essentially all of it is the
exact mass Hessian, not `dSigma/dlambda`.

At the J/psi the covariance's angular term is 0.07 % of `a_res`; scaled by the
correction's own size there (`a_res` moves `alpha` by +0.146e-3 on the gun and
+0.127e-3 on v3) that is **1e-7 on the scale**, below the 1e-6 threshold at
which a refit would have been needed.  `(iii)-(ii)` at the J/psi is -7.7 %
(path-length) / +4.6 % (population), i.e. 7e-6 to 1.1e-5, but (ii) is excluded
there at 10 sigma.

### 9.6 Verdict

* **Does `d sigma_m/d(angles)` from the covariance's path-length dependence
  have to be in `a_res`?  NO -- K_S 4e-7, J/psi 1e-7 on `eps`, both far below
  the 1e-5 target.**
* **Form (ii) must not be used.**  Excluded at 6.8 sigma (K_S) and 10 sigma
  (J/psi).  It is wrong because it drops the exact angular dependence of the
  mass Jacobian, not because it mis-models the detector.
* **The shipped closed form `(1 + f_hit) sigma/m` is right.**  At the J/psi it
  is confirmed to **0.34 +- 0.68 %**; at the K_S, where `f_ang` = 0.70, it is
  confirmed to 11 % (statistics-limited) and sits 0.3 sigma from the
  measurement.  It survives because it is itself an empirical statement --
  `f_hit` is the measured hit share of the mass variance -- and absorbs the
  kinematic and detector responses together, whereas form (iii) rebuilds them
  from parts and inherits the parts' errors (its path-length variant is 14 %
  low at the J/psi, its population variant 2.3 % low).
* **No new maker export is required.**  `Jpsi_covrefmom`, `Jpsi_jacrefmom` and
  `Jpsi_qopref*` are already written by both productions and are all the
  population-regression route needs.  Were `d sigma/d lambda` ever wanted at
  first order rather than from the population, the export would be the per-leg
  `d ln Sigma_aa/d lambda` evaluated inside the fit at fixed hit set -- but
  nothing here motivates it.

### 9.7 Consequence for section 8

The `a_res` band of section 8 (`+0.14/-0.13e-3`, taken as the full range
between `a_res` off and the J/psi closed form, with form (ii) as the central
value) is superseded.  The coefficient is now MEASURED on this sample, and its
uncertainty is the measurement's: 11.2 % statistical plus 3.8 % estimator
closure on `A`, against a total `a_res` correction of
`0.4142 - 0.1480 = 0.266e-3` on `eps`, i.e. **+-0.031e-3**.

```
eps = +0.410 +- 0.068 (stat) +- 0.031 (a_res) +- 0.04 (other)  x 1e-3
```

(the central value is the measured-slope fit; the shipped closed form gives
+0.414, the two first-principles variants +0.455 and +0.465).  The a_res model
is no longer the limiting systematic -- statistics and the unmodelled
nuclear-elastic tail are.

### 9.8 Files

| | |
|---|---|
| `ks_cov_extract.py` | the 6x6 `Sigma`, `dm/du` and the TRUE leg directions, row-aligned to `kspairs_all.npz` |
| `ares_jpsi_extract.py` | the same for a J/psi production (gen matching in the file) |
| `ares_kin.py` | the two-body mass, its gradient and its Hessian in the reference parameters |
| `ares_grad.py` | `grad(sigma_m)`: the exact kinematic term + the two `dSigma/du` models |
| `ares_truth.py` | the Frisch-Waugh-Lovell slope measurement and its bootstrap |
| `ares_angles.py` | the study: gates, the forms, the measurement, the bins, the toy closure, the figures, the `a_res` columns |
| `run_ares_forms.sh` | one closure fit per form |
| `ares_eps_table.py` | the `eps` table above |

Caches: `runs/fixed/kscov_all.npz`, `runs/fixed/kspairs_ares.npz` (the pairs
cache plus one `ares_<form>` column per form), `runs/jpsicov_ideal.npz`,
`runs/jpsimin_ideal.npz`.  Figures:
`~/public_html/ZMass/cvh/260918_ares_angles_fixed/` (K_S) and
`~/public_html/ZMass/cvh/260918_ares_angles/jpsi/` (J/psi).

## 10. What to do next

1. DONE, section 9: the per-candidate `a_res` form.  The coefficient is
   measured from truth, the closed form is confirmed, and the systematic is
   +-0.03e-3.
2. DONE, section 12: the nuclear-elastic family on the mass CF.  It closes at
   the track level on the real tracker; on the mass it has to be CONDITIONED
   on the reconstruction before it can go into the maker (section 12.7).
3. **The radius dependence** is not the tail (section 12.6): it survives the
   family (chi2 10.6/3).  It deserves the full sample split more finely, and
   against the ideal-geometry J/psi, before it is called an effect.
4. **The ideal-geometry J/psi closure** (`fullscale/runs/jpairs_ideal_n600.npz`)
   is the like-for-like reference for the K_S number and has not been fit.

## 11. Before the correlation-sign fix

The same refit (driver, flags, inputs, chunking) on the pre-fix code is
`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/`, its
fits in `runs/`.  Refitting its pairs cache with the current analysis chain
reproduces its published `alpha = +0.260199 +- 0.042427` to every digit
(`runs/results/ctl_all.json`), so the chain is common and every difference
below is the CVH code.

### Candidate by candidate (52 824 truth-matched candidates in both)

| | value |
|---|---|
| median \|dm/m\| | 2.6e-5 |
| p95 \|dm/m\| | 3.4e-4 |
| candidates moving by more than 1e-4 | 21.0 % |
| `d sigma_m / sigma_m`, mean | +2.43e-3 +- 0.02e-3 |
| `d sigma_m / sigma_m`, median | +1.22e-3 |

On all candidates of one input chunk (108, including combinatorial ones) the
same comparison gives median 4.9e-5, p95 6.9e-4, 38.9 % above 1e-4 and
`d sigma_m/sigma_m` +3.75e-3 -- the `DEFECTS_STATE.md` (d) numbers.  The
truth-matched candidates move less than the combinatorial ones.  The fixed
covariance is WIDER, as the sign fix requires (the non-bending offset variance
was low).

### Production

| | pre-fix | fixed |
|---|---|---|
| candidates fit | 347 561 | 347 583 |
| fit failures | 2 205 (0.630 %) | 2 183 (0.624 %) |
| truth-matched | 52 830 | 52 829 |
| core-hours | 141.9 | 90.6 |

The reference-EDM fix is the saving: the reference loop was capped at 10
iterations on every candidate and now stops at convergence.

### Closure (`eps` [1e-3], nominal correction set unless stated)

| sample | pre-fix | fixed | fixed - pre-fix |
|---|---|---|---|
| naive | +0.2869 +- 0.0673 | +0.2878 +- 0.0675 | +0.0009 |
| Jensen only | +0.1463 +- 0.0673 | +0.1480 +- 0.0674 | +0.0017 |
| a_res, no Jensen | +0.5530 +- 0.0673 | +0.5533 +- 0.0675 | +0.0003 |
| **all (a_res closed form + Jensen)** | **+0.4130 +- 0.0673** | **+0.4142 +- 0.0675** | **+0.0012** |
| all, a_res measured from truth | +0.4191 +- 0.0674 | +0.4102 +- 0.0675 | -0.0088 |
| from a B hadron | +0.5569 +- 0.0876 | +0.5570 +- 0.0878 | +0.0001 |
| from a B0 | +0.4870 +- 0.0941 | +0.4872 +- 0.0943 | +0.0002 |
| prompt | +0.1950 +- 0.1052 | +0.1978 +- 0.1054 | +0.0028 |
| r < 2 cm | +0.4804 +- 0.1354 | +0.4826 +- 0.1356 | +0.0021 |
| 2 - 4 cm | +0.0788 +- 0.1343 | +0.0787 +- 0.1345 | -0.0001 |
| 4 - 10 cm | +0.6774 +- 0.1264 | +0.6801 +- 0.1266 | +0.0026 |
| > 10 cm | +0.3806 +- 0.1444 | +0.3801 +- 0.1447 | -0.0005 |
| min-leg p < 0.8 GeV | +0.2658 +- 0.1294 | +0.2645 +- 0.1296 | -0.0013 |
| 0.8 - 1.5 GeV | +0.3623 +- 0.1080 | +0.3654 +- 0.1082 | +0.0031 |
| > 1.5 GeV | +0.5936 +- 0.1155 | +0.5946 +- 0.1157 | +0.0010 |
| residual window 3 sigma | +0.3721 +- 0.0678 | +0.3721 +- 0.0680 | +0.0000 |
| residual window 5 sigma | +0.4088 +- 0.0674 | +0.4096 +- 0.0675 | +0.0007 |
| floating background | +0.4203 +- 0.0676 | +0.4215 +- 0.0677 | +0.0012 |
| chi2/ndof < 1.5 | +0.4081 +- 0.0696 | +0.4098 +- 0.0697 | +0.0017 |
| truth match tight | +0.3866 +- 0.0676 | +0.3872 +- 0.0677 | +0.0005 |
| truth match loose | +0.3954 +- 0.0668 | +0.3967 +- 0.0669 | +0.0013 |

Every fit is EDM-certified (largest EDM 3e-16).  The two samples are the same
candidates, so the differences are far more precise than the errors shown;
all but one are below 0.003e-3.  The exception, the measured-slope row, is not
the refit: it is the truth measurement of `A` (1.0618 -> 1.0712, both +-0.12)
redrawn on slightly different fluctuations, a 0.08-sigma move of `A` that
reaches `eps` through the 0.266e-3 a_res lever.

| diagnostic | pre-fix | fixed |
|---|---|---|
| pull std / robust width | 1.2265 / 0.9566 | 1.2232 / 0.9541 |
| pull mean / median | +0.0216 / +0.0379 | +0.0218 / +0.0382 |
| `\|z\| >= 3` | 1.635 % | 1.615 % |
| chi2/ndof median, > 3 | 0.861, 0.75 % | 0.862, 0.75 % |
| median residual, all [MeV] | +0.215 | +0.218 |
| median residual, r < 1 / 1-2 / 2-3 / 3-4 cm | +0.576 / +0.077 / +0.032 / +0.189 | +0.576 / +0.075 / +0.033 / +0.192 |
| median residual, r 4-6 / 6-10 / 10-20 / 20-60 cm | +0.251 / +0.273 / +0.225 / +0.301 | +0.251 / +0.286 / +0.227 / +0.307 |
| radius-bin chi2 / 3 (p) | 10.87 (0.0125) | 10.94 (0.0120) |
| momentum-bin chi2 / 2, high - low | 3.96, +0.328 +- 0.173 | 3.97, +0.330 +- 0.174 |
| B-daughter - prompt | +0.362 +- 0.137 | +0.359 +- 0.137 |

**Interpretation.**  The correlation-sign fix does not change the K_S closure.
The scale moves by +0.0012e-3, 1.8 % of the statistical error; the radius
structure (the 0.58 MeV residual inside the beam pipe, the dip at 1-3 cm, the
plateau beyond) and its p = 0.012 are unchanged, so they are not a product of
the too-small non-bending offset variance; the tail beyond 3 sigma is
unchanged, so it is not either -- it remains the missing nuclear-elastic
family.  What the fix does change is the width: sigma_m grows by 0.24 % on
average and the pull std falls by 0.3 %, which is the corrected covariance
doing what it should, and it leaves the scale alone because the mass-CF
likelihood's two corrections are built from that same covariance.

Figures: `~/public_html/ZMass/cvh/260918_ksclosure_fixed/` -- `ksfix_shift`
(dm/m and d sigma_m/sigma_m per candidate), `ksfix_pull` and `ksfix_pull_tails`
(fixed/pre-fix ratio panel), `ksfix_resid_vs_radius` (difference panel),
`ksfix_eps_bins`, and the closure figure set of section 8 (`ks_*`) for
the fixed code.

## 12. The nuclear-elastic family on the K_S mass

The in-maker mass CF has hit / MS / ionisation / radiative families and no
`hadElastic`, while a K_S daughter crosses ~15 g/cm^2 of tracker and takes on
average 0.10 elastic collisions per leg of 0.03-0.5 rad.  This section adds the
offline channel of `Documents/Resolution/NUCLEAR_ELASTIC.md` to the K_S mass
term on the real geometry and asks whether it explains the 1.6 % beyond 3
sigma and the radius dependence of section 8.  Code in `nucel/`, caches and fits
in `runs/nucel/`, figures in `~/public_html/ZMass/cvh/260924_ks_nucel/`.

### 12.1 The channel on the mass functional

`nucel/ks_nucel_cf.py`, per candidate, from the raw step records:

    S_ang(tau) = sum_b sum_{s in b} N_s ( g_s(w_b tau) - 1 )
    S_rec(tau) = sum_s N_s ( h_s(wq_s tau) - 1 )

* the blocks b and weights `w_b = sqrt(v_b / sum_s thp2_s)/sigma_m` are the
  maker's own parmtype-10 MS blocks and weights, so a collision rides on the
  influence the fit gives a Moliere kick at the same place.  Checked: the
  Moliere exponent rebuilt from the same records and weights reproduces
  `cfmass_ms` to **9.5e-7** on 2500 candidates (the float32 floor 1.4e-6);
* `N_s = mu(species, Z_s, A_s, p_s) xg_s` per step -- the (Z, A) of every
  record rounded as `cf_nucel_exact.leg_rates` rounds it, the step's own
  momentum, pi+ on the positive leg and pi- on the negative one (the records
  are drained leg by leg; a leg is a segment between momentum jumps, given to
  the leg whose fitted momentum is nearest);
* `g, h, mu` from `nucel/nucel_tables.py`: `nucel_g4driver` (the
  species-correct model `G4ElasticHadrNucleusHE` and dataset
  `G4BGGPionElasticXS`) run on every (species, Z, A, p) bucket, 69 (Z, A) x
  pi+- x 74 momenta 0.18-48 GeV, 5e5 draws each; nearest momentum bucket with
  the angle rescaled by p_bucket/p_step, rate log-log interpolated;
* the recoil sub-channel (`NUCEL_RECOIL`) takes the maker's IONISATION weight
  of the same leg at the same momentum (interpolated in p along the leg), so
  the leg-mean-weight approximation of `cf_nucel_exact` is not needed here; the
  theta-dE independence approximation is kept;
* not centred, gated on `NUCEL_CHANNEL` exactly as the offline channel;
* cache columns `Snuc_ang`, `Snuc_rec_re/_im`, `Snuc_first` (below),
  `nuc_N/Np/Nm/Nfirst/Nbig/Nbigfirst`, `nuc_v`, `nuc_vrec`, `nuc_mrec`;
  `nucel/make_variants.py` turns them into the card families (`Snuc_re/_im`),
  and `fullscale/make_card.py --add-nucel-family` adds family `nucel` with
  `k_nucel` held at 1 (a fifth tabulated family needs no rabbit change).

Kernel validation (the watcher run of the note was not repeated: the per-plane
test of 12.4 is the sim-vs-model check of the same kernels on the real
geometry): pi- on O16 at T = 3000 MeV gives <theta> = 35.20 mrad and
mu = 4.14e-3 cm^2/g, i.e. N = 0.0516 over the toy's 12.4544 g/cm^2, against
the note's 35.2 mrad and 0.0517.  pi+ and pi- agree above ~1 GeV; at 0.18 GeV
the pi+ rate is 13 % lower.  On carbon mu(pi-) runs 1.10e-2 (0.18 GeV) ->
4.1e-3 (0.83 GeV) -> 3.1e-3 cm^2/g (48 GeV) and <theta> 466 -> 118 -> 2.5 mrad.

Per candidate: **N = 0.204** collisions (median 0.187); 0.084 of them (41 %)
have a median kick alone worth more than 3 sigma_m; **0.0149** fall in each
leg's FIRST MS block (the material from the decay vertex up to and including
the first measured module), 0.0141 of them > 3 sigma_m.

### 12.2 The production with step records

`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260924_steprec/`
(PROVENANCE.txt): the 260918 driver, flags, inputs and CMSSW commit, plus
`exportStepRecords=True` and an event list (`nucel/make_evlists.py`: the
52 569 events that carry a truth-matched candidate; every candidate of a
selected event is refit).  `prod/submit_ks_steprec.sh` ->
`prod/array_ks_steprec.sbatch`, 1000 half tasks, all complete, 9.5 GB,
7.82 M propagations with 301 failures.

* smoke (169 candidates, `nucel/smoke_check.py`): `cfmass_*`, `Jpsi_mass`,
  `Jpsi_sigmamass`, `chisqval`, `resinfvarv`, `Jpsi_covrefmom` and 16 more
  branches **bit-identical** to the parent; records non-empty, strides
  msmoliv 10 / ioniurbanv 14 / radstepv 12, ~800 MS records per candidate.
* the pairs cache has the parent's 52 829 rows.

### 12.3 The gates (`nucel/gates.py`, `runs/nucel/gates.log`)

| gate | result |
|---|---|
| (i) family OFF == in-maker composition | all 54 columns x 52 829 rows of the parent cache BIT-IDENTICAL; the family-free fit reproduces the parent's alpha, error and EDM to every digit (+0.260945 +- 0.042498, EDM 4.72e-19) |
| (ii) vgf + sum_b v_b(MS, ioni)/sigma_m^2 = 1 | max deviation **6.3e-7** on 4498 candidates, also against `Jpsi_massvms/vioni` |
| (ii) the family's own second cumulant | > 0 on 52 829/52 829; median **3.4 sigma_m^2** (q10/q90 2.3/4.6) |
| (iii) normalisation / positivity | S_total(0) = 0 exactly; the inverted density integrates to 0.99996 (median; min 0.99983) over \|z\| < 95, to 0.9992 over \|z\| < 40 (the family's mass lives far out); min density / max density -6e-6 (inversion ringing) |

sigma_m stays the fit's Gaussian width: the family is a non-Gaussian EXPONENT,
not a share of it, so the closure (ii) is untouched and the CF's second
cumulant becomes 1 + v_nuc (+ 9e-6 from the recoil).  With v_nuc ~ 3.4 on
99.4 % of the candidates, the second cumulant is not a width here: the channel
is a rare, far tail.

### 12.4 The track level on the real tracker (clean propagation)

`nucel/perplane_pulls.py` + `plot_perplane.py`: one pi- shot 2e5 times through
the real tracker (eta 0.3; pT 0.7 and 1.5 new, `cleanprop/run_campaign.sh` with
`nucel/points_pi_lowpt.txt`, the pT 0.7 model propagated with the 0.05 GeV
momentum floor of `nucel/runCleanPropModelLowP.py`; pT 3 and 10 the existing
260806 samples), per-plane residual in local x (bending) and local y
(non-bending) with H-basis a-vectors, `perplane` acceptance, inelastic veto 0.2;
the model's P(|z| > c) by exact inversion of its CF.  Mean over planes,
split by the plane's acceptance (fraction of rays still crossing it):

| pT | proj. | planes | acceptance | P(\|z\|>5) data | model ON | model OFF |
|---|---|---|---|---|---|---|
| 0.7 | x | 5, r 4.6-27 cm | >= 0.977 | 0.668 % | 0.772 % | 0.229 % |
| 0.7 | x | 10, r 35-109 cm | down to 0.565 | 0.482 % | 1.769 % | 0.192 % |
| 1.5 | x | 4, r 4.2-10 cm | >= 0.974 | 0.540 % | 0.645 % | 0.252 % |
| 1.5 | x | 18, r 27-110 cm | down to 0.393 | 1.071 % | 1.824 % | 0.189 % |
| 3 | x | 6, r 4.2-27 cm | >= 0.970 | 0.720 % | 0.727 % | 0.250 % |
| 3 | y | 6 | >= 0.970 | 0.744 % | 0.744 % | 0.266 % |
| 3 | x | 13, r 35-107 cm | down to 0.875 | 1.595 % | 1.964 % | 0.218 % |
| 10 | x | 7, r 4.6-28 cm | >= 0.977 | 0.625 % | 0.665 % | 0.255 % |
| 10 | x | 13, r 36-110 cm | down to 0.912 | 1.383 % | 1.451 % | 0.220 % |

(full table, both projections and |z| > 3: `runs/nucel/perplane_table.txt`.)

**Where every ray is still observed the channel closes the per-plane tail on
the real tracker** -- to 0-6 % at pT 3 and 10 in both projections and 14-19 %
high at pT 0.7-1.5, against a factor 2.5-3 too low without it.  Where rays are
lost the model over-predicts, and the more rays are lost the more it
over-predicts (pT 10, acceptance >= 0.91: 5 %; pT 0.7, acceptance down to 0.57:
3.7x): a ray is lost when a kick throws it off the next module, so the loss
removes exactly the tail the channel adds.

### 12.5 The mass term (same candidates, same card recipe, `rabbit_fit.py`)

Every fit EDM-certified (largest 3.6e-18).  eps = alpha_mass / <f>, <f> = 0.630.
"first block" is the family restricted to each leg's first MS block (12.1).

| fit | without | **with** (recoil on) | recoil off | first block only |
|---|---|---|---|---|
| alpha_mass, nominal [1e-3] | +0.2609 +- 0.0425 | +0.2752 +- 0.0457 | +0.2686 +- 0.0457 | +0.2649 +- 0.0427 |
| eps, nominal [1e-3] | +0.414 +- 0.068 | +0.437 +- 0.073 | +0.426 +- 0.073 | +0.420 +- 0.068 |
| eps, a_res measured from truth | +0.410 | +0.433 | | |
| eps, naive (no a_res, no Jensen) | +0.288 | +0.290 | | |
| r < 2 / 2-4 / 4-10 / > 10 cm | +0.483 / +0.079 / +0.680 / +0.380 | +0.526 / +0.149 / +0.646 / +0.397 | | +0.484 / +0.090 / +0.683 / +0.391 |
| radius-bin chi2/3 (alpha) | 11.1 (p 0.011) | 6.8 (p 0.077) | | 10.6 (p 0.014) |
| p_min < 0.8 / 0.8-1.5 / > 1.5 GeV | +0.265 / +0.365 / +0.595 | +0.285 / +0.398 / +0.608 | | +0.264 / +0.373 / +0.605 |
| from B / prompt | +0.557 / +0.198 | +0.565 / +0.235 | | +0.561 / +0.207 |
| P(\|z\| >= 3), data 1.615 % | model 0.802 % | model **5.858 %** | | model **1.732 %** |
| P(\|z\| >= 5), data 0.572 % | model 0.126 % | model 2.701 % | | model 0.812 % |
| PIT pull std / robust | 1.050 / 1.017 | 0.856 / 0.886 | | **1.002 / 1.001** |
| P(\|z_PIT\| >= 3) (Gaussian 0.27 %) | 0.895 % | 0.009 % | | **0.240 %** |
| binned chi2/ndof, \|z\| < 10 | 227.5/39 | 1995.7/79 | | **93.3/63** |

(`nucel/ks_nucel_eval.py`: the PIT pull Phi^-1(F_i(z_i)) is N(0, 1) exactly
when the per-candidate density is right; F_i by Gil-Pelaez inversion of the
upsampled CF, at each variant's fitted alpha; the a_res/Jensen corrections,
~0.01 sigma, are not applied there.)

### 12.6 What it answers

* **The tail is the nuclear-elastic channel -- the part the reconstruction
  lets through.**  The unconditioned channel predicts 3.6x the observed tail
  (5.9 % against 1.6 % beyond 3 sigma), uniformly in decay radius, momentum and
  origin.  The same channel restricted to the kicks nothing upstream of the
  mass can see -- the vertex-to-first-module block, 7 % of the collisions --
  reproduces the tail with no free parameter: 1.73 % against 1.62 %, the PIT
  pull N(0, 1) to 0.2 % in width, 0.240 % beyond 3 (Gaussian 0.27 %), chi2/ndof
  93/63 against 228/39 without any family.  Per radius bin the first-block
  model's PIT tail is 0.08-0.37 % everywhere.  The first block is a
  DIAGNOSTIC boundary, not a derived acceptance: its success and the per-plane
  acceptance pattern of 12.4 say the missing piece is the conditioning of the
  channel on the track keeping its hits, which a kick inside the tracker
  typically breaks (in the real sample: pattern recognition, the V0 vertex and
  mass window, the refit chi2 and the minLegHits requirement).
* **The tail is symmetric** (0.78 % high / 0.83 % low); section 8's
  "asymmetric, high-mass side" is corrected.
* **The radius dependence survives.**  With the first-block family the radius
  bins scatter by chi2 10.6/3 (11.1 without); the drop to 6.8 with the
  unconditioned family comes from its inflated errors and wrong tail, not from
  a better description.  The model-free structure of section 8 is a MEDIAN
  residual (+0.58 MeV below 1 cm), which a symmetric tail does not move.
* **eps moves by a tail-sized amount**: +0.006e-3 with the first-block family,
  +0.023e-3 (+0.011 of it the recoil's mean loss) with the unconditioned one --
  0.1 and 0.3 of the statistical error.  The recoil's mean loss is real
  physics (not centred by the reference) and small: -6e-4 sigma_m median.

### 12.7 Consequence for a maker port

The family is ready as physics -- Geant4's rates and kernels, the maker's
weights, closed per plane on the real tracker -- but not as an unconditioned
exponent: on reconstructed candidates it must be conditioned on the kick
leaving the track's hit pattern intact.  The per-step quantity that decides
that is the kick's effect on the DOWNSTREAM hits (the fit's own hit residual
response), not on the mass; a first-principles version is a two-functional
(mass, hit-compatibility) kernel per step, truncated where the pattern
recognition would drop the hits.  Until then the first-block restriction is
the measured boundary.

### 12.8 Files

| | |
|---|---|
| `nucel/nucel_tables.py` | (Z, A) scan of the records and the Geant4 kernel/rate table |
| `nucel/ks_nucel_cf.py` | the family on the mass functional, from the records; writes the pairs cache |
| `nucel/make_variants.py` | base / nuc / nucnr card caches (+ the measured a_res column) |
| `nucel/build_ks_nucel.sh`, `run_nucel_fits.sh`, `run_first_fits.sh` | cards and fits |
| `nucel/gates.py` | gates (i)-(iii) |
| `nucel/ks_nucel_eval.py` | tails, PIT pull, chi2, figures |
| `nucel/perplane_pulls.py`, `plot_perplane.py` | the real-tracker per-plane test |
| `nucel/plot_eps_bins.py` | eps per bin, three variants |
| `nucel/make_evlists.py`, `smoke_check.py` | the production's event selection and its bit-identity check |
| `nucel/runCleanPropModelLowP.py`, `points_pi_lowpt.txt` | the low-pT clean-propagation points |

Caches `runs/nucel/kspairs_{nucel,base,nuc,nucnr,nucfirst}_*.npz`, table
`runs/nucel/nucel_table.npz` (per-bucket files on ceph under the production's
`nucel_kernels/`), results `runs/nucel/results/`, `runs/nucel/eval_all.json`.
Figures: `ks_nucel_pull[_log]`, `ks_nucel_pit`, `ks_nucel_tail_vs_radius`,
`ks_nucel_eps_bins`, `perplane_tail{3,5}_{locx,locy}` (+ `perplane_table.txt`).

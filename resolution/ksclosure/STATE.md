# K_S -> pi+ pi- displaced-vertex momentum-scale closure

CVH two-track refit of K_S -> pi+ pi- from the inclusive B -> J/psi + X MC,
vertex constraint ON, beam-spot constraint OFF, pion mass hypothesis on both
legs; the momentum scale `alpha` is measured on the K_S mass with the same
unbinned mass-CF likelihood and the same two first-principles corrections
(sigma-artefact, Jensen) as the J/psi closure.

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
`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/PROVENANCE.txt`.

The one thing to know: the maker's own gen matching is muon-specific
(`|pdgId| == 13`, `ResidualGlobalCorrectionMakerTwoTrackG4e.cc:6116`), so it
is run with `doGen = False, requireGen = False` -- with `requireGen = True` a
pion channel would be silently emptied -- and the truth match is done OFFLINE
by (run, lumi, event) + decay vertex + both daughter momenta, against
`ks_truth_dump.py`.

## 4. Files

| | |
|---|---|
| `runCvhKs.py` | the cmsRun driver |
| `ks_truth_dump.py` | Geant4 K_S -> pipi truth, one row per decay, keyed by (run,lumi,event) |
| `ks_pairs.py` | the mass-CF pairs cache (cf_inmaker layout) + the truth join |
| `ks_yield.py` | yield / purity / reconstructable-ceiling study |
| `gen_decays.py` | decayer + FSR bookkeeping (K_S, D0, D*, B0, J/psi) |
| `inspect_sample.py`, `dump_b0ks.py` | event-content and candidate-structure dumps |
| `prod/` | slurm arrays for the refit and the truth dump |

Production output:
`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/`
(`task_NNNN/` refit output, `truth/truth_NNNN.npz` truth, `chunks/` input lists).

## 4b. Production (slurm, 500 chunks of 252 files)

`/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/`,
`prod/submit_ks.sh` + `prod/submit_truth.sh`, resubmission with
`prod/resubmit_failed.sh`.  Per chunk (measured on the first completed task,
12 986 events):

```
attempted 716   succeeded 706   failed 10 (1.40 %)
  fail[prop] 10, everything else 0 (no chargeflip, no NaN, no ndof)
skipped before the fit: leghits<8 1219, hits<10 274, ndof<1 114
  -> 2323 candidates in the chunk = 0.179/event, 30.8 % of them fit
propagation: 211 557 calls, 65 failures (0.031 %); pdrain 54, ierr 6,
  fieldbound 3, offsurface 2; 174 backward legs
recovery: 47 leg backtracks, 36 chi2 backtracks, 8 seed inflations, 1 clamp
```

16 min per chunk, 29 MB of output; the whole sample is ~135 core-hours and
~14 GB.  The 1.4 % failure rate is far below the 2.5 % of the 2016 DATA V0
tests -- the B0Ks pairing is a cleaner subset than a V0 skim.

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

## 7. The sigma-artefact prefactor is derived for a momentum-dominated mass

`a_i = (1 + f_hit) sigma_i/m_i` (implemented as `(1 + vgf) sigma/|m|`) comes
from `sigma_m^2 = A m^4 + B m^2 + C`, i.e. from a mass whose resolution is
carried by the two MOMENTA: the hit term gives `sigma_p/p ~ p` hence
`sigma_m ~ m^2`, the MS term `sigma_p/p ~ const` hence `sigma_m ~ m`, and
`d ln sigma_m / d ln m = 1 + f_hit`.

For K_S -> pi pi that premise fails: **`f_ang` has median 0.70**, so 70 % of
`sigma_m^2` comes from the OPENING ANGLE.  An angular fluctuation moves `m`
WITHOUT moving `sigma_m` -- `sigma_ang = (dm/dtheta) sigma_theta ~ (m/theta)
sigma_theta` is invariant under `theta -> theta(1+d)` -- so its contribution to
`d ln sigma_m/d ln m` is zero, not `1 + f_hit`.  Averaging the two fluctuation
sources with their variance shares,

```
a_eff / a_used = (1 - f_ang) [ (1 + vgf)(1 - f_ang) + f_ang ] / (1 + vgf)
```

which is **0.28 (median) / 0.47 (1/sigma^2-weighted)** for the K_S and
**0.89-0.91 at the J/psi** (`f_ang` 0.086 gun / 0.106 data) -- so the published
J/psi correction is itself ~10 % high, comparable to its own error, while the
K_S one is 2-3.5x high.  `make_card.py` gained a scalar `--a-scale` (default 1,
every existing card bit-identical) so the size of this can be measured rather
than assumed.  A per-candidate form is the proper fix and is NOT done here.

## 8. Result (full sample: 500/500 chunks, 52 830 truth-matched candidates)

`alpha_mass` is what `MassCFTerm` fits; `eps = alpha_mass / <f>` with
`<f> = 0.630` is the momentum scale (section 6).  Every fit converges with
EDM < 1e-16, far inside the 1e-3 tolerance.

347 561 candidates were fit, 52 830 of them truth-matched (15.20 %).
Pull `(m_reco - m_gen)/sigma_m`: mean +0.022, median +0.038, std 1.227, robust
width 0.957.  Residual `m_reco - m_gen`: median +0.215 MeV, RMS 8.90 MeV.
chi2/ndof median 0.861 (ndof median 22), 0.75 % above 3.

### The correction ladder

| | alpha_mass [1e-3] | eps [1e-3] |
|---|---|---|
| naive (no corrections) | +0.181 +- 0.042 | +0.287 +- 0.067 |
| Jensen exact only (a_res off) | +0.092 +- 0.042 | +0.146 +- 0.067 |
| **a_res x 0.470 + Jensen exact** | **+0.171 +- 0.042** | **+0.272 +- 0.067** |
| a_res (J/psi form) + Jensen exact | +0.260 +- 0.042 | +0.413 +- 0.067 |
| a_res (J/psi form), no Jensen | +0.348 +- 0.042 | +0.553 +- 0.067 |

**The displaced K_S momentum-scale closure is**

```
eps = +0.27 +- 0.07 (stat) +0.14 / -0.13 (a_res model) +- 0.04 (other)  x 1e-3
```

against the J/psi -> mu mu closure of **+0.006 +- 0.025 x 1e-3** on the same
MC.  The a_res model (section 7) is the limiting systematic, not the
statistics; the band is the full range between a_res off and the J/psi closed
form, with the angular-corrected coefficient as the central value.

### Systematics (nominal = a_res in the J/psi form + Jensen, +0.413)

| variation | eps [1e-3] | shift |
|---|---|---|
| residual window 3 sigma (from 9) | +0.372 | -0.041 |
| residual window 5 sigma | +0.409 | -0.004 |
| floating uniform background | +0.420 | +0.007 |
| chi2/ndof < 1.5 (from 3) | +0.408 | -0.005 |
| truth match tight (0.05/0.05/0.20/1 cm), n 50 564 | +0.387 | -0.026 |
| truth match loose (0.60/0.60/0.90/5 cm), n 56 594 | +0.395 | -0.018 |

Everything except the 3-sigma residual window is a no-op at the 0.03e-3 level.
The 3-sigma window shift, -0.041e-3, IS the unmodelled tail.

### Populations and bins (nominal correction set)

| sample | n | eps [1e-3] |
|---|---|---|
| all | 52830 | +0.413 +- 0.067 |
| from a B hadron | 31956 | +0.557 +- 0.088 |
| from a B0 | 27721 | +0.487 +- 0.094 |
| prompt / fragmentation | 20874 | +0.195 +- 0.105 |
| decay radius < 2 cm | 12392 | +0.480 +- 0.135 |
| 2 - 4 cm | 12992 | +0.079 +- 0.134 |
| 4 - 10 cm | 13508 | +0.677 +- 0.126 |
| > 10 cm | 13938 | +0.381 +- 0.144 |
| min-leg p < 0.8 GeV | 11687 | +0.266 +- 0.129 |
| 0.8 - 1.5 GeV | 17594 | +0.362 +- 0.108 |
| > 1.5 GeV | 23549 | +0.594 +- 0.116 |

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

`|z| >= 3` holds **1.64 %** of candidates against 0.27 % for a Gaussian.  The
log-scale `ks_pull_model_tails` figure shows the per-candidate CF model
describing the core to a few per cent, the data running ~20 % above it at
z = +2 to +3, and ~1 candidate per bin sitting flat out to |z| = 10.  The
in-maker CF has **no nuclear-elastic family** and `S_rad` is identically zero
for pions, so this is where the ~0.05 elastic nuclear collisions per pion
(25-35 mrad each) live.  The excess is ASYMMETRIC, on the high-mass side --
the direction that biases the fitted scale positive -- and it is the only
systematic above that moves the answer.

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
GT; this K_S production runs with `useIdealGeometry=True` (David's 9/17
decision).  The like-for-like comparison is against the ideal-geometry J/psi
production `jpsimc_20M_260917_ideal` launched the same day.

## 9. What to do next

1. **The per-candidate a_res form** (section 7).  It is the limiting
   systematic here (+0.14/-0.13e-3 against 0.07e-3 of statistics) and it also
   makes the published J/psi correction ~10 % high.
2. **The nuclear-elastic CF family for hadrons.**  `cf_nucel_exact.py` exists
   offline and is validated parameter-free on four species; it is not in the
   in-maker CF, and the K_S tail is 6x a Gaussian.
3. **The radius dependence** at p = 0.012 deserves the full sample split more
   finely, and against the ideal-geometry J/psi, before it is called an
   effect.

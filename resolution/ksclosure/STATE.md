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

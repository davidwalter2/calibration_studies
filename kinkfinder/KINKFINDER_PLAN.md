# CVH Kink Finder — Implementation Plan

**Goal:** per-layer likelihood-ratio test for decay-in-flight (K± → μν, π± → μν) inside the CVH
fit, targeting the decay contamination systematic of the B± → J/ψ K± channel in the B-field
calibration. Tests, at every material step, the alternative hypothesis of an unconstrained
kink (Δλ, Δφ) **plus** a momentum step Δ(q/p) — the correlated signature that distinguishes a
decay from a hard elastic scatter.

> **Correction (2026-08-08):** earlier versions of this plan claimed the CMS
> Kalman `trkKink` is blind to the momentum step. **That is wrong.**
> `MuonKinkFinder::getChi2` (RecoMuon/MuonIdentification) compares the forward-
> and backward-predicted states with `mixedFormatVector()` = (q/p, dx/dz, dy/dz,
> x, y) and the full summed covariance (`usePosition=True`, `diagonalOnly=False`
> by default), so q/p IS included. What is genuinely new here is instead:
> (i) the statistic is evaluated inside the CVH fit at every Geant4e material
> step at zero extra cost, reusing the R matrix already formed for the alignment
> gradient/Hessian -- `trkKink` needs its own dedicated KF refit and is filled
> only for muons, not for the hadron tracks this systematic needs;
> (ii) it is a *marginal* score test, whose Hessian accounts for how the rest of
> the track absorbs the kink, rather than a comparison of two independent
> one-sided predictions; (iii) it is **decomposed** into 3-dof / angle-only /
> qop-only sub-tests, which is what revealed that the qop subspace alone is the
> best decay tag -- `trkKink` returns a single 5-dof number mixing position,
> angle and momentum. A direct benchmark against `trkKink` itself has NOT been
> run.

**Working area:** `CMSSW_10_6_26_dev/src/Analysis/HitAnalyzer` (same area as the scalar-potential
and Phase-B work). Port to 15_0 afterwards is mechanical.

---

## 0. Formalism (score-test formulation)

The converged CVH fit minimizes
χ² = Σ_hits r_measᵀ V⁻¹ r_meas + Σ_layers dx0ᵢᵀ Qinvᵢ dx0ᵢ (+ constraints),
where dx0ᵢ is the 5-component difference between the free state at layer i and the propagated
state (`G4e.cc:1421-1422`), and Qᵢ is the Geant4e MS+ionization covariance
(`Geant4ePropagator.cc:1175` tuple element 2, rotated to local via `Hm`, `G4e.cc:1542-1547`).

**Alternative hypothesis at layer i:** the propagated state acquires an unconstrained offset
δᵢ = (Δqop, Δλ, Δφ) (3 params, no prior). Its Jacobian into the residual vector is a 5×3
selector Sᵢ acting only on the 5 rows of material-constraint block i (in the same local frame
as dx0 — apply the same `Hm` rotation used for Qinv).

To quadratic order around the converged solution this is a **score (Rao) test**, needing no
refit:

- gradient:  gᵢ = 2 Sᵢᵀ [Rr]ᵢ   (3-vector)
- marginal Hessian: Hᵢ = 2 Sᵢᵀ [R]ᵢᵢ Sᵢ  (3×3)
- **Δχ²ᵢ = ½ gᵢᵀ Hᵢ⁻¹ gᵢ**  ~ χ²(3) under the null
- best-fit step: δ̂ᵢ = −Hᵢ⁻¹ gᵢ  → estimated kink angles and Δ(q/p) magnitude/sign

where R = V⁻¹ − V⁻¹F C⁻¹ Fᵀ V⁻¹ is **already computed** for the alignment grad/hess
(`G4e.cc:2541`: `R = Vinvsparse − VinvF·Cinvd.solve(FtVinv)`; grad `= 2JᵀRr` at `:2623`,
hess `= 2JᵀRJ` at `:2624`). The kink scan is literally: append 3 sparse columns per material
layer to the J-side of that computation and read off the per-layer diagonal blocks.

Sub-statistics stored alongside the 3-dof Δχ²:
- **Δχ²(angle)**: 2×2 (Δλ, Δφ) block only — sensitive to hard elastic scatters and decays alike;
- **Δχ²(qop)**: 1×1 Δ(q/p) only — the decay-specific direction;
- sign of Δ(q/p)·q: decays *lose* momentum (daughter softer), elastic scatters don't.

Decays between hit i and i+1 map onto material step i. Decays after the last hit are invisible
and harmless (curvature unaffected); decays before the first hit make the whole track the
daughter — not flaggable by any kink finder, caught only by the mass constraint / lineshape.

**Score vs exact LR:** the quadratic approximation underestimates Δχ² for large kinks (which
fail the fit anyway). Optional refinement (Phase 3): for the argmax layer of tracks above
threshold, refit with δ freed (extend `nstateparms` by 3, rerun 2–3 GN iterations) for the
exact LR and unbiased parent-segment parameters. Not needed for a veto.

---

## Phase 1 — Single-track implementation (`ResidualGlobalCorrectionMakerG4e.cc`)

> **STATUS: DONE (2026-07-31), in CMSSW_15_0_19_patch2_dev.** All items below
> implemented and validated:
> - Score-test scan wired after convergence using the existing dense `R` and
>   `Rr = Vinv(r + F dxfree)`; per-step blocks read directly off `R` — zero
>   extra solves. Kink pseudo-params kept out of parmset/detidparms.
> - Formula validated exactly against an explicit refit in a numpy toy
>   (score = LR in the linear model, including sign of deltahat).
> - Injection closure (28 tracks, aligned geometry, dxdz=5e-3 + dqop=5e-4 at
>   step 5): median recovered deltahat = (5.0e-4, 4.7e-3) — ~7% angle dilution
>   from MS-prior absorption; localization at injected layer +-2 for
>   high-significance tracks.
> - Null on J/psi muons (588 tracks, aligned geometry, chisq/ndof median
>   1.02): kinkDchisq median/q90 at 0.73/1.04 x chi2(3) quantiles, q999 ~2.5x
>   (expected MS heavy tail); Delta-chi2 <= chi2 bound satisfied at 99.8%
>   (pseudo-inverse eigenvalue floor 1e-6*lmax).
> - Caveats for users: deltahat components are noise-dominated at steps with
>   small Dchisq; kinkDqop tails are large where H is near-singular — cut on
>   kinkDchisq first, then read deltahat. Ideal-geometry runs are NOT a null
>   (misalignment appears as real kinks, chisq/ndof ~170).
> - Driver: runCvhSingleTrack.py doKinkFinder=True [kinkInjectLayer=N
>   kinkInjectDqop/Dxdz/Dydz=...]; branches kinkDchisq[Angle,Qop], kinkDqop,
>   kinkDxdz, kinkDydz, kinkGlobalR/Z, kinkMax, kinkMaxLayer.

The single-track producer is the right first target: kaon/pion tracks are fitted single-track,
and the sparse path already forms R explicitly.

1. **Config flag** `doKinkFinder` (default `False`) in the cfi + producer, analogous to `dores`.
2. **During the hit loop**, cache per material step: the constraint-row offset `icons`, the 5×3
   selector Sᵢ (identity into the qop/dxdz/dydz components in the local frame — same frame
   convention as the `Qinv` block at `G4e.cc:1542-1568`), and the detid/layer id.
3. **After convergence** (in the block that computes `R`, `chisqval`, `grad`, `hess`,
   `G4e.cc:2541-2624`): build the sparse kink-Jacobian `Jkink` (ncons × 3·nsteps, 5 nonzeros
   per column), then
   `gk = 2·Jkinkᵀ(Rr)`, `Hk = 2·Jkinkᵀ R Jkink` — reusing `Rr` and the `Cinvd` factorization
   (per layer this is 3 extra sparse solves; ~50 small solves/track, negligible vs the fit
   itself, which profiling shows dominates).
   **Do NOT register these as parmset/detidparms global parameters** — they must not leak into
   `jacrefv`/`gradv`/`hesspackedv` consumed by the alignment fit. Keep them in a separate,
   local-only block.
4. **Per-layer extraction:** Δχ²ᵢ (3 dof), Δχ²ᵢ(angle), Δχ²ᵢ(qop), δ̂ᵢ. Guard: skip layers
   where Hᵢ is singular/ill-conditioned (first/last steps have weak constraints).
5. **New branches** (gated on `doKinkFinder`):
   - `kinkDchisq` (vector, per material step), `kinkDchisqAngle`, `kinkDchisqQop` (vectors)
   - `kinkMax`, `kinkMaxAngle`, `kinkMaxQop` (scalars), `kinkLayer` (argmax step index),
     `kinkR`, `kinkZ` (position of argmax step), `kinkDqop` (δ̂ Δ(q/p) at argmax, signed)
   - later: a scalar `kinkMax` ValueMap next to the existing `edmval` ValueMap
     (`G4e.cc:108,3146`) for NanoAOD consumption.
6. **Closure test (Phase-B.5 style, deterministic):** on a few MC tracks, inject a synthetic
   offset δ_inj into the propagated state at one chosen layer inside the fit and verify
   (a) Δχ² appears at that layer with the predicted magnitude ½δ_injᵀHᵢδ_inj (for small δ),
   (b) δ̂ recovers δ_inj, (c) all other layers stay at null level. This pins signs, frames,
   and factor-of-2 conventions before any physics is claimed.
7. **Null calibration:** run on J/ψ muons (data + JPsiToMuMu MC, split=1 files in
   `data/Run2016*`, NOT `data/repacked`). Muons don't decay → `kinkDchisq` should follow
   χ²(3) per layer; `kinkMax` follows the trials-corrected extreme-value distribution.
   Deviations measure residual Q-model mismodeling (also interesting per se for the
   resolution workstream — Josh's doRes non-Gaussian-tail problem).

**Estimate:** the heavy lifting (R, Cinvd, frames) exists; this is ~200 lines + branches.

## Phase 2 — Physics validation on gen-truth MC

> **STATUS (2026-08-08): PHASE 2 CLOSED — full gen-truth efficiency/purity on the
> v3 MC.** The v3 campaign
> (`/ceph/submit/data/group/cms/store/mc/inclusive_btojpsix_2016postvfp_v3/`,
> ~80k files / 4.18M usable events, `CMSSW_10_6_20_patch1` + `keepGenSimTruth`)
> keeps SimTracks + SimVertices, so every gen-matched track is labelled by what
> actually happened to it. Sample: 10k files per species; after quality cuts
> (p > 3 GeV, >= 8 valid hits) and the gen-match arbitration below,
> **74,581 kaons / 203,721 pions / 1,083,059 muons**.
> Producer: new `doSimDecayTruth` flag -> `simTrk*`, `simVtx*` (G4 processType),
> `simDau*` branches. Runner `calibration_studies/kinkfinder/run_v3_species.sh`,
> analysis `kink_truth.py` + `plot_kink_roc.py`, plots
> `~/public_html/ZMass/260808_kink_truth/`.
>
> **Truth labelling.** A vertex only matters if it lies between the first and
> last measurement, so the label uses the path coordinate s = sqrt(R^2+Z^2):
> in-span if s_firststep < s_vtx < s_laststep. processType 201 = decay,
> 121/111/131/151/161 = nuclear, everything else (delta rays, brems) is not
> signal. Composition: kaons **1.05% decay / 3.48% nuclear**, pions
> 0.64% / 4.25%, muons **0.00% / 0.005%** — the muon null validates the whole
> truth chain, as does the K decay daughter mix (63.9% mu vs PDG BR 63.56%).
>
> **The gen match had to be fixed first — this dominated everything.** With the
> pT window opened (mandatory: a decayed kaon's reco pT follows the daughter) a
> soft gen hadron wins the dR match to an unrelated, usually J/psi-muon, track.
> 22% of kaon- and pion-matched tracks were such steals, and they concentrate in
> the decay class (a kaon that decayed is often not reconstructed at all, so its
> gen particle is free to steal): **59% of "decay"-labelled tracks were fakes**.
> Fix = let every species compete for the match and require the winner to be the
> species of the pass. Applied offline here across the three passes (v3 has
> unique run/lumi/event and trackPt/Eta/Phi identify the reco track across
> passes); also implemented in the producer as `genMatchPdgIds` + `genMatchDR`
> + `genPdgId`/`genDR` branches — **source committed but NOT yet compiled**, see
> the note at the end of this block.
> Validation that this removes contamination and not signal
> (`kink_arbitration_validation.png`): rejected decay-labelled tracks have
> pT(reco)/pT(gen) median **4.50** and a max-Dchi2 distribution lying exactly on
> top of clean tracks (median 4.75, q90 11.3), while kept decays sit far above
> (median 8.61, q90 76.0). Diluting effect if skipped: AUC 0.59 instead of 0.72.
>
> **Tagging performance** (kaons, signal = in-span decay, background = clean):
>
> | discriminant | AUC decay | AUC nuclear | eff(decay) at 1% mis-tag |
> |---|---|---|---|
> | 3-dof `kinkMax` | 0.723 | 0.648 | 0.147 |
> | angle only (Delta-lambda, Delta-phi) | 0.718 | 0.646 | 0.142 |
> | **Delta(q/p) only** | **0.738** | **0.678** | **0.184** |
> | 3-dof, momentum-loss steps only | 0.677 | 0.601 | 0.153 |
> | track chi2/ndof (baseline) | 0.700 | 0.647 | 0.139 |
>
> - **The momentum-step direction does add information, and it is the whole
>   gain**: the 1-dof Delta(q/p) test beats both the angle-only test and the
>   3-dof combination. Adding the two angle dof *dilutes* the decay signal
>   (0.738 -> 0.723). Recommendation: **define the decay tag on the qop
>   subspace**, keep the 3-dof score as the generic outlier statistic.
>   The margin over the KF-like angle-only test is real but modest
>   (+0.020 AUC, +30% relative efficiency at 1% mis-tag).
> - Localisation is excellent: tagged-step radius vs true decay radius has
>   median +0.8 cm, IQR 6.3 cm.
> - Efficiency vs true decay radius peaks at ~37% for r = 10-30 cm and falls to
>   ~4% beyond 80 cm (no downstream lever arm).
> - Efficiency vs daughter momentum fraction z rises towards z -> 1 (~35-38%);
>   Delta(q/p)-only is uniformly better and 2-4x better at intermediate z.
>   Physically: small-z decays break the track and are removed by reconstruction
>   rather than tagged.
>
> **Pions make the case far more sharply than kaons** (`kink_roc_decay_pi.png`).
> pi -> mu nu has p* = 30 MeV, so the decay is nearly collinear: the track is
> reconstructed straight through the kink and the *only* visible effect is the
> momentum step. AUCs look similar to the kaon ones (Delta(q/p) 0.727, angle
> 0.714, 3-dof 0.721, chi2 0.695) but the low-mis-tag region -- the one a veto
> actually uses -- separates completely:
>
> | at 1% mis-tag | Delta(q/p) | 3-dof | angle only | chi2/ndof |
> |---|---|---|---|---|
> | pion decays | **0.092** | 0.025 | 0.013 | 0.016 |
> | kaon decays | **0.184** | 0.147 | 0.142 | 0.139 |
>
> **7x the efficiency of the angle-only sub-test on pion decays.** AUC hides
> this entirely -- quote working-point efficiencies, not AUC.
>
> **Distributions of the discriminator variables** (`kink_dpp*_by_truth_*`,
> `kink_angle_by_truth_*`, `kink_score{,qop,angle}_by_truth_*`). Using the
> fractional momentum step delta-hat(q/p)/(q/p) at the candidate step (positive
> = momentum loss), for steps with Delta-chi2 > 20 where deltahat is not
> noise-dominated:
>
> | | clean | nuclear | decay |
> |---|---|---|---|
> | kaons: median / fraction losing p | -0.0011 / 0.490 | +0.0340 / 0.571 | +0.0564 / **0.680** |
> | pions: median / fraction losing p | +0.0002 / 0.502 | +0.0138 / 0.544 | +0.1730 / **0.874** |
>
> The clean samples are symmetric at 0.50 as they must be. **The pion decay
> distribution peaks at ~+0.2 and terminates at the two-body kinematic endpoint
> 1 - (m_mu/m_pi)^2 = 0.43** -- an absolute closure of the deltahat(q/p)
> estimator against decay kinematics, with no free parameter. The kaon endpoint
> (0.95) is nearly unconstraining and K -> pi pi0 dilutes the peak, which is why
> the kaon distribution is much broader. The angular deltahat separates the
> classes far less (kaon median 0.0044 clean vs 0.0050 decay), consistent with
> the AUC ordering.
>
> **Nuclear interactions as signal too.** They distort the curvature just as
> decays do and a veto is equally entitled to remove them, so every ROC now
> shows decay-vs-clean (solid) *and* nuclear-vs-clean (dashed) on the same axes,
> with a third ROC against their union (`kink_roc_<species>.png`,
> `kink_roc_hard_<species>.png`). Efficiency at 1% mis-tag for
> "decay OR nuclear": kaons Delta(q/p) 0.140 vs 3-dof 0.105 vs angle 0.103;
> pions 0.071 / 0.059 / 0.057. Note the *nuclear* class is tagged about equally
> well by every discriminant (pions: 0.064-0.068 for all of them) — a nuclear
> interaction changes angle and momentum together, so there is no qop advantage
> there. The Delta(q/p) advantage is specific to decays, which is exactly the
> design intent.
>
> Plots are organised as **one complete set per species** (kaon, pion) —
> arbitration validation, three score variants, momentum step (all / tagged),
> angular kink, two ROCs, efficiency vs radius and vs z, localisation and
> curvature bias — plus three cross-species plots (per-step null, decay
> survival, species survival).
>
> *Plotting note:* the ROC reference line must be drawn from a dense grid
> (`plot(logspace(...), logspace(...))`). A two-point `plot([lo,hi],[lo,hi])` is
> rendered as a straight segment in *display* space, which on a log axis is not
> tpr = fpr and misleadingly appears to put the curves below the diagonal. The
> ROCs are now log-log, where tpr = fpr is genuinely a straight line.
>
> **Only 21% of in-tracker kaon decays survive into the track sample**
> (`kink_decay_survival.png`): observed in-span decay fraction 1.049% against
> 4.937% expected from 1-exp(-L/lambda) with each track's own measured path
> length; pions 0.638% vs 0.914%, survival 0.697. The species difference is the
> decay kink angle (K->mu nu p* = 236 MeV vs pi->mu nu 30 MeV). **This closes
> the 7/31 puzzle quantitatively**: expected decay-probability ratio K/pi = 5.4
> for these spectra, times the survival ratio 0.212/0.697 = 0.304, gives 1.64 —
> exactly the observed 1.049/0.638.
>
> **Curvature bias — the systematic.** Bias = (q/p)^fit/(q/p)^gen - 1:
>
> | class | mean (clip 1) | median | core mean (|b|<0.02) | out-of-core |
> |---|---|---|---|---|
> | clean | -0.00257 | -0.00080 | -0.000428 | 0.229 |
> | nuclear | -0.00517 | +0.00047 | +0.000234 | 0.592 |
> | decay | -0.06825 | -0.00022 | +0.000707 | 0.657 |
> | all | -0.00334 | -0.00078 | -0.000411 | 0.246 |
>
> Contamination shifts the clipped **mean by -7.7e-4 but the core mean by only
> +1.7e-5**: decays and nuclear interactions produce a broad, nearly symmetric
> tail, not a shift of the peak (`kink_curvature_bias.png`). **Decay-in-flight
> therefore enters the B -> J/psi K calibration as a lineshape/tail effect, not
> as a scale bias**, and a kink veto at 1% mis-tag changes the core mean by
> -5e-6, i.e. nothing. The veto is a tail-cleaning tool, not a scale-systematic
> tool; the systematic should be propagated through the lineshape.
>
> **Per-step null** (`kink_null_perstep.png`): truth-clean kaons, pions and
> muons give identical per-step Delta-chi2 distributions, following chi2(3) in
> the core with a heavy tail above ~13 — MS non-Gaussianity, plus elastic
> scatters that the sim truth cannot label (see caveat).
>
> **Caveat on the truth (limits all numbers above).** Geant4 writes a SimVertex
> only where a secondary track was stored: `fMultipleScattering` has **zero**
> entries and `fHadronElastic` only 72 of 91844. Genuine elastic kinks are
> therefore unlabelled and sit in the "clean" background, so the quoted AUCs and
> efficiencies are **lower bounds** on the true kink-tagging power.
>
> **Production gotchas for anyone else using v3:** 12 zero-length .root files
> (jobs that died during copy-out) abort cmsRun with FileOpenError — the driver
> now sets `skipBadFiles=True`; 16 further files come from jobs whose json
> reports failure, so filter on `exit_reason=="ok"` for a final list. All
> branches are splitLevel=1, so the 10_6-in-15_0 SiStripCluster bug does not
> apply. `first_lumi` varies per job, (run,lumi,event) is unique, and
> `duplicateCheckMode="noDuplicateCheck"` is no longer required.
>
> **UNFINISHED:** the producer-side `genMatchPdgIds` arbitration, `genPdgId` and
> `genDR` branches are edited into `CMSSW_15_0_19_patch2_dev` but **not built** —
> 160 unrelated `runCvhResClosure.py` jobs were running out of that same area and
> a `scram b` would have swapped the plugin .so under them. Build and re-run a
> smoke test before the next production; until then the arbitration only exists
> in the offline analysis. `runCvhSingleTrackJpsiX.py` already passes the new
> parameters, which the current binary silently ignores.

> **STATUS (2026-07-31): first data results in — the tagger sees real decays.**
> - New drivers: `runCvhSingleTrackV0.py` (V0-daughter tracks through the
>   single-track maker, pion hypothesis, p > 3 GeV preselection above the GN
>   momentum floor) and `runCvhSingleTrackMC.py` (JPsiToMuMu MC, gen-matched).
>   Analysis: `calibration_studies/kinkfinder/plot_kink_null.py`; plots at
>   `~/public_html/ZMass/260731_kink_null/`.
> - Samples: 7707 KS->pipi pions (pmlugato split-2 V0 ALCARECO, SingleMuon
>   2016F), 5882 data muons, 2999 MC muons (split-1 JPsiToMuMu repack).
> - **Pion decay signal**: kinkMax survival flattens into a plateau above
>   Dchisq ~ 40 at ~3e-3 while data/MC muon nulls agree. Matched slice
>   (3<p<10 GeV, >=10 hits, |eta|<1.8): pion excess +0.61% at kinkMax>50
>   (+0.27% at >100) vs expected in-tracker pi->mu nu probability 0.40% at
>   <p>=4.4 GeV. Consistent — the tagger works on real decays at data.
> - **New physics background identified**: momentum-loss tails (delta rays /
>   Landau) give q*dqophat > 0 for ALL species — the sign asymmetry at
>   qop-tagged steps is ~0.55 for muons AND pions alike. Per-layer sign alone
>   does not isolate decays; decay/dE-loss-tail separation needs gen truth
>   (below) and likely a magnitude+correlation discriminant, not just sign.
> - Production patch landed: `CustomiseMCJpsiX.keepGenSimTruth` (genParticles
>   + g4SimHits SimTracks/SimVertices) wired into `make_fullchain_cfg.sh` of
>   the BtoJpsiX MC production (pi/K decay vertices only exist as SimVertices
>   — Pythia keeps them stable). Flagged in OPERATOR_NOTES for Pietro's
>   sign-off; existing ~100 rehearsal files do NOT carry these branches.
> - Remaining for Phase 2: gen-truth efficiency/purity once patched MC exists
>   (kaons) — and a V0 MC equivalent for pions if available; sim-based
>   decay-truth branches in the maker (blocked on the same files).
>
> **STATUS UPDATE (2026-07-31 late): gen-matched species passes on the LIVE
> B->J/psi+X production** (253k files / 341 GB / ~3.8M events at
> /ceph/.../group/cms/store/mc/inclusive_btojpsix_2016postvfp — files DO carry
> genParticles; the gen-less "rehearsal" files in the user area are stale).
> Driver: runCvhSingleTrackJpsiX.py (kaon/pi/mu passes, requireGen; NEW
> configurable genMatchPtWindow, default 10 = off, since the legacy 0.5 window
> removes decayed kaons whose reco pT follows the daughter muon).
> - **Reader gotcha (affects every consumer of this production):** all jobs
>   write Run 1/Lumi 1/events 1..800 -> ids collide across files and the
>   default PoolSource duplicate check silently drops everything beyond ~60
>   files. duplicateCheckMode="noDuplicateCheck" is REQUIRED (and rows are
>   not keyable by run/lumi/event).
> - Yields (1200/400/200 files): 2532 kaons, 2950 pions, 5373 muons; tag
>   fractions (kinkMax>50): K 1.03+-0.20%, pi 1.53+-0.23%, mu 0.35+-0.08%.
> - **Two confounders identified vs the naive 7.4x K/pi lifetime ratio:**
>   (1) the B-candidate selection upstream already removes hard kaon decays
>   (broken vertex/kinematics) — opening the match window recovered +20%
>   tracks but did NOT raise the kaon tag rate, so hard decays never reach
>   the track collection (unlike the looser V0 skim where the pion decay
>   plateau is visible); (2) hadronic (nuclear) scattering adds ~1% tag rate
>   for BOTH hadron species — absent for muons — and quasi-elastic scatters
>   also bias curvature, a systematic mechanism to include alongside decays.
> - **Correlated discriminant works at population level:** among tagged
>   steps (Dchisq>50), fraction with significant qop component
>   (Dchisq_qop>10): K 0.72 / pi 0.63 / mu 0.39; momentum-LOSS sign fraction
>   q*dqop>0: K 0.67 / pi 0.57 / mu 0.49 (symmetric null). Full
>   decay-vs-hadronic decomposition still needs SimVertices (keepGenSimTruth
>   batches).
> - Plots: ~/public_html/ZMass/260731_kink_species/ (*_openwin.*);
>   analysis calibration_studies/kinkfinder/plot_kink_species.py (bins in
>   GEN momentum — reco p of decayed hadrons follows the daughter).

**Samples:**
- **Pions, immediately available:** V0 ALCARECO (KS→ππ) — CVH V0 infrastructure exists
  (15_0 tests; 10_6 needs the split=1 repack already understood). MC gen-matching gives pion
  decay vertices. Early testbed while kaon MC finishes.
- **Kaons, primary:** B→J/ψ+X 2016postVFP ALCARECO MC (Pietro-handoff condor production —
  currently blocked on voms proxy; this plan is a consumer of that production). Gen-match
  reconstructed kaon tracks to gen K± and classify by gen decay vertex
  (decayed in tracker: r_dec < 110 cm, |z_dec| < 280 cm).
- **Null:** muons from the same samples.

**Deliverable plots:**
1. Null: per-layer Δχ² vs χ²(3); `kinkMax` data/MC comparison on muons.
2. Efficiency of a `kinkMax` cut vs gen decay radius, gen kink angle, gen p — separately for
   K→μν, K→ππ⁰, π→μν.
3. `kinkLayer` position vs true decay radius (localization check).
4. Curvature bias Δκ/κ of decayed-and-surviving tracks, before/after the kink veto, vs decay
   radius — this is the number that feeds the calibration systematic.
5. ROC: kink veto vs the baseline (track χ²/ndof + nValidHits cuts alone), and vs a
   angle-only statistic (our Δχ²(angle) with Δqop dropped) to quantify
   what the momentum-step direction adds.

## Phase 2b — Decay-in-flight hadrons faking a prompt muon (2026-08-08)

Different question from Phase 2: there the background is truth-clean tracks of the
same species; here it is **real prompt muons**, which is what matters for the
momentum-scale calibration itself, since a hadron that decays and is reconstructed
as a muon enters the J/psi (or Z) sample directly.

Sample: the same 10k v3 files, but hadron tracks fitted with the **muon** mass
hypothesis (`fitAs=mu`, new driver option — the real analysis does not know they are
hadrons) and matched to `ALCARECOTkAlJpsiXLooseMuons`. Trees at
`/ceph/.../cvh/kinkfinder_v3_fakemu_260808/`; analysis
`calibration_studies/kinkfinder/plot_fake_muon.py`; plots
`~/public_html/ZMass/260808_kink_fakemuon/`.

**Muon-fake rate and its origin** (muonMedium):

| | tracks | fake a muon | of which in-span decay |
|---|---:|---:|---:|
| kaon | 74 421 | 0.442 % | **20.1 %** (vs 1.03 % inclusive -> x19.5) |
| pion | 202 819 | 0.305 % | **21.7 %** (vs 0.63 % inclusive -> x34.3) |
| muon | 1 082 756 | 97.8 % | — |

Decay in flight is enriched by a factor 20-34 in the muon-faking sample: it is a
**dominant mechanism** by which a hadron is reconstructed as a muon, even though it
affects only ~1 % of hadron tracks overall.

**Where those decays happen** (of muon-faking hadrons with a decay anywhere):

| | before first hit | in-span (taggable) | after last hit |
|---|---:|---:|---:|
| kaon | 7.9 % | **47.1 %** | 45.0 % |
| pion | 10.3 % | **62.9 %** | 26.8 % |

Note this is a very different split from the inclusive decay population (kaons only
36 % in-span there): faking a muon requires the daughter to be reconstructed and
matched, which biases towards decays inside the tracking volume. The ~45 % of kaon
fakes decaying past the last hit are **invisible to any tracker-based kink finder**
— the tracker sees a pristine hadron track.

**Rejection against 1.06 M real prompt muons:**

| | AUC | rejection at real-muon loss 1 % | 0.5 % | 0.1 % |
|---|---:|---:|---:|---:|
| kaon fakes, 3 dof | **0.878** | **0.470** | 0.394 | 0.227 |
| kaon fakes, angle only | 0.870 | 0.455 | 0.364 | 0.197 |
| kaon fakes, Delta(q/p) | 0.864 | 0.379 | 0.303 | 0.182 |
| pion fakes, Delta(q/p) | 0.795 | **0.284** | 0.179 | 0.104 |
| pion fakes, 3 dof | **0.809** | 0.194 | 0.149 | 0.052 |
| pion fakes, angle only | 0.797 | 0.134 | 0.090 | 0.007 |

**The discriminant ordering flips relative to Phase 2, and the kinematics say why.**
For *kaon* fakes the 3-dof and angle-only tests win: K -> mu nu has p* = 236 MeV, so
the fake carries a large angular kink. For *pion* fakes Delta(q/p) is best by 2.1x
over angle-only at 1 % loss: pi -> mu nu is nearly collinear, so the momentum step is
again the whole signal. Use the 3-dof score if one number must serve both.

**Scanned over the muon-ID ladder** (`plot_fake_muon.py --muonId any muonLoose
muonMedium muonTight`, one load, four working points). At the **Loose** ID — the
realistic analysis selection — 0.540 % of kaon tracks and 0.357 % of pion tracks are
reconstructed as muons, against 98.66 % of real muons:

| muon ID | real-muon eff | kaon fakes rejected at 1 % loss | pion fakes | n(K, pi) |
|---|---:|---:|---:|---|
| any match | 99.31 % | 0.462 | 0.293 | 91, 164 |
| **Loose** | **98.66 %** | **0.475** | **0.283** | **80, 152** |
| Medium | 97.82 % | 0.470 | 0.284 | 66, 134 |
| Tight | 64.43 % | 0.541 | 0.311 | 37, 45 |

Two things are stable and one moves:
- **The rejection is essentially flat across Loose/Medium** (0.47 for kaons, 0.28 for
  pions) — the kink handle is orthogonal to the muon ID, which is what makes it
  worth having: it is not just re-deriving a cut the ID already applies.
- **The decay contamination of the fake sample grows sharply with ID tightness**:
  in-span decays are 15.5 % of any-match kaon fakes, 19.9 % at Loose, 20.1 % at
  Medium, **32.2 % at Tight** (pions 15.5 -> 21.0 -> 21.7 -> 34.4 %). Tightening the
  muon ID preferentially removes the *non-decay* fakes (punch-through,
  mis-association), so what survives is increasingly decay in flight — exactly the
  population this tool addresses. Enrichment over the inclusive hadron rate reaches
  x31 (K) and x54 (pi) at Tight.
- Tight also buys slightly better rejection (0.541 K) but costs 35 % of real muons,
  so it is not a working point anyone would choose for that reason.

**Chain of efficiencies** (kaons, at Loose): 46 % of decay-induced fakes are taggable at all
(in-span), and 47 % of those are rejected at 1 % real-muon loss -> **~22 % of all
decay-induced kaon fakes**. Decay-induced fakes are ~20 % of all muon-faking kaons,
so ~4-5 % of the total hadron-fake population. For pions, 63 % x 28 % -> ~18 % of
decay-induced fakes.

**What the ALCARECO muon object can and cannot do** (checked on the v3 files, 96
muons over 40 events): **all four track references are non-null but DANGLING** —
`innerTrack` 96/96, `muonBestTrack` 96/96, `outerTrack` 85/85, `globalTrack` 83/83
are unavailable, because they point into `generalTracks`, which the ALCARECO drops.
Consequences:
- **The CVH refit is unaffected.** It refits `reco::Track` objects from
  `ALCARECOTkAlJpsiX`, which are present with their hits and clusters; the muon
  object is never the source of the track.
- **Muon identification IS usable**: the selector and type bits, p4 and station
  matches are stored inline on the muon. Measured: real gen muons 99.15 % matched /
  97.6 % Medium / 64.2 % Tight; gen kaons 0.78 % matched / 0.45 % Medium. Sensible
  and discriminating.
- **Only muon -> track navigation is impossible.** Any code calling
  `muon.innerTrack()` / `bestTrack()` throws on this sample — including the maker's
  legacy `doMuons=True` momentum-matching path, which now routes to a ref-safe p4
  match instead.
- **Enter from the candidate, not the muon.** The `VertexCompositeCandidate` leaves
  hold `reco::TrackRef`s into `ALCARECOTkAlJpsiX` (ProductID 3:1449) that DO
  resolve — verified **122/122 J/psi leaves, 0 dangling** — while the muon's own
  refs point at a different product (3:1466). The chain **candidate leaf ->
  TrackRef -> `ALCARECOTkAlJpsiXTrackToMuon` -> reco::Muon** resolves 122/122 and
  yields chamber quality (numberOfMatchedStations 2-4, stationMask) and the ID
  bits. The association must be keyed on the **unselected** collection; the earlier
  `InvalidReference` was only because a `TrackSelector` copy was used as the key.
- Candidate structure is a hard invariant: 61/61 B+ candidates have
  daughter(0) = J/psi (pdgId 443) with exactly two muon leaves and daughter(1) =
  bachelor. Charge order *within* the pair is not fixed. The same dimuon recurs
  across several B candidates (61 candidates / 40 events), so track-level analyses
  must de-duplicate.

**Bug found and fixed while checking this:** `muonIsPF` was missing from the
per-track reset in the G4e maker, so it carried over from the previous track —
true for ~47 % of hadron tracks never matched to a muon. No result above used it
(`muonMedium` was used throughout), but any earlier study that read `muonIsPF` from
this producer is suspect.

**Caveats:** statistics are modest (66 kaon and 134 pion fakes); the loose-muon
collection is the one the J/psi ALCARECO keeps, so absolute fake rates are specific
to this skim and not transferable to a Z -> mumu selection; punch-through and
mis-association fakes (the ~80 % of muon-faking hadrons that did *not* decay) are
untouched by this handle.

## Phase 3 — Application and extensions

1. **B → J/ψ K systematic:** with Phase-2 numbers, compute the surviving decay fraction and
   its κ_K bias after the veto in the B± selection; propagate to the A–ε extraction as the
   decay-in-flight systematic line (companion to the B→J/ψπ contamination systematic).
2. **Two-track/three-track fit:** port the score test to `TwoTrackG4e.cc` (dense path is
   simpler — the kink params are 3 extra rows/cols of `gradfull`/`hessfull` blocks at
   `TwoTrackG4e.cc:1766-1849`, Schur-complemented against `d2chisqdx2`). Needed for the
   eventual 3-track J/ψK fit so the kaon leg carries its kink statistic under the B mass
   constraint.
3. **Optional exact-LR refit** at argmax layer for tagged tracks (recovers parent-segment
   curvature; only if a measurement of the decay fraction itself, or recovery rather than
   veto, becomes interesting).
4. **Side benefit:** the angle-only pulls on muons are a direct diagnostic of MS-tail
   mismodeling — feed observations back into the resolution-corrections workstream
   (`Documents/Resolution/NOTES.md`).

## Phase 4 — Documentation

Add a subsection to AN-XX-XXX under the B→J/ψK calibration inputs: formalism (score test),
null validation, tagging performance, and the resulting decay-in-flight systematic. Follow
the slide/AN accuracy rules: every headline number from the gen-matched MC study, clearly
separated physics vs plumbing.

---

## Risks / open points

- **Frame bookkeeping** is the main correctness risk (local vs curvilinear for dx0 when
  `dolocalupdate`); the injection closure test (Phase 1.6) is designed to catch exactly this.
- **Kink absorption by neighboring layers:** with 5 free params/layer under MS priors, part of
  a real kink leaks into adjacent steps, diluting the single-layer score. The score test with
  marginal Hessian handles the correlations correctly, but expect the localization (kinkLayer)
  to be ~±1 layer; quantified by Phase-2 plot 3.
- **Trials factor:** kinkMax over ~15–20 steps; calibrate the cut on the null MC rather than
  on analytic χ²(3) quantiles.
- **Stripless/pathological layers** (e.g. pixel-only stretches): guard on constraint-block
  conditioning; reuse the existing fit-failure guards.
- **Hard elastic scatters** are an irreducible angle-only background; the Δχ²(qop) component
  and the sign of Δ(q/p) are the discriminators — verify separation on MC (K vs elastic-tag
  from gen).

> **DONE 2026-08-29.** Everything below was carried out; the results are in
> `Documents/Resolution/NOTES_HITRES.md` and
> `~/public_html/calibration_studies/260829_hitres/`. Headline: hit resolution
> is EXCLUDED as a source of the -1.6e-4 momentum-scale bias. The CPE's
> variance is right to a few percent; its core is 8 % too wide in the strips
> and 7 % too narrow in the pixels, and correcting toward the core moves the
> fit AWAY from truth by +0.120 +- 0.020 % of the q/p variance at 5.9 sigma.
> This file is kept as the brief that was worked from, not as an open task.

# Handoff: hit-resolution studies for the CVH fit

Written 2026-08-29. Everything below was read out of
`CMSSW_15_0_19_patch2_dev` on that date; line numbers are from that tree.
Companion notes: `Documents/Resolution/NOTES.md` (the ladder / rung-E work),
`Documents/Resolution/NOTES_CGFFIT.md` (the process-noise programme that
motivates this one, esp. §88).

---

## 1. Why this, why now

The process-noise programme is finished and it came back null. Measured
against gen truth on J/psi MC, no weight tested moves the fitted momentum:
the CGF ionization weight (|dVar|/Var < 2e-5), the alpha convention, the MS
weight (null in the core), the IRLS re-centring (|dVar|/Var < 4.8e-4). See
NOTES_CGFFIT §88 for the table.

The number that did **not** move in any of those arms is a gen bias of
**-1.6e-4 on q/p**, identical to the last digit across every scan. That is
1.6x the *W*-mass target and 16x the Z-mass target (delta p/p ~ 1e-5), so it
is the budget. It has to live in one of: the reference trajectory, the
material model, the **hit resolution**, or the alignment.

This handoff covers the hit-resolution term. Note the framing: the goal is
not "is sigma right" but "does the fit's Gaussian hit likelihood describe the
actual hit errors" -- which is a distributional question, not a scale
question, and the answer decides whether the fix is a rescaling or a
different likelihood.

---

## 2. What the fit actually assumes today

The fit consumes exactly one thing per hit: `preciseHit->localPositionError()`,
inverted into the block `Vinvfull`
(`ResidualGlobalCorrectionMakerG4e.cc:3222` for 2D, `:3237` for 1D). There is
no other hit-error model anywhere in the fit.

Three properties that are easy to get wrong:

1. **It is not a per-module constant.** Line 2866 does
   `cloner.makeShared(hit, tsostmp)` -- the CPE is re-run against the
   propagated track state, so the covariance is a function of the *fitted*
   incidence angle and changes as the fit converges. There is no static
   per-module lookup.

2. **The parameter content is tiny and global.** Via the `WithAngleAndTemplate`
   builder (`:323`):
   - **Strips** (`StripCPEfromTrackAngle.cc:48-66`): sigma is closed-form in
     cluster width `N` and the projected path in strip-pitch units `uProj`.
     `N<=4`: `uerr = P0*uProj*exp(-uProj*P1) + P2`, **three numbers for the
     entire strip tracker** (-0.326, 0.618, 0.300). `N>4`:
     `uerr = P0(subdet) + N*P1(subdet)`, two per subdetector. **Eleven numbers
     total, zero per-module content.** Config:
     `RecoLocalTracker/SiStripRecHitConverter/python/StripCPEfromTrackAngle_cfi.py`.
   - **Pixels** (`PixelCPETemplateReco.cc`): sigma_x, sigma_y interpolated from
     the template for that module's template ID -- so *classes* of modules via
     `SiPixelTemplateDBObject`, not individuals -- as a function of incidence
     angle and charge bin (qbin). With hardcoded escapes: bad/big-pixel
     clusters get flat 55/36 um barrel or 42/39 (`:461-465`), edge clusters get
     configured constants `xEdgeXError_` etc. (`:475-482`).

3. **The pixel off-diagonal is identically zero.**
   `PixelCPETemplateReco.cc:525` returns `LocalError(xerr*xerr, 0, yerr*yerr)`.
   Any xy correlation the fit sees is generated downstream by the maker's
   rotation into the aligned/glued frame, and by the (phi, rho) conversion for
   radial (wedge) strip topologies (`:3004-3089`).

**Species / momentum dependence: no explicit one, two implicit ones.** Nothing
takes a particle hypothesis -- which is why the kaon in B->J/psi K is assigned
the same error model as the muons. But dE/dx enters twice: strips branch on
`dQdx > maxChgOneMIP` (6000) and fall back to the 3-constant legacy formula
above it (a hard threshold that a low-p kaon or proton crosses); and the pixel
template sigma depends on qbin. Charge q=+-1 enters only through the Lorentz
drift, not the error model.

---

## 3. The one piece of prior art, and it is dead code

`ResidualGlobalCorrectionMakerG4e.cc:3210`:

```cpp
const double scalecov = ispixel ? 0.8 : 1.2;
```

Someone's estimate that pixel errors are ~20% over- and strip errors ~20%
under-stated. It feeds only `Vinvfullalt`, which is used **only inside an
`if (false)` block** at `:3549`. It has never been applied to the fit. Treat it
as a hypothesis left by a previous author, not as a result -- but it is a
strong hint about both the sign and the size to expect.

---

## 4. The apparatus that already exists (most of the plumbing is done)

**Samples with PSimHits -- including per-species guns.** The standard
`JPsiToMuMu` MC ALCARECO used by `cgf_mc_closure.py` has **no** PSimHits,
SimTracks or SimVertices (verified 2026-08-29); neither does the B->J/psi+X
ALCARECO (see the comment at `ResidualGlobalCorrectionMakerBase.cc:335`).
Do not start there. Use the private guns instead:

```
calibration_studies/resolution/simprod/filelist_{mu,pi,kaon,proton,jpsi}gun_ul16.txt
  -> /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_*gun_ul16/task_*/step2.root
  160 files x 1000 events per species; 36 PSimHit branches present (verified on the kaon gun)
```

Produced by `simprod/step1_gensim.py` + `step2_digireco.py`, driven by
`run_simprod_hadrons.sh` / `run_simprod_mugun.sh`. **If you produce more:
tighten the Geant4 stepper precision** -- the CMSSW defaults are ~100x loose
and shift the widths by 12% (NOTES.md; this is already set in
`step1_gensim.py`, do not regress it).

**Driver**: `Analysis/HitAnalyzer/test/runCvhResClosure.py`, with
`particle=mu|pi|kaon|proton` (sets the track hypothesis, the gen-match pdgId
and the G4 particle list), `doSimHits=True`, and `fitSimHitPositions=True` for
the null test. Submit via `calibration_studies/slurm/submit_resolution_closure.sh`
or `resolution/run_local_closure.sh`.

**Branches already exported per hit** (no code change needed to start):
- `dxrecsim`, `dyrecsim` -- rec minus sim **in the frame the fit uses**,
  including the phi/rho conversion on wedge modules (filled at `:3410`,
  `-99` when unmatched). The frame handling is the fiddly part and it is done.
- `clusterSizeX`, `clusterOnEdge` -- so binning by hit class is free.
- `simlocalx/y`, `simlocaldxdz/dydz`, `simlocalqop` and their `*prop` variants.

**Switches already wired**:
- `fitSimHitPositions` (`:2936`) -- fit sim hit *positions* with covariances
  unchanged. The clean null test: if the -1.6e-4 survives it, the bias is not
  in the hit positions.
- `keepPixelEdgeHits`, `pixelMinSizeX` (`:1300`) -- the hit-quality selection.
- `doRes` -- registers resolution parameters, see below.

**Global resolution parameters already exist**: parmtype **8** (local x) and
**9** (local y, pixels only), applied as `iV *= exp(parm)` at `:3150-3163`,
one per module per coordinate. `doRes=False` in the muon cfi. This is Josh's
2022 attempt; **it failed on the non-Gaussian tails**. Do not simply re-run it
and expect a different answer -- understand the tails first.

---

## 5. Known defects and traps

1. **The sim-hit match is hardcoded to muons.**
   `ResidualGlobalCorrectionMakerG4e.cc:2115`:
   ```cpp
   if (simHit.detUnitId() == hit->geographicalId()
       && int(simHit.trackId()) == simtrackid
       && std::abs(simHit.particleType()) == 13) {
   ```
   while the *gen* match is properly parameterised (`genMatchPdgId_`,
   `:937-945`). Consequence: **on the kaon/pion/proton guns the sim-hit
   machinery silently no-ops** -- `simhit` stays null, `dxrecsim/dyrecsim` are
   -99, and `fitSimHitPositions` quietly falls back to reco positions because
   `usesimpos = fitSimHitPositions_ && simhit != nullptr` (`:2936`). Verify
   this on a hadron-gun run before anything else, then fix it (one line: use
   `genMatchPdgId_`). It fails silently, which is the dangerous kind.

2. **rec-sim is not the residual the fit needs.** It is the *total* hit error,
   but the fit's weight should describe the hit error *given the track*.
   rec-sim also carries the sim hit's own definition (entry/exit midpoint,
   before drift and charge sharing) and a genuine non-Gaussian tail from
   delta rays and merged clusters. The object to study is the **pull**
   `(rec-sim)/sigma_CPE`, binned in (subdet, N, uProj) for strips and
   (qbin, angle, class) for pixels.

3. **The interesting question is the shape, not the width.** If the pull is
   Gaussian with width != 1, the fix is a rescaling of the CPE parametrisation
   and parmtype 8/9 would have worked. It didn't -- which is evidence the pull
   is *not* Gaussian. Measure the tail fraction and the core width separately
   before proposing any correction.

4. **Do not go per-module.** Fitting 2 x ~15k sigmas repeats the ~15k dBz and
   ~25k material-parameter pattern that the rest of this programme is busy
   reducing. The CPE has *eleven* strip parameters and a template table, so
   the natural target is a correction to its functional form in
   (N, uProj, subdet) and (qbin, angle, class) -- an O(10) object. It is also
   the only version that transfers to data, where there are no sim hits.

5. **MC hit resolution is not validated against data.** AN-21-131 requires
   data/MC resolution matching at the percent level before pT-dependent scale
   closure means anything. Anything derived from sim hits is an MC-internal
   statement until that transfer is done.

---

## 6. Suggested order of work

0. **Sanity**: run `runCvhResClosure.py particle=kaon doSimHits=True` on ~2k
   events and check `dxrecsim != -99`. Expect it to fail (trap 1). Fix, rerun.

1. **Pull study, no code changes**, on the mu gun first: histogram
   `(rec-sim)/sigma_CPE` per subdetector, binned in cluster width and incidence
   angle. Report core width (robust) *and* tail fraction separately. This alone
   decides "rescale" vs "wrong likelihood".

2. **Repeat per species** (mu / pi / kaon / proton) at matched momentum. This
   is the question no one has answered: whether the same CPE covariance is
   right for a kaon as for a muon, given the two implicit dE/dx dependences in
   §2. If it is not, that is directly relevant to the B->J/psi K channel of the
   B-field calibration.

3. **Null test**: `fitSimHitPositions=True`. If the -1.6e-4 gen bias survives,
   the hit *positions* are exonerated and the remaining hit-side suspect is the
   covariance (which this switch leaves untouched) -- a clean factorisation.

4. Only then propose a correction, as a modification of the CPE's functional
   form, and validate it against truth the same way NOTES_CGFFIT §80-88 did:
   paired statistics, `dVar = 2Cov(r,d) + Var(d)`, and **nothing quoted from a
   single sample below 3 sigma on a variance**.

---

## 7. What would count as an answer

A statement of the form: "the CVH hit likelihood is wrong in <this specific
way>, worth <X> on q/p against gen truth, and correcting it moves the fit
toward truth by <Y> at <Z> sigma on a paired comparison." Anything less --
in particular any per-track shift quoted without a truth comparison -- repeats
the mistake catalogued in NOTES_CGFFIT §86-87, where a 1.8-sigma "gain" failed
to replicate.

Note the fit's response floor: a 1e-16 perturbation of the noise rows produces
rms 2.4e-7 on q/p (NOTES_CGFFIT). Shifts at that level are numerical, not
physical.

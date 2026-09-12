# resolution/runs — surviving results, by run tag

`resolution/runs/` is a gitignored working area (which is why this file lives one level up, in the tracked tree): caches, cards, fits and job
logs. This file keeps the results those runs produced that are still quoted,
so that the directories themselves can be pruned. The full derivations are in
the archived development log
`/work/submit/david_w/Documents/Resolution/archive/NOTES_devlog_until_260911.md`
at the dated entries named below; the resulting production settings are in
`calibration_studies/production/PRODUCTIONS.md`.

---

## `clampfix260904` — the Gauss-Newton momentum-floor clamp was a leftover

**The defect.** The clamp exists only to keep the Gauss-Newton state out of the
propagator's refusal region, `Geant4ePropagator.PropagationPtotLimit`. It was
written as a hard-coded **2.0 GeV** when that limit was the cfi default 1.0 GeV;
the drivers later lowered the limit to **0.2 GeV** and the floor was never
lowered with it. A 2 GeV floor against a 0.2 GeV limit is not a divergence
guard — it is a hard non-linearity in the middle of the physical momentum
spectrum: a track whose true momentum is below 2 GeV is pinned at 2 GeV, and
where the reference momentum is already below the floor the rescale
`s = (copysign(1/floor, qop) - qop)/dqop` is negative, `max(s, 0)` makes it a
hard zero, and the whole coupled two-track step freezes at the seed
linearization forever.

**The fix.** `clampMomentumFloor` is a cfi parameter on all three makers
(`existsAs` fallback to 2.0, so an unchanged cfi is bit-identical) and the
drivers derive it as **`1.25 x propagationPtotLimit`** = 0.25 GeV, refusing to
run if the floor is not above the limit. The clamp LOGIC is unchanged. Gates:
floor 2.0 explicit reproduces the pre-fix binary **137/137 (single track) and
212/212 (two track) branches BIT-IDENTICAL**; the new default is bit-identical
single-track (the clamp never fires there).

**What it was worth** (J/psi gun, 300 370 attempted, identical inputs;
`_260903x` = floor 2.0, `_260904f` = floor 0.25):

| | floor 2.0 | floor 0.25 |
|---|---|---|
| clamped fits | 36 863 (**12.27 %**) | 480 (**0.16 %**) |
| `frozen` | 7.025 % | **0.079 %** |
| chi2/ndof > 10 | 12.161 % | **0.143 %** |
| chi2/ndof q99 | 1.055e5 | **2.358** |
| `<m - m_gen>`, ptmin < 0.5 GeV | +262.7 MeV | +273.0 MeV |
| **UNCUT mass fit, alpha [1e-3], single `r`** | **+0.2110 +- 0.0167** | **-0.0027 +- 0.0167** |
| UNCUT, families | +0.2447 +- 0.0177 | +0.0280 +- 0.0175 |

**The published +0.21e-3 gun mass-scale offset was this clamp, and it is gone
with no cut.** The `chi2/ndof < 3` cut, which used to move alpha by 0.214e-3,
becomes a 0.0013e-3 no-op; the `ptmin > 2` control fits are BIT-IDENTICAL
between the two productions (those are exactly the candidates a 2 GeV floor
could never touch); B -> J/psi X v3 and the low-pT single-track closures are
unchanged, as they must be (the single-track clamp fired on 8 of 319 714
tracks). The price is 235 extra gun candidates (0.078 %) lost to genuine
propagation drain and +8.8 % propagation calls.

**Permanent rule: never raise `clampMomentumFloor` into the physical momentum
spectrum, and never hand-tune it per sample.** Tie it to the propagation limit
it exists to bracket.

Productions still in use: `resolution_trackres_{jpsigun_ul16,mugun_lowpt,
btojpsix_v3}_260904f_m0`. Full record: the archived dev log, entry **2026-09-04/05**. Figures:
`~/public_html/ZMass/cvh/260904_clampfix/`.

**Operational lesson that cost a day.** A CephFS EPERM on the login node
returned errors for the whole `/ceph/submit` tree while `cf_mass_likelihood.py`
SKIPS an unreadable input with a warning and still writes a VALID, EMPTY part —
136 of 138 parts held 0 candidates and the merge would have produced a silently
truncated cache. Any sharded extraction must run on compute nodes and **fail
loudly on an empty part**.

---

## `stepdamp260905` — a relative trust region replaces the fixed floor

**Why a floor cannot do the job.** The guard is meant to damp a Gauss-Newton
step that would drive a track to an absurdly small momentum. That is a TRUST
REGION, and a trust region is RELATIVE to where the iterate is. An absolute
floor does nothing for a 40 GeV track that a bad step drives to 0.3 GeV, and it
pins or freezes a 0.3 GeV track that is exactly where it belongs. The proof is
the clamp-scale distribution: **99.52 % (floor 2.0) and 79.58 % (floor 0.25) of
all clamp events had scale EXACTLY ZERO** — the damper was almost never
damping, it was freezing.

**The replacement**, `cvhstep::legStepScaleRel` in
`ResidualGlobalCorrectionMakerBase.h`, shared by all three makers:

* **item 1, relative damping in q/p.** Per iteration a leg's momentum may change
  by at most `maxMomentumStepFactor` = f (default 2): `p_lo = max(absFloor,
  p_ref/f)` if `p_ref > absFloor` else `p_ref/f`, `p_hi = p_ref * f`. `p_lo` is
  ALWAYS strictly below `p_ref`, so the scale can never be 0 and no track is
  ever pinned. Charge-flip semantics kept: a disallowed flip is capped as the
  extreme upward step, which at f = 2 is numerically IDENTICAL to the legacy
  "stop half-way toward q/p = 0". It supersedes the N-track maker's ad-hoc
  `min(floor, 0.5 p_ref)`.
* **item 2, chi2 (Armijo) backtracking.** The chi2 assembled at iteration `k` IS
  the realized chi2 of the step taken at `k-1`; on a sufficient-decrease failure
  the `k-1` linearization is restored, the step halved and the iteration redone.
  `stepBacktrackFromIter` defaults to **2, not 1**: at iteration 0 the GBL
  propagation/kink residuals are identically zero by construction, so chi2(0) is
  a different objective and comparing across that boundary backtracks
  everything.

**The Armijo test had to be re-purposed into a DIVERGENCE TRAP.** The CVH/GBL
iteration does not monotonically decrease `r^T Vinv r` — the realized chi2
approaches the previous value from ABOVE and never crosses it (139.535 ->
147.734 full step, 143.607 at 1/2, 141.561 at 1/4, 140.542 at 1/8), so a
textbook tolerance turns the line search into a permanent step-halver:

| `armijoSlack` | tt fits backtracked / halvings | tt prop calls | st fits / halvings |
|---|---|---|---|
| 0.001 | 110 / 2986 | 142 131 (**6.02x**) | 103 / 2190 |
| 0.05 | 43 / 485 | 42 572 (1.80x) | 22 / 147 |
| 0.25 | 18 / 62 | 26 143 (1.11x) | 5 / 9 |
| **1.0** | **3 / 3** | **23 743 (1.005x)** | **1 / 1** |
| off | 0 | 23 615 | 0 |

**Default `armijoSlack = 1.0`: the chi2 may not more than DOUBLE in one
iteration. NEVER set it to a textbook 1e-4..1e-3.**

**Item 1 does the whole job; item 2 rescues nothing here.** Same 8 gun files
(14 992 candidates), four refits differing only in the step control:

| variant | clamped fits | clamp events (frac scale 0) | btchi2 fits / halvings | chi2/ndof max | chi2>10 | frozen | prop calls |
|---|---|---|---|---|---|---|---|
| legacy (floor 0.25 only) | 26 | 45 (95.6 %) | — | **1.15e6** | 0.160 % | 0.080 % | 1 688 708 |
| **damp only (f = 2)** | 5 | 5 (**0.0 %**) | — | **8.96** | **0.000 %** | **0.000 %** | +0.06 % |
| backtrack only (slack 1) | 26 | 45 (95.6 %) | 225 / 539 | 1.15e6 | 0.160 % | 0.080 % | +1.26 % |
| both | 5 | 5 (0.0 %) | 225 / 539 | 8.96 | 0.000 % | 0.000 % | +1.32 % |

Item 2 is kept on because it is the ONLY guard on the non-momentum directions
(lambda, phi, d0, z0) — the residual 1.15e6 chi2 tail in the legacy variant is
not a q/p runaway — and because it is genuinely inert on healthy fits at slack
1.0. A production that only cares about the muon-pair mass may set
`stepBacktracking=False` and recover the 1.3 %.

**Full re-production `_260905d_m0`** (gun, 300 370 attempted), against the two
earlier vintages:

| | floor 2.0 | floor 0.25 | **step damping** |
|---|---|---|---|
| clamped fits | 36 863 (12.27 %) | 480 (0.16 %) | **112 (0.037 %)** |
| **frac clamp at scale = 0** | **99.52 %** | **79.58 %** | **0.00 %** |
| clamp scale q05/q50/q95 | 0 / 0 / 0 | 0 / 0 / 0.348 | 0.0012 / 0.158 / 0.856 |
| chi2/ndof > 10 | 12.161 % | 0.143 % | **0.014 %** |
| `frozen` | 7.025 % | 0.079 % | **0.007 %** |
| prop calls | 31.168 M | 33.925 M | 34.435 M (**+1.50 %**) |
| **UNCUT alpha [1e-3], single `r`** | +0.2110 +- 0.0167 | -0.0027 +- 0.0167 | **-0.0046 +- 0.0167** |
| UNCUT, families | +0.2447 +- 0.0177 | +0.0280 +- 0.0175 | **+0.0250 +- 0.0175** |
| `k_hit`, UNCUT families | 0.9426 +- 0.0353 | 0.9930 +- 0.0364 | **1.0008 +- 0.0366** |
| `k_ms` | 1.0108 +- 0.0061 | 0.9977 +- 0.0061 | **0.9958 +- 0.0062** |

The `chi2 < 3` cut is now a COMPLETE no-op (0.0001e-3) and `k_hit` lands on
1.000. `k_ioni` (0.6922 +- 0.0532) is the known ionization-tail
over-prediction and is unrelated.

**Failure taxonomy with gen truth** (RUN = the state was driven below half the
seed momentum, i.e. a Gauss-Newton runaway; STOP/HARD = the state agrees with
the seed, seed below/above 1.5 GeV): aborts 441 -> 1773 -> 2213, of which
**RUN 3 (0.68 %) -> 17 (0.96 %) -> 10 (0.45 %)**. The damping leaves only the
physically unfittable class: the absolute runaway count DROPS while the total
rises, because the extra failures are all sub-300-MeV seeds (113 -> 158 ->
617) — muons that genuinely stop in 3-5 m of tracker and that the 2 GeV floor
used to "protect" by freezing them at 2 GeV, i.e. by returning a BIASED fit
instead of a failure. The HARD class (median `p_seed` 18.2 GeV, path 884 cm) is
the known runaway-forward-leg class, flat across all three vintages.

Final defaults (in `production/PRODUCTIONS.md`): `maxMomentumStepFactor=2`,
`stepBacktracking=True`, `stepBacktrackFromIter=2`, `maxChi2Backtrack=4`,
`armijoC=1e-4`, **`armijoSlack=1.0`**, `stepPrintLimit=200`,
`clampMomentumFloor` derived = 0.25 GeV. Re-gated after the rebuild: legacy
settings still 212/212 + 137/137 BIT-IDENTICAL; new defaults single-track
BIT-IDENTICAL, two-track differing on exactly 1 of 29 candidates in 3
`doRes`-export branches only.

Production still in use: `resolution_trackres_{jpsigun_ul16,mugun_lowpt}_260905d_m0`.
Full record: the archived dev log, entry **2026-09-05** (step damping). Figures:
`~/public_html/ZMass/cvh/260905_stepdamp/`.

---

## `cvhcf260905` — the resolution-CF exponents move into the maker

**What it replaced.** The four per-candidate log-CF exponents
(`phi_z(t) = phi_hit exp(S_ms + S_ioni + S_rad + S_del)`) used to be built
OFFLINE from a raw export of every Geant4 step record: **165 kB/track and
329 kB/candidate, 2.2 s each**, i.e. 16 TB and 24 k core-hours at the 40 M
candidates of the full calibration, all to produce 6 x 64 floats. They are now
computed in the maker by `cvhcf::trackExponents`
(`TrackPropagation/Geant4e/{interface,src}/CvhCfExponents.{h,cc}`), reusing
`cvhcgf::blockExponent` for the ionization and radiative channels rather than
transcribing them; the Moliere exponent is the new code.

**Headline, all measured:**

| | value |
|---|---|
| evaluator vs the offline reference, `max_t \|dS\|` | **1.19e-11** (single track) / **1.62e-11** (two track), requirement 1e-6 |
| the floats the MAKER wrote | worst **9.5e-7** = the float32 storage floor at those `\|S\|` (1.26e-6); every other family 100x below |
| end to end, 1891 gun ditrack candidates, same grid | **d(alpha) = 2e-10** in the fit's 1e-3 units = **2e-9 sigma**; every `k` within 1e-8 |
| the 64-vs-448-point grid change | d(alpha) = -2.0e-6 (model `r`) / -4.3e-6 (families) = 2e-5 sigma |
| nothing else moved | 137/137 and 212/212 pre-existing branches BIT-IDENTICAL vs a pre-change reference binary, with 12 new branches in each |
| volume, two-track, `exportStepRecords=False` | **448 kB -> 119 kB per candidate** (54 % of the file) |
| cost | **2.2 s -> 96 ms** per candidate (41.7 ms per single track), a 53x speedup, at 4-10 % of a fit that costs O(1 s) |

The caches agree float32-BIT-IDENTICALLY except one ulp in `Sio_re` and `vgf`
(`Sms`, `Sio_im`, `Srad_re/im`, `z`, `sigma` all exactly 0.0 difference).

**Three decisions worth keeping.** (a) The maker's 64-point tau grid is a
SUBSET of the offline 448-point one (`tau_i = 4 i (14/447)`,
`tau_63 = 7.8926` = `TG[0:256:4]`), so the two can be compared at the same
argument with no interpolation in between — which is what makes the 1e-11 a
measurement of the evaluator rather than of an interpolator. (b) The Moliere
shape tables are LOADED from the reference's own dump
(`data/cvhcf_gshape_elec_v1.bin`), not rebuilt: libstdc++'s
`std::cyl_bessel_j` is not the Cephes `j0` scipy uses, and a C++ rebuild would
differ by more than the tolerance the exponent is held to. (c) The evaluator
reads the EXPORT ARRAYS, not the propagator's logs, so the in-maker pooling is
identical by construction to the offline `extract()` join.

**ONE SWITCH MUST AGREE ON BOTH SIDES.** `cvhcf` exports the Kokoulin-OFF
ionization model, which is what the production builds; `cf_track_resolution`
defaults to Kokoulin ON when the environment is silent. `cfmodel` in the file
says which model it is.

To reproduce:

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
export CMSSW_SRC=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
cd /work/submit/david_w/ZMass/calibration_studies/resolution
./cxx/build_cvhcf.sh                                   # the ctypes shim
python3 cvhcf_validate.py --file <globalcor.root> --ntracks 40 --compare-branches --bench 20
python3 cvhcf_size_260905.py <globalcor.root>          # volume
./cvhcf_e2e_260905.sh refit pairs fit compare          # the end-to-end gate
```

Open, and deliberately so: `exportStepRecords` is still `True` by default (flip
it per campaign, once the model for that campaign is settled — once it is off, a
model change means re-running the fit); the three-track maker
(`ResidualGlobalCorrectionMakerNTrackG4e`) has not been given the export;
ionization regimes 0 and 1 are not exercised by these samples (all regime 2) and
are covered by `cgf_cxx_validate.py` instead.

Full record: the archived dev log, entry **2026-09-05** ("the per-candidate CF
exponents move into the maker"); the maker-side contract is
`Analysis/HitAnalyzer/doc/resolution-cf-export.md`.

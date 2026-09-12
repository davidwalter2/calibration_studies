# STATE_beamline — the BEAM-LINE constraint in the CVH two-track fit (Z -> mumu)

Working checkpoint. Folds into `STATE.md` section 14 at the end.

## Area

| what | where |
|---|---|
| build area | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3` (copy of dev2, own git worktree) |
| branch | `beamline-260913`, off `cvh-exports-clean-260911` @ `23b4c9c7046` |
| maker | `src/Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc` |
| outputs | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/` |
| figures | `~/public_html/ZMass/cvh/260913_beamline/` |

dev2 is BUSY (6 cmsRun DY jobs on submit50/51 from the other agent, started ~11:40).
Never build there. dev3 created 2026-09-12 13:08, builds clean (no-op `scram b` OK on submit52).

## Log

- 13:10 area created and verified.
- 13:31 maker changes built clean in dev3.
- 13:53 four productions launched (see below).
- commit `a6169b7e0d0` on `beamline-260913`.

## THE DOUBLE EMISSION — the finding

`ResidualGlobalCorrectionMakerTwoTrackG4e.cc`, the `if (bsConstraint_)` block
sat INSIDE `for (unsigned int id = 0; id < 2; ++id)` with **no `id == 0`
guard**, while the pointing constraint immediately below it HAS one
(`if (doPointingConstraint_ && id == 0)`, with the comment "Applied once per
iteration (guarded with id == 0) since it is intrinsically a 2-track
constraint, not per-daughter").  The beam block is equally a pair-level
constraint and was not guarded.  Both emissions write

* the SAME residual `dbs0` (with `doVtxConstraint=True` the two legs'
  reference points are both the common vertex, so the vectors are identical),
* the SAME Jacobian `Fbs = Identity` on the SAME state indices 7,8,9,
* the SAME weight `covBSinv`,

and `chisq0val += bschisq` ran twice.  Two identical rows with covariance `S`
are ONE row with `S/2`, so the effective luminous-region covariance was
**halved** and the beam chi2 **double-counted**.  The row budget said so
explicitly: `nbscons = bsConstraint_ ? 3u * 2u : 0u`, with the comment "3
beamspot rows per track".  Meanwhile `ndof` counted `+3` ONCE
(`if (bsConstraint_) { ndofsigned += 3; }`), so the fit was internally
inconsistent: six constraint rows, three degrees of freedom.

A SECOND defect in the same block: the residual was `refFts[0..2] - b0` with
`refFts` the LEG's reference point.  `twoTrackPca2cart` puts the two
reference points at `x_v -+ (d/2) n_hat`, so with index 6 free that is NOT
the vertex and `Fbs = Identity` on 7,8,9 is not its Jacobian.  Fixed to the
MIDPOINT, which is `x_v` identically in both regimes (and its Jacobian is
exactly `I` on 7,8,9, zero on index 6 and on the momenta).  With
`doVtxConstraint=True` and the DCA frozen at zero -- every production -- the
two coincide, so this defect is latent, not active.

`bsConstraint` is `False` in EVERY cfi in the package and in every
production, so nothing shipped was affected.

## The MC beam spot vs the simulated luminous region  (study item a)

Read off `Jpsigen_x/y/z` (the gen production vertex) against the
`offlineBeamSpot` record the maker actually used, exported per candidate as
`Jpsi_bsspot` / `Jpsi_bswidth` / `Jpsi_bsslope`.

DY MiniAOD MC (`dy_vtxon` + the beam smoke, 10 507 gen-matched candidates;
`Jpsigen_*` is **-99 on ~2-4 %** of candidates -- the no-gen-match sentinel,
written on all three at once, which is why a naive rms is meaningless):

| | record | simulation | |
|---|---|---|---|
| x0 | +916.46 um | +916.78 um | **+0.32 um** |
| y0 | +1695.37 um | +1696.35 um | **+0.98 um** |
| z0 | +0.8819 cm | +0.9332 cm | +0.051 cm (1.5 sigma of the mean) |
| sigma_x | 10.83 um | **9.66 um** (1 % trimmed) / 10.10 um (MAD) | record 7-12 % WIDE |
| sigma_y | 10.39 um | **9.59 um** (1 % trimmed) / 9.92 um (MAD) | record 5-8 % WIDE |
| sigma_z | 3.6239 cm | 3.6257 cm | ratio 1.000 |
| dxdz | -5.97e-6 | -1.2e-5 +- 1.0e-5 | consistent |
| dydz | +4.72e-6 | +5.2e-5 +- 2.6e-5 | consistent (1.8 sigma) |

The raw rms of the gen x / y is 39 / 98 um against a MAD of 10.1 / 9.9 um:
**0.05 % of candidates carry a genuinely displaced gen vertex** (a leg matched
to a muon from a heavy-flavour decay), and they are the entire difference.

**The record describes the simulated luminous region.** The centroid agrees to
under a micron, sigma_z to 0.05 %, the slopes within their errors; the
transverse widths are the one mismatch and the record is **7-12 % WIDER** than
the simulation, i.e. the constraint as configured is slightly LOOSE -- a
conservative direction, and a 10 % effect on a term that carries ~20 % of the
beam functional's variance. It is quoted, not corrected: the fit uses the
record, which is what a DATA fit would also use.

### The J/psi gun

`prod_vtxon` (14 969 gen-matched of 14 973): gen vertex median
(+0.091635, +0.169544, +0.9845) cm with MAD (10.12, 9.95 um, 3.653 cm) --
**the gun IS smeared with the same beam spot**, to the same agreement as DY.
So the gun is technically a valid sample for the beam rows, and it is prompt
by construction. It is nevertheless LEFT OUT of the physics conclusions,
because a real J/psi sample is not prompt (the B-decay fraction) and the gun
would certify a constraint that data could not carry.

## The productions

All six read the SAME six DY MiniAOD files
(`production/filelist_dymc_8p5M_260905.txt` lines 1-6) with the
`condor_dymc_v2` configuration verbatim plus `exportVtxResidual=True`, so
every comparison is same-candidate.  Under
`/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/`.

| tag | build | rows | events/task | what for |
|---|---|---|---|---|
| `dy_bs` | dev3 | ON, nominal widths | 4000 | THE STUDY SAMPLE |
| `dy_bsoff` | dev3 | OFF | 4000 | the same-candidate reference (has `Jpsi_covvtx`) |
| `dy_bsoff1200` | dev3 | OFF | 1200 | the gates' reference |
| `dy_bswide` | dev3 | ON, `beamWidthScale=1e6` | 1200 | gate G2: weightless == off |
| `dy_bshalf` | dev3 | ON, `beamWidthScale=1/sqrt(2)` | 1200 | gate G4: == the OLD build |
| `dy_bsold` | **dev2** (the old maker, run only, never built) | ON, nominal | 1200 | gate G4 |

`dy_vtxon` (the earlier rows-OFF production) is NOT used as the reference:
it predates `Jpsi_covvtx`, which the leave-one-out gate needs, and it was
written by a different build.

`run_prod_bs.sh` (with `BSOPTS` / `EXTRA`) and `run_prod_bs_old.sh` (which
sed-patches a COPY of dev2's driver into the job temp dir -- nothing is
written into dev2).

## The export bill, measured (smoke file, 90 candidates)

| block | kB/cand | TB at 7 M Z |
|---|---|---|
| beam: exponents per group | 60.26 | 0.422 |
| beam: exponents flat | 2.82 | 0.020 |
| beam: shares + scalars | 1.77 | 0.012 |
| beam: influence `a_b` | 6.89 | 0.048 |
| beam: the mean-term weights | 0.04 | 0.0003 |
| **beam total** | **71.77** | **0.502** |
| vertex total (for scale) | 36.80 | 0.258 |
| mass total (for scale) | 36.90 | 0.258 |

Two functionals, so the beam block is **1.95x the vertex block** -- it is
almost exactly two copies of it, as it should be.  The whole file is
181.9 kB/cand with all three functionals on.

## The two functionals, and their basis  (a decision, stated)

The maker exports THREE things about the same 2-vector:

* `Jpsi_bsres` (cm) and `Jpsi_bscov` (cm^2, packed xx/xy/yy) -- the leave-one-
  out residual in the GLOBAL transverse frame and its covariance;
* `Jpsi_bsz` -- the LOWER-CHOLESKY whitened pull, `Cov = L L^T`, `z = L^-1 r`,
  in the order (x, y), so `z[0]` is the x pull and `z[1]` the y pull GIVEN x.
  The basis is a choice; the chi2 `Jpsi_bschi2 = z^T z` is not.

**The two CF FUNCTIONALS are the two GLOBAL components** `r_x` and `r_y`, each
standardised by its OWN marginal `sqrt(Cov_kk)` -- not the Cholesky pair.
Two reasons, and one consequence:

* they stay in interpretable units, tied one-to-one to the beam-spot
  parameters `x0` and `y0` (the mean term is then literally `-w[bs row x]`),
  and to the projector identity `Jpsi_bsmeanbs == -P`, which is what makes the
  gate analytic;
* a Cholesky-basis functional would be a candidate-dependent mixture of the
  two, so its influence weights could not be read as a response to a single
  beam-spot parameter.

The consequence is that **the two functionals are CORRELATED** (`Cov_xy` is
not zero; on the smoke sample the correlation is ~0.20), so the two terms may
not simply be multiplied and their QUOTED (Hessian) errors are optimistic.
That is handled the way the vertex-mass joint already is: `fisher_vtx.py`
sums the two terms' per-batch gradients over the SAME candidates, so `J`
carries the within-candidate correlation and the SANDWICH
`(H+P)^-1 J (H+P)^-1` is the variance the estimator actually has. Every error
quoted from the beam channels is the sandwich one, and the
`sandwich/quoted` ratio of the `bs` channel is exactly the over-counting
diagnostic.

**Why one component is much worse measured than the other.** `Cov(r_bs)` is
`C_{-B} + Sigma_{xy|z}`: the vertex covariance WITHOUT the beam rows plus the
luminous region's own transverse spread. Two nearly back-to-back tracks fix
the vertex well PERPENDICULAR to their common direction and poorly ALONG it,
so for a Z pair one transverse direction has `sigma ~ 15 um` (the beam width
itself) and the other can reach hundreds of microns. On the smoke sample the
median `sigma_bs,x` is 25.6 um with a p95 of 248 um. The global (x, y) basis
mixes the good and the bad direction by the pair's phi, which is why the two
pulls look alike in the ensemble even though per candidate they are not.

## The gates  (`gates_bs.py`, the 1200-event legs, 3287 candidates, 3153 on the baseline)

The BASELINE, applied to every gate and to the study: `ndof > 0`,
`chi2/ndof < 3`, `|Jpsi_vtxvchk| < 1e-4`, finite `Jpsi_sigmamass` and
`Jpsi_vtxsig`, and **>= 8 valid hits on the weaker leg** (`minLegHits`, the
gen-background study's recommendation). Every one is a cut on the FIT'S OWN
covariance or arithmetic; none is a cut on a residual.

| gate | measured | |
|---|---|---|
| **G1** `ndof(ON) - ndof(OFF) == 3` | **3 x 3151 of 3151** (unselected: 3 x 3282, plus 2 mis-paired multi-candidate events) | PASS |
| **G2** `beamWidthScale=1e6` reproduces rows-OFF | median **0** (bit-identical) on every export; p99 <= 1.2e-6, p99.9 <= 1.8e-5, **max 6.5e-5** (`Jpsi_vtxz`); `ndof` difference exactly 3 | PASS |
| **G3** the beam chi2 enters ONCE | recomputed / exported `Jpsi_bschi2fit` agree to **3.7e-6** (median), 3.8e-4 (max); chi2 at the linearisation point vs at the optimum agree to 0 (median) | PASS |
| **G4a** the OLD build == the NEW build at `beamWidthScale = 1/sqrt(2)` | 3153 candidates, median **0** on every branch, **max 1.96e-5**, `ndof` difference **exactly 0** | PASS -- **the rows DID enter twice** |
| **G4c** the closure the old build cannot have | OLD `Jpsi_vtxvchk` median **0.0217**, p90 0.132; NEW **9.0e-9**; rows-OFF 8.7e-9 | the independent symptom |
| **G5** `sum_b abs(a_b)^2 == Cov_ii` | `Jpsi_bsvchk` median **8.4e-9**, max 2.9e-5; `resinfbsv` vs `bsvarv` per block **1.7e-7**; the VERTEX functional WITH the beam block registered **8.8e-9**, max 1.8e-6; the MASS functional (material + hit + beam)/sigma_m^2 **1.3e-7**; the three group closures **~5e-16** | PASS |
| **G6** `Jpsi_bsmeanbs == -P` (analytic) | median **7.9e-14**, max **2.3e-11** | PASS |
| **G7** the leave-one-out identity vs the rows-OFF fit | `abs(r_bs(ON)-r_bs(OFF))/sigma` median **1.17e-3**, p90 4.5e-3; `abs(z_bs(ON)-z_bs(OFF))` **1.60e-3** / 7.4e-3; `abs(Cov(ON)-Cov(OFF))/abs(Cov)` **1.1e-3** / 4.7e-3 | PASS -- the vertex residual's own gate is 1.37e-3 / 1.31e-2 |

**G2's residual tail is not the constraint.** With `beamWidthScale = 1e6` the
three rows carry a weight ~1e13 times smaller than a hit's, but they are
STILL AT THE FRONT of the row list, which shifts every subsequent row index
and changes the sparse-LDLT elimination order -- and hence the last bits.
`sigma_v^2 = 1/(h_66 - h_f6^T C h_f6)` is a difference of two nearly equal
numbers, so it is where that shows: 1.2 % of candidates move by more than
1e-6 and the worst by 6.5e-5.

## Cost  (`run_timing_bs.sh`: 400 events, ONE file, ONE host, run SEQUENTIALLY)

186 candidates in all three configurations -- the beam rows change no
candidate's selection.

| configuration | wall | s/candidate | file | kB/candidate |
|---|---|---|---|---|
| `bsConstraint=False` | 886 s | 4.76 | 33.79 MB | 177.4 |
| `+ the three rows` (`exportBsResidual=False`) | 895 s | 4.81 (**+1.0 %**) | 34.27 MB | 179.9 |
| `+ the two functionals` | 1440 s | 7.74 (**+62.6 %**) | 48.13 MB | 252.7 (**+75.3**) |

**The rows are free; the two functionals are not.** The extra 62 % is two
more `cvhcf::trackExponents` calls and two more per-block influence loops --
and each influence loop redoes the `SelfAdjointEigenSolver` of every block's
`dV_b` from scratch. The square root is a property of the BLOCK, not of the
functional: caching it once per block would serve all four functionals and
should remove most of the 62 %. (Not done here -- it changes the mass and
vertex paths too, and this study is not the place to re-certify them.)

At **7 M Z candidates**: the beam block costs **0.527 TB** of export and
**~5.8 kh** of extra CPU over the rows-off configuration.

## WHAT THE BEAM ROWS BUY -- and it is not what was expected

Same-candidate, rows ON against rows OFF, on the controlled pair
(`beamline/timing/{full,off}`: one file, 400 events, 186 candidates, 176 on
the baseline; the 4000-event legs repeat it with ~10 k). `cmp_bson.py`.

**1. The constraint is attractive and adds information, as it must.**
`|x_v - beamspot|` in the transverse plane goes **32.3 -> 6.2 um** (median).
`sigma_m(ON)/sigma_m(OFF) <= 1` on **every** candidate (max ratio 0.994).
The fitted vertex error goes 21.8 -> 9.0 um in x and 24.6 -> 8.9 um in y
(the beam widths are 10.8 / 10.4 um, and 21.8 combined with 10.8 is 9.7 um).
`sigma_z` goes 45 -> 35 um even though the beam's `sigma_z` is 3.6 cm: that
is the transverse constraint propagating through the track directions, not
the beam's z row.

**2. `sigma_m` improves by 4.6 % (median) / 6.5 % (mean)** -- 12.6 % in
variance -- against the vertex constraint's 1.9 %. Flat in the softer muon's
gen `pT` (4.5-9.0 % across the range, no trend).

**3. The gain is on the CURVATURES, not the opening angle.** `Jpsi_fang`,
the angular share of `sigma_m^2`, is **0.0005**: the Z mass error is 99.95 %
the two curvatures. And indeed `sigma(p)/p` improves by **3.4 % / 3.0 %**
(median, mu+ / mu-), and the `sigma_m` ratio PREDICTED from the two
curvatures and their correlation alone,

    sigma_m^2 / m^2 = 1/4 (s_+^2 + s_-^2 + 2 rho s_+ s_-) ,

is **0.95360** (median) against the measured **0.95355** -- agreement to
5e-5. The beam spot acts as an extra measurement at `r ~ 0` with 10 um
resolution: the longest possible inward lever arm, which is exactly what a
curvature wants. `rho(p_+, p_-)` drops 0.053 -> 0.031.

**4. The vertex residual is NOT unchanged.** `|z_v(ON) - z_v(OFF)|` has
median **0.247** and max 3.1, and `sigma_v(ON)/sigma_v(OFF)` is 0.935
(median, and <= 1 on 98.9 % of candidates -- the 1.1 % above are at 1.0016,
numerical noise). The premise that "the beam line constrains the vertex
POSITION and the DCA is a different direction" is WRONG for the two-track
PCA: index 6 (the DCA) and indices 7-9 (the vertex position) are separate
COORDINATES but the hits couple them, so `h_f6` has entries on 7, 8, 9 and
`sigma_v^2 = 1/(h_66 - h_f6^T C h_f6)` moves when `C` does. Both shifts are
exactly the size conditioning predicts:
`rms(m_ON - m_OFF) = sqrt(sigma_OFF^2 - sigma_ON^2)` and
`median |dz_v| ~ sqrt(1 - (sigma_ON/sigma_OFF)^2) = 0.355 x 0.67 = 0.24`.

**5. The beam-spot MEAN TERM.** Per candidate the response to a 5 um
centroid shift is large -- **rms 78.5 MeV on the mass, 0.066 sigma_m** -- but
it is `phi`-random and cancels: the ENSEMBLE MEAN is
**-4.3 +- 5.9 MeV** (176 candidates), i.e. consistent with zero and bounded
at the 12 MeV level. The same shift moves the BEAM PULLS **coherently**
(mean -0.194 per 5 um, equal to `mean|.|`), which is the statement that the
two beam residuals ARE a measurement of the centroid: with N candidates
`sigma(x0) = 5 um / (0.19 sqrt(N))`, i.e. **26 um / sqrt(N)** -- 0.01 um at
7 M. A `1e-4` slope change gives mass rms 54 MeV, ensemble mean
-1.2 +- 4.1 MeV, and a beam-pull rms of 0.15.

So: the beam-spot parameters are OVER-determined by the residuals they
create, the per-candidate response is a resolution effect and not a scale
one, and whether the centroid must FLOAT is decided by the ensemble mean,
which the 10 k sample pins to ~0.8 MeV per 5 um.

## G4b -- THE SIZE OF THE DEFECT, measured

The correct build at nominal widths against the OLD (double-emitting) build,
3153 same candidates, relative differences:

| branch | median | p99 | max |
|---|---|---|---|
| `Muplus_pt` | **5.6e-4** | 5.0e-3 | 4.4e-2 |
| `Muminus_pt` | 5.6e-4 | 4.6e-3 | 2.4e-2 |
| `Jpsi_mass` | **5.9e-4** (54 MeV at the Z) | 3.9e-3 (356 MeV) | 2.3e-2 |
| `Jpsi_sigmamass` | **1.2 %** | 3.3 % | 11.7 % |
| `Jpsi_vtxz` | 5.2e-2 | 1.23 | 1.98 |
| `Jpsi_x` | 1.4e-3 | 7.2e-3 | 1.6e-2 |
| `chisqval` | 2.8 % | 18 % | 28 % |
| `ndof` | **0** (the old build counted ndof right; only the WEIGHT was wrong) | | |

A **5.6e-4 median shift of the fitted muon momentum** -- 5.6x the W-mass
target and 56x the Z-mass one. `bsConstraint` is `False` in every cfi and
every production, so nothing shipped carried it; but it could not have been
turned on.

## The nominal pulls, 1200-event leg (3157 candidates on the baseline)

| | mean | Var | trimmed Var | P(abs(z)>3) | P(abs(z)>5) |
|---|---|---|---|---|---|
| `z_bs,x` | +0.0088 +- 0.0195 | 1.202 | 1.132 | 1.27e-2 | 1.58e-3 |
| `z_bs,y` | +0.0017 +- 0.0183 | 1.055 | 0.990 | 6.3e-3 | 9.5e-4 |

**The mean is zero**, as it must be for a constraint residual. `Var > 1`
means the residual is WIDER than the nominal (Gaussian, fit-`Q`) sigma --
and the beam-spot record being 8-12 % WIDER than the simulated luminous
region pushes the other way (correcting it would take `Var(z_x)` from 1.20 to
~1.27), so the excess is in `C_{-B}`, the fit's own vertex covariance: the
same statement as "the fit's `Q` is Rossi and 14 % low". Which is what the CF
term is for; the CF data/model ratio is the number to quote, not `Var`.

Family composition, nominal: **beam line 0.225 / 0.214, hit 0.604 / 0.613,
MS 0.172 / 0.174, ionization 0.000** (x / y). The beam block carries
**4.0 %** of `sigma_m^2` and **6.2 %** of `sigma_v^2`.

## Checkpoint 2026-09-12 18:10

`dy_bs` (4000 ev x 6) and `dy_bsoff` (4000 ev x 6) COMPLETE at 17:32.
The 1200-event gate legs all complete; the gates pass (above).
dev3 @ `a6169b7e0d0`, built 13:52, working tree clean.
`cvh-exports-clean-260911` has moved to **`dbfe6e4b2c2`** ("minLegHits = 8 by
default") -- the rebase target. My productions ran with `minLegHits = 0` and
the cut is applied OFFLINE in the baseline, which is the same selection.

NEXT: the study chain on `dy_bs` (`run_all_bs.sh all`), then the rebase.

## THE STUDY, 10 254 candidates (`dy_bs` 4000 ev x 6, against `dy_bsoff`)

### The two pulls (`plot_vtx.py`, figures in `~/public_html/ZMass/cvh/260913_beamline/`)

| | N | mean | Var | skew | kurt | corr(sigma, z) |
|---|---|---|---|---|---|---|
| `z_bs,x` | 10254 | **+0.0128 +- 0.0107** | 1.1810 | +0.062 | 5.00 | +0.017 +- 0.010 |
| `z_bs,y` | 10254 | **+0.0043 +- 0.0107** | 1.1810 | +0.011 | 4.92 | +0.009 +- 0.010 |
| `z_v` (for scale) | 10254 | +0.0078 +- 0.0104 | 1.1165 | -0.031 | 3.55 | +0.003 +- 0.010 |

**The mean is zero and there is no skew** -- the two beam residuals are
constraint residuals of exactly the vertex kind. `corr(sigma, z)` is
consistent with zero, so no self-consistent-sigma correction is needed
(the same justification the vertex term uses).

Tails, data / model:

| | 2 sigma | 3 sigma | 4 sigma | 5 sigma |
|---|---|---|---|---|
| `z_bs,x` CF | 1.35 | 3.57 | 10.9 | **16.0** |
| `z_bs,x` Gaussian (variance-matched) | 1.34 | 4.33 | 54.4 | 2621 |
| `z_bs,x` Gaussian (the fit's `Q`) | 1.37 | 4.59 | 60.1 | 3061 |
| `z_v` CF (for scale) | 1.32 | 2.10 | 1.09 | -- |

The CF beats the Gaussian by **160x at 5 sigma** and is still **16x short**.
The vertex residual's CF closes at 5 sigma; the beam one does not. The tail
is NOT the background (the displaced `otherdecay` class is 8 candidates and
contributes 1e-4 of the 1.8e-3 total): it is candidates where
`M = covBS - C_vtx` is nearly singular -- the back-to-back direction in which
the two tracks barely constrain the vertex -- so the leave-one-out
amplification `covBS M^-1` is large. That is a property of the construction
and it is the one place where the beam term is worse described than the
vertex term.

### Composition (median share)

| | beam line | hit | MS | ionization |
|---|---|---|---|---|
| `z_bs,x` | **0.212** | 0.603 | 0.141 | 0.000 |
| `z_bs,y` | 0.200 | 0.609 | 0.143 | 0.000 |
| `z_v` | -- | 0.656 | 0.286 | 0.000 |

Material: `bpix_support6` 0.067, `tib_support` 0.024, `bpix_services` 0.013,
`fpix_support` 0.012, `bpix_active_L1` 0.009 -- the same INNER-tracker weight
as the vertex residual (which has 0.118 / 0.039 / 0.019), diluted by the
beam block's own 0.21. Hit classes: `pix_x_q1` 0.100, `pix_y_q1` 0.097,
`str_N3_lo` 0.049 -- against the vertex residual's much more concentrated
`pix_x_q1` **0.258**, `pix_x_q2` 0.102, `pix_x_q3` 0.077. So the beam
residual sees the innermost pixel classes in BOTH local coordinates where the
vertex residual sees local-x only: the DCA direction `n_hat` is one
direction, the beam residual is two.

The beam block carries **4.0 %** of `sigma_m^2` and **6.1 %** of `sigma_v^2`.

### Against gen truth (`bkg_bs.py`, classes from `genbkg.classify`)

After the baseline: signal 10214 (99.61 %), unmatched 31, otherdecay 8,
dup 1 -- **0.39 % background**.

| class | n | <z_x> | Var z_x | P(abs(z)>3) | P(abs(z)>5) | <chi2_bs> |
|---|---|---|---|---|---|---|
| signal | 10214 | +0.014 | 1.176 | 0.0210 | 0.0030 | 2.29 |
| **otherdecay** | 8 | **-1.24** | **6.10** | **0.250** | **0.125** | **9.97** |
| unmatched | 31 | -0.074 | 0.972 | 0.000 | 0.000 | 1.77 |

**The beam residual sees exactly the class it should**: `otherdecay` -- a leg
matched to a muon from a different (heavy-flavour, DISPLACED) decay -- has 25 %
of its candidates beyond 3 sigma against the signal's 2.1 %. The `unmatched`
class is PROMPT (a pileup muon or one below the gen-pruning threshold) and the
beam line cannot see it.

But **on top of the recommended baseline the beam pulls buy no rejection**,
because what survives the baseline is prompt:

| cut | eff(signal) | eff(bkg) | rejection | signal loss |
|---|---|---|---|---|
| `abs(z_bs) < 3` | 0.97905 | 0.950 | 0.050 | 0.0210 |
| `abs(z_bs) < 5` | 0.99696 | 0.975 | 0.025 | 0.0030 |
| `abs(z_v) < 5` | 1.00000 | 1.000 | 0.000 | 0.0000 |
| beam-row chi2 < 16 | 0.99951 | 1.000 | 0.000 | 0.0005 |

WITHOUT the baseline (3185 candidates of the 1200-event leg, 17 background):
`abs(z_bs) < 5` rejects **53 %** for a 0.73 % signal loss and the beam-row
chi2 < 16 rejects 35 % for 0.095 %, while `abs(z_v) < 5` rejects 0 %. So the
beam residual IS a powerful tag for the pathological candidates -- the
recommended baseline simply catches the same ones first.

**Cosmic-like pairs**: `abs(dphi - pi) < 0.05 AND abs(eta+ + eta-) < 0.05`
tags 21 of 10254 (0.20 %), ALL of them gen-signal, with NARROWER beam pulls
than average. That cut selects a Z produced at rest, not a cosmic: DY MC has
no cosmic background by construction, so this is a null test and it has to be
repeated on data.

### Same-candidate ON vs OFF, 10 237 candidates (`cmp_bson.py`)

* `ndof(ON) - ndof(OFF) = 3` on **10 237 / 10 237**.
* **`sigma_m(ON)/sigma_m(OFF) = 0.9434` (mean) / 0.9580 (median)** -- 5.66 %
  in sigma, **11.0 % in variance**, FLAT in the softer muon's gen `pT`
  (5.37-6.26 % across the whole range). The vertex constraint, for scale,
  buys 1.9 %.
* **The mechanism is the CURVATURES.** `Jpsi_fang`, the angular share of
  `sigma_m^2`, is **9e-5**. `sigma(p)/p` improves **2.98 %** on each leg
  (median). The `sigma_m` ratio predicted from the two curvatures and their
  correlation alone is **0.95802 / 0.94351** (median / mean) against the
  measured **0.95795 / 0.94351** -- the mean agrees to five decimals. The
  beam spot is an extra measurement at `r ~ 0` with 10 um resolution: the
  longest possible inward lever arm, which is what a curvature wants.
* `rms(m_ON - m_OFF) = 861 MeV` against the predicted
  `sqrt(sigma_OFF^2 - sigma_ON^2) = 804 MeV`.
* **The mass MOVES**: `mean(m_ON - m_OFF) = -7.9 +- 3.6 MeV` (1 % trimmed;
  -7.0 +- 2.8 at 5 %, -7.3 +- 9.2 untrimmed), i.e. **-9e-5 relative**. It
  moves TOWARD the truth: `<m - m_gen>` goes **+58.6 +- 11.6 -> +53.1 +- 10.8
  MeV** (1 % trimmed). An 11 % variance reduction on a bias proportional to
  `sigma_m^2` predicts `0.11 x 58.6 = 6.4 MeV`; the measured shift is
  5.5-7.9 MeV. **So the shift is the resolution-proportional (Jensen-type)
  bias shrinking with the resolution, not a new bias** -- but at 9e-5
  relative it is far above the 1e-5 Z-mass target, so turning the rows on is
  NOT a neutral change and has to go through the calibration chain.
* **The vertex residual's DISTRIBUTION is unchanged** (`Var(z_v)` 1.12589 ON
  against 1.12593 OFF) but **candidate by candidate it is not**:
  median `abs(z_v(ON) - z_v(OFF))` = **0.194**, max 6.03, and
  `sigma_v(ON)/sigma_v(OFF)` = 0.944 in the median. The premise that the beam
  line touches the vertex POSITION and the DCA is a different direction is
  WRONG: index 6 and indices 7-9 are separate COORDINATES but the hits couple
  them, so `h_f6` has entries on 7, 8, 9 and
  `sigma_v^2 = 1/(h_66 - h_f6^T C h_f6)` moves when `C` does. Both shifts are
  exactly what conditioning predicts.

### The beam-spot MEAN TERM (10 237 candidates)

Per candidate, for a **5 um** shift of the centroid; `mean` is the ENSEMBLE
mean (a BIAS) and `rms` the per-candidate spread (a RESOLUTION effect):

| functional | mean | rms | mean/sigma |
|---|---|---|---|
| mass, `d x0` | **-0.41 +- 0.89 MeV** | 89.8 MeV | -8.5e-4 |
| mass, `d y0` | **-1.03 +- 0.96 MeV** | 97.3 MeV | -5.3e-4 |
| mass, `d x0` (relative) | -4.2e-6 +- 9.7e-6 | 9.8e-4 | |
| vertex `r_v/sigma_v`, `d x0` | -0.0015 +- 0.0008 | 0.082 | |
| beam `z_bs,x`, `d x0` | **-0.1994 +- 0.0009** | 0.092 | |

and for a **1e-4** slope change (`= the same weight x (z_v - z0)`,
`<abs(z_v - z0)> = 2.89 cm`):

| functional | mean | rms |
|---|---|---|
| mass, `d dxdz` | **-1.63 +- 0.63 MeV** | 63.2 MeV |
| mass, `d dydz` | -0.72 +- 0.66 MeV | 67.1 MeV |
| beam `z_bs,x`, `d dxdz` | -0.0018 +- 0.0016 | 0.157 |

**Read:** the per-candidate response is large (0.066 `sigma_m`) but
`phi`-random, so it cancels; the ensemble mean is consistent with zero and
bounded at **< 2 MeV per 5 um** of centroid and **1.6 +- 0.6 MeV per 1e-4**
of slope. Meanwhile the BEAM PULLS respond COHERENTLY (`mean = mean|.|`
= -0.199 per 5 um), which is the statement that **the two beam residuals ARE
a measurement of the beam-spot centroid**: `sigma(x0) = 5 um/(0.199 sqrt(N))
= 25 um/sqrt(N)`, i.e. 0.01 um at 7 M candidates. The parameters are
over-determined by the residuals they create, so floating them is free.

## The rebase, and the gates re-run on it

`beamline-260913` rebased onto `cvh-exports-clean-260911` @ **`dbfe6e4b2c2`**
("Two-track maker: minLegHits = 8 by default") -- **no conflicts**, the other
agent's edits are in other regions of the file. New head **`0cb6c291354`**.
Rebuilt clean in dev3.

Four fresh 700-event legs with the REBASED build (`rb_on`, `rb_off`,
`rb_wide`, `rb_half`; 1899 candidates) and `dy_bsold` unchanged:

| gate | measured |
|---|---|
| G1 | ndof difference **3 x 1866 / 1866** on the baseline, and **3 x 1899 / 1899** unselected -- with `minLegHits = 8` applied PRE-FIT the mis-paired multi-candidate events are gone |
| G2 | median **0**, max **6.5e-5** |
| G3 | recomputed/exported **4.5e-6** (median); linearisation point vs optimum **0** (median) |
| G4a | OLD == NEW at `1/sqrt(2)`, max **1.6e-5** |
| G4b | the defect: worst **1.95** (`Jpsi_vtxz`) |
| G4c | OLD `vtxvchk` 0.0217, NEW **9.1e-9**, OFF 8.8e-9 |
| G5 | `bsvchk` **7.0e-9**; `resinfbsv` vs `bsvarv` 1.7e-7; vertex **9.0e-9**; mass **1.3e-7**; group closures ~5e-16 |
| G6 | **7.9e-14** |
| G7 | **9.2e-4** / 1.2e-3 / 8.4e-4 (median) |

`logs_bs/gates_bs_rebased.log`. The `skipped[leghits<8]` counter fires 10
times in 700 events of `rb_on/task_0000`, so the new default is active.

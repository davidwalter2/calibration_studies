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

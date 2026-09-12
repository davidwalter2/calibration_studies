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

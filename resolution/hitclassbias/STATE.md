# hit-class LOCATION bias — STATE

Task: measure each hit class's residual LOCATION (not just its width/shape),
propagate the odd part through the per-hit influence weights with NO free
parameter, and compare against the measured track-level and mass-level odd
moments.

## Step 0 — provenance and the exact mechanism (DONE 2026-09-08)

### The two productions and why the study needs BOTH

No single production carries the truth residual AND the influence weight:

| production | `dxrecsim` (truth residual) | `resinfv/resinfbv` (influence) | class vars |
|---|---|---|---|
| `hitres*_<gun>` (`fitFromGenParms=True`, `doRes=False`) | YES | no | YES |
| `resolution_trackres_*` (`fitFromGenParms=False`, `doRes=True`) | no | YES | YES |

`fitFromGenParms=True` is what BOOKS the per-hit validation branches and
`doRes=False` is what removes the parmtype 8-11 blocks, so the two are
mutually exclusive by construction of the maker. The join is therefore on the
hit's own OBSERVABLES, which both trees carry: `hitDetId`, `hitUProj`,
`clusterSizeX`, `clusterChargeBin`, `clusterOnEdge`, `clusterSizeY`,
`hitPitch`, `hitThickness`, `localdxdz`, `localdydz`.

**The productions named in the brief** (`resolution_trackres_mugun260807*`,
`*_csvar` of 260829) have NO per-hit branches at all (`MISS=[dxrecsim, dxerr,
hitDetId, hitUProj, clusterSizeX, clusterChargeBin, reshitidx]`), so they
cannot be used. The usable pair, matched in gun kinematics (mu, pT 20-60 GeV,
the Z-momentum arm) is

* residuals: `hitres2_mugun_ul16` (40 tasks) [+ `hitres_mugun_ul16` as a
  statistics/duplication check], IDEAL geometry (`useIdealGeometry=True`,
  `run_local_hitres.sh`), default OAE field, GT `150X_mcRun2_asymptotic_v1`
  -> any nonzero residual LOCATION is a CPE bias, not misalignment;
* influence: `resolution_trackres_mugun_ul16_260903x_m0`, the tight-stepper
  cache whose per-eta charge-even odd moments (-4.88 / -3.58 / +0.14)e-3 are
  the target.

### The propagation, exactly

`delta(q/p) = sum_b w_b . n_b` with `v_b = w_b^T dV_b w_b` (maker header
`ResidualGlobalCorrectionMakerBase.h` ~line 897). Verified numerically on
track 0 of the trackres production: for rank-1 hit blocks
`v_b = w_b[0]^2 dxerr^2` to 5e-5 relative. Hence a location bias `mu_b` in
PULL units gives

        delta z_b = s_b sqrt(v_b)/sigma * mu_b = s_b sqrt(hitamp2_b) mu_b

with `s_b` the BENDING SENSE = the sign of the fit's q/p response to a
positive residual in that module's local frame. It is charge-INDEPENDENT
(signed curvature is a geometric property of the trajectory), which is why a
fixed local bias produces a charge-EVEN curvature bias -- the observed
symmetry of the target.

`s_b` must come from `resinfbv` (row 0 = `B_b[0,j] = s_b sqrt(v_b)` for a
rank-1 block), NOT from `resinfv[b][0]`: for a 2-D PIXEL block the parmtype-8
and parmtype-9 entries SHARE the block range, so `resinfv[b][0]` is the
local-x weight for both and using it for the y entry is wrong by a factor
200 (measured).

### Sizes already visible (2000 events of one `hitres2_mugun_ul16` file)

MEAN pull `(rec-sim)/sigma` per subdetector, ideal geometry:

| subdet | n | median pull | mean pull | rms |
|---|---:|---:|---:|---:|
| BPix | 3745 | +0.0786 | **+0.1125** | 1.022 |
| FPix | 741 | +0.0322 | +0.0613 | 1.038 |
| TIB | 7154 | +0.0253 | +0.0166 | 0.961 |
| TID | 3290 | +0.0594 | +0.0201 | 0.954 |
| TOB | 7809 | +0.0245 | +0.0120 | 1.003 |
| TEC | 11615 | -0.0020 | +0.0077 | 0.949 |

So the locations are NOT zero, they are subdetector-dependent, and BPix is
0.11 sigma_CPE. `hitres_classes.py:103` throws exactly this away.

## PRE-REGISTERED PREDICTIONS (written BEFORE the per-class measurement)

The coordinator's extra target: at fixed eta, the poorer-resolution half
gives m_Z **-40.9 +- 4.6** (barrel) against -4.6 +- 3.7 for the better half,
and **+65.9 +- 8.7** (endcap) against +34 for the band -- degraded-hit-content
tracks carry the bias in BOTH regions with OPPOSITE signs.

Stated in advance, from the mechanism alone:

* **P1** The bending-sense class location `<s mu>` is nonzero and COHERENT
  (does not average away over modules) in the barrel subdetectors, because
  the Lorentz drift there is `E x B` with `E` along the module normal
  (radial) and `B` along `+z`, i.e. a displacement of FIXED GLOBAL azimuthal
  sense, which is exactly the bending coordinate.
* **P2** Its sign for the DEGRADED classes (single-strip `N=1`, pixel
  edge / single-column, large `|uProj|`) is OPPOSITE between the barrel
  (BPix / TIB / TOB, local x = r-phi) and the forward subdetectors
  (FPix tilted blades, TID / TEC radial strips measuring local phi with the
  `yAxisOrientation` handedness flip).
* **P3** Therefore the propagated track-level `delta z` has opposite sign in
  the two regions for the degraded-hit-content tracks, reproducing
  -40.9 (barrel) against +65.9 (endcap).

P2 is the falsifiable one: if every subdetector's degraded classes have the
SAME bending-sense sign, the mechanism cannot produce the observed sign
reversal and the hypothesis fails on this point.

## Next
Step 1: extract per-hit residual locations + densities per class (18 classes
and the finer subdet / layer / side / local-coordinate / signed-angle key).

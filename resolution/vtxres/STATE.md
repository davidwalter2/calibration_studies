# vtxres — the VERTEX-CONSTRAINT RESIDUAL of the two-track CVH fit, as a CF term

## Purpose

The two-track CVH fit reports a signed track-track distance of closest approach
(DCA) at the common vertex. Its pull is a **pure resolution residual**: the
reference value is zero, so — unlike the mass residual — it needs no kernel, no
theory and no PDG input. This study builds it as a second `MaterialCFTerm` on the
same candidates and the same parameters as the mass term, and answers: does the
export close (gates)? what is the residual made of, and which material and hit
classes does it see? how much better does the full characteristic-function PDF
constrain them than the Gaussian chi2 ACTUALLY does (sandwich errors, never
nominal Fisher ones)? may it be multiplied into the mass term? and what does it
cost in CPU and bytes?

**The vertex constraint is now ON by default** (`doVtxConstraint=True`), and
the whole study was re-run in that regime: **section 12**. Sections 1-11 are
the FREE regime, which is what every production before 2026-09-11 used.

**The BEAM LINE is a third and fourth residual of the same kind**, and its
implementation carried a defect: **section 14**.

**Why have the term at all:** the vertex residual measures the **INNER** tracker
— beam pipe, BPix support and active layers, the innermost PIXEL hit classes —
where the mass residual measures the OUTER one (TIB/TEC/TOB support, the STRIP
hit classes). They are two projections of the same fit that see different
material, they are uncorrelated in the ensemble, and on the innermost pixel hit
classes the vertex term is **4-7x tighter in sigma** than the mass term on the
very same candidates.

---

## The object / model

The two-track CVH state is a 10-dim vertex PCA plus 5 parameters per hit.
`ResidualGlobalCorrectionMakerBase::twoTrackCart2pca` (`Base.cc:2649`) defines

    n_hat   = (p_a x p_b).normalized()          a = track 0, b = track 1
    theta_6 = d0 = n_hat . (x_b - x_a)          the SIGNED track-track PCA distance

and `twoTrackPca2cart` places the two reference points symmetrically about the
common vertex `x_v = statepca.tail<3>()`: `x_a = x_v - d/2 n_hat`,
`x_b = x_v + d/2 n_hat`. `doVtxConstraint_` (maker line 2092) removes index 6
from `freestateidxs`.

### The two regimes

**Index 6 FREE** (`doVtxConstraint=False` — what EVERY production uses): the fit
REPORTS the DCA, and

    sigma_v^2 = C_66 = covstate(6,6),     C = Cinvd^-1
    w_v       = Vinv F_f C e_6            (the mass functional's `wmass` with afull = e_6)
    r_v       = statepcaupd[6],   z_v = r_v / sigma_v

**Index 6 FROZEN** (`doVtxConstraint=True`): `b` is zero on every free index, and
with `F_6 = Ffull.col(6)`, `h_f6 = F_f^T Vinv F_6`, `Cs = C h_f6`,

    sigma_v^2 = 1 / (h_66 - h_f6^T Cs)
    b_6       = -(Vinv F_6) . r                  (minus the half-gradient)
    r_v       = sigma_v^2 b_6                    the unconstrained DCA, relative
                                                 to the FROZEN value
    w_v       = sigma_v^2 (Vinv F_6 - VinvF Cs)

Both give `sum_b |dV_b^{1/2} w_v,b|^2 = sigma_v^2` EXACTLY (one line:
`w_v^T V w_v`). That identity is gate (c).

### The term

The vertex term is the mass term with a **delta kernel at zero**:

    mobs = r_v (cm),   sigma = sigma_v,   m_ref = 0,   no kernel

Everything else is the same `MaterialCFTerm` on the same parameters: the
per-material-group exponent `S_f(tau) = S_f^fix + sum_g exp(k_g) S_{f,g}` and the
per-hit-class Gaussian share `v = v_other + sum_c (1 + eps_c) v_c`, which is what
lets a joint vertex+mass fit float ONE parameter set. Three arms:

| arm | what |
|---|---|
| `cf` | the exported log-CF exponents (the full PDF) |
| `gauss` | each (row, family) replaced by `-1/2 kappa2 tau^2` with kappa2 read off the SAME arrays; imaginary parts dropped (variance-matched) |
| `gaussq` | the variance the FIT used (`Gvqms`, `Gvqio` per group), nothing radiative or delta — i.e. the Gaussian chi2 the CF is measured against. `sum_g (vqms+vqio) + vgf == 1` exactly |

### Signs

`dxfree = -C F^T Vinv r`, so a noise perturbation `n` of the residual rows moves
the free state by `-C F^T Vinv n`: the TRUE influence on `theta_6` is `-w_v`, and
the same minus sits in the mass functional's `wmass` (where the mass code
hard-sets `ioniSign = -1` on top of it). The vertex functional has **no single
global ionization sign** — an energy loss moves the DCA either way depending on
geometry and charge — so it uses the PER-BLOCK `cvhcf::TrackInput::ressgn`
mechanism: for `fam == 11`, `u` is the leading eigenvector of `dV_b` oriented by
its qop component, the signed weight carries `-sign(w_b . u)`, and the leg CHARGE
multiplies it (`ResidualGlobalCorrectionMakerG4e.cc` ~5290). The convention is
**not asserted, it is GATED**: the same rule applied to `wmass` must return -1 on
every ionization block (the validated mass convention), and `Jpsi_vtxsgnchk`
reports the fraction.

### The exported branches (`exportVtxResidual=True`)

Scalars: `Jpsi_vtxres` (r_v, cm — the RAW theta_6, see the `Jpsi_d` defect),
`Jpsi_vtxsig` (sigma_v, cm), `Jpsi_vtxz` (r_v/sigma_v), `Jpsi_vtxdchi2` (z_v^2),
`Jpsi_vtxb6` and `Jpsi_vtxbfree` (the half-gradient at index 6, and
`max_i |g_i| sqrt(C_ii)` over the FREE indices, dimensionless and comparable with
`|z_v|`), `Jpsi_vtxvchk` (the variance-closure gate), `Jpsi_vtxsgnchk` (the sign
gate), the family variance shares `Jpsi_vtxv{gf,hit,ms,ioni}` with
`Jpsi_massv{ms,ioni}` for the MASS functional, and the flags `Jpsi_vtxfree`,
`Jpsi_vtxfirstplus`, `Jpsi_vtxok`.
Per block: `vtxvarv`, `vtxsgnv` (`v_b/sigma_v^2` and the ionization sign) and
`resinfvtxv` / `resinfv` (`a_b = dV_b^{1/2} w_b` for the VERTEX and the MASS
functional, 5/block padded — what the fourth cross cumulant needs).
Exponents: `cfvtx_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` at the vertex weights,
`cfvtx_hitcls` / `cfvtx_hitv` (per-hit-class shares of sigma_v^2), `cfvtx_grp*`
plus the new `cfmass_grp_vqms/vqio` (the per-material-group split and the FIT'S
OWN Q variance per group — the `gaussq` arm).
Gradients: `Jpsi_jacVtx` = `d theta_6^unc / d(global params)`, aligned with
`globalidxv`.

Config switches on `runCvhJpsiGenMC.py`: `exportVtxResidual` (default False),
`vtxConstraintZeroSeed` (default True).

---

## How to run

### Code

| what | where |
|---|---|
| maker | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc` |
| maker branch | `vtxres-cf-260911` @ `4b3984312f6` (branched from `perhit-residual-cf-260910` @ `a0f12c7f66d`) |
| analysis scripts | `/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres/`, branch `resolution-energy-loss-corrections` @ `c931a47` |
| material groups | `.../CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt` |
| scalar-potential init | `/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt` |

### Inputs and outputs (ceph)

    SIM input  /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16/task_*/step2.root
               (filelist: resolution/simprod/filelist_jpsigun_ul16.txt)
    J/psi gun  /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod/task_NNNN/
    DY MC      /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/dy/task_NNNN/
    npz/cards  /ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/runs/
               (symlinked as resolution/runs/vtxres/{vtx,mass,dy_vtx,dy_mass}.npz,
                runs/vtxres/cards, runs/vtxres/fits)

    CONSTRAINT ON (the re-analysis, section 12)
    J/psi gun  .../runs_vtxres_260911/prod_vtxon/task_NNNN/     160 x 2000 ev
    DY MC      .../runs_vtxres_260911/dy_vtxon/task_NNNN/       6 x 4000 ev
    npz/cards  .../runs_vtxres_260911/vtxon/   (symlinked as
                resolution/runs/vtxres_on, same file names)
    logs       resolution/vtxres/logs_on/

The J/psi-gun production is **160 tasks x 2000 events**, the same SIM and the same
fit settings as the mass-card production
`resolution_trackres_jpsigun_ul16_260905d_m0` (`runCvhJpsiGenMC.py`,
`doRes=True fillGrads=True fitFromGenParms=False useIdealGeometry=True
useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0
armijoSlack=1.0 trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False
applyHltFilter=False`), so a joint vertex+mass fit shares candidates. 20-way on
each of submit50/51/52, ~31 min/task, 1.61 GB RSS each, **328 kB/candidate**.
Switches added on top:

    exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False
    doVtxConstraint=False        (index 6 FREE -- what the study above used)

`prod_vtxon` / `dy_vtxon` are the SAME inputs and the SAME settings with
`doVtxConstraint=True`, which is now the maker's default (slurm 6432438 /
6432439, `slurm/array.sbatch`, which writes no `.complete` sentinel -- so the
extraction of that leg runs with `COMPLETE=""`).

The in-maker exponents replace the raw step records the 260905d production had to
carry at ~400 kB/cand, and the per-group split is carried for BOTH functionals.
The DY leg is `production/condor_dymc_v2`'s own configuration verbatim (REAL
geometry, pileup, MiniAOD, `massMin=60 massMax=120`, GT
`106X_mcRun2_asymptotic_v17`) plus `exportVtxResidual=True`.

### The chain

    cd /work/submit/david_w/ZMass/calibration_studies/resolution/vtxres

    ./run_prod.sh 20 0 159                    # the J/psi-gun production
    ./run_prod_dy.sh 10 0 19                  # the Z-like leg

    ./run_all.sh gates 20000                  # gates on production candidates
    NEXT=96000 ./run_all.sh extract           # -> runs/vtxres/{vtx,mass}.npz
    ./run_all.sh extract-dy                   # -> runs/vtxres/dy_{vtx,mass}.npz
    ./run_all.sh cards                        # 13 cards
    ./run_all.sh fits                         # every fit through rabbit_fit.py
    ./run_all.sh fisher                       # H and J, 60 HVPs per arm
    ./run_all.sh eff                          # the sandwich, per channel
    ./run_all.sh recovery ; ./run_all.sh recovery-hit
    ./run_all.sh xcum 20000                   # the vertex-mass correlation
    ./run_all.sh bill                         # the export bill
    ./run_all.sh plots ; ./run_all.sh plots-dy

    # section 13, the gen-background question and the minimum-size cut
    ./run_bkg.sh dy 3 0 5 ; ./run_bkg.sh dy-off 3 0 5   # the two gen legs
    ./run_bkg.sh gate-gun 1 ; ./run_bkg.sh gate-dy 1    # the bit-identity gate
    ./run_all.sh bkg-gate                               # ... and its verdict
    ./run_all.sh extract-bkg                            # -> bkg/runs/*_vtx*.npz
    ./run_all.sh bkg                                    # classes, cuts, mass, figures
    ./run_all.sh bkg-hits                               # the ndof / hit-count report

`run_tf.sh` runs a command in the rabbit TF container against `rabbit-vmass` (and
binds `/ceph/submit` when the host can read it); `run_ladder.sh card_<name>` and
`run_fit.sh <name>` are the per-card entry points. Every matrix and every
fitted value that leaves a term is in PHYSICAL units (`k`, `eps`) — see trap 1 —
so `run_all.sh eff` is the certified path and no unit flag is needed anywhere.

| script | role |
|---|---|
| `gates.py` | closure tests of the maker's own arithmetic; nothing is fitted |
| `extract_vtx.py --functional {vtx,mass}` | the in-maker `cfvtx_grp_*` / `cfmass_grp_*` into the CSR npz the card consumes |
| `vtxterm.py` | the arms and the predicted density, offline |
| `make_vtx_card.py` | the `MaterialCFTerm` cards: vtx / mass / joint, 42 material + 18 hit-class parameters |
| `fisher_vtx.py` | H and J in the layout `hitlik/efficiency.py` reads |
| `xcum_vtx.py` | the vertex-mass correlation at three levels |
| `cost_vtx.py --bill / --timing` | the export bill and the per-candidate cost |
| `plot_vtx.py` | the figures |
| `cmp_vtxon.py` | the same-candidate ON-vs-OFF comparison and the selection flow |
| `plot_vtxon.py` | the constraint-ON figures (gain, correlation, DCA identity, DY) |
| `drop_params.py` | a Fisher npz with uninformative parameters removed (section 12) |
| `report_vtxon.sh` | the whole ON-vs-OFF comparison, read off the two log sets |
| `genbkg.py` | section 13: the gen classification, the cut tables, the mass side, the `--gate` bit-identity check and the `--hits` ndof/hit-count report |
| `run_bkg.sh` | the gen-provenance DY legs (`dy` / `dy-off`) and the two gates (`gate-gun` / `gate-dy`) |

Third-party consumers: `hitlik/efficiency.py` (the sandwich),
`hitlik/recovery.py` (the injections), `hitlik/perhit/certify.py` (value + NLL +
EDM), `matres/pick_models.py` (chooses `tf-trust-krylov` for frozen-parameter
cards, `trust-exact` otherwise).

### Figures

`~/public_html/ZMass/cvh/260912_vtxon/` — the constraint-ON set (section 12):
`sigma_m_gain`, `sigma_m_gain_vs_genpt`, `mass_vertex_correlation`,
`dca_identity`, `dy_mass_shift`, plus the `plot_vtx.py` panels for both
regimes side by side (tags `vtx`/`mass`/`dyvtx` ON and `vtxoff`/`massoff`/
`dyvtxoff`/`vtxfull`/`vtxfulloff` OFF).

`~/public_html/ZMass/cvh/260911_vtxres/` — the free regime, 20 panels plus
`index.php`:
`density_{vtx,mass,dyvtx}[_log]`, `family_shares_*`, `group_shares_*`,
`hitclass_shares_*`, `shares_vs_genpt_{vtx,dyvtx}`, `sigma_check_*`.
NOTES entry: `/work/submit/david_w/Documents/Resolution/archive/NOTES_devlog_until_260911.md`, section
"THE VERTEX-CONSTRAINT RESIDUAL OF THE TWO-TRACK FIT".

---

## Results

### 1. The gates — all pass

Two samples: a 400-event FREE/FROZEN smoke pair (`logs/gates_smoke.log`,
`logs/gates_smoke_cons.log`) for the gates that need both regimes, and 20 000
production candidates (`logs/gates_prod.log`) for the rest.

| gate | number |
|---|---|
| **(a) `b` zero on every free index, nonzero only on 6** (FROZEN run) | `max_i \|g_i\| sqrt(C_ii)` over the FREE indices / `\|b_6\| sigma_v`: median **1.14e-11**, p90 4.1e-10. Absolute: free 3.3e-10, `\|b_6\|` = 64 (median) |
| **(b) the FROZEN fit reproduces the FREE one** | `\|r_v(frozen) - r_v(free)\|/sigma_v` median **1.5e-3**, p90 1.5e-2, max 0.26; relative `\|ratio-1\|` median **2.9e-3**, p90 2.0e-2 (376 matched good pairs of 378) |
| **(b') sigma_v** | `\|sigma_v(frozen)/sigma_v(free) - 1\|` median **8.7e-4**, p90 6.0e-3, max 0.22 |
| **(c) `sum_b \|a_b\|^2 = sigma_v^2`** (20 000 cand) | maker `Jpsi_vtxvchk` median **7.7e-11**, p90 2.6e-9; `\|a_b\|^2` vs `vtxvarv` 6.5e-8 (float32); `sum_c hitv` vs `vtxvgf` 1.9e-9 |
| **(d) `z_v^2 = Delta chi2`** | `<dchi2> = 1.1651` vs `<z_v^2> = 1.1509` (**1.2 % in the mean**); per candidate median 7.2 % (the linearisation); `ndof(frozen) - ndof(free) = 1` on every candidate |
| **(e) the ionization SIGN rule** (20 000 cand) | applied to the MASS influence it returns -1 on **99.78 %** of candidates at 100 % of their blocks, mean fraction **0.99958** |
| per-group closure (20 000 cand) | `cfvtx_grp_closure` max **2.7e-15**, `cfmass_grp_closure` max 1.7e-15 |
| `sum_g (vqms+vqio) + vgf` | **1.000000** (median, p1 and p99, on all 96 160 extracted candidates) |

**A structural fact gate (a) rests on.** In a GBL-type fit the reference
trajectory is built BY PROPAGATING, so every PROCESS-NOISE row of `rfull` is
identically zero at the linearisation point and the vertex state enters the
residual ONLY through the `ihit == 0` propagation rows: `F_6^T Vinv rfull == 0`
bit for bit on every candidate, with `|F_6| = 0.7071` (two entries of +-1/2, one
per leg, since `n_hat` is perpendicular to both momenta) and
`h_66 = 1e5 - 4e6`. The gradient that means anything is the one at the
CONSTRAINED optimum, `rho = rfull + F_f dxfree`, `b_6 = -(Vinv F_6) . rho` — the
same vector the exported `Rr` is built from. Measured with `rho`,
`max_i |g_i| sqrt(C_ii)` over the free indices is 3e-9 .. 7e-9 while
`|b_6| sigma_v = |z_v| = 0.45 .. 1.9`.

**Physics already visible in the gates** (20 000 production candidates):
`r_v` mean **-3.3 um**, rms 463 um; `sigma_v` median **74 um**; `z_v` mean
**-0.0026 +- 0.0075**, Var 1.108; `vtxvgf` (the HIT share of `sigma_v^2`) median
**0.370** with p10 0.081 and p90 0.649 against `cfmass_vgf` median **0.102** for
the MASS functional on the same candidates; `corr(sigma_v, z_v)` =
**+0.00006 +- 0.00708**. The per-block ionization signs are **-1 on 16.1 %** of
211 397 blocks and +1 on 83.9 %: the functional really does change sign block to
block, which is why `ressgn` is needed and a single `ioniSign` would be wrong.

### 2. The distribution — the mean is ZERO and there is NO Landau skew

`plot_vtx.py`, **96 160 candidates**, `logs/plots.log`:

| | data, untrimmed | data, trimmed \|z\| < 5 | model `cf` | model `gauss` | model `gaussq` |
|---|---|---|---|---|---|
| mean | **-0.0060 +- 0.0049** | -- | 0 | 0 | 0 |
| Var | 2.284 | **0.9729** | 1.0383 | 1.0291 | **0.9987** |
| skew | -75.99 | **+0.0091** | **+0.0001** | 0 | 0 |
| kurt | 12 764 | 3.946 | 25.87 | 3.005 | 3.004 |

The trim drops 0.162 % of candidates. Two things. **(i) The mean is zero** — no
kernel, no theory, no PDG input; that is the whole point of the term. **(ii) The
untrimmed moments are set by ~0.2 % of candidates**, so the moments are not the
description: the TAILS table is. `gaussq`'s model variance is 0.9987 rather than
exactly 1 — the 0.13 % is the tau-grid truncation at 7.89, the same number
`hitlik` measured; `norm(cf) = 0.9987` likewise.

**The vertex residual has NO LANDAU SKEW**, and that is physics: the ionization
share of `sigma_v^2` is 0.0000 (3.8e-6), so the one-sided channel that skews the
MASS residual contributes nothing here.

**THE TAILS, `P(|z| > t)`** (96 160 candidates):

| t | data | CF | data/CF | Gaussian (the fit's Q) | data/chi2 |
|---|---|---|---|---|---|
| 1 | 0.28720 | 0.29566 | 0.97 | 0.31693 | 0.91 |
| 2 | 0.04590 | 0.04433 | 1.04 | 0.04545 | 1.01 |
| 3 | 0.00876 | 0.00645 | **1.36** | 0.00270 | **3.25** |
| 4 | 0.00318 | 0.00201 | **1.58** | 0.000063 | **50.3** |
| 5 | 0.00162 | 0.00097 | **1.68** | 0.0000010 | **2829** |

The CF is **1.9x (3 sigma) to 1690x (5 sigma) closer to the data than the chi2**.
The MASS term on the same candidates, for comparison: data/CF 0.88 / 0.85 / 0.83
at t = 3 / 4 / 5, data/chi2 3.17 / 44.0 / 2285.

### 3. The composition — MS- or hit-dominated? BOTH, and it crosses over

Median shares of `sigma_v^2`: **hit 0.370, multiple scattering 0.630, ionization
0.0000**. The MASS functional on the SAME candidates: MS 0.898, ionization
0.0002, hit 0.102.

Against the SOFTER muon's GEN pT (8 quantile bins):

| gen pT [GeV] | <1.17 | 1.17-2.05 | 2.05-2.92 | 2.92-3.91 | 3.91-5.12 | 5.12-6.69 | 6.69-8.94 | >8.94 |
|---|---|---|---|---|---|---|---|---|
| hit share | 0.066 | 0.204 | 0.295 | 0.358 | 0.412 | 0.458 | 0.503 | **0.548** |
| MS share | **0.934** | 0.796 | 0.705 | 0.642 | 0.588 | 0.542 | 0.497 | 0.452 |

So the vertex residual is **MS-dominated below ~7 GeV and hit-dominated above** —
the expected composition (innermost pixel hit resolution plus multiple scattering
in the beam pipe and the first pixel layers), now measured, with the crossover
located.

**Which material and which hits — and it is COMPLEMENTARY to the mass term.**
Mean share of `sigma_v^2` (MS + ionization):

| VERTEX | share | MASS | share |
|---|---|---|---|
| `bpix_support6` | **0.324** | `tib_support` | **0.221** |
| `bpix_active_L1` | 0.074 | `tec_structure` | 0.184 |
| `tib_support` | 0.055 | `tob_support` | 0.135 |
| `bpix_active_L2` | 0.041 | `bpix_support6` | 0.100 |
| **`beampipe`** | **0.031** | `tibtid_services` | 0.069 |
| `fpix_support` | 0.030 | `bpix_services` | 0.041 |
| `bpix_services` | 0.024 | `tid_support` | 0.035 |
| `bpix_active_L3` | 0.017 | `tob_services` | 0.030 |

Hit classes: VERTEX `pix_y_q1` 0.065, `pix_x_q1` 0.043, `pix_y_q2` 0.038,
`pix_y_q3` 0.037, `pix_y_q0` 0.027, `pix_x_q0` 0.025 — **all PIXEL**; MASS
`str_N3_lo` 0.024, `str_N2_lo` 0.019, `str_N1_lo` 0.011, `str_N3_hi` 0.010 —
**all STRIP**.

**The self-consistent-sigma check.** `corr(sigma_v, z_v) = -0.0055 +- 0.0032`,
`corr(sigma_v, |z_v|) = +0.0039` — zero at the sub-percent level, so the vertex
term is built with `self_consistent_sigma=False` and no `a_res`, and the
first-order Jacobian term it would carry is bounded by that correlation. The MASS
functional on the same candidates has `corr(sigma_m, z_m) = +0.0241 +- 0.0032`,
i.e. the effect the mass term's `a_res` correction exists for is real and
7.5 sigma — and the vertex term does not have it.

### 4. The cards — NLL at MC truth (8 000 candidates, 42 material + 18 hit classes)

`logs/cards_all.log`; 13/13 built. Lower NLL(0) = describes the data better:

| card | NLL(0) | vs `cf` | per candidate |
|---|---|---|---|
| `vtx_cf` | **-26 049.892** | -- | -- |
| `vtx_gauss` (variance-matched) | -25 881.262 | +168.63 | +0.0211 |
| `vtx_gaussq` (the fit's Q = the chi2) | -25 876.335 | **+173.56** | **+0.0217** |
| `mass_cf` | **-15 893.497** | -- | -- |
| `mass_gaussq` | -15 312.549 | +580.95 | +0.0726 |
| `joint_cf` | -41 943.389 = exactly `vtx_cf` + `mass_cf` | | |

(`hitlik`'s truth-referenced 4-component term was +0.0155/row against its
Gaussian, so the vertex term's non-Gaussianity is of the same size.)

### 5. The sandwich — what each arm's own error is worth

`fisher_vtx.py` (H by 60 HVPs in 83-113 s per arm, J by 200 batch means,
`|sum_m g_m - g| = 0`) + `hitlik/efficiency.py`. 8 000
candidates, at MC truth, with the parmtype-15 tier priors and 1.0 on every hit
class. Every number is the ACTUAL (sandwich) variance, never the nominal Fisher
one. `logs/eff_vtx_p2.log`.

**sandwich / quoted — does each arm's own error mean anything?**

| arm | marginal | prior-free, material | prior-free, hit classes |
|---|---|---|---|
| **`cf`** | **0.885** | **0.961** | **1.061** |
| `gauss` (variance-matched) | 1.040 | -- | -- |
| **`gaussq`** (the chi2) | **1.099** | **1.543** | **1.177** |

bootstrap/sandwich 0.995-0.998 in every arm (as it must be algebraically).

**EFFICIENCY `sigma^2(chi2, ACTUAL) / sigma^2(CF, ACTUAL)`**

| | marginal | prior-free | what the chi2 CLAIMS |
|---|---|---|---|
| **MATERIAL** | **2.659** (16-84 % 2.225-3.204; min 2.021, max 3.461) | **2.778** (16-84 % 1.851-15.406) | 1.006 |
| **HIT CLASSES** | **1.215** (16-84 % 0.972-1.457) | **1.115** (16-84 % 0.501-1.304) | 0.968 |

**The full PDF constrains the material 2.7x better in variance (1.6x in sigma)
and the hit classes 1.2x, while the chi2 claims parity.** The CF's own quoted
error is right to 4-6 %; the chi2's is optimistic by 54 % in variance on the
material. Per group (marginal): `bpix_services` 3.461, `tib_support` 2.659,
`bpix_support6` 2.021.

### 6. Vertex vs mass vs joint — who measures what

Quoted sigma at MC truth, CF arm, 8 000 candidates,
`logs/eff_{vtx,mass,joint}_p2.log`. Sigmas are PHYSICAL — `k`, the log material
amount, for a group and `eps`, the linear variance scale, for a hit class — so
the prior column is the group's own tier prior.

| parameter | VERTEX | MASS | JOINT | prior |
|---|---|---|---|---|
| `material_bpix_support6` | **0.0399** | 0.0485 | **0.0393** | 0.05 |
| `material_tib_support` | 0.0488 | 0.0453 | **0.0432** | 0.05 |
| `material_tec_structure` | (uninformative) | 0.0438 | **0.0425** | 0.05 |
| `material_tob_support` | (uninformative) | 0.0473 | **0.0463** | 0.05 |
| `material_bpix_services` | 0.0954 | 0.0958 | 0.0918 | 0.1 |
| **`hitres_pix_y_q1`** | **0.167** | 0.958 | **0.166** | 1.0 |
| **`hitres_pix_x_q1`** | **0.206** | 0.800 | **0.204** | 1.0 |
| `hitres_pix_y_q2` | 0.207 | (uninformative, > 0.95) | 0.206 | 1.0 |
| `hitres_pix_y_q3` | 0.195 | 0.948 | 0.193 | 1.0 |
| `hitres_str_N1_lo` | 0.377 | 0.440 | **0.300** | 1.0 |
| `hitres_str_N3_lo` | 0.404 | 0.561 | **0.348** | 1.0 |

**On the innermost PIXEL hit classes the vertex term is 4-6x tighter in sigma
(15-33x in variance) than the mass term**, and the joint is no better than the
vertex alone there — that information comes from the vertex residual and from
nowhere else. On the STRIP classes and the OUTER material the mass term leads and
the joint improves on both. With the correct prior only three material groups
pass the informativeness test at 8 000 candidates (`bpix_support6`,
`tib_support`, `bpix_services`).

### 7. The fits, certified

`logs/fits_all.log` and `logs/fit_<name>.log`, 8 000 candidates, 60 parameters,
`--max-edm 1e-3`. NLL(min) is the total at the minimum (unbinned terms +
prior penalty).

| fit | NLL(min) | EDM | `material_bpix_support6` | `hitres_pix_y_q1` |
|---|---|---|---|---|
| `vtx_cf` | -26 059.934 | **1.3e-12** | +0.07314 +- 0.03912 | **-0.400 +- 0.132** |
| `vtx_gaussq` | -25 890.002 | 2.5e-08 | +0.06205 +- 0.03767 | -0.538 +- 0.125 |
| `mass_cf` | -15 896.129 | 6.2e-14 | -0.00884 +- 0.04853 | -0.181 +- 0.958 |
| `joint_cf` | -41 954.970 | 7.7e-12 | +0.06535 +- 0.03850 | -0.408 +- 0.131 |
| `inj_vtx_cf` | -26 061.175 | 1.2e-16 | +0.05392 +- 0.03873 | -0.414 +- 0.131 |
| `inj_vtx_gaussq` | -25 892.592 | 1.4e-08 | +0.03918 +- 0.03732 | -0.544 +- 0.126 |
| `inj_mass_cf` | -15 895.928 | 3.3e-17 | -0.01180 +- 0.04839 | -0.187 +- 0.958 |
| `inj_joint_cf` | -41 956.049 | 2.1e-15 | +0.04519 +- 0.03810 | -0.422 +- 0.130 |
| `injhit_vtx_cf` | -26 059.921 | 9.2e-18 | | (see the hit injection) |
| `injhit_vtx_gaussq` | -25 889.984 | 2.5e-08 | | |

**8/11 certified; the three that fail are ALL Gaussian arms**: `vtx_gauss`
(Hessian not positive-definite at its own minimum) and `mass_gaussq` /
`joint_gaussq` (a NaN Hessian — open item 1). Same lesson as `hitlik`: a
misspecified likelihood is also a worse-conditioned one (CF 7/8 vs Gaussian 5/8
there).

**The vertex term MEASURES the innermost pixel resolutions.**
`hitres_pix_y_q1 = -0.400 +- 0.132` from the vertex residual against
`-0.181 +- 0.958` from the mass term on the SAME candidates — **7.3x tighter in
sigma, 53x in variance** — i.e. the fit's assumed variance for that class is
~40 % too large, and only the vertex residual can see it.

### 8. The injections

`logs/recovery.log`, `logs/recovery_hit.log`.

**`material_bpix_support6` x1.05 material** (truth = ln(1.05) = `k` +0.0487902;
values are PHYSICAL `k`). `f_pri` is the prior-shrinkage factor
`sigma_post^2 / sigma_lik^2`; `corrected/truth = shift / (f_pri x truth)`:

| channel | baseline | injected | shift | /truth | f_pri | **corrected/truth** | leak rms |
|---|---|---|---|---|---|---|---|
| vertex, CF | +0.07314 | +0.05392 | -0.01922 | -0.394 | 0.400 | **0.985** | 0.034 |
| vertex, fit's Q | +0.06205 | +0.03918 | -0.02287 | -0.469 | 0.443 | **1.058** | 0.118 |
| MASS, CF | -0.00884 | -0.01180 | -0.00296 | -0.061 | 0.063 | **0.957** | 0.013 |
| **JOINT, CF** | +0.06535 | +0.04519 | -0.02016 | -0.413 | 0.419 | **0.986** | 0.034 |

Every channel recovers the 5 % injection to **1.4-5.8 %**. Largest leakage:
-0.12 sigma onto `hitres_pix_x_q1` (CF) and +0.81 sigma onto `hitres_pix_y_q3`
(the fit's-Q arm — another sign that arm is worse conditioned).

**`hitres_pix_x_q2` variance x1.10** (an INNERMOST PIXEL class). `hit_mode` is
LINEAR, so the expected shift is `-eps_inj/(1+eps_inj) x (1+eps_base)`:

| channel | eps_base | injected | shift | expected | **/expected** | leak rms |
|---|---|---|---|---|---|---|
| vertex, CF | -0.1659 +- 0.291 | -0.2375 | -0.0716 | -0.0758 | **0.944** | **0.001** |
| vertex, fit's Q | -0.2978 +- 0.289 | -0.3593 | -0.0615 | -0.0638 | **0.963** | 0.000 |

Leakage below 0.001 sigma on every other parameter.

### 9. Correlation with the MASS term — may the two be multiplied?

`xcum_vtx.py`, 20 000 candidates (19 888 good), `logs/xcum.log`. Three levels:

1. **THE ALGEBRA.** `Cov(r_v, dm) = sum_b a_b^v . a_b^m = e_6^T C a_m` exactly
   (every residual row is covered by a registered block — the `vtxvchk` gate). It
   is NOT zero by construction the way the per-hit complement was (`F^T R = 0`
   there). What makes it vanish is a **MIRROR SYMMETRY**: reflecting the event in
   the plane spanned by the two momenta sends `n_hat -> -n_hat`, hence
   `theta_6 -> -theta_6`, while the momenta — and so the mass — are unchanged;
   only the magnetic field can break it. MEASURED per candidate: median
   **-0.00485**, mean **+0.00013**, rms **0.1924**, p1/p99 -0.418/+0.434 — a real
   |rho| ~ 0.19 of random sign per candidate whose ENSEMBLE average is zero.
2. **THE ENSEMBLE.** `corr(z_v, z_m) = +0.0047 +- 0.0071` with
   `z_m = (m - m_gen)/sigma_m`; `corr(|z_v|, |z_m|) = -0.00009`;
   `corr(z_v^2, z_m^2) = -0.0014`. Consistent with zero at both orders.
3. **THE FOURTH CROSS CUMULANT.** `kappa(v,v,m,m)/sqrt(kappa4_v kappa4_m)` median
   **0.132** (p16-p84 0.069-0.228) — the same size as the per-hit study's
   q/p-vs-others (0.09-0.14) and far below its phi-vs-d0 (0.71).

**THE JOINT VERDICT.** `sandwich/quoted` on the joint is **0.880** against
**0.885** for the vertex alone and **0.432** for the mass alone — i.e. NOT above
1, so the joint does not over-count relative to what its arms already do. The two
residuals are uncorrelated in the ensemble and share 13 % of their fourth
cumulant, so the product-of-marginals form is a good approximation and the
sandwich prices what is left.

### 10. The Z-like check on DY MC (7 841 candidates)

6 files of `production/condor_dymc_v2`'s own configuration plus
`exportVtxResidual=True`; `logs/plots_dy.log`.

**The predicted regime change is confirmed.** Median shares of `sigma_v^2`:

| sample | hit (Gaussian) | multiple scattering | ionization |
|---|---|---|---|
| J/psi gun (muons 1-15 GeV) | 0.370 | **0.630** | 0.0000 |
| **DY (muons 20-143 GeV)** | **0.732** | 0.268 | 0.0000 |

vs GEN pT on DY: hit share 0.587 (20-26 GeV) -> 0.687 -> 0.723 -> 0.743 -> 0.753
-> 0.754 -> 0.763 -> **0.803** (>43 GeV). So the vertex residual is
**MS-dominated at J/psi momenta and hit-dominated at Z momenta**, and on DY the
leading single contribution is one hit class, **`pix_x_q1` at 0.198** of
`sigma_v^2` (0.043 on the gun) — the innermost pixel BENDING-plane measurement,
followed by `pix_x_q2` 0.070, `pix_y_q1` 0.064, `pix_x_q0` 0.062, `pix_x_q3`
0.055. Material drops to `bpix_support6` 0.108 (0.324 on the gun),
`tib_support` 0.038.

**AND A WARNING FOR A DATA FIT.** On DY the residual has a tail the CF does NOT
model: `P(|z|>5)` is **1.86 %** against a CF 0.022 % (**data/CF 85**; data/chi2
32 468), `P(|z|>3)` 3.10 % against 0.41 % (data/CF 7.6), and `Var(z)` is 1123
untrimmed (**1.147 trimmed at 5 sigma**, skew -0.014, kurt 3.99, trim drops
1.86 %). The gun, on ideal geometry, has 0.16 % beyond 5 sigma and data/CF 1.7.
So the extra tail is **not** the resolution model: it is real geometry
(misalignment), pileup, and genuinely displaced or mispaired muons in a
60-120 GeV MiniAOD dimuon sample. **A vertex term on data needs an outlier
(background) component, and the size of it is now measured: ~2 %.**

### 11. Cost and the export bill

**Cost, A/B on ONE PINNED CPU** (`taskset -c 40`, 200 events of the J/psi gun,
identical configuration but for the switch):

| | wall for 200 events | per candidate |
|---|---|---|
| fit + the MASS export | 226 s | 1.13 s |
| + the VERTEX block | 334 s | 1.68 s |
| **the vertex block** | **+108 s** | **+0.54 s (+48 %)** |

It is the same `cvhcf` call at different weights, so it costs what the mass
functional's own call costs.

**Export, compressed, measured** (`cost_vtx.py --bill --file <production file>`;
full scale = 34 M J/psi + 7 M Z = 41 M candidates):

| block | kB/cand | TB at 41 M |
|---|---|---|
| vertex: per-group exponents | 26.37 | 1.107 |
| vertex: flat exponents | 1.44 | 0.060 |
| vertex: shares + scalars | 0.61 | 0.026 |
| vertex: the influence `a_b` (`resinfvtxv`) | 1.63 | 0.069 |
| vertex: the D row (`Jpsi_jacVtx`) | 0.91 | 0.038 |
| **the VERTEX block** | **30.96** | **1.300** |
| the MASS block, for comparison | 29.98 | 1.259 |
| (`hesspackedv`, the quadratic term's, for scale) | 105.3 | 4.42 |

**The vertex term costs 1.03x the mass term's export** — as expected, it is ONE
more component per candidate. The whole production file is 328 kB/candidate; the
offline extraction is 27.36 kB/cand (21.89 group rows, 12.32 hit rows per
candidate).

---

### 12. THE VERTEX CONSTRAINT ON — the study re-run on `prod_vtxon`

Sections 1-11 are the FREE regime (`doVtxConstraint=False`). The maker now
defaults to the constraint ON, freezes state index 6 at `theta_6 = 0`, and
exports `Jpsi_vtxres` (the DCA the unconstrained fit would report, through the
one-step identity), `Jpsi_mass` AND `Jpsi_mass_unc`, and
`Jpsi_covmassvtx` = `cov(m, theta_6)`. The whole chain was re-run on
`prod_vtxon` / `dy_vtxon` — same SIM, same settings, the switch the only
difference — and the free-regime numbers were re-produced with the current
physical-unit code so every comparison is like for like.
Logs `logs_on/`, figures `~/public_html/ZMass/cvh/260912_vtxon/`.

**Every conclusion of sections 1-11 survives. Four things change, all for the
better, and one number in section 10 was wrong for a reason that is now
understood.**

#### 12.1 The gates (20 000 candidates; `logs_on/gates_prod{,_off}.log`)

| gate | ON | OFF |
|---|---|---|
| (a) `max_i \|g_i\| sqrt(C_ii)` (free idx) / `\|b_6\| sigma_v` | median **9.7e-12** | 2.4e+05 (6 is free there) |
| (c) `Jpsi_vtxvchk` | 7.79e-11 | 7.71e-11 |
| (c) mass, `\|sum_b\|a_b\|^2 - (cov+covhit)\|/(.)` | **7.4e-8**; vs `sigma_m^2` 5.1e-8 | — |
| (e) sign rule, candidates at 100 % of blocks | 99.82 % | 99.78 % |
| per-group closure, both functionals | max 3.1e-15 | max 2.7e-15 |
| `sum_g(vqms+vqio)+vgf` on all 96 160 | 1.000000 | 1.000000 |
| `Jpsi_d` | **identically 0** | the raw `theta_6` |
| `\|m_unc - (m_c + cov sigma_v^-2 r_v)\|` | median **6.9e-8 GeV**, max 4.7e-6 | — |
| `sum_c cfmass_hitv` vs `cfmass_vgf` | 9.9e-7 median, 7.6e-6 q99 | 9.7e-7 / 8.6e-6 |

#### 12.2 ON against OFF on the SAME candidate (39 832 matched pairs)

`cmp_vtxon.py`, `logs_on/cmp_vtxon.log`.

| | median | p90 | p99 | max |
|---|---|---|---|---|
| `\|r_v(ON) - r_v(OFF)\|/sigma_v` | **1.37e-3** | 1.31e-2 | 7.95e-2 | 111 |
| `\|sigma_v(ON)/sigma_v(OFF) - 1\|` | **8.3e-4** | 5.3e-3 | 3.3e-2 | 51 |
| `\|m_unc(ON)/m(OFF) - 1\|` | **4.2e-5** | 4.6e-4 | 2.4e-3 | 0.69 |
| `\|m_c(ON)/m(OFF) - 1\|`, no identity | 7.2e-4 | 3.4e-3 | 1.0e-2 | 0.69 |

The one-step identity removes a factor **17** in the median; 2.0 % of
candidates move by more than 0.05 `sigma_v` and 0.05 % by more than 0.5, the
second-order tail of a one-Newton-step projection. `m_c - m_unc` has mean
+0.040 MeV and rms 11.2 MeV.

**THE MASS RESOLUTION GAIN.** `sigma_m(ON)/sigma_m(OFF)` = **0.98109** in the
mean (median 0.99013, p16-p84 0.9626-0.9996) — **1.89 % in sigma, 3.75 % in
variance**. It is PREDICTED candidate by candidate from the ON export alone:
freezing `theta_6` conditions the mass on it, so the ratio must be
`sqrt(1 - rho^2)` with `rho = cov(m,theta_6)/(sigma_m^{unc} sigma_v)`, and the
predicted mean is **0.98103** — agreeing in the mean to 5e-5 and per candidate
to 0.17 % (median). `rho` has mean -0.00016 and **rms 0.19132**, i.e. exactly
the free regime's per-candidate mass-vertex influence correlation (0.1928), and
its distribution lies on top of it (`mass_vertex_correlation.pdf`). The gain
grows with momentum — 1.32 / 1.20 / 1.52 / 1.75 / 1.98 / 2.23 / 2.47 / **2.66**
per cent across the softer muon's GEN `pT` octiles (`sigma_m_gain_vs_genpt.pdf`,
measured over predicted flat at 1.00 in every bin).

**THE PER-CANDIDATE CORRELATION IS GONE.** `corr(w_mass, w_vtx)` in the `V`
metric: mean 0.000000, **rms 0.000000, max 1.2e-5** (free: rms 0.1928,
max 0.997). `xcum_vtx.py` level (1) agrees: per-candidate
`Cov(r_v, dm)/(sigma_v sigma_m)` rms **3.1e-5** against 0.1924. The two
functionals are now orthogonal BY CONSTRUCTION rather than by the mirror
symmetry of section 9.

#### 12.3 The distribution, 96 160 candidates (`logs_on/plots_full_{on,off}.log`)

| | ON | OFF |
|---|---|---|
| mean | +0.00065 +- 0.00324 | -0.00604 +- 0.00487 |
| Var / skew / kurt, UNTRIMMED | **1.00652 / +0.051 / 5.14** | 2.28357 / -75.99 / 12 764 |
| Var / skew / kurt, trimmed `\|z\|<5` | 0.97095 / +0.0100 / 3.929 | 0.9729 / +0.0091 / 3.946 |
| trim drops | 0.100 % | 0.162 % |
| `P(\|z\|>5)` | **0.000998** | 0.001622 |
| data/CF at 3 / 4 / 5 sigma | **1.250 / 1.256 / 1.033** | 1.358 / 1.582 / 1.679 |
| data/chi2 at 3 / 4 / 5 sigma | 2.988 / 39.9 / 1741 | 3.246 / 50.3 / 2829 |

The constraint removes the far tail that made the free-regime moments
unquotable, and the CF now describes the 5-sigma tail to **3 %**.

#### 12.4 Composition — identical

Vertex family shares hit **0.3700** / MS 0.6300 / ionisation 0.0000 (free
0.370 / 0.630 / 0.0000); hit share 0.064 -> 0.547 across GEN `pT` (free
0.066 -> 0.548); `bpix_support6` 0.3234 (0.324), `bpix_active_L1` 0.0736
(0.074), `tib_support` 0.0555 (0.055), `beampipe` 0.0309 (0.031); hit classes
`pix_y_q1` 0.0650 (0.065), `pix_x_q1` 0.0420 (0.043). On DY, hit 0.7297
(0.732), `bpix_support6` 0.1080 (0.108), `pix_x_q1` 0.1992 (0.198).
`corr(sigma_v, z_v)` = +0.0012 +- 0.0071, still zero, so the vertex term still
needs no self-consistent-sigma correction.

The MASS functional moves a little and only where it should: MS 0.9036 (0.8979),
hit 0.0962 (0.1021), `tib_support` 0.2154 (0.2206), `tec_structure` 0.1780
(0.1833), `tob_support` 0.1299 (0.1351) and **`bpix_support6` 0.1151 (0.0999)**
— the constrained mass leans ~15 % more on the innermost support. Its
`corr(sigma_m, z_m)` is **+0.0347 +- 0.0071 ON against +0.0344 +- 0.0071 OFF**
on the same 20 000 candidates, so the two mandatory mass corrections
(self-consistent sigma, Jensen) behave exactly as before.

#### 12.5 A degeneracy the published sandwich carried: `hitres_str_N5_hi`

`H` has a NEGATIVE eigenvalue along `hitres_str_N5_hi` in every arm and both
regimes (CF -7.5 ON / -5.6 OFF, `gaussq` **-9.0 ON / -2.7 OFF**), and the class
carries only 0.27 % of `sigma_v^2` on 3 048 candidates — the SAME occupancy in
the two productions (3 048 / 3 049 rows, `sum v` 8.08 / 8.10). With the 1.0 hit
prior `H + P` then has min eigenvalue **8.3e-3** ON against 0.48 OFF, so
`(H+P)^-1` is enormous along that one direction and inherits into every
marginal error: the raw ON material efficiency reads 609.6. It also inflates
rabbit's EDM, `1/2 g^T (H+P)^-1 g`, on the JOINT fits.

`drop_params.py` removes the parameter from `H`, `J` and `G`, and
`--freezeParameters hitres_str_N5_hi` removes it from the fit — the same
decision, applied identically to every arm and both regimes, and not a scale,
a bound or a clip on anything measured. Everything below is quoted that way,
and the free-regime table is re-quoted the same way beside it.

**What the direction IS, settled with the trust-region preconditioner.** The
negative curvature is a real feature of the likelihood, not a conditioning
artefact. Re-fitting `joint_cf` ON with `hitres_str_N5_hi` FLOATING and
rabbit's spectral preconditioner (`--precondition --preconditionParams '.*'
--preconditionBlocks none --preconditionTransform spectral`, which whitens the
60-parameter block from condition number **1.03e8 to 1** and, unlike `ridge`,
keeps the sign of the negative curvature) the minimiser leaves the
near-stationary point the unpreconditioned fit sits at (EDM 0.5028) and drives
the parameter to **exactly -1.000000** — the boundary of
`H(eps) = max(1 + eps, 0)`, i.e. the hit variance of the class driven to zero,
beyond which the `max` has identically zero gradient. The postfit Cholesky then
fails (`Hessian is not positive-definite`), so there is no EDM and no
covariance at that point: there is **no interior minimum** in that direction,
and no minimiser can produce one. The class simply carries too little
information for the unit prior to create a stationary point before the
boundary. Every other parameter moves by **less than 0.15 sigma** on the way
out (`material_bpix_support6` 0.065 sigma, `hitres_pix_y_q1` 0.14 sigma), which
is why dropping or freezing it costs nothing.

The same options with the parameter FROZEN reproduce the certified row to every
printed digit — NLL -42148.6599, `material_bpix_support6` +0.06699 +- 0.03849,
`hitres_pix_y_q1` -0.39391 +- 0.13244 — with the EDM improved from 5.64e-12 to
**2.13e-16**, so the preconditioner changes no answer.

#### 12.6 The sandwich (8 000 candidates, 59 parameters, physical units)

| | ON | OFF | published (60 par) |
|---|---|---|---|
| CF, median sandwich/quoted | **0.915** | 0.914 | 0.885 |
| `gauss` / `gaussq`, median S/Q | 1.049 / 1.081 | 1.089 / 1.101 | 1.040 / 1.099 |
| MATERIAL efficiency, marginal | **2.256** (2.198-3.067) | **2.632** (2.217-3.173) | 2.659 |
| MATERIAL efficiency, prior-free | 3.310 | 2.778 | 2.778 |
| MATERIAL S/Q prior-free, CF / Gauss | 0.957 / 1.549 | 0.961 / 1.543 | 0.961 / 1.543 |
| HIT efficiency, marginal | **1.238** (1.028-1.434) | 1.197 (0.965-1.377) | 1.215 |
| HIT efficiency, prior-free | 1.186 | 1.104 | 1.115 |
| HIT S/Q, CF / Gauss | 1.034 / 1.204 | 1.055 / 1.170 | 1.061 / 1.177 |
| bootstrap/sandwich | 1.001-1.005 | 0.996-0.998 | 0.995-0.998 |

The CF arm's own quoted and sandwich errors per parameter are unchanged to the
third digit: `bpix_support6` 0.0403/0.0243 ON against 0.0399/0.0237 OFF,
`bpix_services` 0.0956/0.0317 against 0.0954/0.0318, `tib_support`
0.0489/0.0103 against 0.0488/0.0106; `pix_y_q1` 0.1653/0.1591 against
0.1671/0.1614, `pix_x_q1` 0.2019/0.1858 against 0.2060/0.1811. The same three
material groups pass the informativeness test.

#### 12.7 Vertex vs mass vs joint — the information is re-partitioned, not created

Quoted sigma, CF arm, physical units:

| parameter | VTX ON | VTX OFF | MASS ON | MASS OFF | JOINT ON | JOINT OFF |
|---|---|---|---|---|---|---|
| `material_bpix_support6` | 0.0403 | 0.0399 | 0.0481 | 0.0485 | **0.0394** | 0.0393 |
| `material_tib_support` | 0.0489 | 0.0488 | 0.0455 | 0.0452 | **0.0436** | 0.0432 |
| `material_tec_structure` | (unin.) | (unin.) | 0.0445 | 0.0438 | 0.0429 | 0.0425 |
| `material_tob_support` | (unin.) | (unin.) | 0.0474 | 0.0473 | 0.0464 | 0.0463 |
| `material_bpix_services` | 0.0956 | 0.0954 | 0.0961 | 0.0958 | 0.0921 | 0.0918 |
| **`hitres_pix_y_q1`** | **0.1653** | 0.1671 | 0.9364 | 0.958 | **0.1645** | 0.1661 |
| **`hitres_pix_x_q1`** | **0.2019** | 0.2060 | 0.8364 | 0.800 | 0.2002 | 0.2038 |
| `hitres_pix_y_q3` | 0.1900 | 0.1947 | 0.9366 | (unin.) | 0.1890 | 0.1933 |
| `hitres_str_N1_lo` | 0.3767 | 0.3769 | 0.4672 | 0.4396 | **0.3084** | 0.3003 |
| `hitres_str_N3_lo` | 0.4178 | 0.4043 | 0.5662 | 0.5612 | **0.3594** | 0.3478 |
| `hitres_str_N2_hi` | 0.4451 | 0.4527 | 0.7962 | 0.6236 | 0.4228 | 0.3698 |

**`sigma_m` shrinks by 1.9 % and the mass term learns nothing extra.** Its
quoted errors are the same to ~1 % on the material and 3-18 % WORSE on the
outer strip classes, and so are the joint's. That is the expected answer:
conditioning on `theta_6` removes exactly the part of the mass variance the
VERTEX term measures on its own, so the information is re-partitioned between
the two terms rather than created. The vertex term's own numbers do not move,
so its 4-6x advantage over the mass term on the innermost pixel classes stands
verbatim.

**THE JOINT VERDICT** (sandwich/quoted, CF arm, 59 parameters):

| | ON | OFF |
|---|---|---|
| vertex alone | **0.915** | 0.914 (0.885 with the degenerate class) |
| mass alone | **0.358** | 0.432 |
| **joint** | **0.857** | **0.880** |

The joint is BELOW the vertex term's own value in both regimes: **the joint
still does not over-count**, and it is the same statement even though the
per-candidate first-order correlation is now identically zero. The ensemble
correlations stay zero (`corr(z_v,z_m)` = +0.0054 +- 0.0071 ON against
+0.0047 +- 0.0071; `corr(z_v^2, z_m^2)` -0.0023 against -0.0014) and the
FOURTH cross cumulant RISES, **0.178** (p16-84 0.092-0.308) against 0.132 —
which is what is left once the first-order piece is removed.

#### 12.8 The fits and the injections

All 13 cards build and all 13 fits RUN: the free regime's three hard failures
(`vtx_gauss` not positive-definite, `mass_gaussq` / `joint_gaussq` NaN Hessian
— open item 1) are GONE. Under the default `trust-krylov` four stall at
EDM ~0.5 (`joint_cf`, `inj_joint_cf`, `joint_gaussq`, `mass_gaussq`), and they
are not unconverged — `tf-trust-krylov` lands on the same point (`joint_cf`
NLL -42 151.62111 against -42 151.62167, 1.4e-8 relative) with the same EDM.
It is the `hitres_str_N5_hi` direction of 12.5 inflating
`1/2 g^T (H+P)^-1 g`. Freezing that one parameter certifies all of them:

| fit | free EDM | FROZEN EDM |
|---|---|---|
| `joint_cf` ON | 0.5028 | **5.64e-12** |
| `inj_joint_cf` ON | 0.5030 | **1.81e-12** |
| `joint_gaussq` ON | 0.5136 | **1.14e-09** |
| `mass_gaussq` ON | 0.5062 | **9.40e-10** |
| `joint_cf` OFF | 7.69e-12 | 5.76e-12 |

and moves nothing (`joint_cf` ON `material_bpix_support6` +0.0670 +- 0.0385
frozen against +0.0676 +- 0.0385 free). **13/13 certified in both regimes.**

| fit | ON NLL(min) | ON EDM | OFF NLL(min) | OFF EDM | `bpix_support6` ON / OFF | `pix_y_q1` ON / OFF |
|---|---|---|---|---|---|---|
| `vtx_cf` | -26 049.779 | 4.9e-16 | -26 059.934 | 1.3e-12 | +0.0794 / +0.0731 | **-0.400 +- 0.133 / -0.400 +- 0.132** |
| `vtx_gauss` | -25 916.956 | 8.8e-11 | (FAILED) | — | +0.0417 | -0.536 |
| `vtx_gaussq` | -25 912.798 | 2.0e-12 | -25 890.002 | 2.5e-08 | +0.0666 / +0.0621 | -0.510 / -0.538 |
| `mass_cf` | -16 094.041 | 7.1e-19 | -15 896.129 | 6.2e-14 | -0.0133 / -0.0088 | -0.146 +- 0.934 / -0.181 +- 0.958 |
| `mass_gaussq` (frozen) | -15 574.636 | 9.4e-10 | (FAILED) | — | +0.0417 | -0.031 |
| `joint_cf` (frozen) | -42 148.660 | 5.6e-12 | -41 954.485 | 5.8e-12 | +0.0670 / +0.0649 | -0.394 / -0.410 |

The vertex term still MEASURES the innermost pixel resolution:
`hitres_pix_y_q1 = -0.400 +- 0.133` against -0.146 +- 0.934 from the mass term
on the same candidates, **7.0x tighter in sigma** (free 7.3x).
`mass_cf` NLL(0) is 198 lower ON over 8 000 candidates, of which
`8000 x ln(1/0.98109)` = 153 is the smaller `sigma_m` alone.

**Injections**, `material_bpix_support6` x1.05, truth `ln(1.05)` = 0.0487902
physical, `corrected/truth = shift / (f_pri x truth)`:

| channel | OFF shift / `f_pri` | OFF corr/truth | ON shift / `f_pri` | ON corr/truth |
|---|---|---|---|---|
| vertex, CF | -0.01922 / 0.400 | **0.985** | -0.01899 / 0.395 | **0.985** |
| vertex, fit's Q | -0.02287 / 0.443 | 1.058 | -0.02180 / 0.431 | **1.037** |
| MASS, CF | -0.00296 / 0.063 | 0.963 | -0.00348 / 0.075 | 0.951 |
| JOINT, CF (frozen) | -0.02017 / 0.419 | 0.987 | -0.02017 / 0.420 | **0.984** |

Leakage rms 0.034 (CF) in both; the free regime's 0.118 on the fit's-Q arm
(+0.81 sigma onto `pix_y_q3`) drops to **0.035** ON.
`hitres_pix_x_q2` variance x1.10 (linear mode, expected
`-eps_inj/(1+eps_inj) x (1+eps_base)`): vertex CF **0.932** ON / 0.944 OFF,
fit's Q 0.946 / 0.963; leakage < 0.01 sigma everywhere.

#### 12.9 DY — section 10's 1.86 % is COMBINATORICS, not resolution

`logs_on/dy_outliers.log`, `logs_on/cmp_dy.log`, `dy_mass_shift.pdf`.
The constrained fit writes **10 654** candidates against 7 948, and every
OFF candidate is also an ON candidate (OFF-only = 0 at the event level): the
constraint converges on 2 616 events the free fit produced nothing for. After
the extraction selection, 10 325 against 7 841, with 27 event keys the OFF
selection keeps and the ON one drops.

| sample | `P(\|z\|>3)` | `P(\|z\|>4)` | `P(\|z\|>5)` |
|---|---|---|---|
| ON, all 10 325 | 1.36 % | 0.38 % | **0.165 %** |
| OFF, all 7 841 (the section-10 number) | 3.10 % | 2.13 % | **1.862 %** |
| **ON, uniquely matched 7 498** | 1.214 % | 0.293 % | **0.107 %** |
| **OFF, uniquely matched 7 498** | 1.214 % | 0.280 % | **0.107 %** |
| ON, 1 candidate/event (10 201) | 1.225 % | — | 0.108 % |
| ON, >1 candidate/event (124) | 12.10 % | — | **4.84 %** |
| OFF, 1 candidate/event (7 530) | 1.554 % | — | 0.398 % |
| OFF, >1 candidate/event (311) | 40.51 % | — | **37.30 %** |
| OFF-only (29 candidates) | 100 % | — | **89.7 %** |

**On the same candidate the tail is identical.** Section 10's 1.86 % is made
of the 311 candidates that sit in multi-candidate events — combinatorial
dimuon pairings, 116 of the 146 outliers — plus the 29 candidates the
constrained selection rejects. So a vertex term on data still needs an outlier
component, but it is a PAIRING/selection component, not a resolution one, and
with the constraint on the inclusive figure is 0.165 % with data/CF at 5 sigma
**7.5** instead of 85 (data/chi2 2 871 instead of 32 468), and untrimmed
`Var(z)` 1.189 instead of 1 123.

**Why both masses are exported**, measured on the ON DY sample:

| | n | median `\|m_c - m_unc\|` | in units of `sigma_m`, median | p90 |
|---|---|---|---|---|
| core, `\|z_v\| < 3` | 10 185 | 55.6 MeV | 0.052 | 0.289 |
| tail, `\|z_v\| > 5` | 17 | **684 MeV** | **0.511** | **10.76** |

For a tail candidate the unconstrained mass is up to ten sigma from the
constrained one while the core moves by 5 % of a sigma: having both is what
lets that be seen, and vetoed, at analysis level.

#### 12.10 Cost — unchanged

`cost_vtx.py --bill` on `task_0000` of each production: the vertex block
**30.40** kB/candidate ON against 30.41 OFF, the mass block 30.26 against
30.23, the file 327.0 MB against 328.0 MB for the same 2 000 events. The two
new scalars are free.


### 13. IS THE TAIL BACKGROUND?  GEN TRUTH, AND A MINIMUM SIZE FOR A PAIR

Two questions from David, both about candidates the fit should arguably never
have written.

**A.** Section 12.9 argued the DY vertex tail is COMBINATORIAL -- 116 of 146
free-regime outliers sat in the 311 candidates of multi-candidate events.
That is a COUNTING argument. "If the candidates with large vertex residuals
are background we should cut on it, this would also be good for background
reduction. Did you verify that those are background by doing a gen matching?"
No gen matching had been done.

**B.** `ndof == 0` crashed an earlier DY production (`fab515e`). "How are
track pairs with 9 hits treated? I suggest to also add a cut with requiring
more than 9 (10 w/o vertex constraint) hits, these pairs are most likely
anyway bad."

Code: maker `23b4c9c7046` on `cvh-exports-clean-260911`; analysis
`genbkg.py` + `run_bkg.sh` + `run_all.sh` stages
`extract-bkg` / `bkg` / `bkg-hits` / `bkg-gate`.  Productions
`/ceph/.../runs_vtxres_260911/bkg/{dy_vtxon_gen,dy_vtxoff_gen,gate_gun,
gate_dy}`, npz in `bkg/runs/`, figures
`~/public_html/ZMass/cvh/260912_vtxbkg/`, logs `logs/genbkg_*`.

#### 13.1 What was added to the maker

**Gen provenance**, behind the existing `doGen_`.  `Mu*gen_dr` said only that
SOME status-1 gen muon of the right charge sat within dR < 0.1 of the leg.
That cannot tell ONE MUON RECONSTRUCTED TWICE from two different muons, which
is exactly the distinction the tail turns on.  New, per leg:
`Mu{plus,minus}gen_pdgId`, `_idx` (position in the gen collection),
`_motherPdgId` and `_motherIdx` (the first ancestor that is not itself a muon
copy), `_isPrompt`, `_fromHardProcess`; plus `Jpsigen_sameDecay` -- both legs
matched, to DIFFERENT gen particles, whose first non-muon ancestor is the SAME
particle.  The MATCH ITSELF IS UNCHANGED.  `Mu*gen_dr` and the reco/gen pT
ratio were already derivable.

**A minimum size for a pair**, pre-fit: `minNdof` (default 1),
`minPairHits` (default -1 = auto = 10 ON / 11 OFF), `minLegHits` (default 0 =
off).  Counted as `skipped[ndof<N]` / `skipped[hits<N]` /
`skipped[leghits<N]`.  Documented in all 8 two-track cfis and both drivers.

#### 13.2 THE CLASSIFICATION

Criterion, stated so it can be argued with: the leg is MATCHED if the closest
status-1 gen muon of the SAME CHARGE lies within **dR < 0.1** of its fitted
momentum (the maker's own rule).  Then, in order:

  `unmatched`   a leg with no such particle
  `otherdecay`  both matched, but to muons with DIFFERENT ancestors
  `nonres`      one decay, but the ancestor is not a resonance (never seen)
  `dup`         the resonance's two daughters, but ANOTHER candidate of the
                same event matched the SAME two gen particles.  The candidate
                with the smaller dR(+)+dR(-) keeps `signal`; the rest are
                `dup`.  THAT TIE-BREAK USES TRUTH and nothing downstream may
                use it as a selection.
  `signal`      the rest

#### 13.3 THE ANSWER: yes, the tail is background

`dy_vtxoff_gen` (FREE regime, 6 x 4000 events, 10 494 selected candidates).
Quoted first because this is where the tail LIVES.

| class | N | fraction | `P(>3)` | `P(>4)` | `P(>5)` | `P(>10)` |
|---|---|---|---|---|---|---|
| signal | 10 266 | 0.9783 +- 0.0014 | 0.0137 +- 0.0011 | 0.0044 +- 0.0007 | **0.0024 +- 0.0005** | 0.0009 +- 0.0003 |
| dup | 117 | 0.0111 +- 0.0010 | 0.906 +- 0.027 | 0.906 +- 0.027 | **0.872 +- 0.031** | 0.795 +- 0.037 |
| otherdecay | 12 | 0.0011 +- 0.0003 | 0.417 +- 0.142 | 0.333 +- 0.136 | 0.250 +- 0.125 | 0.250 +- 0.125 |
| unmatched | 99 | 0.0094 +- 0.0009 | 0.667 +- 0.047 | 0.626 +- 0.049 | **0.606 +- 0.049** | 0.485 +- 0.050 |

**Of the 190 candidates at `|z_v| > 5`, 86.8 % +- 2.5 % are BACKGROUND**
(102 dup + 60 unmatched + 3 otherdecay, 25 signal); at `|z_v| > 10`,
**94.1 % +- 1.9 %**.  **201 of the 228 background candidates sit in
multi-candidate events**, where half of everything is background (188 signal
against 201 background) -- section 12.9's argument, confirmed by truth.

The `dup` class is what the tail is made of: 112 gen decays reconstructed more
than once, median `|z_v|` **79.1**, p90 362, against 0.67 for signal.  And the
vertex residual is the RIGHT discriminator between two reconstructions of one
decay: the gen-preferred one has the smaller `|z_v|` in **97.3 % +- 1.5 %** of
the 112 cases, against 83.0 % +- 3.6 % for the smaller chi2/ndof.

What the classes ARE:

| | N | med pT+ / pT- | med `\|eta\|` | med mass | med `sigma_v` | pair hits | multi-cand |
|---|---|---|---|---|---|---|---|
| signal | 10 266 | 39.2 / 39.2 | 1.05 | 90.7 | 0.0026 cm | 33 | 188 |
| unmatched | 99 | 24.8 / 25.9 | 1.48 | 75.1 | 0.0071 cm | 31 | 74 |
| dup | 117 | 29.2 / 33.3 | 1.90 | 82.8 | 0.0110 cm | 24 | 117 |

`unmatched` is ONE leg in 98 of 99, and that leg has median reco pT
**7.6 GeV** (p10 4.8, p90 28): a soft track paired with a real Z muon -- a
pileup muon (pileup is not in `prunedGenParticles` at all), a muon from a
hadron decay below the pruning threshold, or a fake.  `dup` is forward and
short.  Mother spectrum: 23 x 10 438, none found x 50, 421 x 2, 411 x 2,
521 x 1, 511 x 1 -- real heavy flavour exists and is negligible.

`dup` and `unmatched` legs are also badly MEASURED, not just mis-paired:
dR(+) median 0.0161 for `dup` against 0.00030 for signal, and reco/gen pT
p5-p95 **0.52-1.53** against 0.969-1.035.

**The CF model describes the SIGNAL class and nothing else** -- data/CF at
3 / 4 / 5 sigma **3.3 / 7.0 / 10.2** for signal, 175 / 889 / 1964 for
background, 238 / 1833 / **4890** for the duplicates.  The residual signal
tail is the FEW-HIT LEGS: gen signal with a weaker leg of <= 8 valid hits has
`P(|z_v|>5)` = 0.104 +- 0.044 on 48 candidates against **0.0020 +- 0.0004**
on the 10 218 with more -- which is the gun's free-regime floor.

#### 13.4 THE CUT

| `\|z_v\| < t` | ALL (10 266 S / 228 B) | 1 cand/ev (10 078 / 27) | >1 cand/ev (188 / 201) |
|---|---|---|---|
| t = 3 | 0.9863 +- 0.0011 / 0.776 +- 0.028 | 0.9864 / 0.815 +- 0.075 | 0.9787 +- 0.0105 / 0.771 +- 0.030 |
| t = 4 | 0.9956 +- 0.0007 / 0.754 +- 0.029 | 0.9957 / 0.741 +- 0.084 | 0.9894 +- 0.0075 / 0.756 +- 0.030 |
| **t = 5** | **0.9976 +- 0.0005 / 0.724 +- 0.030** | 0.9977 / 0.704 +- 0.088 | 0.9894 +- 0.0075 / 0.726 +- 0.031 |
| t = 10 | 0.9991 +- 0.0003 / 0.632 +- 0.032 | 0.9993 / 0.630 +- 0.093 | 0.9894 +- 0.0075 / 0.632 +- 0.034 |

(signal efficiency / background rejection, both against GEN TRUTH).  Purity
0.9783 -> **0.9939 +- 0.0008** at t = 5 inclusively, **0.483 -> 0.772 +-
0.027** in multi-candidate events.

**The floor a cut cannot go below.**  On the J/psi gun, where **99.996 %** of
candidates have both legs gen-matched and only 30 of 55 853 sit in a
multi-candidate event (against 3.7 % on DY), `P(|z_v| > t)` on ~298 000
selected candidates is

| | `>3` | `>4` | `>5` | `>10` |
|---|---|---|---|---|
| `prod_vtxon` ON | 0.0081 +- 0.0002 | 0.0025 +- 0.0001 | **0.00104 +- 0.00006** | 0.0000 |
| `prod` OFF | 0.0088 +- 0.0002 | 0.0032 +- 0.0001 | **0.00161 +- 0.00007** | 0.00031 |

and those 30 multi-candidate gun candidates have `P(|z_v|>5)` = **13.3 %**
against 0.090 % for the 55 823 single-candidate ones.  Even in a
one-resonance-per-event gun the multi-candidate population IS the outlier
population.

#### 13.5 A MINIMUM ON THE WEAKER LEG is a different, complementary cut

`minPairHits` removes **zero** SELECTED candidates.  Its two victims in
`dy_vtxon` are both in ONE event (run 1, lumi 19061, event 14924058), both
`nvalid = (3,6)`, `|z_v|` 58.3 and 0.0, chi2/ndof 1.7e3 and 6.6e4 -- an event
that produces nothing but garbage and that the extraction selection already
threw away.  So `minPairHits` is a crash/CPU guard, not a physics cut.

`minLegHits` is the physics cut.  Free regime, gen truth:

| weaker leg | N | signal | dup | unmatched |
|---|---|---|---|---|
| < 4 | 4 | 1 | 0 | 3 |
| < 6 | 22 | 2 | 8 | 12 |
| < 8 | 148 | 17 | 100 | 31 |

| `minLegHits` | removed | signal eff | bkg rejection | bkg among removed |
|---|---|---|---|---|
| 6 | 22 | 0.9998 +- 0.0001 | 0.088 +- 0.019 | 0.909 +- 0.061 |
| 7 | 80 | 0.9997 +- 0.0002 | 0.338 +- 0.031 | **0.963 +- 0.021** |
| 8 | 148 | 0.9983 +- 0.0004 | 0.575 +- 0.033 | 0.885 +- 0.026 |
| 10 | 264 | 0.9897 +- 0.0010 | 0.693 +- 0.031 | 0.599 +- 0.030 |

**They are NOT the same candidates.**  At `minLegHits = 8` against
`|z_v| > 5`: 113 fail both, **77 fail only the residual cut** (a healthy
weaker leg with a huge residual -- the duplicate pairings) and **35 only the
hit cut** (a thin leg with a small residual -- the degenerate `sigma_v`).
Together:

| cut | signal eff | bkg rejection | purity |
|---|---|---|---|
| `\|z_v\| < 5` | 0.9976 +- 0.0005 | 0.724 +- 0.030 | 0.9939 |
| weaker leg >= 8 | 0.9983 +- 0.0004 | 0.575 +- 0.033 | 0.9906 |
| **both** | **0.9961 +- 0.0006** | **0.811 +- 0.026** | **0.9958** |

**RECOMMENDED: `|z_v| < 5` with `minLegHits = 8`** -- efficiency
**0.9961 +- 0.0006**, rejection **0.811 +- 0.026**, purity 0.9783 ->
0.9958 +- 0.0006.  `|z_v| < 5` alone is the conservative version
(0.9976 +- 0.0005 / 0.724 +- 0.030): it sits a factor 2 above the gun's
pure-signal floor, so it costs essentially only the irreducible resolution
tail.  `|z_v| < 3` buys 5 % more rejection for 14x the signal loss (1.4 %
against 0.24 %) and is not worth it.

#### 13.6 THE CONSTRAINT IS ITSELF A BACKGROUND REJECTOR

The constrained leg `dy_vtxon_gen` looks much cleaner -- 10 325 selected
candidates, **99.30 %** signal, 72 background, `|z_v| > 5` only **17**
candidates of which 41.2 % +- 11.9 % background, `|z_v| > 10` EMPTY -- and
the reason is NOT that the fit is different.  Before the extraction's
`chi2/ndof < 3`:

| | N | background | `\|z_v\|>5` | bkg fraction there |
|---|---|---|---|---|
| ON, no chi2 cut | 10 638 | 318 (2.99 %) | 190 | **0.874 +- 0.024** |
| OFF, no chi2 cut | 10 628 | 310 (2.92 %) | 242 | **0.884 +- 0.021** |
| ON, chi2/ndof < 3 | 10 325 | **72 (0.70 %)** | **17** | 0.412 +- 0.119 |
| OFF, chi2/ndof < 3 | 10 494 | 228 (2.17 %) | 190 | 0.868 +- 0.025 |

**The two regimes contain the SAME background.  `chi2/ndof < 3` rejects 77 %
of it with the constraint on and 26 % with it off**, because a combinatorial
pairing cannot satisfy a common-vertex constraint.  That, not a resolution
difference, is what turned section 12.3's inclusive DY tail from 1.86 % into
0.165 %.  In the constrained regime `chi2/ndof` also beats `|z_v|` at picking
the right duplicate (94.3 % +- 2.5 % against 79.6 % +- 4.3 %) -- the reverse
of the free regime, for the same reason.

The same holds for how well the CF describes gen SIGNAL: data/CF at 5 sigma
is **9.7** (ON) and **11.3** (OFF) before the chi2 cut, and **4.07** (ON)
against 10.2 (OFF) after it.  The constrained regime's better-described tail
is the chi2 selection working, not a different residual.

Consequently, in the ON regime AFTER the chi2 cut the `|z_v|` cut has little
left to reject (t = 5: efficiency 0.9990 +- 0.0003, rejection
0.097 +- 0.035; t = 3: 0.9880 +- 0.0011 / 0.236 +- 0.050), and gen SIGNAL has
`P(|z_v|>5)` = **0.0010 +- 0.0003** -- exactly the gun's ON floor of
0.00104 +- 0.00006.  Before the chi2 cut it is the same picture as the free
regime (t = 5: 0.9977 / 0.522 +- 0.028; `minLegHits = 8`: 0.9984 /
0.654 +- 0.027; both: **0.9961 / 0.837 +- 0.021**).

#### 13.7 THE MASS SIDE: the constraint moves background, not signal

`m_c - m_unc` in units of `sigma_m`, constrained DY leg:

| sample | N | median | p90 | max | mean `m_c - m_unc` |
|---|---|---|---|---|---|
| signal | 10 253 | **0.053** | 0.296 | 12.4 | +3.6 MeV |
| background | 72 | **0.414** | **10.8** | 49.5 | +6 991 MeV |
| background, `\|z_v\|>5` | 7 | **4.26** | 20.3 | 28.5 | +22 210 MeV |
| signal, `\|z_v\|>5` | 10 | 0.435 | 0.611 | 0.978 | +257 MeV |

Within **+-15 GeV of `m_Z`**: the constraint moves **8 background candidates
IN and 1 OUT, net +7 on 26** (a 27 % increase of in-window background), while
signal goes 9 505 -> 9 516 (+20 in, -9 out, +0.1 %).  So on the chi2-selected
sample **the constraint pulls background INTO the Z window**.  On the
unselected sample, where the background is wilder, it is neutral (+25 in,
-28 out, net -3 on 76) because the same candidates are thrown far out.

Either way the veto is available, and it is sharp: for in-window background
the median `|m_c - m_unc|` is **0.51 sigma_m** and the mean is
**+7.55 GeV**, against 0.053 sigma_m and +19 MeV for in-window signal.  That
is the argument for exporting BOTH masses, now with truth behind it.

#### 13.8 TASK B: the ndof arithmetic, as implemented

`ndof = nvalid + nvalidpixel - nstatefree` with `nstatefree = 10`, plus the
constraint rows (+3 beamspot, +1 pointing, +1 vertex, +1 mass on the
constrained pass) -- ONE measurement coordinate per strip hit, TWO per pixel
hit, against the ten state parameters the common vertex costs.  With the
vertex constraint on that is `n_meas - 9`, with it off `n_meas - 10`.

VERIFIED on every written candidate of all four productions:
10 654/10 654 and 300 017/300 017 satisfy `- 9`; 7 948/7 948 and
300 025/300 025 satisfy `- 10`.  **Zero deviations.**  So David's "more than
9 hits (10 without the constraint)" IS `ndof >= 1`, and `minNdof = 1` is that
requirement exactly.

| sample | written | ndof min | nvalid(pair) min | minNdof=1 cuts | minPairHits cuts |
|---|---|---|---|---|---|
| `dy_vtxon` ON | 10 654 | 1 | 9 | 0 | 2 (0.019 %) |
| `dy` OFF | 7 948 | 1 | 9 | 0 | 1 (0.013 %) |
| `prod_vtxon` ON | 300 017 | 3 | 9 | 0 | 1 (0.0003 %) |
| `prod` OFF | 300 025 | 2 | 9 | 0 | 1 (0.0003 %) |

**`ndof == 0` is never written** -- `fab515e` aborts the fit -- and the abort
IS reached: summed over the production logs it fired **1 time in 10 671**
`dy_vtxon` pairs and **2 in 7 964** `dy` pairs, and **0 in 300 370** gun pairs
in either regime.  It is a MiniAOD phenomenon: `slimmedMuons` keeps the hit
pattern but not every RecHit.

**`ndof == 1` is degenerate and is written as GOOD.**  `dy_vtxon` idx 3879:
`ndof` 1, hits (3,6), **`sigma_v` = 32 515 cm**, `z_v` = 7.2e-6,
chi2/ndof = 66 432, and `Jpsi_vtxok` **TRUE**.  The DCA is fixed by the data,
the pull collapses to zero and the candidate dilutes the core.  Neighbours:
`ndof` 2 -> `|z_v|` = 58.3; `ndof` 3 -> `sigma_m` = 1041 GeV; `ndof` 5 ->
`sigma_m` = 73 859 GeV.

**Non-finite exports DO exist, and the cause is the WEAKER LEG:**

| sample | non-finite `Jpsi_sigmamass` | `Jpsi_vtxok` TRUE among them | weaker leg <= 3 |
|---|---|---|---|
| `dy_vtxon` | 5 (0.047 %) | 1 | 5/5 |
| `dy` | 10 (0.126 %) | 2 | 9/10 (<=4: 10/10) |
| `prod_vtxon` | 186 (0.062 %) | 76 | **186/186** |
| `prod` | 525 (0.175 %) | 83 | 462/525 (<=4: 525/525) |

Their PAIR totals are 13-23 hits, so no pair-sum cut can see them.  A further
391 (`prod_vtxon`) / 447 (`prod`) candidates have `sigma_v > 1000 cm`, ALL
flagged `Jpsi_vtxok`, median weaker leg 2 hits.  `Jpsi_vtxres`, `Jpsi_vtxsig`,
`Jpsi_vtxz`, `Jpsi_vtxdchi2`, `Jpsi_mass` and `Jpsi_mass_unc` are finite
everywhere; only `Jpsi_sigmamass` is not, and `cfmass_ok` is False on every
one of them, so nothing downstream consumed them.

**There is no minimum-hit requirement anywhere else in the chain.**  DY:
`slimmedMuons` -> `TrackProducerFromPatMuons` (`innerTrackOnly=False` ->
`muonBestTrack`, `ptMin = -1`, the track must have `extra().isAvailable()`)
-> `DiMuonTrackVertexCandidateProducer` (opposite sign, 60 < m < 120).  Gun:
`useLegacyPairLoop=True`, the all-pairs loop over `generalTracks`.  The
maker's only pre-fit guards were the charge sum and `nhits != 0` per leg.

**The gate.**  The new build re-run on the SAME inputs as the reference
production, candidates matched on (run, lumi, event, nhits, nvalid):

| | matched | bit-identical | missing from ref | ref-only | ndof min | non-finite |
|---|---|---|---|---|---|---|
| `gate_gun` (200 gun ev) vs `prod_vtxon/task_0000` | **192** | **29/29 branches** | 0 | 0 | 10 | 0 |
| `gate_dy` (400 DY ev) vs `dy_vtxon/task_0000` | **186** | **29/29 branches** | 0 | 0 | 12 | 0 |

Both summaries show `skipped[ndof<1]=0  skipped[hits<10]=0
skipped[leghits<0]=0`.

**And the full re-production closes on the reference.**  `dy_vtxon_gen`, the
same 6 x 4000 events with the cut on:

    attempted 10 668  succeeded 10 652
    skipped[ndof<1]=1  skipped[hits<10]=2  skipped[leghits<0]=0

against `dy_vtxon`'s attempted 10 671, succeeded 10 654, `fail[ndof]=1`.  The
three skipped pairs never reach `++fitAttempted_` (10 671 - 3 = 10 668), the
two written ones are exactly the `nvalid == 9` pair, **`ndof` min rises from
1 to 3**, and the SELECTED sample is **10 325 candidates with
`P(|z_v| > 3/4/5/10)` = 0.0136 / 0.0038 / 0.0016 / 0.0000 -- identical to the
reference**, because everything the cut removes the extraction selection had
already removed.

#### 13.9 A CORRECTION to section 12.9

"The constrained fit writes 10 654 candidates against 7 948 ... the constraint
converges on 2 616 events the free fit produced nothing for" has nothing to do
with the constraint: **the two productions ran on different numbers of
events.**  `dy/task_0000/local.log` carries `nEvents=3000`; `dy_vtxon` ran the
slurm array at 4 000.  Per event they agree:

| production | events/task | pairs attempted | attempted / event |
|---|---|---|---|
| `dy` (free, 09-11) | 3 000 | 7 964 | 0.4424 |
| `dy_vtxon` (09-12) | 4 000 | 10 671 | 0.4446 |
| `dy_vtxoff_gen` (free, today) | 4 000 | **10 668** | **0.4445** |

Re-running the same 6 files with the same `doVtxConstraint=False` at 4 000
events writes 10 648 candidates against 10 654 constrained -- a 0.06 %
difference, not 34 %.  The tail FRACTION is unchanged: 190/10 494 = 1.81 %
now against 146/7 841 = 1.86 % then.

#### 13.10 Figures

`~/public_html/ZMass/cvh/260912_vtxbkg/`, one file per panel, PNG twin beside
every PDF, 21 panels: `density_<leg>_{signal,background,dup,unmatched}`
(density + CF model + data/CF ratio panel), `cut_roc_<leg>` (efficiency
against rejection, the three event-multiplicity classes) and
`cut_composition_<leg>` (the `|z_v|` spectrum stacked by class), for
`<leg>` = `dy_vtxon_gen`, `dy_vtxoff_gen` and the `_all` (pre-chi2-cut)
version of each.


### 14. THE BEAM-LINE (LUMINOUS-REGION) CONSTRAINT, AND ITS TWO RESIDUALS

The luminous region is a GAUSSIAN NOISE BLOCK: constraining the common vertex
to the beam line with the beam-width covariance is mathematically ONE EXTRA
HIT shared by the two legs.  This section is what came of implementing it
properly -- the defect that was in the code, the gates, and what the rows and
the two residuals they create are worth.  **The PROMPT channels only** -- Z, DY and
Upsilon -- because the constraint is right for a prompt resonance and wrong
for the non-prompt fraction of a charmonium sample and for every displaced
channel (sections 14.17 and 14.15 point 4). It is ON BY DEFAULT there since
`dbdedfde3c1`.

#### 14.0 Where it lives

| what | where |
|---|---|
| build area | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` |
| branch | `cvh-exports-clean-260911`, FAST-FORWARDED to `beamline-260913` @ `0cb6c291354`, finished on top of it in **`dbdedfde3c1`** |
| maker | `src/Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc` |
| the base-class doc | the "BEAM-LINE (LUMINOUS-REGION) CONSTRAINT" block of `ResidualGlobalCorrectionMakerBase.h` -- the algebra, the two defects, the "+" form, the whitened pair and the mean-term identity |
| analysis scripts | `resolution/vtxres/{gates_bs,gate_cache,gate_defaults_cfg,cmp_bson,bkg_bs,bs_genvtx,width_report}.py`, `run_{prod_bs,prod_bs_old,timing_bs,timing_cache,gate_defaults,ladder_bs,all_bs,all_bsfinal}.sh` |
| the 3x3 covariance (section 14.20) | `resolution/vtxres/{gate_beam3,gate_beam3_card,beam3_gen,beam3_report,beam3_pulls,beam3_plots}.py`, `run_{prod_beam3,all_beam3}.sh`, `mark_complete.sh`; the term is `rabbit/unbinned.py::MaterialCFTerm` (`beam3_params` / `beam3_units` / `beam3`) with `tests/test_beam3.py` |
| the three NEW gates | `gate_defaults_cfg.py` (the producer PSet expanded in BOTH areas, per channel), `gate_cache.py` (bit-identity on EVERY comparable branch, matched on (run, lumi, event, pT rank), with the beam-functional branches declared expected-to-move under `--rows-on`), `width_report.py` (the width floats with an EDM certification that REFUSES to quote an uncertified fit) |
| outputs | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/` |
| figures | `~/public_html/ZMass/cvh/260913_beamline/` (the first pass) and `~/public_html/ZMass/cvh/260913_bsfinal/` (the finish) |
| logs | `resolution/vtxres/logs_bs/` |

Switches: `bsConstraint` (the three rows, ON by default for the PROMPT
channels -- Z, DY and Upsilon -- and OFF for charmonium and every displaced
channel, section 14.17), `beamWidthScale` (default 1.0; 1e6 is the
weightless-rows gate), `exportBsResidual` (the two functionals, default
False).

The FIRST pass was built in a separate area (`dev3`, its own git worktree)
because dev2 was running a production throughout; the finish fast-forwarded
`cvh-exports-clean-260911` onto it (`git merge-base --is-ancestor
dbfe6e4b2c2 0cb6c291354` is true, so it is a real fast-forward) and everything
since is in dev2.

**The build must run on a SUBMIT NODE.** `scram b` from a host scram does not
recognise as el9 (it reports `SCRAM architecture 'el9' on host with operating
system 'linux514'`) fails with ~10 errors of the form `__gthread_cond_t ...
cannot convert '<brace-enclosed initializer list>' to 'unsigned int'` in
`<atomic>` / `std_mutex.h` -- the wrong system headers. `ssh submit50 'scram b'`
builds clean. `/ceph` is likewise unreadable from such a host (Permission
denied) and fine from a submit node, so every production and every read goes
through one.

The chain:

    ./run_prod_bs.sh 6 0 5                  # OUTTAG / BSOPTS / EXTRA select the leg
    ./run_prod_bs_old.sh 6 0 5              # the OLD build, for gate G4
    ./run_timing_bs.sh 400                  # the maker cost, three configurations
    ./run_all_bs.sh gates|cmp|extract|bkg|plots|bill|cards|fits|fisher|eff|recovery|certify

#### 14.2 THE DOUBLE EMISSION — the finding

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

#### 14.3 The MC beam spot vs the simulated luminous region  (study item a)

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

##### The J/psi gun

`prod_vtxon` (14 969 gen-matched of 14 973): gen vertex median
(+0.091635, +0.169544, +0.9845) cm with MAD (10.12, 9.95 um, 3.653 cm) --
**the gun IS smeared with the same beam spot**, to the same agreement as DY.
So the gun is technically a valid sample for the beam rows, and it is prompt
by construction. It is nevertheless LEFT OUT of the physics conclusions,
because a real J/psi sample is not prompt (the B-decay fraction) and the gun
would certify a constraint that data could not carry.

#### 14.4 The productions

All six read the SAME six DY MiniAOD files
(`production/filelist_dymc_8p5M_260905.txt` lines 1-6) with the
`condor_dymc_v2` configuration verbatim plus `exportVtxResidual=True`, so
every comparison is same-candidate.  Under
`/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/`.

| tag | build | rows | events/task | what for |
|---|---|---|---|---|
| **`dy_bs_final`** | **dev2 @ `dbdedfde3c1`** | **ON, all the new exports** | **4000** | **THE STUDY SAMPLE (10 413 candidates, 10 254 on the baseline)** |
| **`dy_bsoff_final`** | dev2 | OFF | 4000 | the same-build rows-OFF reference |
| `gate_defaults/{gun,dy}_{ref,new}` | dev3 / dev2 | off / on | 200 / 400 | the BIT-IDENTITY gates |
| `timing_cache{,_rep}/*` | dev3 / dev2 | both | 400 | the cache cost |
| `dy_bs` | dev3 | ON, nominal widths | 4000 | the FIRST pass's study sample |
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

#### 14.5 The export bill, measured (smoke file, 90 candidates)

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

Re-measured on the FINAL production (`cost_vtx.py` on `dy_bs_final`): the beam
block is **72.80 kB/cand** (0.522 TB at 7 M Z) against the vertex block's 37.30
and the mass block's 37.41. The ten new branches of this pass
(`Jpsi_bscovlo`, `Jpsi_bslinv`, `Jpsi_bsmeig`, `Jpsi_bswidtherr`, the six
direction-split shares) cost **~1 kB/candidate between them** -- the block is
still dominated by the per-group CF exponents (5 x ~12 kB/cand) and
`resinfbsv` (7.07).

Two functionals, so the beam block is **1.95x the vertex block** -- it is
almost exactly two copies of it, as it should be.  The whole file is
181.9 kB/cand with all three functionals on.

#### 14.6 The two functionals, and their basis  (a decision, stated)

The maker exports THREE things about the same 2-vector:

* `Jpsi_bsres` (cm) and `Jpsi_bscov` (cm^2, packed xx/xy/yy) -- the leave-one-
  out residual in the GLOBAL transverse frame and its covariance;
* `Jpsi_bsz` -- the LOWER-CHOLESKY whitened pull, `Cov = L L^T`, `z = L^-1 r`,
  in the order (x, y), so `z[0]` is the x pull and `z[1]` the y pull GIVEN x.
  The basis is a choice; the chi2 `Jpsi_bschi2 = z^T z` is not. **This pair IS
  the two CF functionals** (below), and `Jpsi_bslinv` carries `L^-1`.

**The two CF FUNCTIONALS are the WHITENED PAIR** `z = L^-1 r_bs`, not the two
global components. `Cov(r_bs) = L L^T` with `L` lower triangular in the order
(x, y), so `Cov(z) = I` by construction: each pull has UNIT variance -- the
closure is `sum_b |a_b|^2 == 1` -- and the two are UNCORRELATED.

The first pass used the two GLOBAL components `r_x` and `r_y`, each
standardised by its own marginal `sqrt(Cov_kk)`, for interpretability: each
was then tied one-to-one to a beam-spot parameter and the mean term was
literally `-w[bs row x]`. `sandwich/quoted` for the `bs` channel alone came out
at **1.385** against 0.918 for the vertex term, and that was attributed to the
two components being CORRELATED. **They are not**: on the full sample the model
correlation `Cov_xy/sqrt(Cov_xx Cov_yy)` is **-0.011** and the measured
`corr(z_x, z_y)` is **-0.010 +- 0.010** (the "~0.2" quoted at the time came
from the 90-candidate smoke file; section 14.12). What the two terms DO share
is INFORMATION -- the same hits and the same material seen twice -- and
conditioning the second pull on the first removes that: whitening takes
`sandwich/quoted` to **1.138**, an excess-over-1 smaller by a factor 2.8.

Note what whitening can and cannot touch: the lower-Cholesky first row is the
x marginal, so **`z_1` IS the first pass's global x functional, bit for bit**,
and only the second changes (`Var` 1.1811 -> 1.1094, 5 sigma data/CF 16 ->
6.8).

What the whitening costs, stated: neither pull is the response to a SINGLE
beam-spot parameter any more. `Jpsi_bslinv` (the packed `L^-1`) is exported so
that any influence weight or mean response is one 2x2 multiply from the global
basis, the raw pair `Jpsi_bsres` / `Jpsi_bscov` is still written, and the
mean-term identity survives exactly in the form `Jpsi_bsmeanbs == -L^-1 P`
(gate G6, 1.0e-10).

The residual correlation is still measured rather than assumed: `fisher_vtx.py`
sums the two terms' per-batch gradients over the SAME candidates, so `J`
carries the within-candidate correlation and the SANDWICH
`(H+P)^-1 J (H+P)^-1` is the variance the estimator actually has. Every error
quoted from the beam channels is the sandwich one, and the `sandwich/quoted`
ratio of the `bs` channel is the over-counting diagnostic -- it is what the
whitening had to move.

**HOW `Cov(r_bs)` IS BUILT -- the "+" form, and why it matters.** With
`A = C[7:10,7:10]` the fitted vertex covariance (rows in) and
`M = covBS - A = Cov(rho_B)`,

    C_{-B} = A + A M^-1 A            the rows-OFF vertex covariance
    Cov(e) = C_{-B} + covBS          >= covBS, so positive definite always
    e_B    = rho_B + A M^-1 rho_B    = x_v^{-B} - b0

The same quantities are `covBS M^-1 covBS` and `covBS M^-1 rho_B`, which is how
the first pass wrote them -- and those run the WHOLE answer through `M^-1`.
`M` is a difference of two nearly equal covariances (it degenerates as the
tracks stop constraining the vertex, `A -> covBS`; `Jpsi_bsmeig`, the smallest
eigenvalue of `M` over `covBS`'s own scale, reaches **3e-5**) and `A` carries
the numerical error of a large sparse solve, so the product could lose the
leading order entirely and could return an INDEFINITE covariance. **That is
what produced the 5 sigma tail of the first pass.** In the "+" form the
leading order is explicit, only the correction is amplified, and `Cov(e)` can
no longer come back indefinite. `A M^-1` is an `LLT` solve of `M`, not an
explicit inverse, and `Jpsi_bscovlo` exports `C_{-B}` so the rows-OFF run can
check it DIRECTLY (gate G7a).

**Why one component is much worse measured than the other.** `Cov(r_bs)` is
`C_{-B} + Sigma_{xy|z}`: the vertex covariance WITHOUT the beam rows plus the
luminous region's own transverse spread. Two nearly back-to-back tracks fix
the vertex well PERPENDICULAR to their common direction and poorly ALONG it,
so for a Z pair one transverse direction has `sigma ~ 15 um` (the beam width
itself) and the other can reach hundreds of microns. On the smoke sample the
median `sigma_bs,x` is 25.6 um with a p95 of 248 um. The global (x, y) basis
mixes the good and the bad direction by the pair's phi, which is why the two
pulls look alike in the ensemble even though per candidate they are not.

#### 14.7 The gates  (`gates_bs.py`, the 1200-event legs, 3287 candidates, 3153 on the baseline)

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

#### 14.8 Cost  (`run_timing_bs.sh`: 400 events, ONE file, ONE host, run SEQUENTIALLY)

186 candidates in all three configurations -- the beam rows change no
candidate's selection.

| configuration | wall | s/candidate | file | kB/candidate |
|---|---|---|---|---|
| `bsConstraint=False` | 886 s | 4.76 | 33.79 MB | 177.4 |
| `+ the three rows` (`exportBsResidual=False`) | 895 s | 4.81 (**+1.0 %**) | 34.27 MB | 179.9 |
| `+ the two functionals` | 1440 s | 7.74 (**+62.6 %**) | 48.13 MB | 252.7 (**+75.3**) |

**The rows are free; the two functionals are not.** The extra 62 % is two
more `cvhcf::trackExponents` calls and two more per-block influence loops.

**THE `sqrt(dV_b)` CACHE, AND A CORRECTION.** `dV_b^{1/2}` is a property of the
BLOCK, not of the functional, and it was being re-decomposed in each of the
four influence loops (mass, vertex, beam-x, beam-y). It is now built ONCE per
candidate (`ressqrtdV` / `resionidir`, the latter for the ionization-sign
rule) and read by all four. The gate is bit-identity and it passes:
**264/264 branches on the J/psi gun** (rows off) and **290/313 on DY with the
rows ON**, the 23 movers being exactly the beam-functional outputs the
whitening changes.

But the first version of this section predicted that the cache "should remove
most of the 62 %", and **that was wrong**. Measured (`run_timing_cache.sh`,
400 events, one CPU, sequential):

| configuration | wall |
|---|---|
| `old_off` (no cache, rows off) | 887 s |
| `old_full` (no cache, rows + both functionals) | 1379 s (+55.5 %) |
| `new_off` (cached, rows off) | **853 s** (**-3.8 %**, output BYTE-IDENTICAL) |
| `new_rows` (cached, rows on, no functionals) | 830 s |
| `new_full` (cached, rows + both functionals) | 1356 s (+59.0 % over `new_off`) |

`old_off -> new_off` is the clean comparison -- ONE of two decomposition passes
removed, everything else identical -- and it fixes the scale: **one pass over
all blocks' `dV_b^{1/2}` is worth ~34 s per 400 events, ~4 % of the maker.**
Four passes are therefore ~12 % of the +55 %, not most of it. **The cost of the
two beam functionals is the two extra `cvhcf::trackExponents` calls**, and that
is where any future saving has to come from. The cache is kept because it is
free, bit-identical, and makes each FURTHER functional cheaper.

**THE FOUR FUNCTIONALS NOW SHARE ONE EVALUATOR PASS -- AND IT IS FREE, NOT
CHEAPER** (`run_taugrid_ab.sh`, `cmp_taugrid.py`, dev2 `f7fbef244c6`). Every
`cvhcf` exponent primitive reads the weight only through the PRODUCT `w tau`
(`phi_{aU}(tau) = phi_U(a tau)`), so the mass, vertex and two beam weight sets
are the same primitive evaluated on the CONCATENATED argument list
{ w_{b,k} tau_j }: one `cvhcf::trackExponents(in, nfunc, out)` instead of four
calls, with the pooling by global index, the block gather, `sq2`, the Moliere
step parameters and their `gshape_elec` rows, `delCarveFactor`,
`makeRadSpectrum` and the per-group row selections built ONCE. The arguments
are formed by the same expression, in the same association, the
single-functional path forms them with, so it is not "exact to 1e-7", it is
BITWISE: gun 188 candidates / 264 branches, DY-with-beam-rows 179 / 323,
single-track 403 / 111 -- **zero values differ anywhere**, `Jpsi_vtxvchk`,
`Jpsi_bsvchk` and the mass closures included, the ROOT files the same size.

400 DY events, one pinned cpu, interleaved, first configuration repeated last:

| functionals | old | new | |
|---|---|---|---|
| 1 (mass only) | 570 s | 563 s | **-1.2 %** -- the NULL, `nf = 1` is the same code |
| 2 (+ the vertex DCA) | 911 s | 851 s | -6.6 % |
| 4 (+ the two beam pulls) | 1367 s | 1366 s | -0.1 % |
| 4, repeated at the end | 1359 s | 1347 s | -0.9 % |

**-0.5 % on the full export, i.e. nothing**: the `nf = 1` null is -1.2 % and
the `m4_old` drift over the sequence is 1367 -> 1359, so ~1 % is the noise.
A second lane (cpu 5) decomposes it -- `exportCfExponents=False` runs the same
job in **346 s** (old) / 350 s (new) against **1397 / 1401** with the exponents
on, so the CF block is **1051 s, 75 % of the maker**, and merging four passes
into one removed **none** of it. (That cfoff pair is itself a null this change
cannot touch: 346 vs 350, +1.2 %.)

**The setup was never the cost, so the paragraph above is half right.** The
concatenated grid shares the weight-INDEPENDENT work; what the evaluator
actually spends is per-(functional, tau) transcendentals -- `blockExponent`'s
cos/sin per ionization step, `radBlock`'s `nsteps x nv x ntau` sines,
`interpTau`'s binary search per Moliere step -- and those scale with the number
of functionals however the arguments are grouped. `makeRadSpectrum` is 1/64 of
the radiative channel it feeds; the `gshape_elec` row is comparable to a
two-step block's tau loop, but multiple scattering is the small family. So the
beam functionals' +55 % IS the extra `trackExponents` calls, and **a real
saving has to come from the per-tau kernel or from fewer tau points, not from
merging the passes.** The merge is kept because it is exact, bit-identical, one
call site instead of four, and the right shape for a fifth functional.

A REPEAT of the five configurations on a second CPU is **not usable**: its
walls rise monotonically through the sequence (1139, 1900, 1163, 1309, 2176 --
`new_rows` 58 % above `new_off` while doing strictly more work), i.e. the node
loaded up during the run. **Standing method note: a sequential A/B on a shared
node is only valid if the load is stable across the whole sequence -- test it
by repeating the FIRST configuration at the END, rather than assuming it.**

At **7 M Z candidates**: the beam block costs **0.527 TB** of export and
**~5.8 kh** of extra CPU over the rows-off configuration.

#### 14.9 WHAT THE BEAM ROWS BUY -- and it is not what was expected

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

#### 14.10 G4b -- THE SIZE OF THE DEFECT, measured

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

#### 14.11 The nominal pulls, 1200-event leg (3157 candidates on the baseline) -- FIRST PASS, superseded by 14.12

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

#### 14.12 THE STUDY, 10 254 candidates (`dy_bs_final` 4000 ev x 6, against `dy_bsoff`)

##### The two pulls (`plot_vtx.py`, figures in `~/public_html/ZMass/cvh/260913_bsfinal/`)

| | N | mean | Var | skew | kurt | corr(sigma, z) |
|---|---|---|---|---|---|---|
| `z_1` (= the global x pull, identically) | 10254 | **+0.0128 +- 0.0107** | 1.1810 | +0.062 | 5.00 | +0.017 +- 0.010 |
| `z_2` (the y pull GIVEN x) | 10254 | **+0.0004 +- 0.0104** | **1.1093** | +0.060 | 4.39 | +0.009 +- 0.010 |
| `z_v` (for scale) | 10254 | +0.0078 +- 0.0104 | 1.1165 | -0.031 | 3.55 | +0.003 +- 0.010 |

**The mean is zero and there is no skew** -- the two beam residuals are
constraint residuals of exactly the vertex kind. `corr(sigma, z)` is
consistent with zero, so no self-consistent-sigma correction is needed
(the same justification the vertex term uses).

Tails, data / model:

| | 2 sigma | 3 sigma | 4 sigma | 5 sigma |
|---|---|---|---|---|
| `z_1` CF | 1.35 | 3.57 | 10.9 | **16.0** |
| `z_1` Gaussian (variance-matched) | 1.34 | 4.33 | 54.4 | 2621 |
| `z_1` Gaussian (the fit's `Q`) | 1.37 | 4.59 | 60.1 | 3061 |
| **`z_2` CF** | 1.14 | 2.74 | 6.29 | **6.84** |
| `z_2` Gaussian (the fit's `Q`) | 1.17 | 3.54 | 35.4 | 1156 |
| `z_v` CF (for scale) | 1.32 | 2.10 | 1.09 | -- (no data beyond 5) |

The CF beats the Gaussian by **160x at 5 sigma** on `z_1` and is still **16x**
short; on `z_2` it is 6.8x short. The vertex residual's CF closes at 5 sigma;
the beam one does not.

**WHAT THE TAIL IS NOT.** It is not the background (the displaced
`otherdecay` class is 8 candidates and contributes 1e-4 of the 1.8e-3 total).
And it is **NOT the conditioning**, which the first version of this section
asserted ("candidates where `M = covBS - C_vtx` is nearly singular, so the
leave-one-out amplification `covBS M^-1` is large"). `Jpsi_bsmeig` -- the
smallest eigenvalue of `M` over `covBS`'s own scale -- was added precisely to
test that, and it refutes it: over the sample it runs from **1.8e-7** (p1
2.0e-4) to 0.58 with a median of 0.111, so the degenerate corner is real and
is ~1 % of candidates, but the **26 candidates beyond 5 sigma have `bsmeig`
median 0.244 -- BETTER conditioned than the sample median.**

Rebuilding the innovation in the numerically safe "+" form
(`C_{-B} = A + A M^-1 A`, `Cov(e) = C_{-B} + covBS`, `e_B = rho_B + A M^-1
rho_B`) moved the tail by only ~20 % (`P(|z|>5)` 0.00156 -> 0.00127). It is
kept because it is strictly better conditioned and can no longer return an
indefinite covariance, not because it explained the tail.

**The tail is an open item**, and the place to look next is the same place the
vertex residual's tail turned out to live: the model of the innermost pixel
hits, which carry 0.60 / 0.44 of these two functionals' variance.

##### Composition (median share)

| | beam line | hit | MS | ionization |
|---|---|---|---|---|
| `z_1` | **0.212** | 0.603 | 0.141 | 0.000 |
| `z_2` | **0.374** | 0.440 | 0.150 | 0.000 |
| `z_v` | -- | 0.656 | 0.286 | 0.000 |

The beam block's share RISES with the softer muon's `pT` -- 0.137 -> 0.266 for
`z_1` and 0.280 -> 0.416 for `z_2` across the `pT` bins -- because a stiffer
pair leaves the vertex less well determined by the tracks, so more of the
residual's variance is the luminous region itself. `z_2` carries nearly twice
the beam share of `z_1`: conditioning on x removes the best-determined
direction and leaves the one the tracks constrain worst.

Material: `bpix_support6` 0.067, `tib_support` 0.024, `bpix_services` 0.013,
`fpix_support` 0.012, `bpix_active_L1` 0.009 -- the same INNER-tracker weight
as the vertex residual (which has 0.118 / 0.039 / 0.019), diluted by the
beam block's own 0.21. Hit classes: `pix_x_q1` 0.100, `pix_y_q1` 0.097,
`str_N3_lo` 0.049 -- against the vertex residual's much more concentrated
`pix_x_q1` **0.258**, `pix_x_q2` 0.102, `pix_x_q3` 0.077. So the beam
residual sees the innermost pixel classes in BOTH local coordinates where the
vertex residual sees local-x only: the DCA direction `n_hat` is one
direction, the beam residual is two.

The beam block carries **4.00 %** of `sigma_m^2` and **6.12 %** of `sigma_v^2`.

##### THE CORRELATION BETWEEN THE TWO FUNCTIONALS -- measured, and it was never 0.2

| | on 10 254 candidates |
|---|---|
| MODEL correlation `Cov_xy / sqrt(Cov_xx Cov_yy)` | median **-0.011**, mean -0.005 |
| the GLOBAL pair `corr(z_x, z_y)` | **-0.0103 +- 0.0099** (1.0 sigma) |
| the WHITENED pair `corr(z_1, z_2)` | **+0.0302 +- 0.0099** (3.1 sigma) |

The "~0.20" quoted when the basis was chosen came from the **90-candidate
smoke file**; on the real sample the two global components are already
uncorrelated. Whitening is exact only when the covariance it whitens with is
the TRUE one, and the fit's is ~18 % low (`Var(z) = 1.18`), so the Cholesky
mixing `z_2 = (r_y - L_21 z_1)/L_22` *injects* a small correlation where none
was. It is 0.03, i.e. negligible either way -- but it means **whitening is not
the fix for whatever makes `sandwich/quoted` large**, because there was no
correlation between the two functionals to remove.

##### Against gen truth (`bkg_bs.py`, classes from `genbkg.classify`)

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

##### Same-candidate ON vs OFF, 10 237 candidates (`cmp_bson.py`)

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

##### The beam-spot MEAN TERM (10 237 candidates)

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

#### 14.13 The rebase, and the gates re-run on it

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

##### THE MERGE (2026-09-12)
`cvh-exports-clean-260911` was then **fast-forwarded** onto
`beamline-260913` @ `0cb6c291354` -- verified a real fast-forward
(`git merge-base --is-ancestor dbfe6e4b2c2 0cb6c291354` is true) -- and
everything since (the per-channel defaults, the width floats, the whitened
pair, the "+" form and the `sqrt(dV_b)` cache) is on the clean branch in dev2.

The FINAL productions are `beamline/dy_bs_final` (6 x 4000 DY events, rows ON,
all the new exports; **10 413 candidates, 10 254 on the baseline**) against
`beamline/dy_bsoff` as the rows-OFF reference.  `dy_bsoff` is the FIRST pass's
rows-OFF leg and it is the right reference: same six input files, same 4000
events, same rows-OFF configuration, and the FIT is untouched by this pass --
the gun gate below is 264/264 branches bit-identical between the two builds
with the rows off.  (`beamline/dy_bsoff_final` re-measures it from scratch.)

##### THE BIT-IDENTITY GATES on the merged build (`gate_cache.py`)
| leg | matched | result |
|---|---|---|
| J/psi gun, 200 ev, rows OFF | 188 / 188 | **264 / 264 branches BIT-IDENTICAL** |
| DY, 400 ev, rows ON | 179 / 179 | **290 / 313 bit-identical**; the 23 movers are EXACTLY the beam-functional outputs (`Jpsi_bsmean/bsv*`, `bsvarv`, `resinfbsv`, the 15 `cfbs_*`), plus 10 declared new-only branches |

With the rows off the beam code never runs, so the gun leg certifies BOTH that
the per-channel default left the J/psi alone and that the `sqrt(dV_b)` cache is
bit-identical. The DY leg says the same with the beam code running: nothing
outside the beam block moved.

##### THE DEFAULTS GATE (`gate_defaults_cfg.py`)
The producer PSet expanded in BOTH areas and compared against the wanted value:
**all 8 channels correct, 0 wrong.**

#### 14.14 The terms: cards, fits, the sandwich and the injections

14 cards, every one fitted through `rabbit_fit.py` and certified by value AND
NLL AND rabbit's EDM: **14/14 at EDM < 1e-3**, the worst being
`bs_gaussq` at **1.9e-10** and most at 1e-13 .. 1e-17.

`make_vtx_card.py` needed one generalisation to serve a Z sample at all: the
mass channel's reference mass was hardcoded to the J/psi, so the `|z| < 40`
guard alone (`(m0 - 3.0969)/sigma_m ~ 85` on a Z) kept **325 of 8000**
candidates. `--m-ref 91.1876 --m-window 30` fixes it. Separately,
`fisher_vtx.py`'s `_A` namespace was missing four options the card builder
has since grown (`vtx_norm_window`, `norm_classes`, `norm_tpoints`, `alpha`
and the standard-selection fields), so EVERY Fisher run died mid-term with an
AttributeError -- fixed, and it is a bug the vertex study's own `fisher`
stage shares.

##### The sandwich (8000 shared candidates, 60 parameters, physical units)

| channel | median sandwich/quoted, CF | gauss | gaussq | CF, GLOBAL pair (superseded) |
|---|---|---|---|---|
| `vtx` alone | **0.919** | 1.000 | 1.013 | 0.918 |
| `bs` (the two beam terms) | **1.138** | 1.264 | 1.256 | **1.385** |
| `vtx + bs` | **1.196** | 1.273 | 1.274 | 1.321 |
| `vtx + bs + mass` | **1.194** | 1.271 | 1.272 | 1.322 |

`bootstrap/sandwich` is 0.990-1.003 everywhere, so the sandwich itself is
right.

**THE TWO BEAM TERMS OVER-COUNT.** `bs` alone is at **1.138** where the vertex
term alone is at 0.919. Making the two functionals the WHITENED (Cholesky)
pair took the EXCESS over 1 from 0.385 to 0.138, a factor 2.8, and left `vtx`
untouched at 0.919 as it must.

**But not for the reason the first pass gave.** That reason was "the two
beam functionals are the GLOBAL x and y components, which are correlated
(`Cov_xy` is not zero, the correlation is ~0.2)". On the full sample the model
correlation is **-0.011** and the measured `corr(z_x, z_y)` is
**-0.0103 +- 0.0099** -- the 0.2 came from the 90-candidate smoke file. There
was no correlation to remove. What the whitening removes is the shared
INFORMATION: `z_2` is the y pull *given* x, so it no longer re-uses what `z_1`
already said about the same hits and the same material. That is worth the
factor 2.8.

The remaining **0.138** is the two terms being evaluated on the SAME
candidates with the SAME nuisance parameters, which no change of basis can
undo. **Every error quoted from a beam channel must still be the sandwich
one.**

##### What the beam residuals BUY, per hit class (CF, the ACTUAL / sandwich error)

| class | `vtx` alone | `bs` alone | `vtx + bs` | gain over `vtx` |
|---|---|---|---|---|
| `pix_x_q1` | 0.0661 | 0.1153 | **0.0581** | 12.1 % |
| `pix_x_q2` | 0.1066 | 0.1564 | **0.0997** | 6.5 % |
| `pix_x_q3` | 0.1125 | 0.1799 | **0.1021** | 9.2 % |
| `pix_x_q0` | 0.2035 | 0.2607 | **0.1731** | 14.9 % |
| **`pix_y_q1`** | 0.2880 | **0.1009** | **0.0980** | **2.9x** |
| **`pix_y_q2`** | 0.4642 | 0.2456 | **0.2171** | **2.1x** |
| **`pix_y_q0`** | 0.5133 | 0.1879 | **0.1834** | **2.8x** |
| `str_N3_lo` | 0.4615 | 0.2864 | **0.2644** | 1.7x |

**This is the result.** The vertex residual is ONE direction -- `n_hat`, the
normal to the two momenta, which on a nearly back-to-back pair is essentially
the bending plane, i.e. local x -- and it is 4-8x weaker on the pixel LOCAL-Y
classes than on the local-x ones. The two beam residuals are TWO transverse
directions plus the z conditioning, and they are **2.3-2.8x tighter on
`pix_y_*` than the vertex residual is**, at a cost of 7-10 % on the local-x
classes it already does well. They are complementary, not redundant.

**The MASS adds nothing** on top: `vtx + bs + mass` reproduces `vtx + bs` to
the third digit on every class (0.0601 -> 0.0602). At Z momenta the mass
residual carries no hit-class information the other two do not already have.

**NO MATERIAL GROUP is constrained by any of the four channels on this
sample**: the "MATERIAL GROUPS the term constrains" table is EMPTY for `bs`,
`vtx`, `vtxbs` and `vtxbsmass` alike. At 40 GeV the multiple scattering is too
small for 8000 DY candidates to measure the material past its tier prior --
which is why the vertex study did its material work on the J/psi GUN. The
injection confirms it: `material_bpix_support6` at one prior sigma moves the
posterior by only `f_pri` = 0.026 (`bs`), 0.040 (`vtx`), 0.071 (`vtx + bs`)
of the injection, and the prior-corrected recovery is 0.86 / 0.93 / 0.92.

##### Injections

`hitres_pix_x_q2` x 1.10 (linear mode, so the card value IS `eps = 0.10`):

| arm | baseline | injected | shift | recovery |
|---|---|---|---|---|
| `bs_cf` | 0.0451 +- 0.1444 | -0.0482 +- 0.1316 | **-0.0933** | **0.93** |
| `vtx_cf` | 0.3498 +- 0.1230 | 0.2291 +- 0.1120 | -0.1207 | 1.21 |
| `vtxbs_cf` | 0.2534 +- 0.0925 | 0.1405 +- 0.0842 | -0.1130 | 1.13 |

The BEAM channel recovers an innermost-pixel-class injection at **0.93**, the
joint one at 1.13 -- all three within ~+-0.2 of unity, and unchanged in
character from the first pass (0.996 / 1.21 / 1.14).

**The two width scales do not absorb a hit-class injection**: the largest
leakage onto `beamwidth_x` / `beamwidth_y` is **-0.00 sigma**. They are
orthogonal to the hit classes, which is what makes floating them safe.
(`hitlik/recovery.py` prints `nan` in its `/truth` columns on these cards --
`STATE.md` open item 5, a reader bug, not a fit one -- so the recoveries above
are `|shift| / 0.10` computed here.)

#### 14.15 What a DATA fit needs

1. **The beam-spot record, per IOV, as global MEAN parameters -- DONE, and
   they float** (section 14.20). Four of them -- `x0`, `y0`, `dxdz`, `dydz` --
   per lumi block (the `offlineBeamSpot` record is per-LS and the maker
   already reads it per event, so `Jpsi_bsspot` / `Jpsi_bsslope` /
   `Jpsi_bswidth` carry the right one per candidate); `z0` is inert.
   **No new column in the quadratic term is needed**: the response of ANY
   functional to a centroid shift is minus its own influence weight on the
   beam block's row, and `Jpsi_bsmean{mass,vtx,bs}` are exactly those weights
   (3 floats each, 0.04 kB/candidate). The slope response is the same weight
   times `(z_v - z0)`, and `Jpsi_bsvtx` carries `z_v`. They enter the CF terms
   through the term's own sparse `D` (`make_vtx_card.py --beam3`).
2. **The beam WIDTHS as resolution parameters -- DONE, and they float**, and
   with `--beam3` so does the rest of the covariance (the x-y correlation the
   record does not store, and the two tilts); section 14.20.
   Family 16 is registered as a resolution block with `dV = covBS`, so a scale
   on the luminous region floats in the CF terms exactly as a hit class does.
   There are TWO of them, one per transverse direction, because
   `sigma_x -> sqrt(k_x) sigma_x` sends `covBS -> D covBS D` with
   `D = diag(sqrt(k_x), sqrt(k_y), 1)` and

       v_b(k) = sum_ij w_i w_j C_ij d_i d_j ,   d = (sqrt(k_x), sqrt(k_y), 1)

   whose derivative at `k = 1` is EXACTLY
   `dv/dk_x = w_x^2 C_xx + w_x w_y C_xy + w_x w_z C_xz` -- each cross term
   split half and half between the two directions. Those half-splits are what
   the maker exports (`Jpsi_massvbsx/y`, `Jpsi_vtxvbsx/y`, `Jpsi_bsvbsx/y[2]`);
   they sum to the block's total share IDENTICALLY, so they enter the
   hit-class machinery as two extra classes with a **LINEAR variance scale**
   (`--hit-mode linear`, the card value IS `eps`, physical `k = 1 + eps`).
   The card parameters are `beamwidth_x` / `beamwidth_y`.

   The first whitened pull's `y` share is IDENTICALLY zero
   (`Jpsi_bsvbsx[0] == Jpsi_bsvbs[0]`, `Jpsi_bsvbsy[0] == 0`), which is the
   Cholesky basis telling the truth about itself: `z_1` is the x pull and
   cannot depend on `r_y`.

   It is not a cosmetic parameter: the record's transverse widths are 8-12 %
   wider than the simulated luminous region on this MC (section 14.3), so on
   data they should be floated rather than trusted.

   **THE RECORD'S OWN PRIOR IS TOO TIGHT TO BE USED ALONE.**
   `Jpsi_bswidtherr` (`BeamWidthXError` / `YError`, now exported) is
   **0.2904 um** on widths of 10.834 / 10.388 um, so the prior on a VARIANCE
   scale is `2 err / width` = **0.0536 / 0.0559**. The value the fit must
   return is `eps = (9.66/10.83)^2 - 1 = -0.204` and `(9.59/10.39)^2 - 1 =
   -0.148`: the record's own error is FOUR TIMES smaller than the
   record-vs-simulation mismatch. A fit with that prior measures the TENSION,
   not the width. Both are therefore built and quoted -- `*_cf` with the record
   prior, `*free_cf` with `--beamwidth-prior 0` (no prior at all) -- alongside
   `nobw_bs_cf` with the widths fixed.
3. **The residuals over-determine the mean parameters**, so floating them is
   free: `sigma(x0) = 25 um / sqrt(N)` from the beam pulls alone.
4. **Only for a PROMPT resonance.** A B -> J/psi X decay has `c tau ~ 460 um`,
   so a transverse flight of a few hundred microns against a 10.8 um
   constraint is a 20-30 sigma pull (`chi2` of several hundred) and the fit
   would drag the vertex onto the beam line and mis-measure both momenta. The
   DY MC shows the signature already: the `otherdecay` class -- a leg matched
   to a muon from a different, displaced decay -- has **12.5 %** of its
   candidates beyond 3 sigma against the signal's 2.1 %, and
   `<chi2_bs> = 9.97` against 2.29. **The beam rows stay OFF for J/psi** --
   and for every displaced channel. **They are ON for the Upsilon**: every
   `Upsilon(nS)` is prompt (no b hadron is heavy enough to decay to one), so
   there is no non-prompt component at all. Section 14.17 has the table and
   the full argument.
5. **The mass shift has to go through the calibration chain.** Turning the
   rows on moves the reconstructed Z mass by -8 +- 3.5 MeV (9e-5 relative);
   it is the resolution-proportional bias shrinking, and it is TOWARD the
   truth, but at 9e-5 it is nine times the Z-mass target and must not be
   switched on silently.

#### 14.16 Standing rules this study adds

* **The beam rows are a PAIR-level constraint.** Anything emitted inside the
  `for (id...)` loop of the two-track maker without an `id == 0` guard is
  emitted twice. The pointing constraint has the guard; the beam block did
  not, for years.
* **`ndof` and `ncons` must be counted in the same place.** The defect was
  visible in the source: `nbscons = 3u * 2u` against `ndofsigned += 3`.
* **A residual whose reference is not a state parameter needs its
  leave-one-out form.** The beam rows are MEASUREMENT rows: the FITTED
  residual has covariance `covBS - C_vtx` and is shrunk toward zero; the
  unbiased one is `e_B = covBS (covBS - C_vtx)^-1 rho_B`.
* **Correlated functionals may not be multiplied and then quoted from the
  Hessian.** `bs` alone is at sandwich/quoted 1.385.
* **A leave-one-out residual must be built in the "+" form.** `covBS M^-1 covBS`
  and `A + A M^-1 A + covBS` are the same matrix and are NOT the same
  computation: `M = covBS - A` is a difference of two nearly equal covariances,
  so the first form runs the whole answer through the amplification and can
  return an indefinite matrix, while the second has the leading order explicit
  and is `>= covBS` by construction. Same for the residual itself:
  `rho_B + A M^-1 rho_B`, never `covBS M^-1 rho_B`.
* **Correlated functionals may not be multiplied and then quoted from the
  Hessian.** The global x/y pair was at sandwich/quoted 1.385. Whitening them
  (`z = L^-1 r`, `Cov(z) = I`) is the fix; the cost is that neither pull is
  then the response to a single global parameter, so export `L^-1`.
* **A block's `dV_b^{1/2}` belongs to the BLOCK, not to the functional.**
  Recomputing its eigendecomposition once per functional is the whole cost of
  adding functionals. Cache it once per candidate.
* **A variance-scale parameter's prior must be checked against the mismatch it
  is meant to absorb.** The beam-spot record's `BeamWidthError` is 0.29 um on a
  10.8 um width, i.e. 0.054 on a variance scale, while the record-vs-simulation
  mismatch is 0.20. A fit with that prior measures the TENSION, not the width;
  quote the free fit alongside it.
* **Never `cat X > X` across a shared filesystem.** `/work` is the same
  directory from every submit node; copying a remote log onto itself
  truncates it. (Cost here: `eff_*`, `fisher`, `recovery*` had to be re-run.)
* **Build on a submit node.** A host scram does not recognise as el9 picks up
  the wrong system headers and the build fails inside `<atomic>`; `/ceph` is
  unreadable from there too.
* **Never `scram b` in an area a production is reading.** The relink is the
  same-second-segfault trap; check `ps` on every submit node first.
* **A change of basis invalidates every normaliser downstream.** `gates_bs.py`
  normalised `resinfbsv` by `Cov(r_bs)_kk`; with the whitened pair the
  functional's variance is ONE, so that check read 1.5e5 instead of 1.7e-7.
  `extract_vtx.py` had the same trap (the beam arm's `sigma` is now 1 and its
  residual IS the pull). Both now key on the presence of `Jpsi_bslinv`, so they
  still read the first pass's files correctly.
* **A robust width estimator is not automatically an UNBIASED one.** A
  1 %-trimmed standard deviation is 0.96164 of the true sigma for a Gaussian,
  which is most of the "record is 8-12 % wider" of section 14.3 (section
  14.18).

#### 14.17 THE DEFAULTS, PER CHANNEL -- and why the Upsilon keeps the rows

David, 2026-09-12:

> "Yes merge them and switch them on by default for the Z and off for the
> JPsi, for the Upsilon I'm not sure, what do we expect? I thought there are
> only prompt upsilon no?"

**Yes -- every `Upsilon(nS)` is prompt.** No b hadron is heavy enough to decay
to an Upsilon, so unlike the J/psi there is NO non-prompt component at all, and
the `chi_b` feed-down is at the primary vertex too. The only DISPLACED dimuons
under the Upsilon peak are

* the `b b-bar -> mu mu` continuum -- two muons from two DIFFERENT B vertices,
  which is not one displaced vertex but no common vertex at all, and
* cosmics,

and the constraint REJECTS both rather than being confused by them. The DY MC
shows exactly that signature already (section 14.12): the gen `otherdecay`
class -- a leg matched to a muon from a different, displaced decay -- has 25 %
of its candidates beyond 3 sigma against the signal's 2.1 %. So the Upsilon is
in the same position as the Z and keeps the rows.

| cfi / driver | default | the reason, written at the flag |
|---|---|---|
| `...TwoTrackZMuMuG4e_cfi` | **True** | the Z is PROMPT |
| `...TwoTrackUpsilonMuMuG4e_cfi` | **True** | every Upsilon(nS) is prompt; the displaced dimuons under the peak are `bb -> mu mu` and cosmics, which the constraint REJECTS |
| `...DiMuonG4e_cfi`, `runCvhDimuonMiniAOD.py` | **True** | the Z/DY MiniAOD channel: `diMuonTrackVertexCandidates` has a 50-150 GeV window |
| `...TwoTrackJpsiMuMuG4e_cfi`, `runCvhJpsi.py`, `runCvhJpsiGenMC.py` | False | charmonium is NOT prompt: `B -> J/psi X` at `c*tau ~ 460 um` is a 20-30 sigma pull against an ~11 um constraint |
| `...TwoTrackJpsiKMuMuG4e_cfi` | False | the B vertex is displaced by construction; the POINTING constraint is this channel's analogue |
| `...TwoTrackPiPiG4e_cfi` (K_S) | False | displaced by construction |
| `...TwoTrackKPiG4e_cfi` (D0 / V0) | False | displaced by construction |
| `...TwoTrackProtonPiG4e_cfi` (Lambda) | False | displaced by construction |

`runCvhJpsi.py` and `runCvhJpsiGenMC.py` had `bsConstraint` HARDCODED False;
it is now a registered option (default False) so a prompt-J/psi study can turn
it on deliberately.

**THE TRAP, recorded in the cfi.** `ResidualGlobalCorrectionMakerDiMuonG4e_cfi`
is RESONANCE-AGNOSTIC by design -- its own header says the same config serves
Z / J/psi / Upsilon and the resonance is chosen only by the candidate
producer's mass window. Turning the rows on by default there is right for the
shipped window and WRONG for a charmonium clone, so a `*** TRAP ***` note sits
at the flag. `PhysicsTools/NanoAOD/python/muons_cff.py` clones it bare
(`trackrefitdimuon = ResidualGlobalCorrectionMakerDiMuonG4e.clone()`) with the
Z window, so the NanoAOD `Dimuon` table now gets the rows -- which is the
intent, and which is also why the mass shift of section 14.12 has to go
through the calibration chain.

GATE (`gate_defaults_cfg.py`, the producer PSet expanded in BOTH areas and
compared against the wanted value): **all 8 channels correct, 0 wrong.**
GATE (`gate_cache.py`, the J/psi gun, 200 events, rows off, dev3 against
dev2): **188 / 188 candidates, 264 / 264 branches BIT-IDENTICAL** -- the J/psi
sees no change at all.

#### 14.18 THE LUMINOUS-REGION WIDTHS, FLOATED -- and what it took to make them come out

David, 2026-09-12: *"Also float the widths"*.  `beamwidth_x` / `beamwidth_y`
scale the family-16 block's variance share linearly (section 14.15 point 2 has
the algebra and the exports).  Three things had to be got right before the
answer was the simulation's.

##### 1. The EXPECTATION was biased -- a correction to section 14.3
The gen production vertex has heavy tails (rms **31.8 / 90.8 um** against a
MAD of 10.08 / 9.93 um: 0.05 % of candidates carry a genuinely displaced gen
vertex), so the width has to be estimated robustly.  Section 14.3 used a
**1 %-trimmed standard deviation** and quoted 9.66 / 9.59 um.  **A trimmed
standard deviation is biased LOW**: for a Gaussian trimmed at the 0.5/99.5
percentiles the retained standard deviation is **0.96164** of the true sigma
(computed exactly).  Corrected, the three robust estimators AGREE:

| | 1 % trim (corrected) | 5 % trim (corrected) | MAD |
|---|---|---|---|
| `sigma_x` | 10.047 um | 10.070 um | 10.081 um |
| `sigma_y` | 9.978 um | 9.973 um | 9.925 um |

So the record (10.834 / 10.388 um) is **7.2 % / 4.1 % wider in sigma** --
`eps = -0.140 / -0.077` -- not the -0.204 / -0.148 the uncorrected numbers
give.  **"The record is 8-12 % wider" is really "the record is 4-7 % wider".**

##### 2. The RECORD'S OWN PRIOR is too tight to be used alone
`Jpsi_bswidtherr` is **0.2904 um** on both widths, so `2 err/width` = **0.054**
on a variance scale -- smaller than the -0.14 effect it is supposed to absorb,
and comparable to the statistical error.  A fit with that prior measures the
TENSION, not the width, and it visibly pulls the answer.  **Float them FREE.**

##### 3. It takes the VERTEX residual in the same channel
| fit | EDM | `eps_x` | pull vs -0.140 | `eps_y` | pull vs -0.077 | implied `sigma_x` | implied `sigma_y` |
|---|---|---|---|---|---|---|---|
| `bs_cf` (beam only, record prior) | 7.2e-15 | -0.0099 +- 0.0445 | +2.93 | +0.0355 +- 0.0426 | +2.65 | 10.78 +- 0.24 | 10.57 +- 0.22 |
| `bsfree_cf` (beam only, free) | 1.4e-14 | -0.0031 +- 0.0798 | +1.72 | +0.0899 +- 0.0728 | +2.30 | 10.82 +- 0.43 | 10.84 +- 0.36 |
| `vtxbs_cf` (vtx + beam, record prior) | 4.9e-14 | -0.0567 +- 0.0422 | +1.97 | -0.0257 +- 0.0396 | +1.30 | 10.52 +- 0.24 | 10.25 +- 0.21 |
| **`vtxbsfree_cf`** (vtx + beam, FREE) | **1.7e-15** | **-0.1394 +- 0.0638** | **+0.01** | **-0.0595 +- 0.0567** | **+0.32** | **10.05 +- 0.37 um** | **10.07 +- 0.30 um** |

against a simulated **10.047 / 9.978 um**: a recovery at **0.01 sigma and
0.32 sigma**, EDM-certified at 1.7e-15.  `corr(beamwidth_x, beamwidth_y)` is
+0.09 in that fit.

**Why the beam-only channel cannot do it.**  `Cov(r_bs) = C_{-B} + Sigma_beam`
and the beam block is only 0.21 / 0.37 of it; the rest is the fit's own vertex
covariance, which is ~18 % LOW (`Var(z) = 1.18`).  With only the two beam
terms a width scale and a deficit in `C_{-B}` are nearly degenerate, the 60
material and hit-class parameters take the excess, and the widths sit at the
record.  The VERTEX residual carries **no beam-block share at all** (it is the
DCA direction), so it measures the `C_{-B}` mis-modelling independently; with
that pinned, what is left in the beam terms is the luminous region.

**THE RULE: float the luminous-region widths FREE, and only in a channel that
also carries the vertex residual.**  A hit-class injection leaks onto them at
**-0.00 sigma**, so they are orthogonal to the hit classes and floating them
is safe.

#### 14.19 WHAT IS STILL OPEN

1. **The 5 sigma tail of `z_1`** (data/CF **16.0**). It is not the background,
   it is not the conditioning (`Jpsi_bsmeig` refutes that directly -- the 26
   candidates beyond 5 sigma are BETTER conditioned than the sample median),
   and the "+" form only moved it 20 %. `z_2`, which conditions on x, is at
   6.8. The place to look is the innermost pixel hit model: those classes
   carry 0.60 / 0.44 of the two functionals' variance, and the vertex
   residual -- whose CF closes at 5 sigma -- is one direction where these are
   two.
2. **`sandwich/quoted = 1.138` for the `bs` channel.** Whitening took the
   excess from 0.385 to 0.138; the rest is the two terms living on the same
   candidates with the same nuisances and cannot be removed by a change of
   basis. Quote the sandwich.
3. **The mass shift.** Turning the rows on still moves the reconstructed Z mass
   by **-8 +- 3.5 MeV** (9e-5 relative, section 14.12) -- the
   resolution-proportional bias shrinking with the resolution, toward the
   truth, but nine times the Z-mass target. Now that the rows are ON BY
   DEFAULT for Z / DY and Upsilon, this has to go through the calibration
   chain rather than being noted.
4. **The width floats need the vertex residual in the channel** (section
   14.18). Any data fit of the luminous region must be set up that way, and
   the widths must be FREE, not priored to the record.
5. **`hitlik/recovery.py` prints `nan` in its `/truth` columns** on these
   cards -- a reader bug, not a fit one. STATE open item 5, still open; the
   recoveries in 14.14 are `|shift| / 0.10` computed by hand.
6. **The luminous region's full 3x3** -- section 14.20 for what is now floated and what it closes to.

#### 14.20 THE LUMINOUS REGION AS A FLOATED 3x3 COVARIANCE (`--beam3`)

The constraint's covariance is a PHYSICAL object with six entries, and only
two of them floated.  `covBS` is the CMS beam-spot-fitter form
(`RecoVertex/BeamSpotProducer/src/FcnBeamSpotFitPV.cc`, which the maker copies
verbatim):

    C_xx = k_x sigma_x^2                C_yy = k_y sigma_y^2
    C_xy = rho sqrt(C_xx C_yy)          C_zz = sigma_z^2
    C_xz = dxdz (C_zz - C_xx) - dydz C_xy
    C_yz = dydz (C_zz - C_yy) - dxdz C_xy

and in the maker `rho` is FIXED AT ZERO -- a `FIXME` at the line, because the
beam-spot record does not store it -- while `dxdz`, `dydz`, `x0` and `y0` are
the record's.  `--beam3` floats all of it.

| parameter | what it is | card unit | prior |
|---|---|---|---|
| `beamwidth_x` / `beamwidth_y` | `k = 1 + eps` on `sigma_x^2` / `sigma_y^2` | `eps` | the record's `BeamWidthError`; `--beamwidth-prior 0` = FREE |
| `beamcorr_xy` | `rho = tanh(eta)` | `eta` | **FREE** -- the record has no `rho` to take one from |
| `beamtilt_x` / `beamtilt_y` | offset of `dxdz` / `dydz` | 1e-5 | **FREE** |
| `beamcentre_x` / `beamcentre_y` | offset of `x0` / `y0` | 1e-4 cm = 1 um | **FREE** |

`sigma_z` does not float (the z beam row is weightless against a ~100 um
vertex error, so it is inert) and `z0` is inert for the same reason.

**WHY THE FOUR ARE FREE AND THE TWO WIDTHS ARE NOT.**  `rho` has no prior
because the record carries no value for it.  The tilts and the centre have
none because the fit's own vertices measure them far better than the record
does -- ~10 um per candidate over a 3.6 cm lever arm -- so a record prior
would measure the TENSION rather than the luminous region, which is the same
argument section 14.18 makes for the widths and the reason `--beamwidth-prior
0` is the number to quote.

##### The algebra needs NOTHING new from the maker

Family 16 is registered with `dV = covBS`, so a functional's variance share
from the block is EXACTLY

    Var_bs = w^T covBS w = sum_ab covBS_ab Q_ab ,   Q_ab = w_a w_b

with `w` its own influence weight on the three beam rows.  `Jpsi_bsmean*` IS
`-w` (3 floats per functional), `Jpsi_{mass,vtx}vbs` / `Jpsi_bsvbs` are the
nominal shares, and `Jpsi_bswidth` / `Jpsi_bsslope` / `Jpsi_bsspot` are the
record.  So the whole 3x3 is a re-reading of exports that already exist, and
the MEAN responses were already identities in the base-class doc block:

    d theta / d x0   = -w_x                         =  Jpsi_bsmean*[0]
    d theta / d dxdz = -w_x (z_v - z0)              =  the same x (z_v - z0)

The tilts therefore enter BOTH the covariance (second order) and the mean
(dominant).  They are ONE parameter: `MassCFTerm` de-duplicates a name that
appears in both `beam3_params` and `jac_params`, and the gradient is the sum
of the two contributions (checked exactly, `tests/test_beam3.py`).

##### The gates

| gate | measured |
|---|---|
| **ASSEMBLY** `sum_ab covBS_ab Q_ab / Cov_ii` vs the maker's exported share, all four functionals | median **2.2e-8**, max 1.1e-7 (`gate_beam3.py`, 10 254 candidates) -- the float32 export precision, and it is re-run on every card |
| the same for `d/dk_x` and `d/dk_y` vs `Jpsi_*vbsx` / `*vbsy` | median **2.2e-8** |
| the FORMULA rebuild vs the maker's `covBS -> D covBS D` width convention | median **2.5e-4** of the derivative, ~5e-5 of the share |
| **NOMINAL**, card level: `NLL(0)` beam3 vs the two linear width classes | **BIT-IDENTICAL** on `vtx`, `bsx`, `bsy` |
| **NOMINAL**, gradient on the 60 shared parameters | max **2e-16** relative |
| gradients of all five covariance parameters vs central FD | 1.5e-9 .. 4.3e-6 (the last is `beamcorr_xy`, FD-limited) |
| the full 5x5 Hessian vs FD of the analytic gradient | max **1.2e-8** |
| the in-graph share vs an independent numpy `covBS` over a wide parameter grid | **2e-16** |
| hdf5 round trip, units, role freezing, the truncation normalisation | pass |

**THE TWO WIDTH CONVENTIONS ARE NOT THE SAME, AND THE DIFFERENCE IS STATED.**
The maker's exported `*vbsx` is the derivative under `covBS -> D covBS D` with
`D = diag(sqrt k_x, sqrt k_y, 1)`, which scales the tilt-induced `C_xz` along
with the width.  `--beam3` REBUILDS `covBS` from the scaled width through the
fitter expression, which is what a physical change of `sigma_x` does: there
`C_xz` depends on `k_x` only through `-dxdz sigma_x^2`.  The two differ by
`2 w_x w_z (dxdz)(sigma_z^2/2 + sigma_x^2)`, measured at **2.5e-4** of the
derivative and ~5e-5 of the share, because `w_z/w_x ~ 1e-5`.  The physical one
is used; the gate quotes the difference rather than hiding it.

**THE LINEAR WIDTH SCALE IS EXACT, AND THERE IS NO CLIP.**  The block's
variance is linear in `sigma^2`, so `k = 1 + eps` is not a first-order form of
anything: `eps` IS the fractional change of the variance.  The physical domain
is `eps > -1`; the fits sit at `-0.14 +- 0.06`, fifteen sigma from it, and a
model asked for a negative variance says so rather than being floored into a
different model.  `rho = tanh(eta)` is a smooth bijection onto `(-1, 1)` with
`drho/deta = 1` at the nominal `rho = 0`, so it is a change of variable and
not a bound, and the fitted error needs no transform to be read as one on
`rho`.

##### The closure reference is EXACT, not merely measured

This MC's luminous region is known in closed form.
`BetafuncEvtVtxGenerator` with
`Realistic25ns13TeV2016CollisionVtxSmearingParameters` draws

    Z = Gauss(0, SigmaZ) + Z0
    X = Gauss(0, sigma(Z)/sqrt(2)) + X0        <- `+ Z*fdxdz` is COMMENTED OUT
    Y = Gauss(0, sigma(Z)/sqrt(2)) + Y0           in the source
    sigma(z) = sqrt(emittance (betastar + (z - Z0)^2/betastar))

with `Phi = Alpha = 0` (they are the event's Lorentz boost, not a rotation of
the vertex distribution).  Therefore

* **dxdz = dydz = 0 EXACTLY** -- the simulation has no tilt at all;
* **rho_xy = 0 EXACTLY** -- X and Y are independent draws;
* `sigma_x = sigma_y` identically, marginal rms
  `sqrt(emittance (betastar + SigmaZ^2/betastar)/2)` = **9.9467 um**;
* `X0, Y0, Z0 = 0.09163, 0.16955, 0.9315 cm` to the digit.

The record has `sigma_x != sigma_y`, non-zero tilts and no `rho`, so every one
of those differences is a NUMBER THE FIT MUST RETURN.  `beam3_gen.py` prints
the closed form and re-measures the same quantities on the sample's own gen
vertices (robust: the gen vertex has heavy tails -- rms 32 / 91 um against a
MAD of 10 um -- so the width is a trimmed standard deviation with its Gaussian
bias divided out exactly, the tilt is an IRLS fit, and the errors are
bootstrap).

| parameter | card unit | generator (exact) | gen vertices (10 223) |
|---|---|---|---|
| `beamwidth_x` | eps | **-0.1570** | -0.1401 +- 0.0123 |
| `beamwidth_y` | eps | **-0.0831** | -0.0777 +- 0.0130 |
| `beamcorr_xy` | eta | **0** | -0.0061 +- 0.0098 |
| `beamtilt_x` | 1e-5 | **+0.5970** | +0.4304 +- 0.2771 |
| `beamtilt_y` | 1e-5 | **-0.4718** | -0.4037 +- 0.2671 |
| `beamcentre_x` | um | **-0.1590** | -0.1210 +- 0.1029 |
| `beamcentre_y` | um | **+0.1301** | -0.0458 +- 0.1008 |

(the gen-vertex column measures the same thing with the sample's own
statistical fluctuation, so it is the reference to pull against when the fit
and the reference share the events.)

##### The fits (`dy_bs_final`, 8 000 candidates, `vtx + bsx + bsy`, FREE widths)

Every one through `rabbit_fit.py`, certified by rabbit's EDM.  The `base` card
is the published `vtxbsfree_cf` construction (the two linear width classes);
`b3` is the same candidates and the same 60 material / hit-class parameters
with the block replaced by the 3x3 plus the two centre offsets.

| fit | parameters | EDM | NLL |
|---|---|---|---|
| `base_vtxbs` | 62 | 2.5e-16 | -12480.951 |
| `b3_vtxbs` | 67 | 7.0e-16 | -12491.451 |

`Delta NLL = -10.50` for five more parameters.

| parameter | unit | generator | gen vertices | `base` | `b3` | pull (b3 vs generator) |
|---|---|---|---|---|---|---|
| `beamwidth_x` | eps | -0.1570 | -0.1401 +- 0.0123 | **-0.1394 +- 0.0638** | **-0.1380 +- 0.0640** | +0.30 |
| `beamwidth_y` | eps | -0.0831 | -0.0777 +- 0.0130 | **-0.0595 +- 0.0567** | **-0.0582 +- 0.0569** | +0.44 |
| `beamcorr_xy` | atanh(rho) | 0 | -0.0061 +- 0.0098 | -- | **-0.2970 +- 0.0807** | **-3.68** |
| `beamtilt_x` | 1e-5 | +0.5970 | +0.4304 +- 0.2771 | -- | -0.2530 +- 0.5954 | -1.43 |
| `beamtilt_y` | 1e-5 | -0.4718 | -0.4037 +- 0.2671 | -- | -0.8038 +- 0.5799 | -0.57 |
| `beamcentre_x` | um | -0.1590 | -0.1210 +- 0.1029 | -- | +0.4059 +- 0.2144 | **+2.63** |
| `beamcentre_y` | um | +0.1301 | -0.0458 +- 0.1008 | -- | -0.1451 +- 0.2109 | -1.30 |

**The `base` fit reproduces section 14.18 to the digit** (-0.1394 +- 0.0638 /
-0.0595 +- 0.0567, `corr` +0.09), which is the statement that the
re-extraction and the rebuilt card changed nothing.

**FLOATING FIVE MORE PARAMETERS DOES NOT DISTURB THE WIDTHS**: `eps_x` moves
by 0.0014 and `eps_y` by 0.0013, i.e. **0.02 sigma** each, and their errors
grow by 0.3 %.  The beam block's seven parameters are mutually ORTHOGONAL --
the largest off-diagonal correlation in the block is **+0.091**
(`beamwidth_x`-`beamwidth_y`, which is the pair that was already there) and no
other exceeds 0.09.  So they are seven separate measurements, not one
direction seen seven ways.

**The four parameters that were pinned are measured, and three of the four
close.**  The two tilts and `beamcentre_y` sit within 1.5 sigma of the
generator's exact values; `beamcentre_x` is at +2.6 sigma.  On 8 000
candidates the tilt errors (0.60e-5) are comparable to the record-vs-simulation
offsets themselves (0.60e-5 and 0.47e-5), so the tilt closure is a 1 sigma
statement at this sample size and needs the larger production to become a test.

**`beamcorr_xy` DOES NOT CLOSE: -0.297 +- 0.081 against an EXACT zero.**  It
is the one parameter of the seven that is inconsistent with the simulation,
and the direct check says the luminous region is not the reason: the measured
`corr(z_1, z_2)` on this sample is **-0.0103 +- 0.0099** in the global basis
and **+0.0302 +- 0.0099** in the whitened one (section 14.12), neither of
which is a -0.29 correlation.  The reason is a MODELLING GAP, stated here
rather than absorbed:

* `rho` enters a functional's variance only through `Q_xy = w_x w_y`.  The
  first whitened pull `z_1` IS the x pull, so its `w_y` is identically zero
  (`Jpsi_bsvbsy[0] == 0`, section 14.15) and `rho` cannot move it at all.  So
  `rho` is a knob on the variance of `z_2` and of the vertex residual that
  does NOT touch `z_1` -- and `Var(z_1) = 1.181` against `Var(z_2) = 1.109`
  (section 14.12) is exactly such an imbalance.
* A true `rho` would also make the two beam terms CORRELATED, and the
  likelihood multiplies them as if they were independent, so that half of the
  effect is not in the model at all.

`beamcorr_xy` is therefore reported as what it is -- a measurement of the
`z_1` / `z_2` variance imbalance, not of the luminous region's `rho` -- and it
is the reason the two beam terms' joint treatment (section 14.19 item 2) has
to be settled before `rho` can be read as physical.

##### `beamcorr_xy` -- what it is actually measuring

Freezing it (`--beam3-freeze beamcorr_xy`, which removes the parameter rather
than pinning it with a prior) and re-fitting the same candidates:

| fit | parameters | EDM | NLL |
|---|---|---|---|
| `b3_vtxbs` | 67 | 7.0e-16 | -12491.451 |
| `b3nc_vtxbs` | 66 | 1.0e-13 | -12483.950 |

`Delta NLL = -7.50` for that ONE parameter (`2 Delta NLL = 15.0`, 3.9 sigma,
consistent with its own 3.7 sigma error) -- and **nothing else moves**: the
largest shift over the other 20 floating parameters is **0.10 sigma**
(`beamwidth_x`, -0.0063), the tilts move by 0.10 and 0.02 sigma, the centres
by 0.01 and 0.08.

So `rho` is a genuine, ISOLATED direction that the record's model does not
have, and the data want it at 3.9 sigma -- but it is NOT the luminous
region's `rho`, which this MC sets to zero by construction and which the gen
vertices confirm at `-0.0061 +- 0.0098`.  What it is degenerate with is the
TRANSVERSE ANISOTROPY of the fit's own vertex-covariance deficit: `Var(z_1) =
1.181` against `Var(z_2) = 1.109` (section 14.12), and `rho` is the only
parameter in the block that moves `z_2` and the vertex residual without
touching `z_1` (`z_1` is the x pull, so its `w_y` is identically zero and
`Q_xy = w_x w_y` vanishes).  **On data `beamcorr_xy` must not be read as the
luminous region's correlation** unless the `C_{-B}` mis-modelling is
controlled first, or it will absorb it.

##### The sandwich, with the new parameters

`fisher_vtx.py --beam3` + `hitlik/efficiency.py`, the SAME 8 000 candidates
and the same 67-parameter vector the cards fit.  Adding five parameters leaves
the over-counting exactly where it was:

| channel | median sandwich/quoted, `--beam3` | the same, widths only (section 14.14) |
|---|---|---|
| `bs` | **1.153** | 1.138 |
| `vtx` | **0.920** | 0.919 |
| `vtx + bs` | **1.195** | 1.196 |

`bootstrap/sandwich` is 0.998-1.003, so the sandwich itself is right.  Per
beam parameter, on `vtx + bs`:

| parameter | `bs` | `vtx` | `vtx + bs` |
|---|---|---|---|
| `beamwidth_x` | 1.098 | 1.117 | **1.025** |
| `beamwidth_y` | 1.068 | 1.197 | **1.047** |
| `beamcorr_xy` | 1.129 | 1.072 | **1.151** |
| `beamtilt_x` | 1.095 | 0.940 | **0.990** |
| `beamtilt_y` | 1.106 | 1.050 | **0.975** |
| `beamcentre_x` | 0.913 | 0.995 | **0.816** |
| `beamcentre_y` | 0.976 | 0.961 | **0.891** |

Every one is between 0.82 and 1.15, so the quoted errors on the seven are
honest to ~15 %.  (This needed one change in rabbit: `rechunk` refused to
re-partition a term carrying a per-candidate sparse `D`, and the beam centre
and tilts act through exactly that `D`, so no `--beam3` card could be put
through a per-batch score covariance at all.  It now re-slices the blocks from
the whole matrix -- the same reconstruction `candidate_slice` does -- with the
NLL bit-identical and the gradient to 3e-9 across chunk sizes.)

##### What it does to the MASS term

The same comparison with the constrained MASS channel in the card
(`vtx + bsx + bsy + mass`, the momentum scale `alpha` floating in units of
1e-3 and both mandatory mass corrections on), same candidates, same 60
material / hit-class parameters:

| fit | parameters | EDM | NLL |
|---|---|---|---|
| `base_vtxbsm` | 63 | 2.9e-14 | +10673.711 |
| `b3_vtxbsm` | 68 | 2.3e-06 | +10662.865 |

`Delta NLL = -10.85`, and the beam parameters come out where the beam-only
channels put them (`rho` -0.3034 +- 0.0813, `beamcentre_x` +0.4331 +- 0.2134),
i.e. adding the mass channel does not move them.

| parameter | `base` | `b3` | Delta | Delta / sigma |
|---|---|---|---|---|
| **`alpha`** (1e-3) | +1.51661 +- 0.34480 | +1.52163 +- 0.34477 | **+0.00501** | **+0.01** |
| the 18 hit classes | -- | -- | largest 0.018 | largest **0.04** |

**THE MASS SCALE MOVES BY +5.0e-6, which is 0.46 MeV at the Z** -- 0.015 of
its own statistical error on this sample, and HALF the 1e-5 Z-mass target.
The error on `alpha` is unchanged in the fourth digit (0.34480 -> 0.34477), so
floating five more luminous-region parameters costs the momentum scale
NOTHING in precision.  No hit class moves by more than 0.04 sigma.

Read: the luminous region's shape is nearly orthogonal to the momentum scale,
which is what makes it safe to float -- but +5e-6 is not zero at the Z-mass
target, so the beam parameters belong in the joint fit rather than being
pinned to the record and forgotten.

##### The two beam pulls, before and after

`beam3_pulls.py`, the same 8 000 candidates, the fitted `b3` parameters
applied as the model says they act: the MEAN through `z -> z + sum_k (dz/dp_k)
p_k` and the VARIANCE through `1 + Delta v_bs(p) + sum_c eps_c v_c + Delta
v_mat`.

| | N | Var(z) raw | Var(z + mean) raw | Var trimmed | model Var | raw / model | mean(z) | mean(z + corr) |
|---|---|---|---|---|---|---|---|---|
| `z_1` (`bsx`) | 8000 | 1.1552 | 1.1553 | 1.0787 | 1.1017 | 1.0486 | +0.0167 | +0.0006 |
| `z_2` (`bsy`) | 8000 | 1.1104 | 1.1099 | 1.0467 | 1.0632 | 1.0444 | +0.0031 | +0.0104 |

The variance shares that make the model number: `d v_bs` **-0.031 / -0.027**
(the luminous region shrinks, which is what the negative `eps` means),
`d v_hit` **+0.129 / +0.086**, `d v_mat` +0.004 / +0.005.

**The pulls do NOT become unit-variance, and they were never going to.**  The
excess is `C_{-B}`, the fit's own vertex covariance, which is ~18 % low; the
luminous region is only 0.21 / 0.37 of `Cov(r_bs)` and the fit moves it in the
direction that makes the model NARROWER, not wider.  What closes the gap is
the hit classes, and after both the data/model ratio is **1.049 / 1.044**
against 1.155 / 1.110 at the record.

**The mean terms are not resolved at this sample size.**  The fitted centre
and tilts shift the pulls by an rms of 0.008 / 0.018 -- a hundredth of a sigma
-- and the pull means move from +0.017 / +0.003 to +0.001 / +0.010 against a
statistical error of 0.012 on each.  The closure of the mean parameters is in
the fitted VALUES, not in a visible shift of the residual.

##### Two defects found on the way, both fixed

1. **`rho sqrt(C_xx C_yy)` has a NaN SECOND derivative wherever the record's
   width is zero.**  The value and the gradient stay finite (`0 * inf` only
   appears at second order), so the minimiser converges normally and
   `edmval_cov` dies on `array must not contain infs or NaNs` -- which is
   exactly how it presented.  Measured on the bare expression at `sigma = 0`,
   `rho = 0`: `rho sqrt(vx) sqrt(vy)` gives value 0, gradient 0, **d2 = NaN**;
   `rho sqrt(k_x) sqrt(k_y) sigma_x sigma_y` gives 0, 0, 0.  The two are
   identical for any positive record; the second takes the square root of the
   PARAMETER, which is 1 at the nominal point.
2. **A zero record row is not hypothetical -- the truncation normalisation
   builds them.**  `_vtx_norm_block` gives an EMPTY resolution class the whole
   sample as its members; the first version of the class-level beam block used
   a plain `bincount` and gave those classes zero `Q`, zero record, zero
   share.  The beam channels hit it every time: their `sigma` is identically
   1, the quantile edges collapse and seven of the eight classes come out
   empty.

Either fix alone removes the failure.  Both are kept: the first is about the
expression being differentiable, the second about the class rows being the
model.


---

### 15. THE SELECTION DECISIONS, AND WHERE EACH ONE LIVES

David, 2026-09-12, on the two cuts section 13 recommended:

> "About the minimum number of hits I agree with putting 8 as default.  On the
> vertex residual cutting on 5 sigma also sounds good.  We could keep these
> candidates and only put the cuts downstream, e.g. when running the fits or
> evaluating some other quantity."

(Section 14 is the beam-line agent's.)  Read as: **the leg-hit minimum is a
maker default; the vertex-residual cut is downstream, once, as one documented standard selection.**  They are different
kinds of cut, and that is why they live in different places.

* `minLegHits = 8` removes candidates that carry no usable resolution
  information at all -- 0.885 +- 0.026 of what it removes is a duplicate or
  unmatched pairing by gen truth (section 13.5), and every candidate with a
  non-finite `Jpsi_sigmamass` has a thin leg (13.8).  Cutting them PRE-FIT
  costs nothing downstream and saves the fit.
* `|z_v| < 5` removes 0.24 % of gen SIGNAL.  It is an acceptance cut on the
  resolution tail, so a term fitted on the survivors carries the TRUNCATED
  density and must normalise over the window it cut to.  Keeping the
  candidates in the trees is what makes that possible, and what lets the tail
  still be measured.

#### 15.1 The maker default (`dbfe6e4b2c2`)

`minLegHits` 0 -> **8** in `ResidualGlobalCorrectionMakerTwoTrackG4e.cc` (the
member default, the `existsAs` fallback and the member documentation), in all
**8** two-track cfis and in both drivers.  `minNdof = 1` and
`minPairHits = -1` (auto) are unchanged.  Built in
`CMSSW_15_0_19_patch2_dev2` on `cvh-exports-clean-260911`; nothing was running
from that area and its `git status` was clean.

**The gate** -- the new build re-run on the SAME inputs as the reference,
candidates matched on (run, lumi, event, nhits, nvalid):

| | new / ref | matched | bit-identical | not found in ref | ref-only, explained | unexplained | ndof min | non-finite |
|---|---|---|---|---|---|---|---|---|
| `gate_dy` (400 DY ev) | 179 / 186 | **179** | **29/29** | 0 | **6 / 6** | **0** | 12 -> 13 | 0 |
| `gate_gun` (200 gun ev) | 188 / 192 | **188** | **29/29** | 0 | 0 / 0 | **0** | 10 -> 14 | 0 |

`skipped[leghits<8]` is 8 (DY) and 4 (gun).  On DY the reference's ONE
`fail[kinfit]` is among the 8 -- the fit no longer has to fail on it -- and one
more skipped pair was the only candidate of its event (176 -> 175 events),
which is why 8 skipped gives 6 reference-only candidates.  On the gun all 4
were the only candidate of their event, so there are no reference-only
candidates in common events at all.  Four of the six DY ones sit at `|z_v|` =
35, 74, 88 and 262 -- the tail.

`genbkg.py --gate` grew `--gate-min-leg-hits` so it can attribute those
candidates instead of calling them unexplained.

#### 15.2 The standard selection (`resolution/selection.py`)

    Jpsi_vtxok             the vertex block closed
    finite sigma_m, sigma_v, both > 0
    weaker leg >= 8 hits   on `Mu{plus,minus}_nvalid`
    chi2/ndof < 3          on `chisqval / ndof`  (added Sep 2026, 15.8)
    |z_v| < 5              on `Jpsi_vtxz`, the vertex-constraint pull

-- in that order, so a cut flow reads as a flow.
`standard(table, args) -> (mask, Summary)`; the `Summary` is logged by every
caller, cut by cut, and DISTINGUISHES a cut that removed nothing from one
whose column is absent from one that was switched off.  `ALIASES` maps the
logical columns onto the tree names, `cf_inmaker`'s cache names and the aux
cache names, so no caller has to rename anything.  `--no-standard-selection`
is the escape hatch, for studies OF the tail.

Applied in `extract_vtx.py`, `make_vtx_card.py`, `cmp_vtxon.py`,
`matres/extract_groups.py`, `hitlik/extract_res5.py` (single-track: the
two-track cuts report absent and only the chi2 one bites),
`globalfit/extract.py`, `fullscale/make_card.py` and
`make_joint_card.py` (which forwards the escape hatch and the aux to both
legs).  `oddmoment/aux_gen.py` REPORTS it and does not cut -- its rows are
joined row by row to a pairs cache -- and `cf_inmaker.py` caches the columns
so a card built from a cache can cut on them at all.

`extract_vtx.py` records what it ACTUALLY applied in the npz
(`sel_max_abs_vtxz`, `sel_min_leg_hits`), and `make_vtx_card.py` defaults its
truncation window to the window that was actually cut -- by itself or by the
extraction -- so a truncated sample cannot be fitted with an untruncated
likelihood by forgetting a flag.

#### 15.3 THE TRUNCATED NORMALISATION, and two things that were missing

The fitted density on a sample selected in `|z_v| < w` is `L_i / Z_i` with
`Z_i = Int_{-w}^{+w} L_i dz`.  `Z` depends on the WIDTH -- a wider model
spills more of itself out of the window -- so a fit that leaves it out is
pulled towards a narrower model.  `rabbit.unbinned.MassCFTerm` already had
`norm_window` for the Z MASS window; two things had to be added
(`rabbit-vmass` `c7e6a8f`).

1. **`Z` did not depend on a `MaterialCFTerm`'s parameters.**  `_norm_z`
   summed the flat families, and a `MaterialCFTerm` declares none: its width
   IS the per-group exponents and the per-hit-class variance shares.  `Z` was
   therefore a constant, dropped out of the gradient, and the truncated fit
   was exactly as biased as one with no `Z`.  `MassCFTerm` gains a
   `_norm_extra` hook (a no-op, so nothing existing moves by a float) and
   `MaterialCFTerm` implements it from class-level copies of its blocks:
   `norm["group_families"]` `(K, ngroups, nt)`, `norm["hit_v"]` `(K, ncls)`,
   `norm["vg_other"]` `(K,)`.  A class row is the MEAN of its members, the
   same approximation the class-representative `sigma` already is.

2. **The window has UNITS.**  A Z channel is selected in a MASS window, the
   same for every candidate; a constraint residual is selected on its PULL,
   a window of +-5 `sigma_i` whose physical width differs candidate by
   candidate.  `norm_window_sigma` (default False) makes the class edges
   `d = window * sigma_c`, which a class-level integral represents exactly
   because a class IS a sigma.  This is not a small error: read as
   centimetres, 5 against a `sigma_v` of 0.004-0.04 cm is 125-1250 sigma and
   `NLL(0)` moved by **0.0057**; read as sigma it moves by **21.058**.

#### 15.4 The gate on the J/psi gun

One extraction of `prod_vtxon` (160 x 130 = 20 800 candidates), three cards,
so only the LIKELIHOOD differs between them.  Every fit through
`rabbit_fit.py`, certified by rabbit's EDM.

| arm | candidates | `NLL(0)` | NLL(min) | EDM |
|---|---|---|---|---|
| (a) no `\|z_v\|` cut, untruncated | 20 800 | -68 737.530403 | -68 767.337250 | 1.96e-14 |
| (b) `\|z_v\| < 5` + truncated norm | 20 774 | -68 872.338373 | -68 901.990740 | 9.20e-13 |
| (c) `\|z_v\| < 5`, NO norm | 20 774 | -68 851.280030 | -68 880.647708 | 3.36e-14 |
| mass (a) no cut | 20 800 | -42 115.036875 | -42 121.640416 | 3.54e-13 |
| mass (b) cut | 20 774 | -42 100.524940 | -42 107.374412 | 1.20e-12 |

The mass channel carries no `norm_window`, so its (b) and (c) are the same
card and only (a) vs (b) is fitted.

`NLL(0)`(b) - `NLL(0)`(c) = **-21.058** over 20 774 candidates, i.e.
`<log Z> = -1.014e-3`: the MODEL puts **0.101 %** of itself beyond +-5 sigma
against the measured **0.125 %** on this sample.  Those two agreeing is the
check that the truncation integral is right.

**(a) vs (b) -- the cut with its matching normalisation is consistent.**
Over all 60 parameters the shift is at most **0.07** of the parameter's own
error (median 0.00, rms 0.02, **0/60** above 0.2 sigma).  The largest is
`material_bpix_support6`, the group the vertex term weighs most (0.324 of
`sigma_v^2`): 0.03311 -> 0.03082, sigma 0.03164.

**(c) vs (b) -- the bias of leaving it out.**  On the SAME data,
`material_bpix_support6` goes 0.03082 -> **0.01879**, a shift of
**-0.38 sigma**, towards LESS material -- a narrower model, which is exactly
what a fit does when it is not charged for the tail it no longer has to
describe.  Everything else moves by <= 0.08 sigma (**1/60** above 0.2 sigma,
rms 0.05).  The bias sits where the term's information sits, which is the
worst place for it.

**The cut's effect on the MASS term** (61 parameters, `alpha` floating):
every shift is below **0.12** of its own error (median 0.00, rms 0.02,
**0/61** above 0.2 sigma), and

    alpha [1e-3] = -0.0296 +- 0.0613  (no cut)  ->  -0.0299 +- 0.0613  (cut)

a shift of **-0.0003 = -0.01 sigma**.  The material groups move by at most
0.07 sigma (`material_tib_support` -0.02790 -> -0.03068,
`material_tec_structure` -0.03052 -> -0.03294) and the hit classes by at most
0.12 sigma (`hitres_str_N5_hi`).  **Nothing crosses 0.2 sigma, so the |z_v|
cut is not a systematic of the mass term at this size** and no joint
truncation is needed.

Two of those shifts are nevertheless significant AS SHIFTS -- against
`sqrt(|sigma_b^2 - sigma_a^2|)`, the error on the difference for nested
samples, `material_tib_support` is 3.5 and `material_tec_structure` 2.8 -- so
they are a real pull from the 26 removed candidates and not a fluctuation.
They do not grow with statistics: the removed FRACTION is fixed, so the shift
and the error both scale as `1/sqrt(N)` and the ratio stays at 0.07.

#### 15.5 The DY class table (`sel_dytable.py`)

`bkg/dy_vtxon_gen`, 6 x 4000 events, constraint ON, no chi2 cut (the full
flow, in the order the cuts are applied):

| cut | total | signal | dup | otherdecay | unmatched |
|---|---|---|---|---|---|
| all | 10 638 | 10 320 | 89 | 11 | 218 |
| `Jpsi_vtxok` + finite sigma | 10 638 | 10 320 | 89 | 11 | 218 |
| min leg hits >= 8 | 10 414 | 10 304 | 15 | 11 | 84 |
| `\|z_v\| < 5` | **10 332** | **10 280** | **6** | 9 | **37** |

signal efficiency **0.9961 +- 0.0006**, background rejection
**0.847 +- 0.020**, purity **97.01 -> 99.50 %**.

(The flow WITH `chi2/ndof < 3` in its place, which is what the standard
selection does now, is section 15.8.5b.)  With the extraction's
`chi2/ndof < 3` first the cut has less left to do
(10 350 -> 10 273; signal 0.9976 +- 0.0005, purity 99.09 -> 99.59 %).  In the
FREE regime, where the tail lives, the same flow gives 10 628 -> 10 324
(signal 0.9959 +- 0.0006, purity 97.08 -> 99.54 %) and with chi2 first
10 505 -> 10 271 (signal 0.9962 +- 0.0006, purity 97.72 -> 99.57 %).

`Jpsi_vtxok` and the finiteness cut remove NOTHING on DY.  They bite on the
gun: 567 of 300 017 fail `Jpsi_vtxok`, 76 more have a non-finite `sigma_m`,
and `min leg hits >= 8` removes **5 263** (1.76 %).

#### 15.6 The v2 caches (the productions that predate the vertex export)

`fullscale/make_card.py` on `runs/gpairs_v2_n50.npz`:

* without `--selection-aux`, all four cuts log `column absent: NOT APPLIED`
  and the card keeps **645 473** candidates -- BIT-IDENTICAL to before, cut
  flow line by line;
* with `--selection-aux runs/auxgen_jpsiv2.npz` (which carries `nv_p`/`nv_m`),
  `min leg hits >= 8` removes **555** (0.085 %) and the card keeps
  **644 956**, 517 fewer after the downstream window / chi2 / sigma cuts;
* the DY v2 cache has **3 896** of 487 742 (**0.799 %**) with a weaker leg
  below 8.

Neither v2 production carries `Jpsi_vtxz`, so the residual cut is genuinely
unavailable there and is REPORTED as such rather than silently skipped.
`cf_inmaker.py` now caches the columns, so the next cache needs no aux.

#### 15.7 Running it, and the figures

    ./run_sel.sh extract|cards|fits|report        (SELROOT on ceph)
    python3 sel_dytable.py --npz <dy gen npz> [--chi2 3]
    python3 sel_plots.py --npz <gun vtx npz> --dy <dy gen npz> \
        --fits <R>/fits --groups <materialGroups50.txt>

`~/public_html/ZMass/cvh/260912_selection/`, one file per panel with a PNG
twin: `zv_spectrum_{gun,dy}` (the pull with the cut marked and the fraction
beyond it), `cutflow_dy` (the gen-class composition through the flow) and
`shift_{vtx_a_vs_b,vtx_b_vs_c,mass_a_vs_b}` (every parameter's shift in units
of its own error, with the 0.2 sigma marks).

Inputs and outputs under
`/ceph/.../runs_vtxres_260911/selection/` -- `gate_{gun,dy}/` (the maker
gate), `runs/{vtx,mass}.npz`, `runs/cards/`, `runs/fits/`.


### 15.8 THE OTHER ACCEPTANCE CUT: `chi2/ndof < 3`, MEASURED THE SAME WAY

Section 16 showed that the two-track fit's own reduced chi2 is the variable
the residual tails are made of, and that `chi2/ndof < 3` removes exactly the
excess-chi2 population: **1.0 % of gen SIGNAL on DY, 0.33 % on the J/psi gun,
and every single gen-signal candidate it removes has chi2 probability below
1e-3** (16.12).  It is therefore the same KIND of object as `|z_v| < 5` -- an
acceptance cut on a population the resolution model does not describe -- and
it was owed the same two things: one place, and a measurement.  It had
neither.  This section gives it both, on MC only (David: "for data there is a
lot more going on, misalignment and miscalibration of B field and material
etc., so maybe first focus on MC only").

**Where it was.**  Re-implemented in nine scripts with TWO defaults: 3.0 in
`fullscale/make_card.py`, `fullscale/make_joint_card.py` and
`vtxres/extract_vtx.py`, 0 (no cut) in `matres/extract_groups.py` and
`hitlik/extract_res5.py`, and again with its own literal in
`globalfit/extract.py`, `globalfit/make_global_term.py`,
`matres/make_material_card.py`, `hitlik/make_hitlik_card.py` (twice, with
different values on its two tables) and a dozen plot and Fisher scripts.  What
a card had been cut to depended on which script had built it.

**Where it is.**  `resolution/selection.py`, cut 4 of 5, in the order

    Jpsi_vtxok  ->  finite sigma  ->  leg hits >= 8  ->  chi2/ndof < 3  ->  |z_v| < 5

with ONE default, `selection.MAX_CHI2_NDOF = 3.0`, the value every certified
full-scale and vertex result was produced with.  `selection.chi2ndof(table)`
takes the ratio where a table has it (`chi2ndof` in an extraction npz,
`normchi2` in an aux cache) and builds it from `chisqval / ndof` where it does
not, with ONE convention: **`ndof <= 0` is +inf, i.e. it FAILS any finite
cut**.  A fit with no degrees of freedom has no chi2 to speak of; the older
per-script copies' `chisq / max(ndof, 1)` silently admits it.  The maker's
`minNdof = 1` means this costs nothing in practice, and the gates below show
it removes zero candidates on every production in this tree.

**The one difference from `|z_v| < 5`, and it matters.**  A cut on a residual
obliges the term that consumes the survivors to normalise over the window it
cut to (15.3).  This one does not, and not because it is being neglected:
under the resolution model the candidate's reduced chi2 is `chi2_k/k` at
`k ~ 29`, and `P(chi2/ndof > 3) = 6.6e-8`.  The model's own truncation factor
is 1 to seven digits, so there is nothing to normalise.  **The cut's entire
effect is a change of sample COMPOSITION, by candidates the model does not
describe** -- which is why it has to be MEASURED and cannot be corrected for.

#### 15.8.1 The callers, and what the cut removes in each

`--max-chi2-ndof` is now `selection.add_args`'s flag everywhere (same name,
`0` = off), and every caller logs the flow cut by cut.

| caller | what changed | `chi2/ndof < 3` removes |
|---|---|---|
| `vtxres/extract_vtx.py` | own flag + own cut deleted; records `sel_max_chi2_ndof` in the npz | was already 3.0 |
| `vtxres/make_vtx_card.py` | own flag + two own cuts deleted (single and joint index) | idempotent after the extraction |
| `vtxres/cmp_vtxon.py` | `chi2/ndof < 3` left `SEL_CUTS` for the helper | -- |
| `matres/extract_groups.py` | own flag deleted, helper applies it to BOTH functionals | default 0 -> 3.0 |
| `hitlik/extract_res5.py` | own flag deleted, now calls `selection.standard` (single-track: the two-track cuts report absent) and LOGS it | default 0 -> 3.0 |
| `hitlik/make_hitlik_card.py` | `--mass-max-chi2-ndof` + the per-hit `--max-chi2-ndof` take `selection.MAX_CHI2_NDOF`; the mass table goes through the helper | 0 -> 3.0 on the per-hit table |
| `globalfit/extract.py` | own flag deleted; helper applies to two-track AND single-track | default 0 -> 3.0 |
| `globalfit/make_global_term.py`, `matres/make_material_card.py` | default 0 -> `selection.MAX_CHI2_NDOF` | no-op on an npz already cut |
| `fullscale/make_card.py` | own flag + own cut deleted; `selection_table` now also carries `chisqval`/`ndof` so the `--selection-aux` path does not lose the cut | was already 3.0 |
| `fullscale/make_joint_card.py` | default from `selection.MAX_CHI2_NDOF`, still forwarded to both legs | was already 3.0 |
| `vtxres/fisher_vtx.py`, `plot_vtx.py`, `xcum_vtx.py`, `vtxterm.load`, `tail_common`, `sel_dytable.py`, `sel_plots.py`, `qmsmodel/*`, `hitlik/perhit/*`, `fullscale/plot_inputs.py`, `qopbias.py`, `zchannel/kern_from_selected.py` | literal 3.0 -> `selection.MAX_CHI2_NDOF` | -- |

`oddmoment/aux_gen.py` REPORTS and does not cut, as before; the chi2 line now
appears in its report because the aux cache's `normchi2` is an alias.

What the cut removes, measured (all MC):

| sample | candidates | `chi2/ndof < 3` removes | `< 5` | `< 10` |
|---|---|---|---|---|
| J/psi gun `prod_vtxon` | 20 800 | **64 (0.308 %)** | 10 (0.048 %) | 0 |
| DY `dy_vtxon_gen` | 10 402 | **120 (1.154 %)** | 62 (0.596 %) | 54 (0.519 %) |
| full-scale J/psi v2 cache | 7 923 460 | **69 644 (0.879 %)** | -- | -- |
| full-scale Z (DY v2), after the mass window | 3 656 047 | **26 387 (0.722 %)** | -- | -- |

The two samples are not the same object: the gun's worst candidate has
`chi2/ndof = 7.5`, DY's has **4.5e5**.  That is the runaway-fit population
`globalfit/extract.py` has always warned about, and it is why the cut is not
optional on a real production.

#### 15.8.2 The bit-identity gates -- nothing a certified result rests on moved

Same input, same cut value, HEAD code against the new code.  `cmp_outputs.py`
compares array by array; `cmp` compares bytes.

| gate | what | result |
|---|---|---|
| full-scale **Z** card | `zpairs_dyv2.npz`, `--maxn 300000`, `--max-chi2-ndof 3` (the value it always used) | **BYTE-IDENTICAL**, md5 `7a194db1...` both sides |
| full-scale **J/psi** leg | `jpairs_v2_n600.npz`, cut 3.0; `make_joint_card.build_jpsi` calls `make_card.select` and builds its own term, so the SELECTION is the gate | **7 687 678 selected**, all five arrays (`idx`, `m`, `sigma`, `srel`, `w`) identical |
| `matres/extract_groups.py` | `--max-chi2-ndof 0`, the OLD default | **BIT-IDENTICAL**, 28/28 arrays (`provenance` carries the argv and is skipped) |
| `hitlik/extract_res5.py` | `--max-chi2-ndof 0`, the OLD default | **BIT-IDENTICAL**, 35/35 arrays |

And what 3.0 removes in the two that used to default to 0, quoted rather than
assumed: on the J/psi-gun two-track production `matres` keeps
**1 830 of 1 835** (5 removed, 0.27 %; `P(chi2/ndof > 3)` = 0.25 %,
median 0.926, max 5.83), and on the single-muon-gun track production `hitlik`
removes **0 of 400** tracks.  Neither is a large number, which is the point:
the two extractions had been running with no cut at all and nobody could say
what that was worth.

Two conventions were checked while doing it.  `ndof <= 0 -> +inf` removes
**zero** candidates on every production here (the maker's `minNdof = 1`), so
the change of convention is free.  And `fullscale/make_card.py
--selection-aux` would have LOST the chi2 cut when the aux table replaced the
cache table; `selection_table` now carries `chisqval`/`ndof` through, which is
why the Z gate is byte-identical rather than nearly so.

#### 15.8.3 The MASS term on the J/psi gun (pure signal, 20 800 candidates)

One extraction with the cut OFF, four cards that differ only by the cut value,
`|z_v| < 5` and the rest of the standard selection in force on all of them,
`alpha` FREE and **both mandatory mass-term corrections ON**
(`--mass-corrections`; the self-consistent resolution `a_res` and the exact
Jensen map -- `f_ang` is not exported by `extract_vtx.py`, so `s^2` is the
isotropic `(sigma/m)^2`).  Every fit through `rabbit_fit.py`, certified by
rabbit's EDM.

| arm | candidates | NLL(min) | EDM | `alpha` [1e-3] | mass scale [MeV] |
|---|---|---|---|---|---|
| no chi2 cut | 20 769 | -41 953.322056 | 2.9e-14 | +0.0238 +- 0.0615 | +0.074 +- 0.190 |
| `chi2/ndof < 3` | 20 710 | -41 961.613116 | 1.3e-17 | +0.0231 +- 0.0614 | +0.072 |
| `chi2/ndof < 5` | 20 761 | -41 984.006918 | 1.2e-15 | +0.0225 +- 0.0615 | +0.070 |
| `chi2/ndof < 10` | 20 769 | -41 953.322056 | 2.9e-14 | +0.0238 +- 0.0615 | +0.074 |

`< 10` removes NOTHING on the gun (the worst candidate is at 7.46), and its
fit reproduces the no-cut one to the last printed digit of the NLL and of
every parameter -- the null test of the whole machinery.

**Every shift is small.**  Over the 61 parameters, no-cut -> cut:

| arm | max \|shift\|/sigma_own | median | rms | > 0.2 sigma | `alpha` shift |
|---|---|---|---|---|---|
| `< 3` | **0.24** (`material_tib_support`) | 0.01 | 0.06 | **1/61** | -0.0007 = -0.01 sigma, **-0.002 MeV** |
| `< 5` | 0.07 (`hitres_pix_y_q1`) | 0.00 | 0.02 | 0/61 | -0.0014 = -0.02 sigma, -0.004 MeV |
| `< 10` | 0.00 | 0.00 | 0.00 | 0/61 | 0.0000, 0.000 MeV |

ONE parameter crosses 0.2 sigma at the standard cut, and it is a material
group, not `alpha`: `material_tib_support` -0.0212 -> -0.0315 against its own
error 0.0426, i.e. **0.24 sigma_own** -- and 4.6 sigma of the error on the
DIFFERENCE, so it is a real pull from the 59 removed candidates and not a
fluctuation.  **Said plainly: at `chi2/ndof < 3` the cut is a 0.24 sigma
systematic on one TIB material amount.**  Its mass-scale consequence is the
number in the table: the momentum scale moves by **0.002 MeV on a J/psi**,
0.01 of `alpha`'s own error, because the material groups that move are not
the ones `alpha` is correlated with.  At `< 5` nothing crosses 0.2 at all.

#### 15.8.4 The VERTEX term on the gun -- the three-way, where the two cuts meet

`|z_v| < 5` truncates the density and is normalised for (15.3); `chi2/ndof < 3`
truncates nothing the model can see.  The three arms are therefore

    (a)  chi2/ndof < 3  +  |z_v| < 5  + truncated normalisation   20 710
    (b)  no chi2 cut    +  |z_v| < 5  + truncated normalisation   20 769
    (c)  both off, untruncated likelihood                         20 800

| arm | NLL(min) | EDM |
|---|---|---|
| (a) | -68 673.046814 | 7.9e-15 |
| (b) | -68 831.724374 | 7.5e-14 |
| (c) | -68 669.452528 | 2.0e-13 |
| chi2 < 5 + (b) | -68 811.988361 | 1.0e-11 |
| chi2 < 10 + (b) | -68 831.724374 | 1.4e-12 |

| comparison | max \|shift\|/sigma_own | median | rms | > 0.2 sigma |
|---|---|---|---|---|
| (b) -> (a), i.e. the chi2 cut alone | **0.30** (`hitres_pix_y_q0`) | 0.00 | 0.07 | **2/60** |
| (b) -> chi2 < 5 | 0.05 | 0.00 | 0.01 | 0/60 |
| (b) -> chi2 < 10 | 0.00 | 0.00 | 0.00 | 0/60 |
| (c) -> (b), i.e. the `\|z_v\|` cut with its normalisation | 0.14 (`material_bpix_support6`) | 0.01 | 0.02 | 0/60 |
| (c) -> (a), both cuts | 0.29 | 0.01 | 0.07 | **3/60** |

The two cuts do not fight each other: (c) -> (a) is (c) -> (b) plus
(b) -> (a) to within a hundredth of a sigma on every parameter.  The chi2 cut
moves **pixel hit-resolution classes** (`hitres_pix_y_q0` -0.30,
`hitres_pix_x_q1` -0.22) and not the material, which is what section 16.13
predicts: the excess-chi2 candidates are the ones carrying an outlying
MEASUREMENT, so what they inform is the hit-noise model.  **Two classes cross
0.2 sigma, so on the vertex term the cut IS a systematic at that size**, and
the honest statement is that the hit-class scales are not measured to better
than 0.3 of their error until the excess-chi2 population is MODELLED rather
than cut (16.12's route 2).

#### 15.8.5 DY MC, gen SIGNAL only -- the number the mass measurement would see

`bkg/dy_vtxon_gen` (6 x 4000 events), the vertex constraint ON, the gen-signal
mask from `genbkg.classify` applied identically to every arm
(`gensig_mask.py`, **10 300 of 10 402**), so the comparison is about the
physics and not about the background the cut also happens to take.  The mass
channel has `m_ref = 91.1876`, a WIDE kernel (the gen Z mass, rms 6.9 GeV
against `sigma_m` = 1.08 GeV), both corrections ON in the FLUCTUATION form,
and `alpha` free.

| arm | candidates | NLL(min) | EDM | `alpha` [1e-3] | mass scale [MeV] |
|---|---|---|---|---|---|
| no chi2 cut | 10 276 | +30 590.004805 | 1.6e-13 | +1.0024 +- 0.3121 | **+91.410 +- 28.5** |
| `chi2/ndof < 3` | 10 228 | +30 457.670460 | 3.8e-18 | +1.0029 +- 0.3131 | **+91.452** |
| `chi2/ndof < 5` | 10 271 | +30 578.451793 | 4.2e-15 | +1.0009 +- 0.3123 | +91.266 |
| `chi2/ndof < 10` | 10 272 | +30 580.383561 | 8.5e-19 | +1.0026 +- 0.3122 | +91.421 |

**THE ANSWER TO THE QUESTION THIS STUDY WAS ASKED.**  The cut that removes the
population carrying a +195 MeV resolution-proportional bias (16.11) moves the
FITTED mass scale by **+0.043 MeV**, which is 0.001 of the arm's own
statistical error (28.5 MeV) and 0.02 of the error on the difference.  At `<5`
it is -0.144 MeV and at `<10` +0.011 MeV.  **The chi2 cut is not hiding a
model deficiency that the mass measurement would otherwise see**, at this
statistics and in this likelihood -- not because the excess-chi2 candidates
are harmless, but because the mass term already carries the two corrections
that make the bias proportional to the candidate's own `sigma_m^2`, and the
excess-chi2 candidates enter with their own larger `sigma_m`.  (The +91 MeV
scale itself is a separate matter and is NOT the +57 MeV trimmed MEAN of
16.11: a 61-parameter likelihood with the two corrections is not the sample
mean, and the two are not comparable.)

Over the 61 parameters the mass term moves by at most **0.12 sigma_own**
(`hitres_str_N1_lo`, median 0.00, rms 0.02, **0/61** above 0.2) at `< 3`, and
at most 0.04 at `< 5` and `< 10`.  **Nothing on the DY mass term crosses
0.2 sigma.**

The DY VERTEX term, the same three-way:

| comparison | max \|shift\|/sigma_own | median | rms | > 0.2 sigma |
|---|---|---|---|---|
| (b) -> (a), the chi2 cut alone | **0.29** (`hitres_pix_x_q1`) | 0.00 | 0.05 | **2/60** |
| (b) -> chi2 < 5 | 0.12 | 0.00 | 0.02 | 0/60 |
| (b) -> chi2 < 10 | 0.12 | 0.00 | 0.02 | 0/60 |
| (c) -> (b), the `\|z_v\|` cut with its normalisation | **1.06** (`hitres_pix_y_q3`) | 0.00 | 0.17 | **7/60** |
| (c) -> (a), both | 1.06 | 0.01 | 0.18 | **8/60** |

Read this the right way round: on DY it is **the `|z_v|` cut, not the chi2
cut**, that dominates the vertex term -- 7 of 60 parameters above 0.2 sigma
and one at 1.06 -- because DY's vertex-residual tail is far heavier than the
gun's (16.1: slope 0.49 against 0.14).  The chi2 cut adds one more parameter
above the threshold.  Both are hit-resolution classes again.

#### 15.8.5b The DY gen-class table, with the chi2 cut in its place

`bkg/runs/dy_vtxon_gen_vtx_all.npz`, nothing pre-applied, the full flow in the
order `selection.standard` applies it (`sel_dytable.py`):

| cut | total | signal | dup | otherdecay | unmatched |
|---|---|---|---|---|---|
| all | 10 638 | 10 320 | 89 | 11 | 218 |
| `Jpsi_vtxok` + finite sigma | 10 638 | 10 320 | 89 | 11 | 218 |
| min leg hits >= 8 | 10 414 | 10 304 | 15 | 11 | 84 |
| **`chi2/ndof < 3`** | **10 286** | **10 241** | **2** | 9 | **34** |
| `\|z_v\| < 5` | 10 273 | 10 231 | 2 | 9 | 31 |

Signal efficiency of the WHOLE selection **99.14 +- 0.09 %**, purity
**97.01 -> 99.59 %**.  In its place in the flow the chi2 cut costs **63 of
10 304 signal (0.61 %)** -- less than the 1.0 % of 16.12, because the leg-hit
minimum has already taken some of the same candidates -- and removes **63 of
110 remaining background** (13 `dup`, 50 `unmatched`), so at this point in the
flow it is as much a background veto as an acceptance cut.  `|z_v| < 5` then
costs 10 more signal and 3 more background.

The two samples' chi2 spectra are not the same object at all:

| | median `chi2/ndof` | p99 | max | P(> 3) | P(> 5) | P(> 10) |
|---|---|---|---|---|---|---|
| J/psi gun (20 800) | 0.911 | 2.27 | **7.5** | 0.308 % | 0.048 % | 0 |
| DY (10 402) | 0.966 | 3.17 | **4.5e5** | 1.154 % | 0.596 % | 0.519 % |
| the MODEL, `chi2_k/k` at `k = 29` | 1.00 | 1.52 | -- | **6.6e-8** | 2e-15 | -- |

The model's `P(chi2/ndof > 3)` is **6.6e-8**: seven orders of magnitude below
what is measured, and the reason the cut needs no truncation factor while the
population it removes is entirely real.

#### 15.8.6 Two defects found on the way

1. **`make_vtx_card.py --max-abs-z 40` is a RESOLUTION-pull guard and was
   cutting the Z LINESHAPE.**  It exists because the `gauss`/`gaussq` arms'
   density underflows float64 beyond |z| ~ 40, and on a delta-kernel J/psi
   channel `|m - m_ref|/sigma_m` IS a resolution pull, where it costs 2 in
   16 000.  On a DY MASS channel the same quantity is `(m - 91.1876)/1.08`,
   i.e. the Z lineshape: it dropped **99 of 10 276** gen-signal candidates
   (1 %) on the observable being fitted.  The DY mass cards here are built
   with `--max-abs-z 0`; the `cf` arm's density carries the kernel and does
   not underflow.  (The DY VERTEX channel keeps the guard -- there `z` is a
   pull -- and it removes 3 of 10 300, only in the untruncated arm.)
2. **rabbit's `auto` correction form reads `term.kernel`, which
   `make_vtx_card.py` never sets.**  The card passes its physics kernel as a
   tabulated `phik` and leaves `kernel` at the default `DeltaKernel`, so
   `Fitter` moved the DY mass term to the RESIDUAL form -- exact at a delta
   kernel, and wrong at a wide one, where it would evaluate the width at the
   candidate's distance from `m_ref` (the lineshape, +-10 GeV) instead of at
   its resolution fluctuation.  On the gun the default is the truth (every
   candidate has the same gen mass).  The DY mass fits are therefore run with
   `rabbit_fit.py --unbinnedDeltaKernelForm fluctuation`, which `run_chi2.sh`
   applies by card name; `make_vtx_card.py --corr-form` sets the card's own
   declaration.

#### 15.8.7 Running it, and the figures

    ./run_chi2.sh extract|extract-dy|genmask|cards|cards-dy|fits|report
    ./run_chi2_gates.sh z|jpsi|matres|hitlik        (the bit-identity gates)
    python3 gensig_mask.py --npz <dy npz> -o <mask npz>     (gen signal only)
    python3 sel_dytable.py --npz <dy gen npz>               (the class flow)
    ./run_tf.sh python3 chi2_plots.py --gun ... --dy ... --fits ... --groups ...

Inputs and outputs under
`/ceph/.../runs_vtxres_260911/chi2cut/` -- `runs/{gun,dy}_{vtx,mass}.npz`,
`runs/cards/`, `runs/fits/`, `gates/`.
Figures `~/public_html/ZMass/cvh/260913_chi2cut/`, one file per panel with a
PNG twin: `chi2_spectrum_{gun,dy}` (the measured reduced chi2 against the
`chi2_k/k` the model implies, with the three cut values marked),
`chi2_cutflow_dy` (the gen-class composition with the chi2 cut in its place),
`chi2_shift_<sample>_<channel>_<arm>` (every parameter's shift in units of its
own error) and `chi2_alpha` (the fitted mass scale against the cut value, in
MeV).


### 16. THE NON-BACKGROUND TAIL OF THE CONSTRAINT RESIDUALS -- IT IS THE FIT'S OWN CHI2

Section 14.12 left the beam residual's 5 sigma tail open ("the place to look
next is the model of the innermost pixel hits"), and section 13 left a
gen-SIGNAL `z_v` tail on DY of data/CF 4-10 at 5 sigma against 1.0-1.7 on the
J/psi gun.  This section answers both, on `dy_bs_final` (10 413 candidates,
10 254 on the published baseline) with the J/psi gun `prod_vtxon` (55 155) as
the control.

**THE ANSWER.  Every residual of a candidate is a linear functional of the
same noise vector whose squared length is the fit's `chisqval`, so the
conditional variance of a pull is the candidate's own reduced chi2, and the
pull's marginal density is a SCALE MIXTURE of normals whose mixing density is
the measured `chi2/ndof`.  The tail is that mixture; the model failure is not
in the residual at all, it is that the DY chi2 distribution is far wider than
the fit's covariance implies -- an UNMODELLED per-candidate noise-scale spread
of 13-15 % on DY against 4-5 % on the J/psi gun.**  Nothing about the luminous
region, the hit classes, the leg length or the alignment is needed to produce
it, and none of those four survives its own test -- the alignment by a DIRECT
test, the same DY events refitted with the ideal geometry, which moves the
spread by 0.3 % of itself (section 16.10).

#### 16.1 The identity, measured

The fitted `Var(z | chi2/ndof)` against `chi2/ndof`, with the constraint rows
removed from BOTH the numerator and `ndof` (1 row for the vertex constraint,
3 for the beam line) so the trivial self-correlation cannot produce it
(`tail_mixture.py`, `logs_tail/mixture_dy.log`; figure `ladder_all`):

| | slope | intercept |
|---|---|---|
| `z_1` | **+1.0616 +- 0.0562** | -0.0454 +- 0.0527 |
| `z_2` | +0.8861 +- 0.0505 | +0.1205 +- 0.0484 |
| `z_v` | +0.4927 +- 0.0535 | +0.5939 +- 0.0551 |
| `z_v`, J/psi gun | +0.1360 +- 0.0275 | +0.9601 +- 0.0261 |

The model says slope 1 and intercept 0 and that is what `z_1` gives.  The
slope measures HOW MUCH OF THE RESIDUAL'S VARIANCE LIVES IN THE NOISE THE CHI2
ACTUALLY RESOLVES -- `chisq0val` carries the hit, the process-noise, the beam
and the pointing terms, but a fluctuation the fit absorbs into its own free
parameters leaves no chi2 behind.  It is 1 for the beam-line x pull and 0.14
for the gun's vertex pull, and **that, not the hit model, is why the gun
closes at 5 sigma and DY does not: the gun's residual is largely made of noise
its own chi2 does not see, so it cannot inherit the chi2's tail.**

Note the correlation itself is unremarkable in the core -- Spearman
`corr(chi2/ndof, z_1^2)` is **+0.158** and Pearson on `|z_1| < 3` is +0.171,
of the order of the `1/sqrt(ndof)` = 0.19 a correct model gives.  It is the TAIL that is
entirely chi2: a candidate at `|z_1| > 5` has a median reduced `chi2/ndof` of
**3.05**, i.e. **61 units of excess chi2 on 29 ndof**.

#### 16.2 The chi2 is far too wide, on DY and (less) on the gun

The chi2 probability of the fit must be FLAT on [0, 1].  It is not
(`logs_tail/chi2src.log`, figure `chi2prob_dy_gun`):

| `P(chi2 prob < x)` | 0.05 | 0.01 | 1e-3 | 1e-5 |
|---|---|---|---|---|
| DY, gen signal | 0.1427 | 0.0853 | **0.0512 +- 0.0022** | 0.01825 |
| J/psi gun | 0.0644 | 0.0318 | **0.0156 +- 0.0005** | 0.00573 |
| a correct model | 0.05 | 0.01 | 0.001 | 1e-5 |

DY is **51x** the expectation at 1e-3 and the gun **16x**; DY is 3.3x the gun.
The core scale is much milder: the median `chi2/ndof` is 1.0140 against the
0.9754 a `chi2_27` implies (DY, +2.0 % in sigma) and 0.9092 against 0.9763
(gun, -3.5 % in sigma).  So the picture is **a ~2 % mis-scaling plus a ~5 %
population of candidates with one genuinely bad measurement**, not a uniform
inflation.

#### 16.3 The scale mixture multiplies the Gaussian tail by a thousand, and leaves a factor of a few

Folding the MEASURED `chi2/ndof` into a Gaussian as
`P(|z| > t) = E_s[2 Phi(-t/s)]`, `s = sqrt(chi2/ndof)`, with no free parameter
(`tail_mixture.py`; figures `density_z1_dy`, `density_z2_dy`, `density_zv_dy`,
each with a data/mixture ratio panel):

| t | Gaussian | mixture | `z_1` | `z_2` | `z_v` |
|---|---|---|---|---|---|
| 2 | 0.04550 | 0.06102 | 0.06689 (1.10) | 0.05708 (0.94) | 0.06533 (1.07) |
| 3 | 0.00270 | 0.00867 | 0.01621 (1.87) | 0.01311 (1.51) | 0.01107 (1.28) |
| 4 | 6.3e-5 | 0.00159 | 0.00718 (4.52) | 0.00447 (2.81) | 0.00243 (1.53) |
| 5 | 6.0e-7 | 0.00059 | 0.00427 (7.26) | 0.00223 (3.80) | 0.00146 (2.48) |

(gen signal, NO chi2 cut).  The mixture takes `P(|z_1| > 5)` from 6.0e-7 to
5.9e-4 -- a factor **980** -- against a measured 4.3e-3, so on the log scale
that is **78 % of the distance from a Gaussian to the data**, with a factor
2.5 (`z_v`) to 7.3 (`z_1`) left.  That remainder is the ONE-SCALE
approximation: the excess chi2 is one bad measurement, not a uniform
rescaling, so a functional that happens to weigh that measurement heavily sees
more than `sqrt(chi2/ndof)`.  Section 16.3b closes the rest.

#### 16.3b ONE NUMBER REPRODUCES THE PUBLISHED data/CF AT 5 SIGMA

First, the published ratios are reproduced here exactly, on the same npz and
the same `zg = linspace(-40, 40, 3201)` grid `plot_vtx.py` uses (the far-tail
integral is grid-sensitive: extending the grid to +-60 inflates the CF's own
tail 13x, so the range is part of the definition):

| | data | CF | data/CF | published |
|---|---|---|---|---|
| `z_1` at 5 sigma | 0.00176 | 1.10e-4 | **15.98** | 16.0 |
| `z_2` at 5 sigma | 0.00078 | 1.14e-4 | **6.84** | 6.84 |
| `z_1` at 4 / 3 sigma | | | 10.95 / 3.57 | 10.9 / 3.57 |

Now take the 1 %-trimmed spread of `chi2/ndof` on the SAME baseline and
subtract, in quadrature, the `sqrt(2/ndof)` a correct model implies.  What is
left is the UNMODELLED part of the per-candidate noise-variance scale:

| | trimmed rms | `sqrt(2/ndof)` | EXTRA | on sigma |
|---|---|---|---|---|
| DY, gen signal, baseline | 0.3741 | 0.2733 | **0.255** | **12.8 %** |
| J/psi gun, baseline | 0.2822 | 0.2708 | **0.080** | **4.0 %** |
| (DY with the chi2 cut lifted) | 0.4014 | 0.2734 | 0.294 | 14.7 % |

Feed that extra spread, and ONLY that, into a lognormal scale mixture, with
each functional's sensitivity set by its own measured LADDER SLOPE (effective
spread `lambda x s`):

| | 3 sigma | 4 sigma | **5 sigma** | measured data/CF at 5 sigma |
|---|---|---|---|---|
| DY `z_1` (`lambda` 1.06) | 1.51 | 3.71 | **18.4** | **16.0** |
| DY `z_2` (`lambda` 0.89) | 1.36 | 2.79 | **10.4** | **6.84** |
| DY `z_v` (`lambda` 0.49) | 1.11 | 1.47 | **2.6** | (no data above 5) |
| gun `z_v` (`lambda` 0.14) | 1.00 | 1.00 | **1.01** | **1.0-1.7** |

**One number -- a 12.8 % unmodelled spread of the noise scale on DY, 4.0 % on
the gun -- reproduces `z_1`'s 5 sigma ratio to 15 %, `z_2`'s to 50 %, and the
J/psi gun's closure, with no free parameter and in the right order.**  At 3
and 4 sigma it UNDER-predicts by a factor 2-3, which says the true scale
distribution has more weight at moderate scales than a lognormal of the same
variance; the shape of the mixing density is a modelling choice still to be
made, its WIDTH is measured.

#### 16.4 What the tail candidates ARE

`tail_hypD.py --dump`.  On the published baseline (`chi2/ndof < 3`, gen
signal) there are 17 candidates at `|z_1| > 5` and 8 at `|z_2| > 5` -- section
14.12's 26 -- and 4 at `|z_v| > 5` once the `|z_v| < 5` cut is lifted.  Core
against tail (medians):

| | core | tail | |
|---|---|---|---|
| `chi2/ndof` | **0.964** | **2.505** (p90 2.86) | against the cut at 3 |
| `max \|eta\|` | 1.53 | **2.14** | |
| `sigma_m` | 1.03 GeV | 1.47 GeV | |
| `sigma_v` | 24.3 um | 21.9 um | not worse |
| weaker leg `nvalid` | 15 | 16 | not thinner |
| `bsmeig` | 0.112 | 0.240 | BETTER conditioned |
| event multiplicity | 1 | 1 | not combinatorics |
| `nTrueInt` | 21.8 | 23.7 | |

and, as fractions of the tail set against the baseline rate:

| | `\|z_1\|>5` (17) | `\|z_2\|>5` (8) | baseline |
|---|---|---|---|
| chi2 prob < 1e-3 | **0.882 +- 0.078** | **1.000** | 0.0437 |
| `chi2/ndof` > 1.5 | 0.941 | 1.000 | -- |
| `max \|eta\|` > 1.8 | **0.882 +- 0.078** | **1.000** | 0.352 |
| gen vertex > 3 sigma of the beam line | 0.059 | 0.125 | 0.0025 |

With the chi2 cut LIFTED the same picture holds on 44 gen-signal `|z_1| > 5`
candidates: 0.886 +- 0.048 at chi2 prob < 1e-3, 0.795 +- 0.061 at
`max |eta| > 1.8`, and **0.000** with a displaced gen vertex.  (The lifted set
is also 58.5 +- 4.8 % BACKGROUND -- 106 candidates of which 44 gen signal --
so `chi2/ndof < 3` is doing background rejection as well, exactly as section
13.6 found for the vertex constraint.)

The `|z_v| > 5` set is a different, milder population: only 0.60 +- 0.13 of
it has chi2 prob < 1e-3.

#### 16.5 Hypothesis A -- the luminous region.  REFUTED for the bulk.

The pull compares the fitted vertex with the beam-spot RECORD, so a gen vertex
far from the record's line gives a large pull with no tracking problem.  Split
the residual into the two pieces it is made of,
`r_bs = (x_gen - x_line(z_gen)) + (x_LOO - x_gen) = delta_lum + delta_reco`
-- using the LEAVE-ONE-OUT vertex `x_line + r_bs`, NOT `Jpsi_bsvtx`, which is
the CONSTRAINED vertex and differs by a median 13 um (`tail_hypA.py`).  On
gen SIGNAL the split closes: `Var 0.195 + 0.993 = 1.188` against
`Var(z_1) = 1.176`, `corr = -0.002`.  And

* the SIMULATED luminous region is Gaussian: the gen offsets over their own
  MAD have kurtosis **2.93 / 2.95** and `P(> 4 sigma) = 0` on 10 214
  candidates.  `delta_lum` has `P(|z| > 3) = P(|z| > 4) = P(|z| > 5) = 0`;
* the RECORD describes it: record sigma / simulated MAD = **1.0754 / 1.0470**,
  centroid to -0.11 +- 0.10 um, and the slope residuals are
  `dxdz` **+4.4e-6 +- 2.7e-6** and `dydz` -4.0e-6 +- 2.7e-6 (robust IRLS),
  i.e. **0.015 sigma_BS** over the whole `|z| < 3.6 cm` range.  `<z_1>` is
  flat in the gen vertex `z` over eight octiles;
* the TAIL is entirely in `delta_reco`: `P(> 3/4/5) = 0.0108 / 0.0038 /
  0.0019` against `0 / 0 / 0` for `delta_lum`, and the `|z_1| > 5` candidates
  have a median `|delta_reco,x|` of **125 um** against the baseline's 15 um
  while their `|delta_lum|` is the baseline's;
* **of section 14.12's 26 candidates beyond 5 sigma** (`|z_1| > 5` or
  `|z_2| > 5` on the published baseline; 25 of them gen signal), only
  **2 -- 0.077 +- 0.052 -- have a gen vertex beyond 3 sigma of the record's
  line** and 1 beyond 5 sigma, against a baseline rate of 0.0025 +- 0.0005.

So A accounts for **0.08 +- 0.05** of the published 26, 0.06 +- 0.05 of the
`z_1` tail and **0.00** of it once
the chi2 cut is lifted.  (The RAW rms of the gen offset is 32 / 91 um against
a MAD of 10.1 / 9.9 -- section 14.3's displaced-vertex population -- but those
candidates are the `otherdecay` class, 8 of the 10 254 on the baseline, and
they are not gen signal.)

#### 16.6 Hypothesis C -- thin legs, truncated MiniAOD hit lists.  REFUTED.

* Nothing is truncated: `nvalid == nhits` on **10 214/10 214**, and the number
  of modules the fit actually attaches an alignment parameter to is >= the
  number of valid hits on 10 214/10 214.
* The tail candidates' weaker leg is NOT thin: median `nvalid` 16 against the
  sample's 15, a HIGHER BPix-L1 rate (1.00 against 0.90), the same innermost
  radius (4.14 cm) and the same reco/gen `pT` spread
  (p1-p99 **0.939-1.053** against 0.946-1.060) -- so the legs are neither
  short nor mis-measured.
* `P(|z_1| > 5)` against the weaker leg's `nvalid` is 0/86 (<= 9),
  0.0020 +- 0.0020 (10-11) and 0.0017 +- 0.0004 (>= 12) -- FLAT.  Section
  13.3's hit-count dependence lived BELOW `minLegHits = 8`, which the maker
  now applies pre-fit.
* Nor does the tail follow the missing BPix L1 (0/1057 at `|z_1| > 5` without
  it, 0.0019 with it) or the pixel-hit count.

#### 16.7 Hypothesis D -- a pixel hit-noise family.  Not by itself.

`P(|z_1| > 5)` across quintiles of the PIXEL share of the residual's variance
is 0.0005-0.0044 with no trend, and the J/psi gun -- the same hit model, a
LARGER median pixel share (0.877 against 0.857) -- has the smaller tail.  What
the classes do show is that a tail candidate's variance moves OUT of the
innermost pixel class: `pix_x_q1` carries 0.44 of `Var(z_1)` in the core and
0.24 in the `|z_1| > 3` set, with the difference going to the strip classes.
`npixDemoted` (the edge / single-column hits the maker already drops) has no
effect on the chi2 either.

#### 16.8 Hypothesis E -- pileup / wrong-vertex association.  NOT APPLICABLE.

The residuals are built from the two legs and the beam-spot record only; no
primary-vertex collection enters them, so there is no vertex to associate
wrongly.  Pileup does enter through hit OCCUPANCY, and that is measured below.

#### 16.9 Where the excess chi2 comes from

`P(chi2 prob < 1e-3)`, DY gen signal 0.0512 +- 0.0022 against the gun's
0.0156 +- 0.0005 (`tail_chi2src.py`; figures `chi2excess_vs_eta`,
`chi2excess_vs_pileup`):

* **the MOMENTUM is not it.**  Inside the gun, which spans 2-20 GeV, the
  excess is FLAT in the softer leg's `pT` -- 0.0151 below 2 GeV, 0.0138 above
  12 -- and DY is flat across its own 25-60 GeV range (0.047-0.054).  So the
  gun's 40 GeV analogue would still be 0.014, not 0.051.
* **pileup carries a factor 1.2-1.5.**  DY runs 0.0352 +- 0.0096
  (`nTrueInt < 12`) to 0.0680 +- 0.0085 (above 32); extrapolated to zero
  pileup it would be ~0.030-0.035, still twice the gun.
* **the rest is FORWARD and DY-specific.**  DY rises 0.0393 (`|eta| < 0.9`)
  to 0.0780 (`> 2.2`) while the GUN FALLS, 0.0162 to 0.0108.  At the lowest
  pileup AND `|eta| < 1.4` DY is still 0.042 +- 0.016, ~2.7x the gun.
* the two factorise, and neither closes the gap.  `P(chi2 prob < 1e-3)` in
  (pileup x `|eta|`) cells:

  | `nTrueInt` | `\|eta\| < 1.8` | `\|eta\| > 1.8` |
  |---|---|---|
  | 0-20 | 0.0426 +- 0.0039 | 0.0547 +- 0.0060 |
  | 20-26 | 0.0393 +- 0.0043 | 0.0604 +- 0.0069 |
  | 26+ | 0.0512 +- 0.0050 | 0.0819 +- 0.0086 |
  | J/psi gun | **0.0166 +- 0.0006** | **0.0130 +- 0.0009** |

  so even DY's most favourable cell -- central, lowest pileup -- is **2.6x**
  the gun.  Pileup buys a factor ~1.2-1.5 and `eta` ~1.3-1.6; a factor ~2.6
  is DY-specific and unexplained by either.  What is left to differ is the
  provenance of the hits themselves: the gun is 15_0-native GEN-SIM-RECO with
  the ideal geometry, DY is a 106X `slimmedMuons` MiniAOD track read in 15_0.
  `ndof == 0` was already traced to that (section 13.8: "a MiniAOD phenomenon,
  because `slimmedMuons` keeps the hit pattern but not every RecHit"), and a
  cross-release cluster/CPE difference is the obvious suspect.  It is NOT
  tested here.
* and `eta` is not only the chi2: at FIXED `chi2/ndof` in 1.3-3,
  `P(|z_1| > 5)` is 0.0017 +- 0.0012 for `|eta| < 1.8` and
  **0.0235 +- 0.0060** for `|eta| > 1.8`.
* it is NOT a weighting concentration: the largest single noise block's share
  of `Var(z_1)` is 0.20 / 0.18 / 0.18 across the three `|eta|` bands and the
  tail does not follow it.
* turning the BEAM ROWS on raises the median reduced `chi2/ndof` from 0.963
  (`dy_bsoff`) to 1.014 and `P(prob < 1e-3)` from 0.037 to 0.051: about a
  quarter of the DY chi2 excess is the tension between the tracks' preferred
  vertex and the beam line, which is the constraint doing its job.

#### 16.10 Hypothesis B -- the alignment

The two productions DO refit with different tracker geometries, and the maker
exports the difference itself: the `runtree` carries per module
`dx/dy/dz = r_ideal - r_aligned` and `dtheta`, the angle between the two
surfaces' local x axes.  On the J/psi gun (`useIdealGeometry=True`) all four
are **identically zero**; on DY (`False`) they are not.  Both global tags carry
the SAME alignment tag (`TrackerAlignment_2016_ultralegacymc_v1`) -- it is the
maker switch that differs -- and in MC the Geant4 tracker is the IDEAL one, so
the aligned geometry displaces every reconstructed hit from the position the
particle actually crossed.  Projected on the MEASUREMENT direction
(`tail_align.py`, `tail_geom.py`; figure `misalignment_locx`):

| | rms `\|d_locx\|` | p99 | max | kurtosis | / sigma_hit |
|---|---|---|---|---|---|
| BPix | 2.40 um | 8.19 | 9.4 | 3.5 | 0.20 |
| FPix | 2.72 | 6.92 | 8.5 | 3.0 | 0.23 |
| TIB | 3.84 | 8.14 | **142.5** | **694** | 0.17 |
| TOB | 4.67 | 12.44 | 29.9 | 4.8 | 0.13 |
| TID | 3.05 | 7.71 | 11.9 | 3.3 | 0.10 |
| TEC | 9.91 | 10.41 | **731.6** | **4664** | 0.25 |

with `dtheta` rms 0.144 mrad (2.9 um at a module edge), 99.6 % of modules
moved by more than 1 um, 7 906 by more than 10 and 117 by more than 100.
That is a realistic residual-misalignment scenario and it is a genuine,
heavy-tailed, UNMODELLED hit-position noise -- so it is the natural candidate
for the excess chi2.  Three tests, all negative:

* **no STRUCTURE.**  The MEAN of each pull in 12 bins of either leg's `phi`
  and 8 of its `eta` is consistent with a constant (`chi2/ndof` against a
  constant 0.37-2.61, amplitude <= 0.14 sigma peak-to-peak).  Fitted as
  harmonics of the leg `phi` (0.5 %-trimmed, gen signal), **this is the
  DY alignment scenario's d0 bias, measured**:

  | | h1 | h2 | h3 | constant |
  |---|---|---|---|---|
  | vertex error `dx` [um] | 0.94 +- 0.79 | 1.41 +- 0.79 | 0.86 +- 0.79 | +0.60 +- 0.56 |
  | vertex error `dy` [um] | 1.50 +- 0.78 | 0.75 +- 0.78 | 0.81 +- 0.78 | +0.41 +- 0.55 |
  | `z_1` | 0.011 +- 0.014 | 0.014 +- 0.014 | 0.022 +- 0.014 | +0.013 +- 0.010 |
  | `z_v` | 0.009 +- 0.014 | 0.006 +- 0.014 | 0.007 +- 0.014 | +0.010 +- 0.010 |

  -- every harmonic consistent with zero, so the scenario's coherent vertex
  bias is **< 3 um at 95 % CL in any `phi` harmonic** and **< 0.05 sigma** on
  the pull, and the global offset is < 1.7 um.  In `eta` a linear and a
  quadratic term are likewise consistent with zero (< 1.5 sigma each).  The
  r-phi part of the MODULE displacements does carry a coherent sinusoid, but
  only ~2 um in amplitude, and it does not propagate into a vertex bias.
  **So there is essentially nothing here for the joint fit's alignment
  parameters to absorb** -- which is the other half of the answer: the
  misalignment is not a mean term either.
* **no EXPOSURE.**  `P(|z_1| > 5)`, `P(|z_1| > 3)` and `Var(z_1)` are flat --
  if anything falling -- against the worst `|d_locx|` the candidate's hits
  carry, taken over all hits, over the PIXEL hits and over the hits at
  `rho < 20 cm`; medians 44.9 um for the tail against 49.4 for the sample.
  The TEC exposure moves `P(chi2 prob < 1e-3)` only 0.049 -> 0.063.
* **THE DIRECT ONE, and it settles it.**  The SAME DY events refitted with
  `useIdealGeometry=True` (`run_prod_tail.sh`, dev2 @ `dbdedfde3c1`, the build
  that wrote `dy_bs_final`, 6 x 700 events, with a same-build same-events
  `GEOM=False` control so nothing but the switch differs).  Matched candidate
  by candidate on `(run, lumi, event, genidx+, genidx-)` -- the reconstructed
  `pT` moves between the two geometries, so a key that uses it matches nothing
  -- 1 878 pairs pass the baseline in both:

  | | median `chi2/ndof` | p90 | `P(prob<1e-3)` | EXTRA spread | `Var(z_1)` | `P(\|z_1\|>5)` |
  |---|---|---|---|---|---|---|
  | DY, UL16 MC alignment | 1.0221 | 1.6789 | 0.0538 +- 0.0052 | **0.306 (15.3 %)** | 4.141 | 0.0048 |
  | DY, **IDEAL geometry** | 1.0222 | 1.6730 | 0.0527 +- 0.0052 | **0.307 (15.4 %)** | 4.173 | 0.0053 |
  | J/psi gun (for scale) | 0.9092 | 1.3498 | 0.0156 +- 0.0005 | 0.107 (5.3 %) | -- | -- |

  **Nothing moves.**  The per-candidate `chi2/ndof` shifts by a median
  **+0.0007**, 92 of the 101 candidates with `prob < 1e-3` are still there,
  31 of the 32 at `|z_1| > 3` are still there (median `|z_1|` 3.62 against
  3.73), and the `eta` dependence is unchanged (0.051 / 0.048 / 0.060 aligned
  against 0.050 / 0.040 / 0.063 ideal across `|eta|` 0-1.4 / 1.4-1.8 /
  1.8-2.4).  **Removing the realistic misalignment entirely does not move the
  unmodelled noise-scale spread from DY's 15 % towards the gun's 5 %** --
  it moves it by +0.3 % of itself, and in the wrong direction.

A FULL-SIZE repeat (`dy_ideal`, 6 x 4000 events, launched at the same time)
is still running; `finish_tail_ideal.sh` is detached on submit51 and will
write `logs_tail/{extract_dy_ideal,ideal_full,mixture_ideal}.log` and
`/ceph/.../tail/dy_ideal.npz` when it lands.  It is a confirmation with 5.4x
the statistics, not the test -- the 1 878 matched pairs above already fix the
spread to 0.1 %.

**B is REFUTED, directly and not by inference.**  The DY refit does carry a
realistic misalignment; it is simply not what the chi2 excess or the tail is
made of.  (Figures `chi2ndof_ideal`, `chi2prob_dy_ideal`, `ladder_ideal`,
`density_z1_dy_ideal`.)

#### 16.11 THE CONSEQUENCE: it reaches the MASS

`(m - m_gen)` by chi2 class, gen signal, 1 %-trimmed (figure `mass_vs_chi2`):

| class | N | `<m - m_gen>` | `Var((m-m_gen)/sigma_m)` | `P(\|pull\| > 5)` |
|---|---|---|---|---|
| chi2 prob >= 0.01 | 9 422 | **+51.1 MeV** | 0.873 | 0.00064 |
| chi2 prob 1e-3..0.01 | 352 | +50.0 | 1.047 | 0.00284 |
| **chi2 prob < 1e-3** | **527** | **+195.2 MeV** | **1.278** | **0.01898** |
| all | 10 301 | +56.9 | 0.890 | 0.00165 |

`<m - m_gen>` rises monotonically with `chi2/ndof`, from +10 +- 22 MeV at 0.75
to +204 +- 135 MeV at 2.9.  **That is exactly what the mass term's
resolution-proportional (Jensen) bias must do** -- it is proportional to
`sigma_m^2`, and the REALISED `sigma_m^2` is the model one times
`chi2/ndof`.  The MEAN of `chi2/ndof` on DY is **1.19** (1 %-trimmed 1.11, median
1.01), so a Jensen correction evaluated at the MODEL resolution is 11-19 %
short on the ensemble, **+6 to +10 MeV** on a +55 MeV bias; and measured directly, the 5.1 %
bad-chi2 population moves the inclusive mean from +51.1 to +56.9 MeV, i.e.
**+5.8 MeV**.  At the Z-mass target of 1e-5 (0.9 MeV) that is not negligible.
`chi2/ndof < 3` removes 3.1 MeV of it.

#### 16.12 What this means for the likelihood, and for a DATA fit

**It is not a mean term.**  There is no `phi` or `eta` structure to absorb:
the alignment parameters of the joint fit have nothing to take here at the
level that matters (<= 0.14 sigma peak-to-peak, consistent with zero).

**The `chi2/ndof < 3` cut IS a physics cut, and it cuts exactly this
population.**  By gen truth:

| | N | signal eff of `chi2/ndof < 3` | bkg rejection | of the SIGNAL it removes, `prob < 1e-3` |
|---|---|---|---|---|
| DY | 10 412 | **0.99039 +- 0.00096** (99 removed) | 0.649 +- 0.045 (111 bkg) | **99/99 = 1.000** |
| J/psi gun | 55 155 | **0.99674 +- 0.00024** (180 removed) | n/a (no bkg) | **180/180 = 1.000** |

-- so the cut is a **1.0 % acceptance loss on DY signal against 0.33 % on the
gun**, the ratio being exactly the ratio of their excess-chi2 rates, and
**every single gen-signal candidate it removes is an excess-chi2 candidate**.
It is therefore NOT a background veto on DY (it rejects only 0.65 of 111
background candidates); it is an acceptance cut on the unmodelled-noise
population, and it buys little: `P(|z_v| > 5)` on gen signal goes
0.00146 +- 0.00038 -> 0.00098 +- 0.00031, and `P(|z_v| > 3)`
0.01107 -> 0.00980.  On the beam pull it does more -- `P(|z_1| > 5)` 0.00427
-> 0.00166 -- because `z_1` is the functional most correlated with the chi2.
Like `|z_v| < 5`, it TRUNCATES the density and a term fitted on the survivors
has to normalise for it; unlike `|z_v| < 5` nothing in the stack does.

**It is not a selection in the sense of removing something wrong**, except
that the same cut takes 61 of the 62 background candidates that sit at
`|z_1| > 5`.  Quoting the
tail without saying which chi2 cut is in force is meaningless:
`P(|z_1| > 5)` is 0.00166 with the cut and 0.00427 without, and the
`|z_1| > 5` SET goes from 6 % background to 58 %.

**It is a MODEL term, and the model it needs is a per-candidate NOISE SCALE.**
The residual terms all condition on a covariance that is right on average and
wrong candidate by candidate by a factor whose distribution the fit already
measures.  Three routes, in increasing order of honesty:

1. mix the CF density over a scale `s` with a fitted (not measured) prior --
   one or two extra parameters shared by every residual term, since the SAME
   `s` multiplies the mass, the vertex and the two beam functionals;
2. add an explicit unmodelled-hit-noise family to the resolution
   parameterisation -- section 16.13 sizes it at ~4e-3 of MEASUREMENTS beyond
   4 sigma, i.e. ~11 % of candidates carrying one -- which is the physical
   statement and folds into the material / hit-class block David asked for;
3. condition each term on the candidate's own `chi2`.  **This is a trap**: it
   is the same pairing that makes the mass likelihood biased (RESOLUTION.md
   2.1, the sigma artefact) -- `chi2` and `z` are built from one noise
   realisation, so normalising a residual by its own chi2 removes the
   information along with the nuisance.  If it is used at all it must be a
   PROFILED scale, not a plug-in one.

**A DATA fit.**  On data the beam-spot record IS the measurement of the
luminous region, fitted per lumi section from the tracks, so the MC question
"does the record describe the truth" has no data analogue -- what remains is
(i) the record's own statistical error and its per-lumi variation, which the
maker already reads per event, and (ii) the width scale, which section 14.18
already floats FREE (`sigma_x = 10.05 +- 0.37 um`, `sigma_y = 10.07 +- 0.30`)
because the record's quoted `BeamWidthError` of 0.29 um is smaller than the
mismatch.  Both stay as they are.  What does NOT carry over is the chi2
distribution: **data will have a wider one than this MC** (real misalignment,
real dead/noisy channels, real cluster splitting), so the scale-mixture term
must be fitted ON DATA and not taken from MC -- and the `chi2` distribution
itself becomes a data/MC comparison worth making before any residual term is
quoted on data.  The `chi2/ndof < 3` cut must be applied identically and
declared, exactly as `|z_v| < 5` is.

#### 16.13 The size of one bad measurement

Reading the excess as a single outlying measurement,
`chi2 - E[chi2] > t^2`, gen signal, no chi2 cut:

| | median excess | p90 | p99 | `P(> 3 sigma^2)` | `P(> 4 sigma^2)` | `P(> 5 sigma^2)` |
|---|---|---|---|---|---|---|
| DY | +0.37 | +18.0 | +54.6 | 0.2084 | **0.1148** | 0.0653 |
| gun | -2.52 | +9.7 | +35.5 | 0.1088 | **0.0483** | 0.0214 |

i.e. an implied per-MEASUREMENT outlier rate at 4 sigma of **4.3e-3** on DY
against **1.7e-3** on the gun, on ~27 measurements per candidate.  That is the
same order as the per-hit study's innovation tail (`P(|z| > 4)` 5.1e-4 data
against 1.5e-4 CF), so the two studies are almost certainly looking at one
phenomenon from two sides -- but the two-track chi2 sees it on EVERY
measurement of the pair, and the hit-noise family the resolution model needs
has to be fitted at that rate, not at the CF's.

#### 16.14 The classification of the tail, in one table

Gen-signal `|z_1| > 5`, published baseline (17 candidates) and with the chi2
cut lifted (44):

| cause | baseline | no chi2 cut | evidence |
|---|---|---|---|
| the fit's own excess chi2 | **0.88 +- 0.08** | **0.89 +- 0.05** | chi2 prob < 1e-3, against 0.044 / 0.051 inclusive |
| ... concentrated at `\|eta\| > 1.8` | 0.88 +- 0.08 | 0.80 +- 0.06 | against 0.35 inclusive |
| A, a displaced true vertex | 0.06 +- 0.05 | **0.00** | gen vertex > 3 sigma of the record's line |
| C, a thin or truncated leg | 0.00 | 0.00 | `nvalid == nhits`, reco/gen `pT` normal, flat in `nvalid` |
| D, the pixel hit classes alone | 0.00 | 0.00 | flat in the pixel variance share; the gun has MORE pixel share and LESS tail |
| B, the tracker misalignment | **0.00** | **0.00** | flat in the worst `\|d_locx\|` carried; and the SAME events refitted with the IDEAL geometry keep 92/101 of the bad-chi2 candidates and 31/32 of the `\|z_1\|>3` ones, spread 15.3 % -> 15.4 % |
| E, pileup vertex association | n/a | n/a | no vertex collection enters the residual |
| background (not gen signal) | 0.06 +- 0.06 (1/18) | **0.58 +- 0.05** (62/106) | `genbkg.classify` |

The rows are not exclusive by construction -- "excess chi2" is the MECHANISM
and the others are candidate CAUSES of it -- but only the first is populated.


#### 16.14b THE CAUSE, stated plainly, and what is left open

The two-track fit's covariance is right ON AVERAGE and wrong CANDIDATE BY
CANDIDATE by a factor whose spread is 13-15 % in sigma on DY and 4-5 % on the
gun, and which the fit itself measures as `chi2/ndof`.  Every constraint
residual inherits that factor in proportion to how much of its variance lives
in the noise the chi2 resolves (the ladder slope), and the 5 sigma tails are
exactly what that mixture predicts.  The physical realisation is an outlier
rate of ~4e-3 per MEASUREMENT beyond 4 sigma on DY against ~1.7e-3 on the gun
(section 16.13).

What it is NOT: the luminous region (0.08 +- 0.05 of the published 26), the
leg length or a truncated MiniAOD hit list (0.00), the pixel hit classes
(0.00, and the gun has MORE pixel share and LESS tail), the tracker
misalignment (0.00, by direct refit), the momentum (flat inside the gun over
2-20 GeV), or a wrong vertex association (no vertex collection enters).

What it IS, decomposed as far as this study can take it:

| | factor on `P(chi2 prob < 1e-3)` |
|---|---|
| in-time pileup (`nTrueInt` 12 -> 32+) | 1.2-1.5 |
| forward, DY-specific (`\|eta\|` 0.9 -> 2.2+; the GUN falls over the same range) | 1.3-1.6 |
| **residual, present in DY's most favourable cell (central, lowest pileup)** | **2.6** |

The residual 2.6x is what separates a 106X `slimmedMuons` MiniAOD track of a
FULL Z EVENT refit in 15_0 from a 15_0-native single-J/psi gun track in an
otherwise empty event.  The candidates, none of them tested here, are the
event's own hadronic activity (merged and mis-assigned clusters, which
`nTrueInt` does not count), the MiniAOD hit content itself (section 13.8
already traced `ndof == 0` to `slimmedMuons` not keeping every RecHit), and a
cross-release cluster/CPE difference.  **The next experiment is a DY-like
sample produced and refit in ONE release with pileup on and off**, which
separates all three at once; failing that, the same maker run on a 15_0-native
Z sample.

#### 16.15 How to reproduce it

    cd resolution/vtxres
    # the compact per-candidate cache (ceph is only readable from a submit node)
    python3 tail_extract.py --files '<prod>/task_*/globalcor_*.root' --out <npz>
    python3 tail_hypA.py   --npz <dy npz>                 # the luminous region
    python3 tail_geom.py   --a <dy file> --b <gun file>   # the two geometries
    python3 tail_align.py  --npz <dy npz> --files '<prod>/task_*/*.root' \
                           --ref <gun file>               # the misalignment
    python3 tail_hypBC.py  --npz <dy npz> --files ... --geom ...
    python3 tail_hypD.py   --npz <dy npz> --gun <gun npz> # classes + the dump
    python3 tail_scan.py / tail_chi2.py / tail_chi2src.py / tail_subdet.py
    python3 tail_mixture.py --npz <dy npz> --gun <gun npz>   # THE CLOSURE
    ./run_prod_tail.sh 6 0 5            # the ideal-geometry re-production
    python3 tail_ideal.py --real <npz> --ideal <npz>
    python3 tail_plots.py --npz ... --gun ... --align ... [--ideal ...]

`tail_common.baseline()` reproduces `extract_vtx.py`'s selection EXACTLY
(finite `sigma_v > 0`, `Jpsi_bsok`, `Jpsi_vtxok`, `cfmass_ok`,
`sigma_m > 0`, `chi2/ndof < 3`, `|vtxvchk| < 1e-4`, weaker leg >= 8,
`|z_v| < 5`), so every number here sits on the same 10 254 candidates section
14.12 published; `chi2 = 0.0` lifts the chi2 cut and `max_abs_vtxz = 0.0` the
residual cut, which is what a study OF the tail must do.

Caches under `/ceph/.../runs_vtxres_260911/tail/`, logs in
`resolution/vtxres/logs_tail/`, figures in
`~/public_html/ZMass/cvh/260913_tail/` (18 panels, one file per panel, PNG
twin for each PDF, `index.php` in place):
`density_{z1,z2,zv}_dy`, `density_zv_gun` (data against a Gaussian and against
the scale mixture, each with a data/mixture ratio panel), `ladder_all`
(`Var(z | chi2/ndof)` with the slope-1 line), `chi2prob_dy_gun` (flat if the
model is right, with a ratio panel), `chi2ndof_density` (against `chi2_ndof`,
with a ratio panel), `scalespread_enhancement` (the one-number closure against
the published data/CF), `chi2excess_vs_eta`, `chi2excess_vs_pileup`,
`genvtx_pull_{x,y}` (the luminous region against a Gaussian, with a ratio
panel), `misalignment_locx`, `mass_vs_chi2`, and the ideal-geometry set
`chi2ndof_ideal`, `chi2prob_dy_ideal`, `ladder_ideal`, `density_z1_dy_ideal`.

#### 16.16 What this changes in the standing picture

* Section 14.12's "the place to look next is the model of the innermost pixel
  hits" is **wrong as posed**, and open item 2 of this file ("a non-Gaussian
  HIT model") is answered: the missing ingredient is not a shape, it is a
  per-candidate SCALE.  The innermost pixel hits are where the
  residual's variance sits, but the tail is not theirs: it is a per-candidate
  noise-scale fluctuation that the fit measures for us in `chisqval`, and the
  gun -- with a LARGER pixel share -- has the SMALLER tail.
* Section 13's "the residual signal tail is the FEW-HIT LEGS" is superseded
  inside the standard selection: with `minLegHits = 8` applied there is no
  hit-count dependence left at all.
* The `chi2/ndof < 3` cut is not a technicality.  It removes 61 % of the
  gen-signal `|z_1| > 5` tail and 61 of the 62 background candidates that sit
  there; every tail number must be quoted with it stated.
* Any residual-term fit that quotes `sandwich/quoted` or a tail fraction is
  quoting it on a model that is right on average and wrong per candidate by
  `sqrt(chi2/ndof)`.  Whether that is what the beam channel's 1.138 is made of
  has not been tested here and is the obvious next thing to try.


## Defects found and fixed

1. **`Jpsi_d`'s charge re-sign destroys the sign it claims to define.**
   `theta_6` is INVARIANT under swapping the two legs (`twoTrackCart2pca` flips
   both `n_hat` and `x_b - x_a`), so the raw `theta_6` is already a well-defined
   signed DCA. `Jpsi_d = firstplus ? theta_6 : -theta_6` (maker line 4390)
   multiplies an invariant by the charge of leg 0 — and leg 0 is the mu+ on only
   **50.2 %** of candidates, so it RANDOMIZES the sign rather than defining it.
   Measured on 20 000 production candidates: `skew(z_v)` is **+0.40** in the RAW
   (swap-invariant) convention and **-0.33** in the `Jpsi_d` one (+0.075 vs
   -0.378 on the 400-candidate smoke — the skew is tail-dominated, but the sign
   flip is not).
   **Fix:** `Jpsi_vtxres` carries the RAW `theta_6` and `Jpsi_vtxfirstplus` is
   exported so the `Jpsi_d` convention can still be formed downstream. This
   matters because the CF exponents are built from `w_v` in the raw convention:
   re-signing the residual and not the weights would put the Landau skew on the
   wrong side. Anyone using `Jpsi_d` for a signed quantity should know this.

2. **`doVtxConstraint` constrains `d = d_seed`, not `d = 0`.** It freezes index 6
   at its SEED value, and nothing zeroed `statepca[6]` before the first iteration
   (`refftsarr` comes from `midPropagated`/perigee seeds whose PCA distance is not
   zero, maker 2285-2335). The path is dormant — every production runs
   `doVtxConstraint=False` — which is presumably why it was never caught.
   **Fix:** the `vtxConstraintZeroSeed` switch (default True).

3. **`imap_unordered` in the extraction broke the row correspondence.** The
   vertex and the mass npz were assembled in whatever order the 8 workers
   finished, so row `i` of one was NOT the same candidate as row `i` of the
   other. The joint card's `(run, lumi, event)` gate caught it.
   **Fix:** the extraction uses ordered `imap`, and `make_vtx_card.common_index`
   takes the INTERSECTION of the two selections up front.

4. **The Gaussian arms underflow float64.** A Gaussian density at `|z| = 40` is
   `exp(-800) = 1e-348`, below the smallest float64, so `gauss`/`gaussq` returned
   `NLL(0) = inf` and no comparison was possible.
   **Fix:** a `--max-abs-z 40` guard, applied IDENTICALLY to every arm and
   channel, dropping **0.012 %** of candidates. It IS a cut on the residual and is
   labelled as such; the UNCUT tails are reported separately. The candidates it
   removes are NOT fit failures by any convergence measure (`Jpsi_vtxbfree`,
   `Jpsi_vtxvchk`, `edmval`, the fitted vertex position and chi2/ndof are all
   normal), and NOT the `n_hat` degeneracy either — their muon opening angles are
   LARGER than the bulk's (median `sin theta` 0.264 vs 0.051), refuting the
   collinear-`p_a x p_b` hypothesis. They are a real tail: a fitted DCA of
   centimetres on a prompt J/psi gun.

---

## Traps and standing rules

1. **One unit convention: everything that leaves a term is PHYSICAL.**
   `matres/groups.py` is the single definition. `card_group_units` gives the
   card unit of a material parameter (`1/gprior` when the card is whitened, 1
   otherwise) and `term_units` / `card_units` READ it off a term or a card;
   `matrix_to_physical` / `gradient_to_physical` convert. Every Fisher, score
   and Hessian matrix, every fitted value and error, every prior, injection and
   table that leaves a term is converted to physical `k` (the log material
   amount of a group) or `eps` (the linear variance scale of a hit class)
   before it is written or printed, so no downstream tool has to be TOLD which
   convention it is holding and `efficiency.py --prior-power` /
   `recovery.py --prior-sigma` are gone.

   Historical note: while the two conventions coexisted, this pipeline's
   matrices were in CARD units and its efficiency calls carried
   `--prior-power 2`. Those logs (`logs/eff_*_p2.log`) reproduce exactly what
   the physical-unit code now produces with no flag at all
   (`logs_on/eff_off_*.log`: 0.885 / 2.659 / 2.778 / 1.215 / 1.115 / 0.961 /
   1.543 / 1.061 / 1.177).

2. **The `--max-vchk 1e-4` cut is a cut on the FIT'S OWN COVARIANCE, not on the
   residual.** `extract_vtx.py --max-vchk 1e-4` drops **0.245 %** of candidates
   (49 in 20 000). Every one of them has a `sigma_v` of order 10^2 m (the printed
   examples run 76 m to 462 m) — a fit whose vertex direction is unconstrained.

3. **Never quote the untrimmed moments of this residual IN THE FREE REGIME.**
   They are set by ~0.2 % of candidates on the gun and ~2 % on DY; the tails
   table is the description. With the vertex constraint ON that tail is gone
   (untrimmed Var 1.0065, skew +0.05, kurt 5.14) and the moments are quotable
   — section 12.

4. **The ionization sign is per block, not global.** 16 % of vertex blocks carry
   -1. A single `ioniSign` would be wrong; use `cvhcf::TrackInput::ressgn`.

5. `/ceph` is NOT readable from submit82 — use submit50/51/52. `run_tf.sh` binds
   ceph only when the host can read it.

6. **Do not `scram b` in `..._dev2` while a production runs from it**, and do not
   edit a bash script that is executing. (The `.py` modules can be edited freely.)

7. `/work/submit` quota is 500 G and nearly full: everything above ~0.5 G lives
   on ceph under `runs_vtxres_260911/runs/` and is symlinked into
   `resolution/runs/vtxres/`.

---

## Open items

None blocking; each is a new study.

1. ~~**The mass term's Gaussian arm returns a NaN Hessian.**~~ GONE with the
   vertex constraint on (section 12.8): `mass_gaussq` and `joint_gaussq` both
   run and certify there. The free-regime symptom is recorded below. `mass_gaussq` and
   `joint_gaussq` have a finite NLL and gradient at theta = 0 but `H` is NaN;
   with `floor="clip"` instead of `"softplus"` the Hessian is finite and the NLL
   is `inf`, so the density is going to <= 0 (or underflowing) for some
   candidates and the softplus floor's SECOND derivative is what NaNs there.
   Localised to the MASS term with a Gaussian arm — the vertex term's three arms
   and the mass term's CF arm are all fine — and no headline needs it.

2. ~~**A non-Gaussian HIT model.**~~ ANSWERED by section 16, and the answer
   is that the missing ingredient is not a SHAPE but a per-candidate SCALE:
   `Var(z | chi2/ndof) = chi2/ndof` with a measured slope +1.06 +- 0.06, and
   the DY `chi2` distribution carries a 12.8-15 % UNMODELLED noise-scale
   spread against the gun's 4-5 %, which reproduces every published data/CF at
   5 sigma (18.4 predicted against 16.0 for `z_1`, 1.01 against 1.0-1.7 for
   the gun) with no free parameter. The sentence below about the few-hit legs
   is SUPERSEDED -- that population lives below `minLegHits = 8`, which the
   maker now cuts pre-fit, and inside the standard selection the tail has no
   hit-count dependence at all. The original text: Both arms treat the hit
   noise as exactly Gaussian. On the gun the CF is within 1.4-1.7 of the data out to 5 sigma in
   the free regime and 1.03-1.26 with the constraint on; on DY it is a factor
   85 short free and 7.5 constrained, and the residual DY excess is
   combinatorial rather than a resolution effect (section 12.9) -- section 13
   proves that by gen matching (86.8 +- 2.5 % of the free-regime `|z_v| > 5`
   candidates are background). What is LEFT after the background is removed is
   still a factor **10.2** free / 4.07 constrained on gen signal, against
   1.03-1.68 on the gun, so the DY signal tail is a real, unmodelled excess
   and this item stands. Its likeliest cause is in hand: gen signal with a
   weaker leg of <= 8 valid hits has `P(|z_v|>5)` = 0.104 +- 0.044 against
   0.0020 +- 0.0004 for the rest.

6. ~~**Should `minLegHits` be on by default?**~~ DECIDED and DONE, section
   15: `minLegHits = 8` is the maker default (`dbfe6e4b2c2`), gated
   bit-identical on the surviving candidates of both gate samples. The
   companion `|z_v| < 5` is NOT a maker cut -- it is the downstream standard
   selection (`resolution/selection.py`) with the matching truncated
   normalisation.

3. **The concatenated-tau trick is not ported.** Every exponent primitive depends
   on `weight * tau` alone, so the mass and the vertex functional could share ONE
   `cvhcf` pass; as it stands the second call costs the same as the first
   (+0.54 s/candidate, +48 %).

4. ~~Unify the prior-unit convention~~ — DONE. `hitlik_term.build` takes
   `group_units`, the card builders pass it in, and `matres/groups.py` is the
   one definition (trap 1).

5. **`hitlik/recovery.py` prints `nan` in its `/truth` columns on these cards**
   (it reads the injected truth under a key `make_vtx_card.py` does not write).
   The recovery numbers above were obtained from its own `shift` and `f_pri`
   columns.

6. `run_all.sh certify` writes `logs/certify.log`, which is absent; the certified
   values, NLLs and EDMs above come from `logs/fits_all.log` and
   `logs/fit_<name>.log`.

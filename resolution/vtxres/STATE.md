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

2. **A non-Gaussian HIT model.** Both arms treat the hit noise as exactly
   Gaussian. On the gun the CF is within 1.4-1.7 of the data out to 5 sigma in
   the free regime and 1.03-1.26 with the constraint on; on DY it is a factor
   85 short free and 7.5 constrained, and the residual DY excess is
   combinatorial rather than a resolution effect (section 12.9).

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

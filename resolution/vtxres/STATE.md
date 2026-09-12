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
    doVtxConstraint=False        (index 6 FREE -- what every production does)

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
`run_fit.sh <name>` are the per-card entry points. Every efficiency call on this
pipeline must carry `--prior-power 2` (see the unit trap below); `run_all.sh eff`
does not add it, so the certified numbers come from the explicit calls whose logs
are `logs/eff_{vtx,mass,joint}_p2.log`.

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

Third-party consumers: `hitlik/efficiency.py` (the sandwich),
`hitlik/recovery.py` (the injections), `hitlik/perhit/certify.py` (value + NLL +
EDM), `matres/pick_models.py` (chooses `tf-trust-krylov` for frozen-parameter
cards, `trust-exact` otherwise).

### Figures

`~/public_html/ZMass/cvh/260911_vtxres/` — 20 panels plus `index.php`:
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
`|sum_m g_m - g| = 0`) + `hitlik/efficiency.py --prior-power 2`. 8 000
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

1. **The `--prior-power` unit trap.** `hitlik/efficiency.py` applies the prior in
   whatever units the FISHER MATRICES carry, and the two pipelines differ:
   * `hitlik/fisher_cmp.py` builds through `hitlik_term.build`, which sets
     `group_units = ones` — the parameter IS the physical `k`, one tier prior is
     `gprior`, and `--prior-power 1` (the default) is RIGHT there;
   * `make_*_card.py` OVERRIDES `term.group_units` to `1/gprior`, where one tier
     prior is `gprior**2`; `vtxres/fisher_vtx.py` builds through
     `make_vtx_card.build_term`, i.e. in CARD units, so **only THIS pipeline
     needs `--prior-power 2`**.

   Every vtxres number in this file was produced at `--prior-power 2`. Getting it
   wrong the other way is diagnosable: re-evaluating the stored k-unit matrices
   (`runs/hitlik/fisherHJ20k.npz`, `runs/perhit/fisherHJ{,_all}.npz`) at
   `--prior-power 2` gives them a 20x-too-tight prior, `(H+P)^-1` collapses onto
   it and the informativeness test rejects every material group — the material
   table comes out EMPTY. At `--prior-power 1` they reproduce every published
   hitlik/perhit number exactly (1.825 / 2.093, 0.974 / 1.338, 1.204 / 1.221;
   1.097 / 1.171, 0.529 / 0.563; 1.486 / 1.393; 1.510 / 1.657). Making `build`
   and the card share one convention would remove the trap.

2. **The `--max-vchk 1e-4` cut is a cut on the FIT'S OWN COVARIANCE, not on the
   residual.** `extract_vtx.py --max-vchk 1e-4` drops **0.245 %** of candidates
   (49 in 20 000). Every one of them has a `sigma_v` of order 10^2 m (the printed
   examples run 76 m to 462 m) — a fit whose vertex direction is unconstrained.

3. **Never quote the untrimmed moments of this residual.** They are set by ~0.2 %
   of candidates on the gun and ~2 % on DY; the tails table is the description.

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

1. **The mass term's Gaussian arm returns a NaN Hessian.** `mass_gaussq` and
   `joint_gaussq` have a finite NLL and gradient at theta = 0 but `H` is NaN;
   with `floor="clip"` instead of `"softplus"` the Hessian is finite and the NLL
   is `inf`, so the density is going to <= 0 (or underflowing) for some
   candidates and the softplus floor's SECOND derivative is what NaNs there.
   Localised to the MASS term with a Gaussian arm — the vertex term's three arms
   and the mass term's CF arm are all fine — and no headline needs it.

2. **A non-Gaussian HIT model.** Both arms treat the hit noise as exactly
   Gaussian. On the gun the CF is within 1.4-1.7 of the data out to 5 sigma; on
   DY it is a factor 85 short, and a data fit will need an outlier component
   whose size is now measured (~2 %).

3. **The concatenated-tau trick is not ported.** Every exponent primitive depends
   on `weight * tau` alone, so the mass and the vertex functional could share ONE
   `cvhcf` pass; as it stands the second call costs the same as the first
   (+0.54 s/candidate, +48 %).

4. **Unify the prior-unit convention** between `hitlik_term.build` and
   `make_*_card.build_term` so that `--prior-power` is no longer needed.

5. **`hitlik/recovery.py` prints `nan` in its `/truth` columns on these cards**
   (it reads the injected truth under a key `make_vtx_card.py` does not write).
   The recovery numbers above were obtained from its own `shift` and `f_pri`
   columns.

6. `run_all.sh certify` writes `logs/certify.log`, which is absent; the certified
   values, NLLs and EDMs above come from `logs/fits_all.log` and
   `logs/fit_<name>.log`.

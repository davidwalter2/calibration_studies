# STATE_bkg.md -- the gen-background question and the minimum-size cut

Working checkpoint for the two linked tasks on the two-track fit's
vertex-constraint residual.  Folded into `STATE.md` section 13 when done.

## The two questions

**A.** Section 12.9 argued the DY vertex tail is COMBINATORIAL (116 of 146
free-regime outliers sit in the 311 candidates of multi-candidate events).
David: "If the candidates with large vertex residuals are background we should
cut on it ... Did you verify that those are background by doing a gen
matching?"  No gen matching had been done.  Do it.

**B.** ndof == 0 crashed an earlier DY production (fixed in `fab515e`).
David: "How are track pairs with 9 hits treated?  I suggest to also add a cut
with requiring more than 9 (10 w/o vertex constraint) hits."

## Where things are

| what | where |
|---|---|
| maker | `CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc`, branch `cvh-exports-clean-260911` |
| analysis | `resolution/vtxres/genbkg.py` (new), `extract_vtx.py` (extended), `run_bkg.sh` (new) |
| outputs | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/bkg/{dy_vtxon_gen,gate_gun,gate_dy}/` |
| figures | `~/public_html/ZMass/cvh/<YYMMDD>_vtxbkg/` |

## DONE

### The maker (built on submit50, el9, 0 errors)

* **gen provenance**, behind the existing `doGen_`: `Mu{plus,minus}gen_pdgId`,
  `_idx`, `_motherPdgId`, `_motherIdx`, `_isPrompt`, `_fromHardProcess`, and
  `Jpsigen_sameDecay`.  The MATCH itself is unchanged (closest status-1 gen
  muon of the same charge within dR < 0.1); what is new is the IDENTITY and
  the ANCESTRY, which is what separates one muon reconstructed twice from two
  different muons.  `Mu*gen_dr` and the reco/gen pT ratio were already
  derivable from the existing branches.
* **minimum size of a pair**, pre-fit, three knobs:
  `minNdof` (default 1), `minPairHits` (default -1 = auto = 10 ON / 11 OFF),
  `minLegHits` (default 0 = OFF, the evidence-driven one; see below).
  Counted in the fit summary as `skipped[ndof<N]`, `skipped[hits<N]`,
  `skipped[leghits<N]`.  Documented in all 8 two-track cfis and in both run
  configs.

### Task B -- the arithmetic, measured

`ndof = nvalid + nvalidpixel - nstatefree`, `nstatefree = 10`, plus the
constraint rows (+3 bs, +1 pointing, +1 vertex, +1 mass on the constrained
pass).  With the vertex constraint on that is `n_meas - 9`; off, `n_meas - 10`
-- so David's "more than 9 (10 w/o constraint) hits" IS `ndof >= 1`.

VERIFIED on every written candidate of all four productions:
`ndof == nvalid + nvalidpixel - 9` for 10654/10654 (`dy_vtxon`) and
300017/300017 (`prod_vtxon`); `- 10` for 7948/7948 (`dy`) and 300025/300025
(`prod`).  Zero deviations.

| sample | written | ndof min | nvalid(pair) min | minNdof=1 cuts | minPairHits cuts |
|---|---|---|---|---|---|
| `dy_vtxon` ON | 10 654 | 1 | 9 | 0 | 2 (0.019 %) |
| `dy` OFF | 7 948 | 1 | 9 | 0 | 1 (0.013 %) |
| `prod_vtxon` ON | 300 017 | 3 | 9 | 0 | 1 (0.0003 %) |
| `prod` OFF | 300 025 | 2 | 9 | 0 | 1 (0.0003 %) |

`ndof == 0` is never WRITTEN -- `fab515e` aborts the fit there -- so
`minNdof = 1` is a no-op on existing output by construction and its value is
that it moves the rejection BEFORE the fit (and makes the abort configurable
rather than hard-wired).  `minPairHits` removes the `nvalid == 9` pairs, and
they are extreme: `|z_v|` = 58.3 (ON) and 96.2 (OFF).

**ndof == 1 IS degenerate, as predicted.** `dy_vtxon` idx 3879: ndof 1,
nvalid (3,6), `sigma_v` = **32 515 cm**, `z_v` = 7.2e-6, chi2/ndof = 66 432,
and `Jpsi_vtxok` is **TRUE**.  The DCA is fixed by the data, the pull collapses
to zero and the candidate is written as good.  The neighbours are as bad:
ndof 2 -> `|z_v|` = 58.3; ndof 3 -> `sigma_m` = 1041 GeV; ndof 5 ->
`sigma_m` = 73 859 GeV.

**Non-finite exports EXIST**, and the cause is the WEAKER LEG, not the pair:

| sample | non-finite `Jpsi_sigmamass` | of those, `Jpsi_vtxok` TRUE | weaker leg <= 3 |
|---|---|---|---|
| `dy_vtxon` | 5 (0.047 %) | 1 | 5/5 = 100 % |
| `dy` | 10 (0.126 %) | 2 | 9/10 = 90 % (<=4: 100 %) |
| `prod_vtxon` | 186 (0.062 %) | 76 | 186/186 = 100 % |
| `prod` | 525 (0.175 %) | 83 | 462/525 = 88 % (<=4: 100 %) |

Their PAIR totals are 13-23 hits, so no pair-sum cut can see them.
`sigma_v > 1000 cm` on 391 (`prod_vtxon`) / 447 (`prod`) candidates, ALL with
`Jpsi_vtxok` TRUE, median weaker leg 2 hits -- a degenerate vertex written as
good, which pushes `z_v` to zero and narrows the core.

**The weaker leg is also where the tail is** (selected sample,
`P(|z_v| > 5)`):

| sample | weak <= 3 | weak <= 5 | inclusive |
|---|---|---|---|
| `prod_vtxon` | 0.0070 +- 0.0040 (N 429) | 0.0017 +- 0.0010 (N 1811) | 0.00104 +- 0.00006 |
| `prod` | 0.0140 +- 0.0057 (N 428) | 0.0055 +- 0.0017 (N 1812) | 0.00161 +- 0.00007 |
| `dy_vtxon` | 0/1 | 0.083 +- 0.080 (N 12) | 0.00165 +- 0.00040 |
| `dy` | 1/2 | 0.643 +- 0.128 (N 14) | 0.0186 +- 0.0015 |

The gun is pure signal, so on the gun this is a RESOLUTION-MODEL effect (a
few-hit leg has a badly estimated `sigma_v`), not background.

### Task B -- the current selection (there ISN'T one)

Nothing in either chain requires a minimum number of hits, on a leg or on a
pair, and the maker's only pre-fit guards are the charge sum
(`fitSkippedSameSign_`) and `nhits != 0` per leg:

* **DY**: `slimmedMuons` -> `TrackProducerFromPatMuons`
  (`innerTrackOnly=False` -> `muonBestTrack`, `ptMin = -1` i.e. no cut, the
  track must have `extra().isAvailable()`) -> `DiMuonTrackVertexCandidate-
  Producer` (opposite sign, 60 < m < 120 GeV).  The few-hit legs are a MiniAOD
  slimming artefact: `slimmedMuons` keeps the hit PATTERN but not every
  RecHit.
* **J/psi gun**: `useLegacyPairLoop=True`, i.e. the all-pairs loop over
  `generalTracks` with no candidate producer at all.

So `minNdof` / `minPairHits` / `minLegHits` are the FIRST hit requirement in
the chain.

### Task B -- the gate (`genbkg.py --gate`)

The new build re-run on the SAME inputs as the reference production, every
candidate matched on (run, lumi, event, nhits, nvalid) and compared field by
field:

| | matched | bit-identical | missing from ref | ref-only | ndof min | non-finite |
|---|---|---|---|---|---|---|
| `gate_gun` (200 gun ev) vs `prod_vtxon/task_0000` | **192** | **29/29 branches** | 0 | 0 | 10 | 0 |
| `gate_dy` (400 DY ev) vs `dy_vtxon/task_0000` | **186** | **29/29 branches** | 0 | 0 | 12 | 0 |

Both fit summaries show the new counters at zero:
`skipped[ndof<1]=0  skipped[hits<10]=0  skipped[leghits<0]=0`.
Committed as `23b4c9c7046` on `cvh-exports-clean-260911`.

### The ndof <= 0 abort DOES fire, and only on DY

Summed over the productions' own logs:

| sample | attempted | succeeded | `fail[ndof]` |
|---|---|---|---|
| `dy_vtxon` | 10 671 | 10 654 | **1** (9.4e-5) |
| `dy` | 7 964 | 7 948 | **2** (2.5e-4) |
| `prod_vtxon` | 300 370 | 300 017 | 0 |
| `prod` | 300 370 | 300 025 | 0 |

Before `fab515e` each of those three would have aborted the PROCESS and lost
the whole task's output.

### The gun floor -- what a `|z_v|` cut costs with NO background at all

Full gun production, extraction selection applied (297 867 / 298 010
candidates):

| | `P(>3)` | `P(>4)` | `P(>5)` | `P(>10)` |
|---|---|---|---|---|
| `prod_vtxon` (ON) | 0.0081 +- 0.0002 | 0.0025 +- 0.0001 | **0.00104 +- 0.00006** | 0.0000 |
| `prod` (OFF) | 0.0088 +- 0.0002 | 0.0032 +- 0.0001 | **0.00161 +- 0.00007** | 0.00031 |

The gun is pure signal by construction, so **0.104 %** is the irreducible
signal loss of a `|z_v| < 5` cut in the constrained regime -- and it is the
SAME number as the DY same-candidate figure of section 12.9 (0.107 %).  The
DY tail, once the combinatorics is removed, IS the gun's resolution-model
floor.

### Task A -- the FREE-regime DY answer (10 494 candidates, gen truth)

`bkg/dy_vtxoff_gen`, 6 x 4000 events, the extraction selection
(`Jpsi_vtxok`, `cfmass_ok`, sigmas > 0, chi2/ndof < 3, `|vtxvchk|` < 1e-4).
This regime is quoted first because it is where the tail LIVES (190 candidates
at `|z_v| > 5`, against 17 with the constraint on), so the composition can be
measured rather than bounded.

**The match criterion** is the maker's own, unchanged: the closest status-1
gen muon of the SAME CHARGE within dR < 0.1 of the leg's fitted momentum.
The classification adds the identity and the ancestry (see `genbkg.py`).

| class | N | fraction | `P(>3)` | `P(>4)` | `P(>5)` | `P(>10)` |
|---|---|---|---|---|---|---|
| signal | 10 266 | 0.9783 +- 0.0014 | 0.0137 +- 0.0011 | 0.0044 +- 0.0007 | **0.0024 +- 0.0005** | 0.0009 +- 0.0003 |
| dup | 117 | 0.0111 +- 0.0010 | 0.906 +- 0.027 | 0.906 +- 0.027 | **0.872 +- 0.031** | 0.795 +- 0.037 |
| otherdecay | 12 | 0.0011 +- 0.0003 | 0.417 +- 0.142 | 0.333 +- 0.136 | 0.250 +- 0.125 | 0.250 +- 0.125 |
| unmatched | 99 | 0.0094 +- 0.0009 | 0.667 +- 0.047 | 0.626 +- 0.049 | **0.606 +- 0.049** | 0.485 +- 0.050 |
| nonres | 0 | | | | | |

**THE ANSWER.** Of the 190 candidates at `|z_v| > 5`, **86.8 % +- 2.5 % are
BACKGROUND by gen truth** (102 dup + 60 unmatched + 3 otherdecay, 25 signal).
At `|z_v| > 10` it is **94.1 % +- 1.9 %**. Yes: the large-vertex-residual
candidates ARE background, and cutting on `|z_v|` is background rejection.

**The cut table** (`|z_v| < t`), signal efficiency and background rejection
both against gen truth:

| | | ALL (10 266 S / 228 B) | 1 cand/event (10 078 S / 27 B) | >1 cand/event (188 S / 201 B) |
|---|---|---|---|---|
| t = 3 | eff / rej | 0.9863 +- 0.0011 / 0.776 +- 0.028 | 0.9864 / 0.815 +- 0.075 | 0.9787 +- 0.0105 / 0.771 +- 0.030 |
| t = 4 | | 0.9956 +- 0.0007 / 0.754 +- 0.029 | 0.9957 / 0.741 +- 0.084 | 0.9894 +- 0.0075 / 0.756 +- 0.030 |
| **t = 5** | | **0.9976 +- 0.0005 / 0.724 +- 0.030** | 0.9977 / 0.704 +- 0.088 | 0.9894 +- 0.0075 / 0.726 +- 0.031 |
| t = 10 | | 0.9991 +- 0.0003 / 0.632 +- 0.032 | 0.9993 / 0.630 +- 0.093 | 0.9894 +- 0.0075 / 0.632 +- 0.034 |

Purity 0.9783 -> **0.9939 +- 0.0008** at t = 5 inclusively, and
**0.483 -> 0.772 +- 0.027** in multi-candidate events, where half the
candidates are background (188 signal against 201 background).

**The multi-candidate argument of section 12.9 is CONFIRMED by truth**: 201 of
the 228 background candidates sit in multi-candidate events, 27 in
single-candidate events (0.27 % of those).

**The duplicate pairings are what the tail is made of.** 112 gen decays were
reconstructed more than once (229 candidates). Their `|z_v|` median is
**79.1**, p90 362, `P(>5)` 87 %, against median 0.67 for signal. And the
vertex residual is the RIGHT discriminator for choosing between them: the
gen-preferred candidate is the one with the smallest `|z_v|` in
**97.3 % +- 1.5 %** of those 112 decays, against **83.0 % +- 3.6 %** for the
smallest chi2/ndof.

**What the classes look like**, showing the duplicates are badly reconstructed
tracks rather than a second good fit:

| class | dR(+) med/p99 | pT+/genpT+ med / p5 / p95 |
|---|---|---|
| signal | 0.00030 / 0.00229 | 1.0005 / 0.969 / 1.035 |
| dup | **0.01607 / 0.08607** | 0.999 / **0.516 / 1.528** |
| unmatched | 0.00041 / 0.03048 | 0.997 / 0.965 / 1.063 |

Mother spectrum of the plus leg: 23 (the Z) x 10 438, none found x 50,
421 x 2, 411 x 2, 521 x 1, 511 x 1 -- real heavy-flavour muons exist and are
negligible.

**The CF model describes the SIGNAL class**, and nothing else:
data/CF at 3 / 4 / 5 sigma is **3.3 / 7.0 / 10.2** for signal against
175 / 889 / 1964 for background and 238 / 1833 / **4890** for the duplicates.
(The inclusive free-regime figure in section 12.3 was 85 at 5 sigma.)

The residual signal tail is again the FEW-HIT LEGS: gen signal with a weaker
leg of <= 8 valid hits has `P(|z_v|>5)` = **0.104 +- 0.044** on 48 candidates,
against **0.0020 +- 0.0004** on the 10 218 with more -- and that is the gun's
free-regime floor (0.00161 +- 0.00007).

**Without the chi2/ndof < 3 and `|vtxvchk|` cuts** (the extraction's own,
and chi2/ndof is correlated with the residual being measured, so the table is
quoted both ways): 10 628 candidates, background **2.92 % +- 0.16 %**,
`|z_v| > 5` background fraction **88.4 % +- 2.1 %** (against 86.8 %), signal
efficiency at t = 5 **0.9973 +- 0.0005**, rejection 0.690 +- 0.026. The chi2
cut removes ~80 background candidates already; it changes no conclusion.

Figures, `~/public_html/ZMass/cvh/260912_vtxbkg/`:
`density_dy_vtxoff_gen_{signal,background,dup,unmatched}` (density + CF model
+ data/CF ratio panel, one file per panel), `cut_roc_dy_vtxoff_gen`
(efficiency against rejection, both against truth, the three event-
multiplicity classes) and `cut_composition_dy_vtxoff_gen` (the `|z_v|`
spectrum stacked by class).

### A CORRECTION to section 12.9

Section 12.9 says "the constrained fit writes 10 654 candidates against 7 948
... the constraint converges on 2 616 events the free fit produced nothing
for". **It has nothing to do with the constraint: the two productions ran on
DIFFERENT NUMBERS OF EVENTS.** `dy/task_0000/local.log` carries
`nEvents=3000`, `dy_vtxon` ran the slurm array at 4000. Per event the two
agree exactly:

| production | events/task | pairs attempted | attempted / event |
|---|---|---|---|
| `dy` (free, 09-11) | 3 000 | 7 964 | 0.4424 |
| `dy_vtxon` (09-12) | 4 000 | 10 671 | 0.4446 |
| `dy_vtxoff_gen` (free, today, 4 000) | 4 000 | **10 668** | **0.4445** |

Re-running the SAME 6 files with the SAME `doVtxConstraint=False` at 4 000
events writes 10 648 candidates (10 494 after the extraction selection)
against 10 654 (10 325) constrained -- a 0.06 % difference, not 34 %. The
"2 616 events the free fit produced nothing for" were the 1 000 events per
task that were never read.

What survives unchanged is the tail FRACTION: 190/10 494 = **1.81 %** now
against 146/7 841 = 1.86 % then, and the same-candidate figure of 0.107 %.

## RUNNING (started 12 Sep ~11:45)

* `bkg/dy_vtxon_gen/task_000{0..5}` -- the 6 DY files re-run with the gen
  provenance on, CONSTRAINT ON (the free leg `dy_vtxoff_gen` is DONE and
  analysed above).

## NEXT

1. `extract_vtx.py --functional vtx` on `bkg/dy_vtxon_gen` -> npz with the
   gen keys; then `genbkg.py --classes --cuts --mass --density`.
2. Commit the scripts; fold this file into `STATE.md` section 13 and
   `RESOLUTION.md` 2.6 / 9.  (The maker is committed: `23b4c9c7046`.)
3. Merge `run_bkg2.sh` back into `run_bkg.sh` once the production it is
   running has finished, and delete it.

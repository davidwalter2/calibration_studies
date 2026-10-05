# STATE — the exact marginal by the conditional-Kalman recursion (`rbgrid.py`)

Resumption note; the method and every gate's numbers are README section 20.

## Code (trklik/solver/)
- `rbgrid.py` — the recursion: `build_legs` + `_precondition` (whitened states),
  `backward_filter`, `remaining_information` (trim look-ahead through later
  kicks), `KickLaw`, `_levy_kernel` (compact core step, integer compensation),
  `_reframe_kick` (local Lagrange re-grid), `leg_measure`, `LegMixture`,
  `RBGridLikelihood` (the objective `minimise` / `mass_profile` use).
- Gates: `rb_census.py` (A0), `rb_gate_a1.py` (A1), `rb_mc.py` + `rb_gate_a2.py`
  (A2), `rb_bartlett.py` (score identities), `tail_toy.py --arm-R` +
  `rb_tail_report.py` (A3; `--ab` pairs two arm-R runs, `--forms` splits the
  elimination, `--bins` by sigma_m/m and ionisation share), `rb_converge.py`
  (lattice settings), `rb_cost.py` (A4), `rb_tail_toy.sbatch` (A3 on slurm).
- Outputs: `runs/rbgrid/` — `census.log`, `a1.log`, `a2.log`, `bartlett.log`,
  `a3_full/` (153 600 replicas, `code.md5`), `forms/`, `ab4/` (lattice A/B),
  `converge.log`, `a4_cost.log`.

## Where it stands
- A0-A2 and the score identities pass: the recursion is the likelihood.
- A3 does not close: R - G = -0.059 +- 0.007 e-3 (R - H -0.063 +- 0.003);
  lattice settings move it by <= 0.003; the residual is in the nuisance
  elimination (profile -0.09 vs G, modified-profile corrections +0.07).
- Cost 2.6 s per data candidate (one core), 90 % in `_assign_local`.

## Open
- The modified profile's residual with a skewed, heavy-tailed likelihood:
  a higher-order or simulation-based correction, validated in this toy
  (target +-0.006 e-3), before any real-record use.
- The momentum scaling of the block covariances (`use_scale`) is not in the
  recursion; MS stays Gaussian; ditracks need the legs joined at the vertex.
- Gate 2b on real records not run.

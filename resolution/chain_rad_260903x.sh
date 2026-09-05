#!/bin/bash
# Extractions of the 2026-09-03 (_260903x) productions WITH the new radiative
# (brems + pair) CF term. Serial by sample -- extract_parallel.sh takes one
# BLAS-pinned python per shard and two samples at once has previously put this
# user over `ulimit -u` and killed running cmsRun jobs (finish_muon_closure_
# 260902.sh). Offline Kokoulin OFF throughout (project convention since
# 2026-09-03: >7x cost, ~1e-3 effect).
#
# NORMALISATION: `var` on BOTH arms. Since the `ioniqscalev` export
# (2026-09-03) `var` is exact for CgfQoPMode=1 as well, so the mode-1 samples
# no longer need `--ioni-norm raw` -- that is the whole point of the export.
#
# `nomsrad` IS EXTRACTED WITH THE TERM ON, not with --no-rad. Its simulation
# has no bremsstrahlung and no pair production, so the term must NOT be in its
# physics model -- but the refit propagator still logs radiative steps (its
# own physics list is nominal), so having the arrays in the cache makes both
# arms available from ONE extraction: `--krad 0` is the physics arm and
# `--krad 1` is the sanity check that must get WORSE. --no-rad remains the
# extraction-side switch and is exercised in radterm_validate / the one-file
# gate; it writes zeros and rad_model=0.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export CVH_IONI_KOKOULIN=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/rad260903x; mkdir -p "$LOG"
NSH=${NSH:-160}

ex() { # <production tag> <cache tag> [extra args]
  [ -s "runs/cf_trackres_$2.npz" ] && { echo "=== skip $2 (exists)"; return; }
  echo "=== extract $2 ($(date +%H:%M:%S)) ==="
  EXTRA_ARGS="${3:-}" ./extract_parallel.sh "$CEPH/resolution_trackres_$1" \
      "runs/cf_trackres_$2.npz" "$NSH" > "$LOG/ext_$2.log" 2>&1 \
    && echo "    [ok] $2 ($(date +%H:%M:%S))  $(ls -la runs/cf_trackres_$2.npz | awk '{print $5}')" \
    || echo "    [FAIL] $2 -- see $LOG/ext_$2.log"
}

ex mugun_lowpt_260903x_m0          mugun_lowpt_260903x_m0_k0          ""
ex mugun_ul16_260903x_m0           mugun_ul16_260903x_m0_k0           ""
ex mugun_lowpt_260903x             mugun_lowpt_260903x_k0             ""
ex mugun_ul16_260903x              mugun_ul16_260903x_k0              ""
ex mugun_lowpt_noms_260903x_m0     mugun_lowpt_noms_260903x_m0_k0     ""
ex mugun_lowpt_nomsrad_260903x_m0  mugun_lowpt_nomsrad_260903x_m0_k0  ""
echo "=== ALL EXTRACTIONS DONE ($(date +%H:%M:%S)) ==="

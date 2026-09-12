#!/bin/bash
# GEN LEG KINEMATICS for both v2 productions, aligned to the pairs caches.
#
# `oddmoment/aux_gen.py` already does exactly this and validates its own
# alignment: it matches the cache row by row on `z` and asserts that `z` AND
# `sigma` come back BIT-IDENTICAL, then reports `max|f_hit - cache vgf|`. That
# is a stronger contract than a run/lumi/event join, which cannot detect a
# reordering within an event.
#
# It writes, per cache row:
#   Muplusgen_pt/eta/phi, Muminusgen_pt/eta/phi   (charge is the +/- label)
#   Jpsigen_pt/eta/phi, Jpsigen_mass
#   Muplus_pt/eta, Muminus_pt/eta, Muplus_nvalid, Muminus_nvalid
#   fhit, fms, fioni, fother   -- the per-family MASS-VARIANCE shares, built
#       from `resinfvarv` grouped by `parmtype`, i.e. from the fit's own Q
#       matrix. NOTE these are the well-defined quantities that the CF-exponent
#       second derivative is NOT (the Moliere and Landau families
#       have no finite second moment, so -S''(0) is cut-dependent). `f_ioni`
#       from here is the number `e = f_hit - f_ioni` actually needs.
#
# ORDERING: the alignment is a sequential scan, so the cache must be a
# SUBSEQUENCE of the tree in order. `zpairs_dyv2_full.npz` and
# `jpairs_v2_n600.npz` are built by a sequential pass with no random `--maxn`,
# so they are. A cache built with `--maxn N` and a random draw is NOT and will
# fail the alignment loudly rather than silently mis-join.
#
# *** NEEDS /ceph AND A REAL SLURM ***: run it from an ordinary submit login
# shell.  A sandboxed shell gets permission denied on the productions and has
# no slurm.conf.
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
OM=/work/submit/david_w/ZMass/calibration_studies/resolution/oddmoment
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUT=$FS/runs
NPROC=${NPROC:-32}
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
cd "$OM"
for spec in \
    "dyv2:$CEPH/dymc_8p5M_260906_v2:$OUT/zpairs_dyv2_full.npz" \
    "jpsiv2:$CEPH/jpsimc_20M_260906_v2:$OUT/jpairs_v2_n600.npz" ; do
  tag=${spec%%:*}; rest=${spec#*:}; dir=${rest%%:*}; cache=${rest#*:}
  echo "=== $tag  $(date +%H:%M:%S)  $dir"
  python3 -u aux_gen.py --files "$dir" --cache "$cache" --nproc "$NPROC" \
      --out "$OUT/auxgen_$tag.npz" 2>&1 | tee "$FS/logs/auxgen_$tag.log"
done
echo "=== done $(date +%H:%M:%S)"
echo "-> $OUT/auxgen_dyv2.npz  and  $OUT/auxgen_jpsiv2.npz"

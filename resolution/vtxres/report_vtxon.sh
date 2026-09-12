#!/bin/bash
# The whole ON-vs-OFF comparison, read off the two sets of logs and fits.
# Everything it prints was produced by the pipeline, nothing is recomputed.
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
RON=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/vtxres_on
ROFF=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/vtxres
G=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
cd $HERE
clean() { grep -av "WARNING\|absl\|oneDNN\|cuInit\|findfont" | sed 's/\x1b\[[0-9;]*m//g'; }

echo "############ 1. NLL(0) of every card"
for c in vtx_cf vtx_gauss vtx_gaussq mass_cf mass_gaussq joint_cf joint_gaussq; do
  a=$(grep -ah "NLL(0)" logs_on/card_$c.log 2>/dev/null | tr '\n' ' ')
  b=$(grep -ah "NLL(0)" logs/card_$c.log 2>/dev/null | tr '\n' ' ')
  printf "%-14s ON  %s\n%-14s OFF %s\n" "$c" "$a" "" "$b"
done

echo; echo "############ 2. the fits, certified"
for R in $RON $ROFF; do
  echo "--- $R"
  ./run_tf.sh python3 -u ../hitlik/perhit/certify.py --fits $R/fits \
     --params material_bpix_support6 hitres_pix_y_q1 hitres_pix_x_q1 \
              hitres_str_N3_lo --groups $G 2>&1 | clean
done

echo; echo "############ 3. the sandwich, per channel"
for ch in vtx mass joint; do
  echo "=== $ch  (ON then OFF)"
  grep -a "median sandwich/quoted\|EFFICIENCY sigma\|PRIOR-FREE (standalone)\|sandwich/quoted, PRIOR-FREE" \
    logs_on/eff_$ch.log 2>/dev/null | clean
  echo "  --"
  grep -a "median sandwich/quoted\|EFFICIENCY sigma\|PRIOR-FREE (standalone)\|sandwich/quoted, PRIOR-FREE" \
    logs_on/eff_off_$ch.log 2>/dev/null | clean
done

echo; echo "############ 4. quoted sigma per parameter, cf arm"
for ch in vtx mass joint; do
  echo "=== $ch ON"; sed -n '/MATERIAL GROUPS/,/EFFICIENCY/p;/HIT CLASSES/,/EFFICIENCY/p' logs_on/eff_$ch.log | clean | head -40
  echo "=== $ch OFF"; sed -n '/MATERIAL GROUPS/,/EFFICIENCY/p;/HIT CLASSES/,/EFFICIENCY/p' logs_on/eff_off_$ch.log | clean | head -40
done

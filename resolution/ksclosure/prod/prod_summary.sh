#!/bin/bash
# Aggregate the maker's per-task fit summary over a whole K_S production.
#   ./prod_summary.sh [prod dir]
set -euo pipefail
OUT=${1:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed}
grep -h "fit summary" "$OUT"/logs/slurm_*.out 2>/dev/null | awk '
{
  for (i = 1; i <= NF; i++) {
    split($i, kv, "=")
    if (length(kv) == 2) { gsub(/\(.*/, "", kv[2]); s[kv[1]] += kv[2] }
  }
  n++
}
END {
  printf "tasks with a summary: %d\n", n
  printf "attempted %d  succeeded %d  failed %d (%.3f %%)\n",
         s["attempted"], s["succeeded"], s["failed"],
         100.0 * s["failed"] / (s["attempted"] ? s["attempted"] : 1)
  for (k in s) if (k != "attempted" && k != "succeeded" && k != "failed")
    printf "  %-24s %d\n", k, s[k]
}'
echo "--- wall time per task (s) ---"
grep -h "^>>> .* finished" "$OUT"/logs/slurm_*.out >/dev/null 2>&1 || true
python3 - "$OUT" <<'PY'
import glob, os, sys, datetime, statistics
out = sys.argv[1]
t = []
for fn in glob.glob(os.path.join(out, 'logs', 'slurm_*.out')):
    a = b = None
    for line in open(fn, errors='ignore'):
        if line.startswith('>>> ') and ' host=' in line:
            a = line.split('>>> ')[1].split(' host=')[0].strip()
        elif line.startswith('>>> ') and 'finished' in line:
            b = line.split('>>> ')[1].split(' finished')[0].strip()
    if a and b:
        fmt = '%a %b %d %H:%M:%S %Y'
        try:
            fa = datetime.datetime.strptime(' '.join(a.split()[:4] + a.split()[5:]), fmt)
            fb = datetime.datetime.strptime(' '.join(b.split()[:4] + b.split()[5:]), fmt)
            t.append((fb - fa).total_seconds())
        except ValueError:
            pass
if t:
    print(f'n {len(t)}  median {statistics.median(t):.0f} s  '
          f'mean {statistics.mean(t):.0f} s  max {max(t):.0f} s  '
          f'total {sum(t)/3600:.1f} core-h')
else:
    print('no complete task timings yet')
PY

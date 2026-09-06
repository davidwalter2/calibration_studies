#!/bin/bash
# Turn the run_scan.sh output directories into the throughput/memory table.
#
# The event count comes from the driver's own FwkReport lines rather than the
# requested nEvents, because a chunk can end early; the candidate count comes
# from the maker's fit summary, and it is the candidates -- not the events --
# that the CPU is spent on.
set -uo pipefail
BASE=${1:?usage: summarize.sh <scan outbase> [label]}
LABEL=${2:-$(basename "$BASE")}

printf '%-8s %8s %8s %9s %9s %8s %9s %9s %10s %9s\n' \
  threads events cands wall_s cpu_s cpu/wall ev/s/core cand/s/core peakRSS_GB RSS/core_GB
for d in "$BASE"/t*; do
  [[ -d "$d" ]] || continue
  N=$(basename "$d"); N=${N#t}
  T="$d/time.log"; [[ -f "$T" ]] || continue
  wall=$(grep -o 'Elapsed (wall clock) time.*: .*' "$T" | tail -1 | awk -F': ' '{print $NF}')
  wall_s=$(awk -F: -v w="$wall" 'BEGIN{n=split(w,a,":"); if(n==3) print a[1]*3600+a[2]*60+a[3]; else if(n==2) print a[1]*60+a[2]; else print w}')
  usr=$(grep -o 'User time (seconds): .*' "$T" | tail -1 | awk '{print $NF}')
  sys=$(grep -o 'System time (seconds): .*' "$T" | tail -1 | awk '{print $NF}')
  rss=$(grep -o 'Maximum resident set size (kbytes): .*' "$T" | tail -1 | awk '{print $NF}')
  # FwkReport prints every 100th record, so its last index UNDERCOUNTS by up
  # to 99. The requested nEvents is exact and is echoed by /usr/bin/time -v in
  # the "Command being timed" line; fall back to the report only if absent.
  evmax=$(grep -o 'nEvents=[0-9]*' "$T" | head -1 | grep -o '[0-9]*')
  [[ -n "$evmax" ]] || evmax=$(grep -o 'Begin processing the [0-9]*' "$T" | tail -1 | grep -o '[0-9]*')
  # One fit summary PER STREAM, so the candidates have to be SUMMED, not
  # taken from the last line -- at N threads the last line is 1/N of the job.
  cand=$(grep -ho 'fit summary  attempted=[0-9]*' "$d/cmsrun.log" "$T" 2>/dev/null \
         | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}')
  awk -v n="$N" -v ev="${evmax:-0}" -v c="${cand:-0}" -v w="$wall_s" -v u="$usr" -v s="$sys" -v r="$rss" 'BEGIN{
    cpu = u + s;
    eff = 0; if (w > 0) eff = cpu / w;
    evc = 0; if (w > 0 && n > 0) evc = ev / w / n;
    cac = 0; if (w > 0 && n > 0) cac = c / w / n;
    gb  = r / 1048576.0;
    printf "%-8s %8d %8d %9.1f %9.1f %8.2f %9.3f %11.3f %10.2f %9.2f\n",
      n, ev, c, w, cpu, eff, evc, cac, gb, gb / n;
  }'
done

#!/usr/bin/env python3
"""Fold poor-man's-profiler gdb samples into a leaf/inclusive symbol census."""
import re, sys, collections
frames_per_sample = []
cur = []
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('=== SAMPLE'):
        if cur: frames_per_sample.append(cur)
        cur = []; continue
    m = re.match(r"#(\d+)\s+(?:0x[0-9a-f]+ in )?(.+?) \(", line)
    if m:
        cur.append((int(m.group(1)), m.group(2).strip()))
if cur: frames_per_sample.append(cur)

def norm(s):
    s = re.sub(r"<.*?>", "<>", s)
    return s.split('(')[0].strip()

leaf = collections.Counter(); incl = collections.Counter()
n = 0
for fr in frames_per_sample:
    if not fr: continue
    # only the main thread's stack (frames restart at #0)
    stack, started = [], False
    for lvl, sym in fr:
        if lvl == 0:
            if started: break
            started = True
        stack.append(norm(sym))
    if not stack: continue
    n += 1
    leaf[stack[0]] += 1
    for s in set(stack): incl[s] += 1
print(f"samples={n}")
print("\n-- LEAF (self time) --")
for s, c in leaf.most_common(25): print(f"{100*c/n:6.1f}%  {c:4d}  {s}")
print("\n-- INCLUSIVE --")
for s, c in incl.most_common(45): print(f"{100*c/n:6.1f}%  {c:4d}  {s}")

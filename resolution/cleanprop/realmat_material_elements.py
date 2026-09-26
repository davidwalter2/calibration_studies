"""Elemental mass fractions of a toy geometry's materials, resolved from the
CMSSW DDD XML (all 'mixture by weight'): an export-independent cross-check of
the `materials` table G4ePropagationExport writes, and of how much hydrogen
the rounded single-element representation (round(effZ), round(effA)) drops.

usage: python realmat_material_elements.py <toy tracker.xml>   (-> material_elements.json)
"""
import glob, re, sys, json
B = "/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_15_0_19/src/Geometry"
files = glob.glob(B + "/*/data/**/*.xml", recursive=True)
elem, comp = {}, {}
for f in files:
    s = open(f, errors="ignore").read()
    m = re.search(r'<MaterialSection\s+label="([^"]+)"', s)
    if not m:
        continue
    ns = m.group(1)[:-4] if m.group(1).endswith(".xml") else m.group(1)
    for e in re.finditer(r'<ElementaryMaterial\s+name="([^"]+)"[^>]*?atomicNumber="([\d.]+)"', s):
        elem.setdefault(f"{ns}:{e.group(1)}", float(e.group(2)))
    for c in re.finditer(r'<CompositeMaterial\s+name="([^"]+)"(.*?)</CompositeMaterial>', s, re.S):
        fr = re.findall(r'fraction="([\d.eE+-]+)"\s*>\s*<rMaterial\s+name="([^"]+)"', c.group(2))
        comp.setdefault(f"{ns}:{c.group(1)}", [(float(a), b if ":" in b else f"{ns}:{b}") for a, b in fr])
def resolve(name, w=1.0, out=None):
    out = {} if out is None else out
    if name in elem:
        z = int(round(elem[name])); out[z] = out.get(z, 0.0) + w; return out
    if name not in comp:
        raise KeyError(name)
    tot = sum(f for f, _ in comp[name])
    for f, sub in comp[name]:
        resolve(sub, w * f / tot, out)
    return out
X = sys.argv[1]
s = open(X).read()
mats = sorted(set(re.findall(r'<rMaterial name="([^"]+)"', s)))
res = {}
for m in mats:
    try:
        r = resolve(m)
    except KeyError as e:
        print(f"{m:44} unresolved: {e}"); continue
    res[m] = r
    top = sorted(r.items(), key=lambda t: -t[1])[:4]
    print(f"{m:44} H {100*r.get(1,0):5.2f} %   " + "  ".join(f"Z{z}:{100*w:.1f}%" for z, w in top))
json.dump({k: {str(z): w for z, w in v.items()} for k, v in res.items()},
          open("material_elements.json", "w"), indent=1)

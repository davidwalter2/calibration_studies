"""THE STANDARD TWO-TRACK SELECTION, in one place.

David, 2026-09-12, on the two cuts the gen-truth study recommended
(`vtxres/STATE.md` section 13, `Documents/Resolution/RESOLUTION.md` 2.6):

    "About the minimum number of hits I agree with putting 8 as default.  On
     the vertex residual cutting on 5 sigma also sounds good.  We could keep
     these candidates and only put the cuts downstream, e.g. when running the
     fits or evaluating some other quantity."

So the two cuts live in two DIFFERENT places, deliberately:

* **the minimum on the weaker leg is a MAKER default** (`minLegHits = 8`,
  applied pre-fit, so a thin pair costs no CPU and is never written).  A leg
  below 8 valid hits is background -- 0.885 +- 0.026 of what it removes is a
  duplicate or unmatched pairing by gen truth -- and it is what every
  candidate with a non-finite `Jpsi_sigmamass` has in common.  Applying it in
  the maker cannot bias anything downstream because the candidates it removes
  carry no usable resolution information at all.

* **the cut on the vertex residual is DOWNSTREAM**, here.  `|z_v| < 5` removes
  0.24 % of gen SIGNAL, so it is a genuine acceptance cut on the resolution
  tail, and a term that cuts on a residual must NORMALISE ITS DENSITY OVER THE
  WINDOW IT CUT TO (`make_vtx_card.py --vtx-window`, `rabbit.unbinned`'s
  `norm_window`).  Keeping the candidates in the trees is what makes that
  possible: the tail can still be STUDIED (with `--no-standard-selection`),
  and any term that wants the cut applies it here, once, with the same
  numbers.

WHAT THE STANDARD SELECTION IS

    |z_v| < 5              on `Jpsi_vtxz`, the vertex-constraint pull
    weaker leg >= 8 hits   on `Mu{plus,minus}_nvalid`
    Jpsi_vtxok             the vertex block closed
    finite sigma_m, sigma_v, both > 0

Every cut is SKIPPED, loudly, when its column is absent: the v2 J/psi and DY
productions predate `exportVtxResidual`, so they carry no `Jpsi_vtxz` and the
residual cut is simply unavailable there.  The summary says so rather than
silently applying nothing, and a caller that must have it can test
`summary.applied("|z_v| < 5")`.

USAGE

    import selection
    ...
    selection.add_args(parser)
    ...
    mask, summ = selection.standard(tab, args)
    summ.log(logger.info)           # every caller logs it
    idx = np.flatnonzero(mask)

`tab` is anything with `__getitem__` and `__contains__` over column names: a
dict of arrays, an `uproot` `arrays(library="np")` result, or an `np.load`
NpzFile (its `.files` is honoured as well).
"""

import numpy as np

__all__ = ["MAX_ABS_VTXZ", "MIN_LEG_HITS", "Summary", "standard",
           "add_args", "from_args", "column", "merge"]

#: the standard cut on the vertex-constraint residual (units of sigma_v).
#: `vtxres/STATE.md` 13.5: efficiency 0.9976 +- 0.0005 for a background
#: rejection of 0.724 +- 0.030; `|z_v| < 3` buys 5 % more rejection for 14x
#: the signal loss and is not worth it.
MAX_ABS_VTXZ = 5.0

#: the standard minimum on the WEAKER leg's valid hits.  This is the MAKER
#: default since `cvh-exports-clean-260911`; it is repeated here so that a
#: table read from an OLDER production gets the same selection.
MIN_LEG_HITS = 8

# Column aliases.  The same quantity is called different things by the tree,
# by `cf_inmaker`'s cache and by the aux caches; one map, so a caller never
# has to rename its columns to be selectable.
ALIASES = {
    "vtxz":   ("Jpsi_vtxz", "vtxz", "z_v"),
    "vtxok":  ("Jpsi_vtxok", "vtxok"),
    "vtxsig": ("Jpsi_vtxsig", "vtxsig", "sigma_v"),
    "nvp":    ("Muplus_nvalid", "nvp", "nv_p", "nvalid_plus"),
    "nvm":    ("Muminus_nvalid", "nvm", "nv_m", "nvalid_minus"),
    "sigmam": ("Jpsi_sigmamass", "sigmam", "sigma_m", "sigmamass"),
}


def _keys(tab):
    f = getattr(tab, "files", None)          # np.load NpzFile
    if f is not None:
        return set(f)
    k = getattr(tab, "keys", None)
    if k is not None:
        return set(k())
    return set()


def column(tab, name, keys=None):
    """The array of the logical column `name`, or None if the table has none.

    `name` may be a logical name (a key of `ALIASES`) or a literal column.
    """
    keys = _keys(tab) if keys is None else keys
    for cand in ALIASES.get(name, (name,)):
        if cand in keys:
            return np.asarray(tab[cand])
    return None


class Summary:
    """The per-cut counts of one application of the standard selection.

    Cuts that could not be applied are recorded too (`n` is None), so a log
    tells the difference between "nothing failed it" and "the column was not
    there".
    """

    def __init__(self, n0, enabled=True):
        self.n0 = int(n0)
        self.enabled = bool(enabled)
        self.steps = []          # (label, n_kept or None)
        self.why = {}            # label -> "absent" | "off", for the None ones

    def add(self, label, keep=None, why="absent"):
        self.steps.append((label, None if keep is None else int(keep.sum())))
        if keep is None:
            self.why[label] = why

    @property
    def n(self):
        for _, k in reversed(self.steps):
            if k is not None:
                return k
        return self.n0

    def applied(self, label):
        return any(l == label and k is not None for l, k in self.steps)

    def removed(self, label):
        """How many candidates THIS cut removed (0 if it was not applied)."""
        prev = self.n0
        for l, k in self.steps:
            if k is None:
                continue
            if l == label:
                return prev - k
            prev = k
        return 0

    def lines(self):
        if not self.enabled:
            return [f"standard selection DISABLED (--no-standard-selection); "
                    f"{self.n0} candidates kept"]
        out = [f"standard two-track selection on {self.n0} candidates"]
        prev = self.n0
        for label, k in self.steps:
            if k is None:
                tag = ("switched off" if self.why.get(label) == "off"
                       else "column absent")
                out.append(f"    {label:26s} {'--':>9s}  ({tag}: NOT APPLIED)")
                continue
            frac = 100.0 * k / self.n0 if self.n0 else 0.0
            out.append(f"    {label:26s} {k:9d}  ({frac:6.2f} %, -{prev - k})")
            prev = k
        out.append(f"    {'kept':26s} {self.n:9d}  "
                   f"({100.0 * self.n / self.n0 if self.n0 else 0.0:6.2f} %)")
        return out

    def log(self, emit=print):
        for line in self.lines():
            emit(line)

    def __str__(self):
        return "\n".join(self.lines())


def standard(tab, args=None, *, enabled=None, max_abs_vtxz=None,
             min_leg_hits=None, n=None):
    """Apply the standard two-track selection to a candidate table.

    Returns `(mask, summary)`.  `mask` is a boolean array over the rows of
    `tab`; `summary` is a `Summary` the caller MUST log.

    `args` is an argparse namespace carrying the options `add_args` installs;
    the keyword arguments override it (and are what a non-CLI caller uses).
    """
    cfg = from_args(args)
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if max_abs_vtxz is not None:
        cfg["max_abs_vtxz"] = float(max_abs_vtxz)
    if min_leg_hits is not None:
        cfg["min_leg_hits"] = int(min_leg_hits)

    keys = _keys(tab)
    if n is None:
        for probe in ("vtxz", "sigmam", "nvp"):
            c = column(tab, probe, keys)
            if c is not None:
                n = len(c)
                break
    if n is None:
        raise ValueError("selection.standard: cannot size the table -- pass "
                         "n=, or give it one of "
                         + ", ".join(sum(ALIASES.values(), ())))

    keep = np.ones(int(n), bool)
    summ = Summary(n, cfg["enabled"])
    if not cfg["enabled"]:
        return keep, summ

    # 1. the vertex block closed.  `Jpsi_vtxok` is the fit's own flag.
    ok = column(tab, "vtxok", keys)
    if ok is None:
        summ.add("Jpsi_vtxok")
    else:
        keep &= np.asarray(ok, bool)
        summ.add("Jpsi_vtxok", keep)

    # 2. finite, positive resolutions.  A non-finite sigma_m is the signature
    #    of a leg with one or two hits (STATE.md 13.8); a non-positive sigma_v
    #    is a vertex direction the fit never constrained.
    fin = np.ones(int(n), bool)
    anyfin = False
    for nm in ("sigmam", "vtxsig"):
        c = column(tab, nm, keys)
        if c is None:
            continue
        anyfin = True
        c = np.asarray(c, np.float64)
        fin &= np.isfinite(c) & (c > 0.0)
    if not anyfin:
        summ.add("finite sigma_m, sigma_v")
    else:
        keep &= fin
        summ.add("finite sigma_m, sigma_v", keep)

    # 3. the minimum on the WEAKER leg.  A maker default since
    #    `cvh-exports-clean-260911`; re-applied here so an OLDER production
    #    gets the same sample.
    nvp, nvm = column(tab, "nvp", keys), column(tab, "nvm", keys)
    lab_h = (f"min leg hits >= {cfg['min_leg_hits']}"
             if cfg["min_leg_hits"] > 0 else "min leg hits (off)")
    if cfg["min_leg_hits"] <= 0 or nvp is None or nvm is None:
        summ.add(lab_h, why="off" if cfg["min_leg_hits"] <= 0 else "absent")
    else:
        keep &= (np.asarray(nvp, np.int64) >= cfg["min_leg_hits"]) & \
                (np.asarray(nvm, np.int64) >= cfg["min_leg_hits"])
        summ.add(lab_h, keep)

    # 4. the vertex residual.  THE cut that truncates a density: any term
    #    that consumes the survivors must normalise over this window.
    z = column(tab, "vtxz", keys)
    lab_z = (f"|z_v| < {cfg['max_abs_vtxz']:g}"
             if cfg["max_abs_vtxz"] > 0 else "|z_v| cut (off)")
    if cfg["max_abs_vtxz"] <= 0 or z is None:
        summ.add(lab_z, why="off" if cfg["max_abs_vtxz"] <= 0 else "absent")
    else:
        keep &= np.abs(np.asarray(z, np.float64)) < cfg["max_abs_vtxz"]
        summ.add(lab_z, keep)

    return keep, summ


def merge(summaries):
    """Sum a list of per-file `Summary` objects into one.

    A cut that was unavailable in ANY file is reported unavailable overall --
    a half-applied cut is not a selection.
    """
    tot = None
    for s in summaries:
        if s is None:
            continue
        if tot is None:
            tot = Summary(s.n0, s.enabled)
            tot.steps = [[l, k] for l, k in s.steps]
            tot.why = dict(s.why)
            continue
        tot.n0 += s.n0
        for i, (l, k) in enumerate(s.steps):
            if k is None or tot.steps[i][1] is None:
                tot.steps[i][1] = None
            else:
                tot.steps[i][1] += k
    if tot is not None:
        tot.steps = [tuple(x) for x in tot.steps]
    return tot


def add_args(p, prefix=""):
    """Install the standard-selection options on an argparse parser."""
    d = "--" + prefix
    g = p.add_argument_group("standard two-track selection")
    g.add_argument(f"{d}no-standard-selection", action="store_true",
                   help="do NOT apply the standard two-track selection "
                        "(|z_v| < 5, weaker leg >= 8 hits, Jpsi_vtxok, finite "
                        "sigma). The escape hatch for studies OF the tail "
                        "itself -- a term fitted on the survivors without the "
                        "matching truncated normalisation is biased "
                        "(resolution/selection.py)")
    g.add_argument(f"{d}max-abs-vtxz", type=float, default=MAX_ABS_VTXZ,
                   help="the standard cut on the vertex-constraint residual, "
                        "in units of its own sigma. 0 disables it. THE TERM "
                        "THAT CONSUMES THE SURVIVORS MUST NORMALISE OVER THIS "
                        "WINDOW (default: %(default)g)")
    g.add_argument(f"{d}min-leg-hits", type=int, default=MIN_LEG_HITS,
                   help="minimum valid hits on the WEAKER leg; the maker "
                        "default since cvh-exports-clean-260911, repeated "
                        "here for older productions. 0 disables "
                        "(default: %(default)d)")
    return p


def from_args(args, prefix=""):
    """The config dict `standard` uses, read off an argparse namespace."""
    def get(name, default):
        return getattr(args, prefix + name, default) if args is not None \
            else default
    return dict(
        enabled=not get("no_standard_selection", False),
        max_abs_vtxz=float(get("max_abs_vtxz", MAX_ABS_VTXZ)),
        min_leg_hits=int(get("min_leg_hits", MIN_LEG_HITS)),
    )

#!/usr/bin/env python3
"""Per-material-group decomposition of the CVH resolution-CF exponents.

WHY
---
The unbinned mass likelihood (``cf_masslik_fit.py``, ``rabbit.unbinned``)
currently scales its resolution by four ad-hoc per-FAMILY knobs
``k_hit, k_ms, k_ioni, k_rad``.  Those are not physical parameters: the fit has
no reason to believe that all of the tracker's multiple scattering is wrong by
one common factor while all of its ionization straggling is wrong by another.

The physical parameters already exist.  The CVH fit carries 42 **material
groups** (parmtype 15, ``Analysis/HitAnalyzer/data/materialGroups50.txt``) with
one dimensionless log-amount parameter ``k_g`` each, and the propagator applies
that parameter to the *amount of material of the step*:

* mean loss   (``G4ErrorEnergyLossForCVH::AlongStepDoIt``)::

      xifact = exp(dxieff),  dxieff = <leg offset> + k_{g(step)}
      dE     = xifact * (dE/dx)_0 * ds

* step MS covariance and step ionization variance
  (``Geant4ePropagator::PropagateError``, the "M1" block)::

      matStepFact  = exp(k_{g(step)})
      errMS(step) *= exp(dms)   * matStepFact
      errI (0,0)  *= exp(dioni) * matStepFact

so ``k_g`` IS the log of the group's material amount, and it is the same
parameter that the global hit-chi2 (quadratic) term already floats.

THE RELATION IMPLEMENTED HERE
-----------------------------
Every step-level log-CF exponent of the CVH resolution model is **linear in the
amount of material of that step at fixed composition**:

======================  ==========================================================
Moliere multiple scat.  chi_c^2 ∝ x and Omega_0 = chi_c^2/chi_a^2 ∝ x at fixed
                        chi_a (chi_a depends on Z and p only), and
                        ``ms_step_exponent`` returns sum_steps (chi_c^2/chi_a^2)
                        g(...)  -- linear in x.
Urban ionization        the mean excitation/ionization numbers a_1, a_2, a_3 all
                        carry the step length, and ``ioni_step_exponent``
                        returns sum_steps a_j (e^{i th} - 1 - i th) -- linear.
Radiative (brems+pair)  the mean number of emissions is ∝ x; ``rad_exponent``
                        is a sum over steps with per-step weights -- linear.
delta-ray recoil        xi ∝ x, ``delta_step_exponent`` is a sum over steps.
======================  ==========================================================

Therefore, with the fit's influence weights **held fixed** (the same two-step
convention the in-fit CGF block uses -- "the fit never differentiates through
the weights", ``Geant4ePropagator.cc`` M1 comment), the decomposition

    S_f(tau)        = sum_g S_{f,g}(tau)                       [EXACT, by step]
    S_f(tau; k)     = sum_g exp(k_g) * S_{f,g}(tau)            ["exp"    mode]
                    = sum_g (1 + k_g) * S_{f,g}(tau)           ["linear" mode]

is exact for an amount change at fixed composition, and ``k = 0`` reproduces the
production's own exponents bit-for-bit.  ``exp`` is the mode that agrees with
the C++ (``matStepFact = exp(k_g)``); ``linear`` is its first-order form.

The Gaussian hit share is treated the same way but per HIT CLASS,

    v_hit(eps) = v_other + sum_c (1 + eps_c) v_c ,   phi_hit = exp(-t^2 v/2)

with ``v_c`` the summed exported influence variance ``resinfvarv`` of the
parmtype-8/9 blocks of class c, and ``v_other`` the non-hit Gaussian remainder
(beamspot / vertex constraint), which no parameter scales.

STEP -> GROUP RESOLUTION
------------------------
``msmoliv`` carries the group id in column 9 (``MoliereMsStep::stepGroup``), so
the MS and delta channels split exactly.  ``radstepv`` has no group column but
its log is pushed under the *same* ``if (ioniStepLogging_ && thisPathLength >
0.)`` guard as the Moliere log, so the two are 1:1 within a block (verified:
identical row counts in every block of every file inspected) and the radiative
channel inherits the group by row position -- also exact.

``ioniurbanv`` has neither a group column nor a 1:1 correspondence: the Urban
record is skipped when ``meanLoss = length*dedx < minLoss``
(``G4UniversalFluctuationForExtrapolator::SampleFluctuations``), so the Urban
rows are an ORDER-PRESERVING SUBSEQUENCE of the Moliere rows.  ``pair_ioni_rows``
recovers it by taking the ``n_ioni`` Moliere rows of largest areal density
``xg``, in their original order.  Measured on the 260905d J/psi gun (5057
blocks): 51.5 % of blocks are already 1:1; in the rest the *dropped* rows carry
a median 2.4e-6 and at most 3.7e-5 of the block's total ``xg``, and the ratio
``xg[n_ioni-1]/xg[n_ioni]`` never falls below 1.5.  So the residual assignment
ambiguity is bounded by ~4e-5 of the block's material.
"""

import numpy as np

# msmoliv column layout (stride 10)
MS_EFFZ, MS_EFFA, MS_XG, MS_P, MS_BETA, MS_THP2, MS_DOX0, MS_ZZP1, MS_LNSW, MS_GROUP = range(10)

# GeomDetEnumerators::SubDetector, which is what the runtree's `subdet`
# branch carries (`subdet = det->subDetector()`, Base.cc:1092).  Verified
# against the module counts of the UL16 (phase-0) tracker in the runtree:
# 768 / 672 / 2724 / 5208 / 816 / 6400.
SUBDET_NAMES = {0: "bpix", 1: "fpix", 2: "tib", 3: "tob", 4: "tid", 5: "tec"}


def read_groups(fname):
    """(names, prior_sigmas) keyed by group id, from a materialGroups tier file.

    Identical to ``globalfit.make_global_term.read_groups`` -- duplicated here
    only so this module has no import-time dependency on that one.
    """
    names, priors = {0: "other"}, {0: 0.2}
    if not fname:
        return names, priors
    for line in open(fname):
        if line.startswith("RULE"):
            f = line.split("\t")
            names[int(f[1])] = f[2]
            priors[int(f[1])] = float(f[10])
    return names, priors


def group_param_names(ngroups, groups_file):
    """rabbit parameter names for the parmtype-15 groups, IDENTICAL to
    ``make_global_term.name_params`` (including the repeated-name
    disambiguation), so a card can float one set of parameters across the
    quadratic and the mass term."""
    gnames, gpriors = read_groups(groups_file)
    names = [f"material_{gnames.get(g, f'group{g}')}" for g in range(ngroups)]
    sigmas = np.array([gpriors.get(g, 0.2) for g in range(ngroups)], dtype=np.float64)
    seen = {}
    for i, nm in enumerate(names):
        seen.setdefault(nm, []).append(i)
    for nm, idxs in seen.items():
        if len(idxs) > 1:
            for i in idxs:
                names[i] = f"{nm}{i}"
    if len(set(names)) != len(names):
        raise ValueError("duplicate material parameter names")
    return names, sigmas


# ---------------------------------------------------------------------------
# UNITS
#
# A card may float a RESCALED material parameter, because the minimiser is
# better conditioned on it.  The physical quantity is always the log material
# amount ``k_g`` of the group (and, for a hit class, the linear variance scale
# ``eps_c``), and exactly one factor relates the two,
#
#     physical = value * units
#
# which is the convention ``rabbit.unbinned.MaterialCFTerm`` itself applies
# (``k = values[group_params] * group_units``).  The factor is READ from the
# object that carries it -- a built term, or the ``global_index_map`` a card
# writes -- never assumed by the consumer.  Everything that LEAVES a term
# (Fisher/Hessian/score matrices, fitted values and errors, priors, injected
# amounts, tables) is reported in physical units, so no tool needs to be told
# which convention it is looking at.


def card_group_units(ngroups, groups_file, whiten=True):
    """``units`` of the parmtype-15 material parameters of a card.

    A whitened card floats ``value = k_g * gprior_g`` (the whitening
    ``make_global_term.param_scales`` applies to the quadratic term's gradient
    and Hessian, ``theta_card = theta_raw * s``), so ``k_g = value / gprior_g``
    and ``units = 1 / gprior_g``.  Without whitening the parameter IS ``k_g``
    and ``units = 1``.

    This is the ONE definition of the card unit; every card builder and every
    Fisher driver takes it from here, so a term built by hand and a term built
    inside a card cannot drift apart.
    """
    _, gpriors = group_param_names(ngroups, groups_file)
    if not whiten:
        return np.ones(ngroups)
    return 1.0 / np.maximum(gpriors, 1e-300)


def units_for(param_names, group_params=(), group_units=None,
              hit_params=(), hit_units=None):
    """``units`` per parameter, in ``param_names`` order (1 where unknown)."""
    u = {}
    if group_units is not None:
        u.update(zip(group_params, np.asarray(group_units, dtype=np.float64)))
    if hit_units is not None:
        u.update(zip(hit_params, np.asarray(hit_units, dtype=np.float64)))
    return np.array([u.get(p, 1.0) for p in param_names], dtype=np.float64)


def term_units(term):
    """``units`` of a ``MaterialCFTerm``, read off the term itself."""
    return units_for(term.param_names,
                     getattr(term, "group_params", ()),
                     getattr(term, "group_units", None),
                     getattr(term, "hit_params", ()),
                     getattr(term, "hit_units", None))


def card_units(card, param_names):
    """``units`` of the parameters of a card, from its ``global_index_map``.

    ``card`` is the decoded auxiliary dict (``group_params``/``group_units``,
    or the flat ``params``/``units`` pair).  Missing entries are 1.
    """
    if "params" in card and "units" in card:
        return units_for(param_names, list(card["params"]), card["units"])
    return units_for(param_names, list(card.get("group_params", [])),
                     card.get("group_units"))


def to_physical(values, units):
    """A value vector (or its errors) in physical units."""
    return np.asarray(values, dtype=np.float64) * np.asarray(units)


def matrix_to_physical(M, units):
    """A Hessian / score covariance in physical units.

    ``value = physical / u`` so ``d/dvalue = u d/dphysical`` and a second
    derivative picks up ``u_i u_j``; dividing it out is the inverse.
    """
    u = np.asarray(units, dtype=np.float64)
    return np.asarray(M, dtype=np.float64) / np.outer(u, u)


def gradient_to_physical(g, units):
    """A score / gradient vector in physical units."""
    return np.asarray(g, dtype=np.float64) / np.asarray(units)


def pair_ioni_rows(xg, n_ioni):
    """Indices (ascending) of the Moliere rows that produced an Urban record.

    The Urban record is written iff ``meanLoss = length * dedx >= minLoss``;
    ``meanLoss`` is monotone in the step's areal density at fixed material and
    the excluded rows are vacuum-like (xg ~ 1e-16), so the ``n_ioni`` rows of
    largest ``xg`` are the ones with records.  Any subset of an ordered list,
    read in order, is an order-preserving subsequence, so no further sorting is
    needed.

    Returns ``None`` when the request is impossible (n_ioni > len(xg)).
    """
    n_ms = len(xg)
    if n_ioni > n_ms:
        return None
    if n_ioni == n_ms:
        return np.arange(n_ms)
    # argpartition is O(n); take the n_ioni largest then restore file order
    sel = np.argpartition(xg, n_ms - n_ioni)[n_ms - n_ioni:]
    sel.sort()
    return sel


def ioni_pair_quality(xg, n_ioni):
    """(gap, lost_frac) diagnostics of ``pair_ioni_rows`` for one block.

    gap        = xg of the smallest KEPT row / xg of the largest DROPPED row
                 (inf when nothing is dropped)
    lost_frac  = sum of xg over the dropped rows / total xg
    """
    n_ms = len(xg)
    if n_ioni >= n_ms:
        return np.inf, 0.0
    srt = np.sort(xg)[::-1]
    tot = max(float(srt.sum()), 1e-300)
    return (float(srt[n_ioni - 1]) / max(float(srt[n_ioni]), 1e-300),
            float(srt[n_ioni:].sum()) / tot)


# ---------------------------------------------------------------------------
# sparse per-group container
# ---------------------------------------------------------------------------
class GroupStore:
    """Per-candidate ragged store  (candidate -> [(group, per-family arrays)]).

    Layout on disk (npz), for ``n`` candidates, ``nnz`` (candidate, group)
    pairs and ``nt`` grid points:

        grp_ptr   int64 (n+1,)      CSR row pointer into the nnz axis
        grp_id    int16 (nnz,)      material group index of each row
        S<fam>    float32 (nnz,nt)  that group's contribution to family <fam>

    ``sum over the rows of candidate i`` reproduces the flat per-family
    exponent of ``cf_inmaker`` / ``globalfit.extract`` exactly.
    """

    def __init__(self, families, nt):
        self.families = list(families)
        self.nt = int(nt)
        self.ptr = [0]
        self.gid = []
        self.buf = {f: [] for f in self.families}

    def add_candidate(self, per_group):
        """per_group: dict {group_id: {family: (nt,) array}}"""
        for g in sorted(per_group):
            self.gid.append(g)
            row = per_group[g]
            for f in self.families:
                a = row.get(f)
                self.buf[f].append(
                    np.zeros(self.nt, np.float32) if a is None
                    else np.asarray(a, dtype=np.float32))
        self.ptr.append(len(self.gid))

    def finish(self):
        out = {
            "grp_ptr": np.asarray(self.ptr, dtype=np.int64),
            "grp_id": np.asarray(self.gid, dtype=np.int16),
        }
        for f in self.families:
            out["S" + f] = (np.stack(self.buf[f]) if self.buf[f]
                            else np.zeros((0, self.nt), np.float32))
        return out

    @property
    def nnz(self):
        return len(self.gid)


def concat_group_stores(parts, families):
    """Concatenate the finish() dicts of several shards, fixing the pointers."""
    ptrs, gids = [], []
    bufs = {f: [] for f in families}
    off = 0
    for p in parts:
        ptr = p["grp_ptr"]
        ptrs.append((ptr[1:] if len(ptrs) else ptr) + (off if len(ptrs) else 0))
        off += int(ptr[-1])
        gids.append(p["grp_id"])
        for f in families:
            bufs[f].append(p["S" + f])
    out = {"grp_ptr": np.concatenate(ptrs) if ptrs else np.zeros(1, np.int64),
           "grp_id": np.concatenate(gids) if gids else np.zeros(0, np.int16)}
    for f in families:
        out["S" + f] = np.concatenate(bufs[f]) if bufs[f] else np.zeros((0, 1), np.float32)
    return out


def expand(store, family, ngroups=None):
    """Dense ``(n, ngroups, nt)`` view of one family -- diagnostics only."""
    ptr, gid = store["grp_ptr"], store["grp_id"]
    S = store["S" + family]
    n = len(ptr) - 1
    ng = int(gid.max()) + 1 if ngroups is None else ngroups
    out = np.zeros((n, ng, S.shape[1]), np.float64)
    for i in range(n):
        for j in range(ptr[i], ptr[i + 1]):
            out[i, gid[j]] += S[j]
    return out


def contract(store, family, w):
    """``sum_g w[g] S_{i,g}`` -> ``(n, nt)``; ``w`` is a (ngroups,) vector."""
    ptr, gid = store["grp_ptr"], store["grp_id"]
    S = store["S" + family].astype(np.float64)
    n = len(ptr) - 1
    seg = np.repeat(np.arange(n), np.diff(ptr))
    out = np.zeros((n, S.shape[1]))
    np.add.at(out, seg, w[gid][:, None] * S)
    return out


def amount_weights(k, mode="exp"):
    """The per-group multiplier of the exponent, from the parmtype-15 k_g."""
    k = np.asarray(k, dtype=np.float64)
    if mode == "exp":
        return np.exp(k)
    if mode == "linear":
        return 1.0 + k
    raise ValueError(f"unknown amount mode {mode!r}")

"""The process-noise channels of the CF model as ROW functions.

ONE implementation of every channel, shared by
  * the clean-propagation model (`cf_propagation_test.model_phi`,
    `cgf_channels.block_cf_exponent`): rows weighted by the exact transport of
    each step to the plane, expanded into sub-steps (`transport_entries`);
  * the offline fit-level CF (`cf_track_resolution.extract`,
    `cf_mass_likelihood`): rows weighted by the fit's block influence weights,
    one entry per record;
  * the in-maker C++ (`cvhcf`, TrackPropagation/Geant4e), which implements the
    same functions and is validated against them.

A ROW is one exported Geant4 step record: `ioniurbanv` (ionisation, knock-on),
`msmoliv` (scattering, nuclear elastic), `radv`/`radstepv` + spectrum
(radiation).  The channel functions take the records and a list of ENTRIES,
each (record index, q/p weight wq, angular weight wb, fraction):

  wq  z per unit q/p, the track charge included (the records' q/p maps are
      for a positive charge);
  wb  z per radian: the length of the projected angular weight
      (w_lambda, w_phi / cos lambda) -- the kicks are isotropic in 2D, so a
      channel sees only that length;
  fraction  of the record's material / rate the entry carries (1/NSUB for a
      sub-step, 1 for a whole record).

The caller owns the weights and the sub-step rule; the physics of a record is
evaluated once however many entries share it.
"""

import numpy as np

import cf_brems_exact
import cf_knockon
import cf_track_resolution as ctr


# ---------------------------------------------------------------- entries
def transport_entries(leg, A_end, A_start, avec, sigma, nsub, interp):
    """(rid, wq, wb, frac) of a clean-propagation leg's rows at one plane.

    A_end / A_start: (nrow, 5, 5) transports of each row's noise to the plane
    from the row's end / start; `avec` the plane's functional; `nsub` equal
    sub-steps at the centres of their parts, the transport interpolated
    linearly in between.  `interp`:
      "vector"  interpolate the weight VECTOR, then take q/p and the length
                (what a kick at that point sees exactly);
      "length"  interpolate the two lengths (the scattering channel's
                historical rule; identical when the start and end weight
                vectors are parallel).
    """
    q = np.sign(leg["refqop"]) or 1.0
    coslam = max(leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0, 1e-3)
    v1 = np.einsum("i,sij->sj", avec, A_end)
    v0 = np.einsum("i,sij->sj", avec, A_start)
    ns = max(int(nsub), 1)
    n = len(v1)
    rid = np.repeat(np.arange(n), ns)
    f = np.tile((np.arange(ns) + 0.5) / ns, n)
    if interp == "vector":
        v = v0[rid] + f[:, None] * (v1[rid] - v0[rid])
        wq = q * v[:, 0] / sigma
        wb = np.hypot(v[:, 1], v[:, 2] / coslam) / sigma
    elif interp == "length":
        b1 = np.sqrt(v1[:, 1] ** 2 + (v1[:, 2] / coslam) ** 2) / sigma
        b0 = np.sqrt(v0[:, 1] ** 2 + (v0[:, 2] / coslam) ** 2) / sigma
        wb = b0[rid] + f * (b1[rid] - b0[rid])
        w1q, w0q = q * v1[:, 0] / sigma, q * v0[:, 0] / sigma
        wq = w0q[rid] + f * (w1q[rid] - w0q[rid])
    else:
        raise ValueError(interp)
    return rid, wq, wb, np.full(len(rid), 1.0 / ns)


def end_weights(leg, A_end, avec, sigma):
    """(wq, wb) of each row at its END -- no sub-steps (the ionisation rows)."""
    q = np.sign(leg["refqop"]) or 1.0
    coslam = max(leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0, 1e-3)
    v = np.einsum("i,sij->sj", avec, A_end)
    return q * v[:, 0] / sigma, np.hypot(v[:, 1], v[:, 2] / coslam) / sigma


# ---------------------------------------------------------------- channels
def ioni_rows(tau, rows, wq):
    """The ionisation channel (Urban excitations + the record's knock-on law,
    linear map): each record at its own q/p weight, folded into the record's
    q/p-per-MeV column so the vectorised exponent runs once."""
    if not len(rows):
        return np.zeros(len(tau), dtype=np.complex128)
    steps = np.array(rows, dtype=np.float64, copy=True)
    steps[:, 10] *= wq
    return ctr.ioni_step_exponent(steps, 1.0, tau)


def ms_rows(tau, rows, rid, wb, frac, scale=1.0):
    """Multiple scattering: one isotropic 2D Moliere kick per entry, the
    entry's share `frac` of the record's material, at angular weight wb.
    Entries with wb <= 0 carry nothing."""
    S = np.zeros(len(tau), dtype=np.complex128)
    if not len(rid):
        return S
    rows = np.asarray(rows)
    if np.all(frac == 1.0) and np.all(wb == wb[0]) and len(rid) == len(rows) \
            and np.array_equal(rid, np.arange(len(rows))):
        # one weight for the whole record set (a fit block): vectorised
        return scale * ctr.ms_step_exponent(rows, float(wb[0]), tau) \
            if wb[0] > 0.0 else S
    for r, w, f in zip(rid, wb, frac):
        if w <= 0.0:
            continue
        rec = rows[r:r + 1].copy()
        if f != 1.0:
            rec[:, 2] *= f            # xg -> chi_c^2 scales with material
        S += scale * ctr.ms_step_exponent(rec, w, tau)
    return S


def rad_rows(tau, recs, spec, vg, rid, wq, wb, frac, exact_qop=None):
    """Radiation: every emission the joint event of its q/p change (exact 1/p
    map under `exact_qop`) and the primary's recoil against the photon at
    angular weight wb (`cf_brems_exact.rad_exponent`).  The entry's share of
    the record scales the step length (the spectrum's normalisation)."""
    if exact_qop is None:
        exact_qop = bool(cf_knockon.QOP_EXACT)
    if not len(rid):
        return np.zeros(len(tau), dtype=np.complex128)
    r = np.array(np.asarray(recs)[rid], dtype=np.float64, copy=True)
    r[:, cf_brems_exact.R_STEPCM] *= frac
    return cf_brems_exact.rad_exponent(tau, r, np.asarray(spec)[rid], vg,
                                       weights=wq, exact_qop=exact_qop,
                                       bweights=wb)


def knockon_rows(tau, rows, rid, wq, wb, frac, part="all"):
    """The hard knock-on collision exactly (`cf_knockon.step_correction`): the
    joint law of the energy loss and the deflection and the exact 1/p map, as
    the correction to the ionisation channel's linear map and the scattering
    channel's independent electron term, per record's law (regimes 2-5).
    part "map" / "joint" split it into the exact-map and the joint piece."""
    return cf_knockon.knockon_rows(tau, rows, rid, wq, wb, frac, part)


# ---------------------------------------------------------------- the fit
def fit_pairing(uim, ridx, uvm, wms):
    """The fit's block weights on the rows that need both a q/p and an angular
    weight.  The radiative rows are one per Geant4 step, PARALLEL to the MS
    rows (`msmoliv`), and carry their leg's ionisation block index, so

      wb_rad[s]  = wms[uim[s]]  -- the angular weight of radiative row s;
      beta[g]    = the angular weight of ionisation block g: the xg-weighted
                   rms of wms over the MS rows paired with g (the knock-on
                   rate goes with the material).

    `wms` {MS global index: angular weight}."""
    wb = np.array([wms.get(int(g), 0.0) for g in uim], dtype=np.float64)
    xg = np.asarray(uvm, dtype=np.float64)[:, 2] if len(uvm) else np.zeros(0)
    beta = {}
    for g in np.unique(ridx):
        m = ridx == g
        den = xg[m].sum()
        beta[int(g)] = float(np.sqrt((xg[m] * wb[m] ** 2).sum() / den)) if den > 0 else 0.0
    return wb, beta


def check_parallel(rrec, uvm, uim, ridx):
    """The pairing joins the radiative and MS rows BY POSITION: refuse rows
    that are not one per Geant4 step on both sides (same count, same
    momentum)."""
    rrec, uvm = np.asarray(rrec), np.asarray(uvm)
    if not (len(rrec) == len(uvm) == len(uim) == len(ridx)) or (
            len(uvm) and not np.array_equal(rrec[:, cf_brems_exact.R_P], uvm[:, 3])):
        raise ValueError("radiative rows are not parallel to the MS rows")


def fit_families(tau, sig, ms_blocks, io_blocks, rad=None):
    """The fit-level CF families of one track or leg from the ROW functions.

    ms_blocks  [(g, msmoliv rows, |w|)]      angular block weights
    io_blocks  [(g, ioniurbanv rows, w)]     signed q/p block weights
    rad        None, or dict(uim, ridx, uvm, rrec, rspc, rvg, wrad) with
               `wrad` {ionisation index: signed radiative q/p weight} and
               rrec / rspc the radiative rows parallel to uvm.
    Weights are in the functional's units; `sig` standardises them.

    Returns {Sms, Sio, Srad, Skx, Skj, Sdel}: scattering, ionisation (linear
    map), radiation (exact 1/p map and recoil angle), the knock-on exact-map
    and joint pieces on the ionisation rows (`cf_knockon`; zero with their
    switch off), and Sdel = 0 -- the scattering channel's electron term and
    the knock-on joint piece carry the delta-ray recoil."""
    nt = len(tau)
    out = dict(Sms=np.zeros(nt, dtype=np.complex128),
               Sio=np.zeros(nt, dtype=np.complex128),
               Srad=np.zeros(nt, dtype=np.complex128),
               Skx=np.zeros(nt, dtype=np.complex128),
               Skj=np.zeros(nt, dtype=np.complex128),
               Sdel=np.zeros(nt, dtype=np.complex128))
    wms = {}
    for g, rows, w in ms_blocks:
        wms[int(g)] = abs(w)
        n = len(rows)
        out["Sms"] += ms_rows(tau, rows, np.arange(n), np.full(n, abs(w) / sig),
                              np.ones(n))
    beta, wb_rad = {}, None
    if rad is not None and len(rad["ridx"]):
        check_parallel(rad["rrec"], rad["uvm"], rad["uim"], rad["ridx"])
        wb_rad, beta = fit_pairing(rad["uim"], rad["ridx"], rad["uvm"], wms)
    for g, rows, w in io_blocks:
        n = len(rows)
        wq = np.full(n, w / sig)
        out["Sio"] += ioni_rows(tau, rows, wq)
        if cf_knockon.active():
            wbk = np.full(n, beta.get(int(g), 0.0) / sig)
            ones = np.ones(n)
            if cf_knockon.QOP_EXACT:
                out["Skx"] += knockon_rows(tau, rows, np.arange(n), wq, wbk, ones,
                                           part="map")
            if cf_knockon.KNOCKON_JOINT:
                out["Skj"] += knockon_rows(tau, rows, np.arange(n), wq, wbk, ones,
                                           part="joint")
    if rad is not None and len(rad["ridx"]):
        vf, spf = cf_brems_exact.refine_spectra(rad["rrec"], rad["rspc"],
                                                rad["rvg"], cf_brems_exact.RAD_NSUB)
        for g, wr in rad["wrad"].items():
            m = np.flatnonzero(rad["ridx"] == g)
            if not len(m):
                continue
            out["Srad"] += rad_rows(tau, rad["rrec"][m], spf[m], vf, np.arange(len(m)),
                                    np.full(len(m), wr / sig), wb_rad[m] / sig,
                                    np.ones(len(m)))
    return out

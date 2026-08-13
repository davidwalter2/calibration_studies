"""Geant4 truth labelling for the CVH kink-finder validation.

Consumes the trees written by runCvhSingleTrackJpsiX.py with
doKinkFinder=True doSimDecayTruth=True on the v3 inclusive B->J/psi+X MC
(the first campaign that keeps SimTracks + SimVertices).

Truth model
-----------
The gen-matched track is linked to its Geant4 SimTrack, and every SimVertex
that SimTrack produced is stored with its G4 process subtype.  A vertex only
distorts the fit if it happens *between the first and the last measurement*,
so the label is applied on the path-ordered coordinate s = sqrt(R^2 + Z^2),
which increases monotonically along an outgoing track: a vertex counts when
s_firststep < s_vtx < s_laststep.

  decay    processType 201            -- K/pi -> mu nu, K -> pi pi0, ...
  nuclear  processType 111/121/131/   -- hadron elastic / inelastic / capture /
                       151/161           at-rest / charge exchange
  em       everything else            -- delta rays, brems: the parent
                                         survives, so these are NOT signal

A track with no in-span decay or nuclear vertex is "clean" -- the ROC
background.
"""

import awkward as ak
import numpy as np
import uproot

PROC_DECAY = 201
PROC_NUCLEAR = (111, 121, 131, 141, 151, 161)

# tracker acceptance used for the in-tracker bookkeeping printouts
TRACKER_R = 120.0
TRACKER_Z = 300.0

BRANCHES = [
    "kinkMax", "kinkMaxLayer", "kinkDchisq", "kinkDchisqAngle", "kinkDchisqQop",
    "kinkDqop", "kinkDxdz", "kinkDydz", "kinkGlobalR", "kinkGlobalZ",
    "muonPt", "muonLoose", "muonMedium", "muonTight",
    "muonIsGlobal", "muonIsTracker", "muonIsPF",
    "trackPt", "trackEta", "trackPhi", "trackCharge", "normalizedChi2",
    "nHits", "nValidHits", "nValidPixelHits",
    "run", "lumi", "event",
    "genPt", "genEta", "genPhi", "genCharge", "genPdgId", "genDR",
    "genParms", "refParms", "refCov",
    "simTrkFound", "simTrkPdgId", "simTrkP", "simTrkPt", "simTrkEta", "simTrkPhi",
    "simTrkVtxX", "simTrkVtxY", "simTrkVtxZ",
    "simVtxX", "simVtxY", "simVtxZ", "simVtxProcType", "simVtxNDau",
    "simDauVtx", "simDauPdgId", "simDauPx", "simDauPy", "simDauPz",
]


def load(files, branches=None, treename="tree"):
    """Concatenate the per-chunk trees.

    Requested branches that the files do not have are dropped, so trees
    written before genPdgId/genDR existed still load.
    """
    branches = branches or BRANCHES
    have = set(uproot.open(f"{files[0]}:{treename}").keys())
    missing = [b for b in branches if b not in have]
    if missing:
        print(f"[kink_truth] branches not in input, skipping: {missing}")
    return uproot.concatenate(
        [f"{f}:{treename}" for f in files],
        expressions=[b for b in branches if b in have], library="ak")


def _max_or(arr, default):
    """Row-wise max of a jagged array, `default` for empty rows."""
    return ak.to_numpy(ak.fill_none(ak.max(arr, axis=1), default))


def label(arr):
    """Return a dict of flat per-track numpy arrays: truth labels,
    discriminants and the kinematics used for binning."""
    out = {}
    n = len(arr)

    # ---- path coordinate of the material steps and of the sim vertices ----
    s_step = np.sqrt(arr.kinkGlobalR**2 + arr.kinkGlobalZ**2)
    s_first = ak.to_numpy(ak.fill_none(ak.min(s_step, axis=1), np.inf))
    s_last = ak.to_numpy(ak.fill_none(ak.max(s_step, axis=1), -np.inf))

    r_vtx = np.sqrt(arr.simVtxX**2 + arr.simVtxY**2)
    s_vtx = np.sqrt(r_vtx**2 + arr.simVtxZ**2)
    inspan = (s_vtx > s_first) & (s_vtx < s_last)

    proc = arr.simVtxProcType
    is_decay = proc == PROC_DECAY
    is_nuclear = ak.zeros_like(proc, dtype=bool)
    for p in PROC_NUCLEAR:
        is_nuclear = is_nuclear | (proc == p)

    has_decay = ak.to_numpy(ak.any(is_decay & inspan, axis=1))
    has_nuclear = ak.to_numpy(ak.any(is_nuclear & inspan, axis=1))

    out["has_decay"] = has_decay
    # a track with both is counted as a decay (the decay comes first in the
    # overwhelming majority of cases and is the process under study)
    out["has_nuclear"] = has_nuclear & ~has_decay
    out["clean"] = ~has_decay & ~has_nuclear
    out["sim_found"] = ak.to_numpy(arr.simTrkFound) == 1

    # ---- properties of the (first) signal vertex --------------------------
    # "first" = innermost along the track. Non-signal vertices are pushed to
    # s = +inf so argmin selects the innermost signal vertex; rows with no
    # vertex at all are padded so the index stays in range (they are masked
    # to NaN afterwards anyway).
    sig = (is_decay | is_nuclear) & inspan
    s_masked = ak.fill_none(ak.pad_none(ak.where(sig, s_vtx, np.inf), 1), np.inf)
    isel = ak.argmin(s_masked, axis=1, keepdims=True)

    def first_of(jagged, default=np.nan):
        v = ak.to_numpy(ak.fill_none(ak.firsts(ak.pad_none(jagged, 1)[isel]), default))
        return np.where(out["clean"], default, v)

    out["vtx_s"] = first_of(s_vtx)
    out["vtx_r"] = first_of(r_vtx)
    out["vtx_z"] = first_of(arr.simVtxZ)
    out["vtx_proc"] = first_of(proc, -1).astype(int)
    out["s_first"] = s_first
    out["s_last"] = s_last

    # ---- leading charged daughter of that vertex --------------------------
    # simDauVtx indexes the vertex list, so carry the slot index through.
    vtx_slot = ak.to_numpy(ak.fill_none(ak.firsts(
        ak.pad_none(ak.local_index(sig, axis=1), 1)[isel]), -1))
    dau_p = np.sqrt(arr.simDauPx**2 + arr.simDauPy**2 + arr.simDauPz**2)
    charged = np.abs(arr.simDauPdgId) != 0
    for neutral in (12, 14, 16, 22, 111, 130, 2112, 311):
        charged = charged & (np.abs(arr.simDauPdgId) != neutral)
    at_vtx = (arr.simDauVtx == vtx_slot[:, None]) & charged
    out["dau_p"] = np.where(out["clean"], np.nan,
                            _max_or(ak.where(at_vtx, dau_p, -np.inf), np.nan))
    out["dau_p"] = np.where(np.isfinite(out["dau_p"]), out["dau_p"], np.nan)

    simp = ak.to_numpy(arr.simTrkP)
    out["sim_p"] = simp
    # momentum fraction carried by the leading charged daughter: the variable
    # the Delta(q/p) component of the score test is directly sensitive to
    out["zfrac"] = out["dau_p"] / np.where(simp > 0, simp, np.nan)

    # ---- discriminants ----------------------------------------------------
    q = ak.to_numpy(arr.trackCharge).astype(float)
    out["D3"] = ak.to_numpy(arr.kinkMax)
    out["Dangle"] = _max_or(arr.kinkDchisqAngle, 0.0)
    out["Dqop"] = _max_or(arr.kinkDchisqQop, 0.0)
    # 3-dof score restricted to steps where the fitted momentum step is a
    # LOSS (q * delta(q/p) > 0), the decay signature
    loss = (arr.kinkDqop * q[:, None]) > 0
    out["Dloss"] = _max_or(ak.where(loss, arr.kinkDchisq, -np.inf), 0.0)
    out["chi2"] = ak.to_numpy(arr.normalizedChi2)

    # ---- quantities at the highest-scoring material step ------------------
    # This step is the kink *candidate*: its position localises the vertex and
    # its deltahat is the estimated kink. NB deltahat is noise-dominated where
    # the step's Delta-chi2 is small, so cut on kink_dchisq before reading it.
    ml = ak.to_numpy(arr.kinkMaxLayer)
    nst = ak.to_numpy(ak.num(arr.kinkGlobalR))
    idxml = ak.singletons(ak.Array(np.clip(ml, 0, np.maximum(nst - 1, 0))))

    def at_argmax(jagged):
        return np.where(nst > 0, ak.to_numpy(ak.fill_none(ak.firsts(
            ak.pad_none(jagged, 1)[idxml]), np.nan)), np.nan)

    out["kink_r"] = at_argmax(arr.kinkGlobalR)
    out["kink_z"] = at_argmax(arr.kinkGlobalZ)
    out["kink_dqop"] = at_argmax(arr.kinkDqop)
    out["kink_dxdz"] = at_argmax(arr.kinkDxdz)
    out["kink_dydz"] = at_argmax(arr.kinkDydz)
    out["kink_dchisq"] = at_argmax(arr.kinkDchisq)
    out["kink_dchisq_qop"] = at_argmax(arr.kinkDchisqQop)
    out["kink_dchisq_angle"] = at_argmax(arr.kinkDchisqAngle)
    # magnitude of the estimated angular kink (small-angle: the two local
    # slope steps add in quadrature)
    out["kink_angle"] = np.hypot(out["kink_dxdz"], out["kink_dydz"])
    out["nsteps"] = nst

    # ---- kinematics and curvature bias ------------------------------------
    genparms = ak.to_numpy(arr.genParms)
    refparms = ak.to_numpy(arr.refParms)
    out["gen_qop"] = genparms[:, 0]
    out["ref_qop"] = refparms[:, 0]
    # p_gen / p_fit - 1: positive = the fit lost momentum w.r.t. the parent
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dqop_rel"] = refparms[:, 0] / genparms[:, 0] - 1.0
    # Fractional momentum step of the estimated kink: delta(q/p) / (q/p).
    # Dimensionless and comparable across momenta; POSITIVE = momentum loss,
    # which is what a decay (softer daughter) produces.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["kink_dpp"] = out["kink_dqop"] / refparms[:, 0]
    out["gen_p"] = ak.to_numpy(arr.genPt * np.cosh(arr.genEta))
    out["gen_pt"] = ak.to_numpy(arr.genPt)
    out["gen_eta"] = ak.to_numpy(arr.genEta)
    out["track_pt"] = ak.to_numpy(arr.trackPt)
    out["track_eta"] = ak.to_numpy(arr.trackEta)
    out["nvalid"] = ak.to_numpy(arr.nValidHits)
    for b in ("muonPt", "muonLoose", "muonMedium", "muonTight",
              "muonIsGlobal", "muonIsTracker", "muonIsPF"):
        if b in arr.fields:
            out[b] = ak.to_numpy(arr[b])

    # match quality, recomputed here so trees written before the genDR
    # branch existed are handled identically
    gphi, tphi = ak.to_numpy(arr.genPhi), ak.to_numpy(arr.trackPhi)
    dphi = np.abs(np.mod(tphi - gphi + np.pi, 2 * np.pi) - np.pi)
    out["gen_dr"] = np.hypot(ak.to_numpy(arr.trackEta) - ak.to_numpy(arr.genEta), dphi)
    out["pt_ratio"] = ak.to_numpy(arr.trackPt) / ak.to_numpy(arr.genPt)

    # identity of the *reconstructed* track, stable across species passes:
    # (run, lumi, event) is unique in the v3 production and trackPt/Eta/Phi
    # are properties of the input KF track, untouched by the CVH hypothesis.
    out["key"] = np.rec.fromarrays(
        [ak.to_numpy(arr.run).astype(np.int64),
         ak.to_numpy(arr.lumi).astype(np.int64),
         ak.to_numpy(arr.event).astype(np.int64),
         ak.to_numpy(arr.trackPt).astype(np.float32).view(np.int32),
         ak.to_numpy(arr.trackEta).astype(np.float32).view(np.int32),
         ak.to_numpy(arr.trackPhi).astype(np.float32).view(np.int32)],
        names="run,lumi,event,pt,eta,phi")
    out["n"] = n
    return out


def arbitrate(species):
    """Cross-species arbitration of the gen match, done offline.

    Each pass matches its own species only, so with a wide pT window a soft
    gen hadron can win the dR match to an unrelated (usually J/psi muon)
    track. Because every pass runs over the same files and the reconstructed
    track is identifiable across passes, the same arbitration the producer
    now does internally can be applied here: a track is kept in the pass
    whose gen candidate is closest in dR.

    Returns {species: boolean mask of tracks this species legitimately owns}.
    """
    keys = [d["key"] for d in species.values()]
    drs = [d["gen_dr"] for d in species.values()]
    allk = np.concatenate(keys)
    uniq, inv = np.unique(allk, return_inverse=True)
    best = np.full(len(uniq), np.inf)
    np.minimum.at(best, inv, np.concatenate(drs))
    out, off = {}, 0
    for name, d in species.items():
        m = len(d["gen_dr"])
        ids = inv[off:off + m]
        out[name] = d["gen_dr"] <= best[ids]
        off += m
    return out


# in-tracker path length used for the analytic decay-probability closure
NOMINAL_PATH_M = 1.0
MASS = {211: 0.13957, 321: 0.49368}
CTAU = {211: 7.804, 321: 3.712}


def decay_prob(p, pdgid, path_m=NOMINAL_PATH_M):
    """1 - exp(-L/(p/m c tau)): the probability that a hadron of momentum p
    decays over `path_m` metres of flight."""
    lam = p / MASS[abs(pdgid)] * CTAU[abs(pdgid)]
    return 1.0 - np.exp(-path_m / lam)


def roc(score, is_sig, is_bkg, npoints=400):
    """Signal efficiency vs background efficiency, sweeping the threshold.
    Returns (fpr, tpr, auc) with both curves starting at (1, 1)."""
    s = np.asarray(score, dtype=float)
    sig = s[is_sig & np.isfinite(s)]
    bkg = s[is_bkg & np.isfinite(s)]
    if len(sig) == 0 or len(bkg) == 0:
        return np.array([]), np.array([]), np.nan
    qs = np.unique(np.quantile(np.concatenate([sig, bkg]),
                               np.linspace(0.0, 1.0, npoints)))
    thr = np.concatenate([[-np.inf], qs, [np.inf]])
    tpr = np.array([(sig > t).mean() for t in thr])
    fpr = np.array([(bkg > t).mean() for t in thr])
    o = np.argsort(fpr)
    auc = np.trapz(tpr[o], fpr[o])
    return fpr, tpr, auc


def threshold_at_fpr(score, is_bkg, target_fpr):
    """Score cut giving `target_fpr` mis-tag rate on the background sample."""
    b = np.asarray(score, dtype=float)[is_bkg]
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.inf
    return float(np.quantile(b, 1.0 - target_fpr))


def binomial_err(k, n):
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, np.asarray(k, dtype=float) / np.where(n > 0, n, 1), np.nan)
    return p, np.where(n > 0, np.sqrt(np.clip(p * (1 - p), 0, None) / np.where(n > 0, n, 1)), np.nan)

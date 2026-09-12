#!/usr/bin/env python3
"""Shared definitions for the hit-class LOCATION-bias study.

WHY THIS MODULE. `hitres_classes.py` centres every class on its own median
and throws the median away ("a per-hit bias belongs to alignment"). That is
the assumption under test: alignment fits MODULE POSITIONS, and a CPE bias
that depends on the CLUSTER (edge, single column, N=1 strip, charge bin,
incidence angle) is not a rigid module shift, so it survives the alignment
and enters the fit as a curvature bias. Here the location is KEPT.

THE SIGN CONVENTION, which is the whole difficulty. `dxrecsim` is
`rec - sim` in the MODULE's own local frame (local x for pixels and barrel
strips, local phi for the radial-strip wedges -- filled in
ResidualGlobalCorrectionMakerG4e.cc), and `dxerr` is the
square root of the same coordinate's variance as the fit used it, so
`pull = dxrecsim/dxerr` is exactly the residual in the fit's measurement dof.
The map to curvature is

    delta(q/p) = sum_b w_b . n_b,     v_b = w_b^T dV_b w_b,

so a location bias `mu_b` in PULL units contributes

    delta z_b = sign(w_b) sqrt(v_b) mu_b / sigma = s_b sqrt(hitamp2_b) mu_b.

`s_b` is the BENDING SENSE: it is the sign of the fit's response of the
signed curvature q/p to a positive residual in that module's local frame, and
it is charge-INDEPENDENT (signed curvature is a property of the trajectory's
geometry, not of the charge), which is why a fixed local bias is a
charge-EVEN curvature bias. `s_b` is read from `resinfbv` (B_b = M_b dV^{1/2},
row 0 = the q/p functional), whose single nonzero entry per rank-1 hit block
is exactly `s_b sqrt(v_b)`.

DetId bit layout: Geometry/TrackerCommonData/data/trackerParameters.xml of
CMSSW_10_6_26 (Phase-0, which is the 2016 UL geometry these guns were
simulated in).
"""
import numpy as np

# subdet -> (name, (field, startbit, mask) ...)
SUBDET = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}


def decode(detid):
    """(subdet, layer_or_disk, side) per hit.

    layer: BPix layer 1-3, FPix disk 1-2, TIB layer 1-4, TID wheel 1-3,
    TOB layer 1-6, TEC wheel 1-9.
    side:  +1 / -1 for the endcap subdetectors (TrackerTopology side 1 = -z,
    2 = +z), 0 for the barrel ones.
    """
    d = np.asarray(detid, dtype=np.uint32)
    sd = ((d >> 25) & 0x7).astype(np.int8)
    lay = np.zeros(len(d), dtype=np.int8)
    side = np.zeros(len(d), dtype=np.int8)
    m = sd == 1
    lay[m] = ((d[m] >> 16) & 0xF)
    m = sd == 2
    lay[m] = ((d[m] >> 16) & 0xF)
    side[m] = np.where(((d[m] >> 23) & 0x3) == 2, 1, -1)
    m = sd == 3
    lay[m] = ((d[m] >> 14) & 0x7)
    m = sd == 4
    lay[m] = ((d[m] >> 11) & 0x3)
    side[m] = np.where(((d[m] >> 13) & 0x3) == 2, 1, -1)
    m = sd == 5
    lay[m] = ((d[m] >> 14) & 0x7)
    m = sd == 6
    lay[m] = ((d[m] >> 14) & 0xF)
    side[m] = np.where(((d[m] >> 18) & 0x3) == 2, 1, -1)
    return sd, lay, side


def ring(detid):
    """TID / TEC ring (0 elsewhere) -- the radial position of a wedge."""
    d = np.asarray(detid, dtype=np.uint32)
    sd = (d >> 25) & 0x7
    r = np.zeros(len(d), dtype=np.int8)
    m = sd == 4
    r[m] = ((d[m] >> 9) & 0x3)
    m = sd == 6
    r[m] = ((d[m] >> 5) & 0x7)
    return r


def is_stereo(detid):
    d = np.asarray(detid, dtype=np.uint32)
    sd = (d >> 25) & 0x7
    return np.where(sd >= 3, (d & 0x3) == 1, 0).astype(np.int8)


# ---------------------------------------------------------------------------
# The canonical 18 classes of `hitres_classes.py`, reproduced here vectorised
# so that the two studies cannot drift apart.
CLASSES18 = ([f"pix_{a}_q{q}" for a in ("x", "y") for q in range(4)]
             + [f"str_N{n}_{u}" for n in range(1, 6) for u in ("lo", "hi")])
CLS18 = {c: i for i, c in enumerate(CLASSES18)}


def class18(sd, N, uproj, qbin, isy):
    """Vectorised `hitres_classes.class_of`."""
    sd = np.asarray(sd)
    isy = np.broadcast_to(np.asarray(isy), sd.shape)
    q = np.clip(np.asarray(qbin).astype(int), 0, 3)
    n = np.clip(np.asarray(N).astype(int), 1, 5)
    hi = (np.asarray(uproj) >= 0.25).astype(int)
    out = np.where(sd <= 2,
                   np.where(isy, 4, 0) + q,
                   8 + 2 * (n - 1) + hi)
    return out.astype(np.int16)


# ---------------------------------------------------------------------------
# The FINER key. The 18 classes are blind to the two things a CPE location
# bias must depend on: WHERE the module is (its layer and z side, which set
# the drift geometry and the incidence) and the SIGN of the local track
# angle (a drift/charge-sharing bias is odd in it). Both are in the trees.
DXDZ_EDGES = np.array([-np.inf, -0.30, -0.15, -0.06, -0.02, 0.02, 0.06, 0.15, 0.30, np.inf])
DYDZ_EDGES = np.array([-np.inf, -0.8, -0.4, -0.15, 0.15, 0.4, 0.8, np.inf])
NDXDZ = len(DXDZ_EDGES) - 1
NDYDZ = len(DYDZ_EDGES) - 1


def angbin(v, edges):
    return np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2).astype(np.int8)


def fine_key(sd, lay, side, cls18, adx, ady):
    """A single integer key, injective on (subdet, layer, side, class, angles)."""
    s = (side + 1).astype(np.int64)          # 0,1,2
    return (((((sd.astype(np.int64) * 10 + lay) * 3 + s) * 18
              + cls18) * NDXDZ + adx) * NDYDZ + ady)


def unpack_key(k):
    k = np.asarray(k, dtype=np.int64)
    ady = k % NDYDZ
    k //= NDYDZ
    adx = k % NDXDZ
    k //= NDXDZ
    c = k % 18
    k //= 18
    s = k % 3 - 1
    k //= 3
    lay = k % 10
    sd = k // 10
    return sd, lay, s, c, adx, ady


# ---------------------------------------------------------------------------
# ORIENTATION GROUP. The bending sense `s_b` is nearly random inside a
# (subdet, layer): +-0.03 to 0.06 in BPix, TOB and TEC. That is not noise, it
# is the local FRAME flipping between the two module orientations of a layer
# (BPix inner/outer ladders, TIB internal/external strings, TOB and TEC
# forward/backward rods and petals, FPix panels). A residual bias that is
# fixed in the LOCAL frame therefore has to be measured in the same
# orientation group in which `s_b` is measured, or the two average away
# separately and the product is meaningless.
#
# Every field below is a DetId bit field, exact, and `hitDetId` is present in
# BOTH trees -- so this is a join on the module's identity, not a proxy.
def orient_group(detid):
    """A small integer labelling the module's frame orientation class."""
    d = np.asarray(detid, dtype=np.uint32)
    sd = ((d >> 25) & 0x7).astype(np.int64)
    g = np.zeros(len(d), dtype=np.int64)

    m = sd == 1                                   # BPix: layer, ladder parity
    g[m] = ((d[m] >> 16) & 0xF) * 4 + ((d[m] >> 8) & 0xFF) % 2 * 2 \
        + (((d[m] >> 2) & 0x3F) > 4)
    m = sd == 2                                   # FPix: side, disk, panel
    g[m] = ((((d[m] >> 23) & 0x3) * 4 + ((d[m] >> 16) & 0xF)) * 4
            + ((d[m] >> 8) & 0x3)) * 2 + ((d[m] >> 10) & 0x3F) % 2
    m = sd == 3                                   # TIB: layer, fw/bw, int/ext, stereo
    g[m] = ((((d[m] >> 14) & 0x7) * 4 + ((d[m] >> 12) & 0x3)) * 4
            + ((d[m] >> 10) & 0x3)) * 4 + (d[m] & 0x3)
    m = sd == 4                                   # TID: side, wheel, ring, fw/bw, stereo
    g[m] = (((((d[m] >> 13) & 0x3) * 4 + ((d[m] >> 11) & 0x3)) * 4
             + ((d[m] >> 9) & 0x3)) * 4 + ((d[m] >> 7) & 0x3)) * 4 + (d[m] & 0x3)
    m = sd == 5                                   # TOB: layer, rod fw/bw, stereo
    g[m] = (((d[m] >> 14) & 0x7) * 4 + ((d[m] >> 12) & 0x3)) * 4 + (d[m] & 0x3)
    m = sd == 6                                   # TEC: side, wheel, ring, petal fw/bw, stereo
    g[m] = (((((d[m] >> 18) & 0x3) * 16 + ((d[m] >> 14) & 0xF)) * 8
             + ((d[m] >> 5) & 0x7)) * 4 + ((d[m] >> 12) & 0x3)) * 4 + (d[m] & 0x3)
    return sd * 100000 + g

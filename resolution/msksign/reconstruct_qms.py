#!/usr/bin/env python3
"""Rebuild the propagator's accumulated MS block from ITS OWN per-step log, for
both sign conventions of the within-step angle-offset term, and report the
deficit the negative one carries.

`G4ePropagationExport` writes, per leg: `dQMS` (the propagator's accumulated
5x5), `stepjacc` (the cumulative transport J_{start->s} after every step),
`stepnms` (the size of the Moliere log after every step) and `msmoliv`
(thp2 = the step's projected-angle variance as it enters Q, column 5).

The recursion the propagator runs is  Q <- J_step Q J_step^T + q_step, so

    Q_end = sum_s  J_{s->end} q_s J_{s->end}^T ,   J_{s->end} = J_end J_s^{-1}

with q_s the 5x5 of `PropagateErrorMSC`. Reproducing `dQMS` from that is the
check that the model of the block is the code's; switching the sign of the
(1,4) entry then gives the size of the defect, leg by leg.
"""
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prodfiles import cumulative_total, reshape_records  # noqa: E402

MS_THP2 = 5   # column of thp2 in msmoliv


def qstep(thp2, lcm, cla, sign):
    """`PropagateErrorMSC`'s 5x5, given DD = thp2 and the step length."""
    q = np.zeros((5, 5))
    S2 = thp2
    S1 = thp2 * lcm * lcm / 3.0
    S3 = thp2 * lcm / 2.0
    q[1, 1] = S2
    q[1, 4] = q[4, 1] = sign * S3
    q[2, 2] = S2 / cla / cla
    q[2, 3] = q[3, 2] = S3 / cla
    q[3, 3] = S1
    q[4, 4] = S1
    return q


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/work/submit/david_w/ZMass/calibration_studies/resolution/runs/msksign_260918/subdiv/model_legacy_L10.0.root"
    t = uproot.open(path)["propExport/legs"]
    br = ["ok", "dQMS", "stepjacc", "stepnms", "msmoliv", "msmolistride", "radv", "refp", "refpt"]
    br = [b for b in br if b in t.keys()]
    a = t.arrays(br, library="np")
    print(f"{path}\n")
    print(f"{'leg':>4} {'nms':>5} {'path[cm]':>9} {'Q44 code':>12} {'Q44 rebuilt-':>13}"
          f" {'Q44 rebuilt+':>13} {'-/+':>7} {'Q14 -/+':>8} {'Q33 code':>12} {'Q33 rebuilt':>12}")
    r44, r14 = [], []
    for i in range(t.num_entries):
        if not a["ok"][i]:
            continue
        nms = np.asarray(a["stepnms"][i], dtype=int)
        if nms.size == 0 or nms[-1] == 0:
            continue
        J = np.asarray(a["stepjacc"][i], dtype=float).reshape(-1, 5, 5)
        # strides come from the file: the declared branch when the producer
        # writes one, else the CUMULATIVE per-step counters (`stepnms` is the
        # log size AFTER each step, so the total is its last element)
        nms_tot = cumulative_total(nms)
        ms = reshape_records(a["msmoliv"][i], nrec=nms_tot,
                             stride=(int(np.atleast_1d(a["msmolistride"][i])[0])
                                     if "msmolistride" in a else None),
                             branch="msmoliv", min_cols=6)
        rad = reshape_records(a["radv"][i], nrec=nms_tot, branch="radv",
                              min_cols=7) if "radv" in a else None
        if rad is None or rad.shape[0] != ms.shape[0]:
            continue
        lcm = rad[:, 6]                       # step(cm), 1:1 with the Moliere log
        cla = a["refpt"][i] / a["refp"][i]    # cos(lambda) at the leg end
        Jend = J[-1]
        out = {}
        for sign in (-1.0, +1.0):
            Q = np.zeros((5, 5))
            prev = 0
            for k, n in enumerate(nms):
                if n <= prev:
                    prev = n
                    continue
                js = k                        # the step that pushed record n-1
                T = Jend @ np.linalg.inv(J[js])
                Q += T @ qstep(ms[n - 1, MS_THP2], lcm[n - 1], cla, sign) @ T.T
                prev = n
            out[sign] = Q
        Qc = np.asarray(a["dQMS"][i], dtype=float).reshape(5, 5)
        print(f"{i:>4} {nms[-1]:>5} {lcm.sum():>9.4f} {Qc[4,4]:>12.5e} {out[-1][4,4]:>13.5e}"
              f" {out[1][4,4]:>13.5e} {out[-1][4,4]/out[1][4,4]:>7.4f}"
              f" {out[-1][1,4]/out[1][1,4]:>8.4f} {Qc[3,3]:>12.5e} {out[-1][3,3]:>12.5e}")
        r44.append(out[-1][4, 4] / out[1][4, 4])
        r14.append(out[-1][1, 4] / out[1][1, 4])
        # the code must equal the rebuilt with the sign the binary was run with
        rel = abs(Qc[4, 4] - out[-1][4, 4]) / abs(out[-1][4, 4])
        if i == 1:
            print(f"     reproduction of the code's own dQMS(4,4) with the "
                  f"legacy sign: rel {rel:.2e}; with the fixed sign: "
                  f"{abs(Qc[4,4]-out[1][4,4])/abs(out[1][4,4]):.2e}")
    r44, r14 = np.array(r44), np.array(r14)
    print(f"\nover {len(r44)} legs:  Q(4,4) legacy/fixed  median {np.median(r44):.4f}"
          f"  p05 {np.percentile(r44,5):.4f}  min {r44.min():.4f}")
    print(f"                     Q(1,4) legacy/fixed  median {np.median(r14):.4f}"
          f"  p05 {np.percentile(r14,5):.4f}  min {r14.min():.4f}")


if __name__ == "__main__":
    main()

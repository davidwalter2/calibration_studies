#!/usr/bin/env python3
"""Figures for the CF-exponent compression study.  One file per figure."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubhtml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "figure.constrained_layout.use": True})

# The small artifacts of the study (spec_*.npz, phys_*.npz, the json
# summaries and the logs) are kept with the code.
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = pubhtml.figdir("cfcompress")
TAGS = ["jpsigun", "btojpsix", "trk_lowpt", "trk_ul16"]
LBL = {"jpsigun": "J/psi gun (mass pairs)", "btojpsix": "B->J/psi X v3 (mass pairs)",
       "trk_lowpt": "mu gun low pT (single track)", "trk_ul16": "mu gun UL16 (single track)"}
COL = {"Sms": "C0", "Sdel": "C5", "Sio_re": "C1", "Sio_im": "C2",
       "Srad_re": "C3", "Srad_im": "C4"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    try:
        from wums import plot_tools
        plot_tools.save_pdf_and_png(OUT, name, fig)
    except Exception:
        fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
        fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ->", f"{OUT}/{name}.png")


def fig_spectra():
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    for ax, tag in zip(axs.ravel(), TAGS):
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        fams = [str(x) for x in d["fams"]]
        s0 = d["ALL/sv"][0]      # ONE normalisation for every curve, so the
        for f in fams:           # panel shows ABSOLUTE importance (the error
            sv = d[f"{f}/sv"]    # criterion is absolute on the exponent)
            ax.semilogy(np.arange(1, 41), sv[:40] / s0, label=f, color=COL[f])
        sv = d["ALL/sv"]
        ax.semilogy(np.arange(1, 41), sv[:40] / s0, "k--", lw=2,
                    label="ALL (joint)")
        ax.set_title(LBL[tag], fontsize=15)
        ax.set_xlabel("component"); ax.set_ylabel("singular value / first of the joint block")
        ax.set_ylim(1e-8, 2); ax.grid(alpha=.3)
        ax.legend(fontsize=11, ncol=2)
    fig.suptitle("SVD spectrum of the per-candidate CF exponents "
                 "(centred, n = 40000)", fontsize=17)
    save(fig, "svd_spectra")


def fig_rank_error():
    for metric, mlab in (("ewgt", r"max$_t$ $W(t)\,|\Delta S|$"),
                         ("eabs", r"max$_t$ $|\Delta S|$")):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        for ax, tag in zip(axs.ravel(), TAGS):
            d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
            fams = [str(x) for x in d["fams"]]
            r = d["ALL/ranks"]
            for f in fams:
                ax.loglog(r, d[f"ALL/{metric}_{f}_max"], "-o", ms=4,
                          color=COL[f], label=f + " (max)")
                ax.loglog(r, d[f"ALL/{metric}_{f}_q999"], ":", color=COL[f],
                          alpha=.6)
            ax.axhline(1e-4, color="k", ls="--", lw=1)
            ax.axhline(1e-3, color="k", ls=":", lw=1)
            ax.text(1.1, 1.15e-4, r"$10^{-4}$", fontsize=11)
            ax.set_title(LBL[tag], fontsize=15)
            ax.set_xlabel("rank r of the joint basis  [scalars / candidate]")
            ax.set_ylabel(mlab)
            ax.grid(alpha=.3); ax.legend(fontsize=10, ncol=2)
        fig.suptitle("Reconstruction error vs rank -- one coefficient vector "
                     "per candidate for all families (solid: worst candidate, "
                     "dotted: 99.9 %)", fontsize=15)
        save(fig, f"rank_error_{metric}")


def fig_perfamily():
    """Joint (ALL) basis vs an independent basis per family, at equal TOTAL
    scalars per candidate."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, tag in zip(axs, ["jpsigun", "btojpsix"]):
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        fams = [str(x) for x in d["fams"]]
        r = d["ALL/ranks"]
        wa = np.max([d[f"ALL/ewgt_{f}_max"] for f in fams], axis=0)
        ax.loglog(r, wa, "-o", color="k", lw=2, label="ALL joint basis (r scalars)")
        # per-family: IO block (2 fam) + MS + RAD block, each with its own basis
        tot, err = [], []
        for i, rr in enumerate(r):
            e = max(d[f"Sms/ewgt_Sms_max"][i],
                    d[f"IO/ewgt_Sio_re_max"][i], d[f"IO/ewgt_Sio_im_max"][i],
                    d[f"RAD/ewgt_Srad_re_max"][i], d[f"RAD/ewgt_Srad_im_max"][i])
            tot.append(3 * rr); err.append(e)
        ax.loglog(tot, err, "-s", color="C3",
                  label="per-block bases MS+IO+RAD (3r scalars)")
        ax.axhline(1e-4, color="k", ls="--", lw=1)
        ax.set_xlabel("scalars per candidate"); ax.set_ylabel(r"worst family, max$_t$ $W|\Delta S|$")
        ax.set_title(LBL[tag], fontsize=14); ax.grid(alpha=.3); ax.legend(fontsize=11)
    fig.suptitle("One joint basis is at least as good as one basis per family, at equal storage",
                 fontsize=16)
    save(fig, "joint_vs_perfamily")


def fig_physbasis():
    f = f"{SCRATCH}/phys_jpsigun.npz"
    if not os.path.exists(f):
        return
    d = np.load(f, allow_pickle=True)
    k = d["klist"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, b in zip(axs, ["MS", "IO", "RAD"]):
        for meth, c, lab in (("pca", "k", "PCA of the data matrix"),
                             ("phys", "C3",
                              "physics quadrature (Levy / Moliere nodes)"),
                             ("poly", "C0", "cumulant / Taylor")):
            key = f"{b}/{meth}/ewgt_q999"
            if key in d.files:
                ax.loglog(k, d[key], "-o", ms=4, color=c, label=lab)
        if f"{b}/fine/ewgt_q999" in d.files:
            ax.axhline(float(d[f"{b}/fine/ewgt_q999"]), color="C3", ls=":",
                       label=f"full physics dict (K={int(d[f'{b}/fine/K'])})")
        ax.axhline(1e-4, color="k", ls="--", lw=1)
        ax.set_title(b, fontsize=15)
        ax.set_xlabel("scalars per candidate  k")
        ax.set_ylabel(r"99.9 % of max$_t$ $W|\Delta S|$")
        ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.suptitle("Data-driven PCA vs the exact physics dictionary "
                 "(J/psi gun, test split)", fontsize=16)
    save(fig, "physbasis_vs_pca")


def fig_weight():
    fig, ax = plt.subplots(figsize=(9, 6))
    for tag in TAGS:
        d = np.load(f"{SCRATCH}/spec_{tag}.npz", allow_pickle=True)
        ax.semilogy(d["TG"], d["Wmed"], label=LBL[tag])
    ax.axhline(1e-3, color="k", ls=":")
    ax.set_xlabel(r"$t\,\sigma$  (the standardized grid TG)")
    ax.set_ylabel(r"median $W(t) = e^{{\rm Re}S_{\rm tot}(t)}|\phi_K|$")
    ax.set_ylim(1e-12, 2); ax.grid(alpha=.3); ax.legend(fontsize=11)
    ax.set_title("The integrand envelope: where an exponent error matters",
                 fontsize=15)
    save(fig, "weight_envelope")


def fig_nll():
    f = f"{SCRATCH}/nllimpact_jpsigun_n20000.json"
    if not os.path.exists(f):
        return
    rows = json.load(open(f))
    ref = rows[0]
    kinds = {"pca_all": ("k", "joint PCA (r scalars)"),
             "pca_fam": ("C0", "per-family PCA (5r scalars)"),
             "levy": ("C3", "physics quadrature")}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for kind, (c, lab) in kinds.items():
        rr = [r for r in rows if r.get("kind") == kind]
        if not rr:
            continue
        b = [r["bytes"] for r in rr]
        axs[0].loglog(b, [abs(r["x"][0] - ref["x"][0]) * 1e-3 for r in rr],
                      "-o", color=c, label=lab)
        axs[1].loglog(b, [abs(r["nll"] - ref["nll"]) for r in rr], "-o",
                      color=c, label=lab)
    axs[0].axhline(1e-5, color="k", ls="--",
                   label=r"target $|\Delta\alpha| < 10^{-5}$")
    axs[0].axhline(1.77e-6, color="C2", ls=":",
                   label=r"$\sigma_\alpha/10$ (full 300k sample)")
    axs[0].set_ylabel(r"$|\Delta\alpha|$")
    axs[1].set_ylabel(r"$|\Delta$NLL$|$ at the minimum")
    for ax in axs:
        ax.set_xlabel("bytes per candidate")
        ax.grid(alpha=.3); ax.legend(fontsize=10)
    fig.suptitle("End-to-end: shift of the fitted momentum scale vs storage "
                 "(J/psi gun, 20000 candidates)", fontsize=15)
    save(fig, "nll_impact")

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for j, (ax, pn) in enumerate(zip(axs, ["alpha", "k_hit", "k_ms", "k_ioni"])):
        for kind, (c, lab) in kinds.items():
            # the `levy` point is DROPPED here: the 6-scalar physics quadrature
            # makes the fit DIVERGE (k_ms -> 1034, NLL -> nan), which is a
            # failure mode, not a measurement, and its 4e4 sigma would flatten
            # every other curve.  Reported in the text instead.
            if kind == "levy":
                continue
            rr = [r for r in rows if r.get("kind") == kind]
            if not rr:
                continue
            ax.semilogx([r["bytes"] for r in rr],
                        [(r["x"][j] - ref["x"][j]) / ref["err"][j] for r in rr],
                        "-o", color=c, label=lab)
        ax.axhline(0, color="k", lw=.8)
        for y in (-0.1, 0.1):
            ax.axhline(y, color="k", ls=":", lw=.8)
        ax.set_title(pn, fontsize=14)
        ax.set_xlabel("bytes per candidate")
        ax.set_ylabel(r"shift / $\sigma$")
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=9)
    fig.suptitle("Parameter shifts of the compressed fit, in units of the "
                 "statistical error of the same 20000-candidate fit "
                 "(the diverged 6-scalar physics quadrature is off scale and "
                 "not shown)", fontsize=14)
    save(fig, "param_shifts")


def fig_fullshift():
    """The FULL 299712-candidate first-order shift -- the number that decides."""
    f = f"{SCRATCH}/nllfullshift_jpsigun.json"
    if not os.path.exists(f):
        return
    rows = json.load(open(f))
    ref = rows[0]
    rr = [r for r in rows if r.get("kind") == "pca_all"]
    b = [r["bytes"] for r in rr]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    axs[0].loglog(b, [abs(r["d1"][0]) * 1e-3 for r in rr], "-o", color="k")
    axs[0].axhline(1e-5, color="C3", ls="--", label=r"target $10^{-5}$")
    axs[0].axhline(ref["err"][0] * 1e-3, color="C2", ls=":",
                   label=r"$\sigma_\alpha$ (full 300k fit)")
    axs[0].set_ylabel(r"$|\Delta\alpha|$"); axs[0].legend(fontsize=10)
    axs[1].loglog(b, [r["maxdS"] for r in rr], "-o", color="k")
    axs[1].axhline(1e-4, color="C3", ls="--", label=r"$10^{-4}$")
    axs[1].set_ylabel(r"max over 299712 candidates of $|\Delta S|$")
    axs[1].legend(fontsize=10)
    axs[2].loglog(b, [abs(r["dnll_at_ref"]) for r in rr], "-o", color="k")
    axs[2].set_ylabel(r"$|\Delta$NLL$|$ at $\theta^*_{\rm full}$")
    for ax in axs:
        ax.set_xlabel("bytes per candidate"); ax.grid(alpha=.3)
    fig.suptitle("Full sample (299712 J/psi-gun candidates): joint-PCA rank "
                 "r = 8, 16, 32, 48, 64", fontsize=15)
    save(fig, "fullshift")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pubhtml.ensure_index(OUT)
    for fn in (fig_spectra, fig_rank_error, fig_perfamily, fig_physbasis,
               fig_weight, fig_nll, fig_fullshift):
        try:
            fn()
        except Exception as e:
            print(f"!! {fn.__name__}: {type(e).__name__}: {e}")
    print("figures in", OUT)

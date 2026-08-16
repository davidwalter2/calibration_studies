import hist

from wums import ioutils
from wums import boostHistHelpers as hh

# Candidate names for the dilepton/boson mass axis, newest histmaker naming
# first. Older files used "mass"; the 260519+ histmaker uses "massVgen"
# (gen boson) / "ewMll" (dressed/bare leptons).
_MASS_AXES = ("mass", "massVgen", "ewMll", "mll")


def load_results_h5py(h5file):
    if "results" in h5file.keys():
        return ioutils.pickle_load_h5py(h5file["results"])
    else:
        return {k: ioutils.pickle_load_h5py(v) for k, v in h5file.items()}


def _rename_axis_to_mass(h):
    """Return a copy of 1D hist `h` with its single axis renamed to 'mass'.

    Downstream code (plot_lineshape, analytic_gen_distributions, ...) keys on
    an axis literally named 'mass', so normalize the name here regardless of
    the histmaker convention in the input file.
    """
    ax = h.axes[0]
    new_ax = hist.axis.Variable(
        ax.edges, name="mass", label=(getattr(ax, "label", None) or "mass")
    )
    h2 = hist.Hist(new_ax, storage=h.storage_type())
    h2.view(flow=False)[...] = h.view(flow=False)
    return h2


def get_hist(result, sample, histname="nominal_gen", mass_axis=None,
             pt_min=None, pt_max=None, pt_axis="ptVgen",
             y_max=None, y_axis="absYVgen"):
    """Project the multi-dim ``histname`` of ``sample`` to a 1D mass hist.

    If the input has a transverse-momentum axis (``pt_axis``, default
    ``"ptVgen"``) and ``pt_min``/``pt_max`` are provided, the boson pT is
    restricted to ``[pt_min, pt_max]`` before projecting. Likewise, if a
    rapidity axis (``y_axis``, default ``"absYVgen"``) is present and
    ``y_max`` is provided, the boson rapidity is restricted to
    ``|Y| < y_max`` -- this matters when the upstream histmaker is run
    without a rapidity cut and ``absYVgen`` extends past the desired cut.
    The mass axis is auto-detected from ``_MASS_AXES`` (or pass
    ``mass_axis=`` explicitly) and renamed to ``"mass"`` so downstream code
    keys uniformly on that name.
    """
    h = result[sample]["output"][histname].get()

    if pt_axis in [a.name for a in h.axes] and (pt_min is not None or pt_max is not None):
        lo = complex(0, pt_min) if pt_min is not None else None
        hi = complex(0, pt_max) if pt_max is not None else None
        h = h[{pt_axis: slice(lo, hi, hist.sum)}]

    if y_axis in [a.name for a in h.axes] and y_max is not None:
        h = h[{y_axis: slice(complex(0, 0.0), complex(0, y_max), hist.sum)}]

    if mass_axis is None:
        names = [a.name for a in h.axes]
        mass_axis = next((c for c in _MASS_AXES if c in names), None)
        if mass_axis is None:
            raise ValueError(
                f"No mass-like axis in '{histname}' for '{sample}' "
                f"(axes={names}); pass mass_axis= explicitly."
            )

    h = h.project(mass_axis)

    weight_sum = result[sample]["weight_sum"]
    xsec = result[sample]["dataset"]["xsec"]
    scale = xsec / weight_sum

    h = hh.scaleHist(h, scale)

    if h.axes[0].name != "mass":
        h = _rename_axis_to_mass(h)

    return h
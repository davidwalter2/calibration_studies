from wums import ioutils
from wums import boostHistHelpers as hh

def load_results_h5py(h5file):
    if "results" in h5file.keys():
        return ioutils.pickle_load_h5py(h5file["results"])
    else:
        return {k: ioutils.pickle_load_h5py(v) for k, v in h5file.items()}


def get_hist(result, sample, histname="nominal_postfsr"):

    h = result[sample]["output"][histname].get().project("mass")

    weight_sum = result[sample]["weight_sum"]
    xsec = result[sample]["dataset"]["xsec"]
    scale = xsec / weight_sum
    
    return hh.scaleHist(h, scale)
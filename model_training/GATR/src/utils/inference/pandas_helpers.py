import gzip
import pickle
# import mplhep as hep

# hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=25)
import numpy as np
import pandas as pd


def open_hgcal(path_hgcal, neutrals_only):
    with gzip.open(
        path_hgcal,
        "rb",
    ) as f:
        data = pickle.load(f)
    sd = data["showers_dataframe"]
    # if neutrals_only:
    #     sd = pd.concat(
    #         [
    #             data[data["pid"] == 130],
    #             data[data["pid"] == 2112],
    #             data[data["pid"] == 22],
    #         ]
    #     )
    # else:
    sd = data
    matched = sd.dropna()
    ms = data["matched_showers"]

    return sd, ms


def open_mlpf_dataframe(path_mlpf, neutrals_only=False):
    data = pd.read_pickle(path_mlpf)
    # if neutrals_only:
    #     sd = pd.concat(
    #         [
    #             data[data["pid"] == 130],
    #             data[data["pid"] == 2112],
    #             data[data["pid"] == 211],
    #         ]
    #     )
    # else:
    sd = data
    mask = (~np.isnan(sd["pred_showers_E"])) * (~np.isnan(sd["reco_showers_E"]))
    matched = sd[mask]
    return sd, matched

def open_tracking_dataframe(path_tracking):
    data = pd.read_pickle(path_tracking)
    
    return data
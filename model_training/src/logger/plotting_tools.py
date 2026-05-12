import plotly.express as px
import dgl
import torch
import pandas as pd
import numpy as np
import os
import wandb
import matplotlib
matplotlib.rc('font', size=15)

import matplotlib.pyplot as plt

def PlotCoordinates(
    g,
    path,
    outdir,
    num_layer=0,
    predict=False,
    egnn=False,
    features_type="ones",
    epoch="",
    step_count=0,
):
    name = path
    graphs = dgl.unbatch(g)

    for i in range(0, 1):
        graph_i = graphs[i]

        if path == "input_coords":
            coords = graph_i.ndata["original_coords"]
            labels = graph_i.ndata["particle_number"]
            features = torch.ones_like(coords[:, 0]).view(-1, 1)

        if path == "final_coords":
            coords = graph_i.ndata["final_coords"]
            labels = graph_i.ndata["reco_labels"]
            features = graph_i.ndata["particle_number"]

        if path == "embedded_coords":
            coords = graph_i.ndata["embedded_coords"]
            labels = graph_i.ndata["particle_number"]
            features = torch.sigmoid(graph_i.ndata["beta"])

        data = {
            "X": coords[:, 0].view(-1, 1).detach().cpu().numpy(),
            "Y": coords[:, 1].view(-1, 1).detach().cpu().numpy(),
            "Z": coords[:, 2].view(-1, 1).detach().cpu().numpy(),
            "labels": labels.view(-1, 1).detach().cpu().numpy(),
            "features": features.view(-1, 1).detach().cpu().numpy(),
        }

        df = pd.DataFrame(
            np.concatenate([data[k] for k in data], axis=1),
            columns=list(data.keys()),
        )

        df["labels"] = df["labels"].astype(int).astype(str)

        unique_labels = sorted(df["labels"].unique(), key=lambda x: int(x))
        colors = px.colors.qualitative.Light24 
        color_map = {
            label: colors[i % len(colors)]
            for i, label in enumerate(unique_labels)
        }

        fig = px.scatter_3d(
            df,
            x="X",
            y="Y",
            z="Z",
            color="labels",          
            size="features",
            size_max=10,
            template="plotly_white",
            color_discrete_map=color_map,
            category_orders={"labels": unique_labels},  
            labels={"labels": "Particle label"},
        )

        fig.update_layout(
            legend=dict(
                title="Particle label",
                itemsizing="constant",   
                tracegroupgap=2,
            )
        )

    return fig

def shuffle_truth_colors(df, qualifier="truthHitAssignementIdx", rdst=None):
    ta = df[qualifier]
    unta = np.unique(ta)
    unta = unta[unta > -0.1]
    if rdst is None:
        np.random.shuffle(unta)
    else:
        rdst.shuffle(unta)
    out = ta.copy()
    for i in range(len(unta)):
        out[ta == unta[i]] = i
    df[qualifier] = out


def efficiency_purity_plot(df, minX, maxX, binStep, applyConstraints = False, maxR = 0.05, minDeltaMC = 0.02, minNumHits = 3, minTheta = 10, maxTheta = 170, genStatus = [0,1]):
        
        bins = np.exp(np.arange(np.log(minX), np.log(maxX), binStep))

        def theta_to_float(x):
            if x is None:
                return None
            elif isinstance(x, (list, np.ndarray)):
                return float(x[0]) if len(x) > 0 else None
            else:
                return float(x)

        df["theta"] = df["theta"].apply(theta_to_float)
        
        df_valid = df.copy()
        df_valid = df_valid.dropna()
        mask = df_valid["trackLabel"].apply(lambda x: isinstance(x, list) and 0 in x)
        df_valid.loc[mask, "trackLabel"] = df_valid.loc[mask, "trackLabel"].apply(lambda _: [])

        df_valid = df_valid[df_valid["trackLabel"].apply(bool)]

        df_valid["theta_deg"] = np.degrees(df_valid["theta"])
        
        if applyConstraints: 
            df_valid = df_valid[(df_valid["theta_deg"] > minTheta) & (df_valid["theta_deg"] < maxTheta)]
            df_valid = df_valid[df_valid["genStatus"].isin(genStatus)]
            df_valid = df_valid[(df_valid["pT"] > 0)]
            df_valid = df_valid[(df_valid["numSIhits"] + df_valid["numCDChits"] > minNumHits)]
            # df_valid = df_valid[(df_valid["R"] < maxR)]
            # df_valid = df_valid[(df_valid["deltaMC"] > minDeltaMC)]
        else:
            df_valid = df_valid[(df_valid["theta_deg"] > minTheta) & (df_valid["theta_deg"] < maxTheta)]
            
        bin_indices = np.digitize(df_valid["pT"], bins)    
        
        efficiencies = []
        purities = []
        bin_centers = []
        errors_eff = []
        errors_pur = []

        for i in range(1, len(bins)):
            df_bin = df_valid[bin_indices == i]
            n_total = len(df_bin)

            bin_centers.append((bins[i-1] + bins[i]) / 2)

            if n_total == 0:
                efficiencies.append(np.nan)
                purities.append(np.nan)
                errors_eff.append(0)
                errors_pur.append(0)
                continue

            eff_arr_raw = df_bin["hitEfficiency"].to_list()
            pur_arr_raw = df_bin["hitPurity"].to_list()

            eff_arr = []
            pur_arr = []

            for idx, el in enumerate(eff_arr_raw):
                if not isinstance(el, (list, np.ndarray)):
                    continue  

                max_eff_idx = 0
                max_eff = -np.inf  

                for sub_idx, sub_el in enumerate(el):
                    if sub_el > max_eff:
                        max_eff = sub_el
                        max_eff_idx = sub_idx

                eff_arr.append(el[max_eff_idx])
                pur_arr.append(pur_arr_raw[idx][max_eff_idx])
                
            pur_arr = np.array(pur_arr, dtype=float)
            eff_arr = np.array(eff_arr, dtype=float)
            

            # Means
            eff = np.mean(eff_arr)
            pur = np.mean(pur_arr)

            efficiencies.append(eff)
            purities.append(pur)

            # Standard error on the mean
            errors_eff.append(np.std(eff_arr, ddof=1) / np.sqrt(n_total))
            errors_pur.append(np.std(pur_arr, ddof=1) / np.sqrt(n_total))

        bin_centers = np.array(bin_centers)
        efficiencies = np.array(efficiencies)
        purities = np.array(purities)
        errors_eff = np.array(errors_eff)
        errors_pur = np.array(errors_pur)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)

        colors = ["#238A8DFF", "#440154FF"]
        markers = ["s", "o"]
        labels = ["Hit efficiency", "Hit purity"]

        # Efficiency
        ax.scatter(bin_centers, efficiencies, marker=markers[0], label=labels[0], s=35)
        yerr_low, yerr_up = limit_error_bars(efficiencies, errors_eff, upper_limit=1)
        ax.errorbar(bin_centers, efficiencies, yerr=[yerr_low, yerr_up], linestyle='none', capsize=4)

        # Purity
        ax.scatter(bin_centers, purities, marker=markers[1], label=labels[1], s=35)
        yerr_low_p, yerr_up_p = limit_error_bars(purities, errors_pur, upper_limit=1)
        ax.errorbar(bin_centers, purities, yerr=[yerr_low_p, yerr_up_p], linestyle='none', capsize=4)

        ax.set_xscale("log")
        ax.set_xlim([minX, maxX])
        ax.set_ylim([0.01, 1.01])
        ax.set_xlabel("$p_T$ [GeV]")
        ax.set_ylabel("Hit efficiency / purity")
        ax.legend(loc="lower right")

        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto', numticks=10))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle=':', linewidth=0.5, color='black')
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')

        if applyConstraints:
            textbox_text = (
                r"$Z/\gamma^* \rightarrow q\bar{q} (q = u,d,s)$" "\n"
                r"$m_Z = 91~\mathrm{GeV}$" "\n"
                rf"${minTheta}^\circ < \theta < {maxTheta}^\circ$" "\n"
                rf"$N_\mathrm{{hits}} = N_\mathrm{{SI}} + N_\mathrm{{CDC}} > {minNumHits}$" "\n"
                # rf"$R < {maxR}$" "\n"
                # rf"$\Delta_\mathrm{{MC}} > {minDeltaMC}$" "\n"
                rf"$genStatus \in {genStatus}$"
            )
            
            ax.text(
                0.55, 0.5, textbox_text,
                transform=ax.transAxes,
                fontsize=22,
                verticalalignment='center',
                horizontalalignment='left',
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="none",
                    edgecolor="none"
                    )
            )
            
        else:
            textbox_text = (
                r"$Z/\gamma^* \rightarrow q\bar{q} (q = u,d,s)$" "\n"
                r"$m_Z = 91~\mathrm{GeV}$" "\n"
                rf"${minTheta}^\circ < \theta < {maxTheta}^\circ$"
            )
            
            ax.text(
                0.62, 0.33, textbox_text,
                transform=ax.transAxes,
                fontsize=22,
                verticalalignment='center',
                horizontalalignment='left',
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="none",
                    edgecolor="none"
                    )
            ) 

        return fig

def limit_error_bars(y, yerr, upper_limit=1):
        yerr_upper = np.minimum(y + yerr, upper_limit) - y
        yerr_lower = yerr
        return yerr_lower, yerr_upper
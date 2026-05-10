import plotly.express as px
import dgl
import torch
import pandas as pd
import numpy as np
import os
import wandb

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

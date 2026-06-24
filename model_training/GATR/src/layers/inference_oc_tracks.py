import dgl
import torch
import os
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb
from sklearn.cluster import DBSCAN, HDBSCAN
import sys
from collections import Counter

def hfdb_obtain_labels(X, device, eps=0.1):
    # hdbscan gives -1 if noise.. 1 if +
    # hdb = HDBSCAN(min_cluster_size=8, min_samples=8, cluster_selection_epsilon=eps).fit(
    #     X.detach().cpu()
    # )
    hdb = HDBSCAN(min_cluster_size=4, cluster_selection_epsilon=eps).fit(
        X.detach().cpu()
    )
    # hdb = DBSCAN(min_samples=5, eps=0.2).fit(X.detach().cpu())
    labels_hdb = hdb.labels_ + 1  # noise class goes to zero
    labels_hdb = np.reshape(labels_hdb, (-1))
    labels_hdb = torch.Tensor(labels_hdb).long().to(device)
    return labels_hdb


def evaluate_efficiency_tracks(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    ct=False,
    clustering_mode="clustering_normal",
    tau=False
):
    number_of_showers_total = 0
    if not ct:
        batch_g.ndata["coords"] = model_output[:, :-1]
        batch_g.ndata["beta"] = model_output[:, -1]
    else:
        batch_g.ndata["model_output"] = model_output
        
    graphs = dgl.unbatch(batch_g)
    
    batch_id = y[:, -1].view(-1)
    df_list = []
    df_hits = []
    for i, g in enumerate(graphs):
        
        mask = batch_id == i
        dic = {}
        dic["graph"] = g
        dic["part_true"] = y[mask]

        
        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        # betas = dic["graph"].ndata["beta"]
        X = dic["graph"].ndata["coords"]
        if ct:

            # labels can start at -1 in which case the 0 is the 'noise'
            labels_ = graphs[i].ndata["model_output"].long() + 1
            map_from = list(np.unique(labels_.detach().cpu()))
            labels = map(lambda x: map_from.index(x), labels_)
            labels = (
                torch.Tensor(list(labels))
                .long()
                .to(dic["graph"].ndata["coords"].device)
            )
            
        else:
            if clustering_mode == "clustering_normal":
                
                clustering1 = get_clustering(betas, X, tbeta=0.5, td=0.007)
                # clustering1 = DPC_custom(X, betas, model_output.device)
                map_from = list(np.unique(clustering1.detach().cpu()))
                cluster_id = map(lambda x: map_from.index(x), clustering1)
                clustering_ordered = (
                    torch.Tensor(list(cluster_id)).long().to(model_output.device)
                )
                
                if torch.unique(clustering1)[0] != -1:
                    clustering = clustering_ordered + 1
                else:
                    clustering = clustering_ordered

                labels = clustering.view(-1).long()
            elif clustering_mode == "dbscan":
                labels = hfdb_obtain_labels(X, betas.device)
        
        pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks, tracks_dict, fileIDs, eventIDs, = match_tracks(labels, dic) 
        
        df_event = generate_tracks_dataframe(fileIDs, eventIDs, pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks, tracks_dict)
        df_list.append(df_event)

        df_hits_event = dataframe_position_labels(labels, dic, X, betas) 
        df_hits.append(df_hits_event)
        
        if len(df_list) > 0:
            df_batch = pd.concat(df_list)
        else:
            df_batch = []
        if store:
            store_at_batch_end(
                path_save, df_batch, local_rank, step, epoch, predict=predict
            )
            
        if len(df_hits) > 0:
            df_batch_hits = pd.concat(df_hits)
        else:
            df_batch_hits = []
        if store:
            
            store_at_batch_end_hits(
                path_save, df_batch_hits, local_rank, step, epoch, predict=predict
            )
            
    
    return df_batch, df_batch_hits
        
def match_tracks(labels, dic):
    
    pids = []
    partIndices = []
    deltaMCs = []
    energies = []
    pTs = []
    thetas = []
    genStatus = []
    numSIhits = [] 
    numCDChits = []
    trackLabels = []
    hitEfficiencies = []
    hitPurities = []
    fakeTrackIndices = []
    fileIDs = []
    eventIDs = []
    
    part_true = dic["part_true"]
    graphInfo = dic["graph"]
    
    fileID = graphInfo.ndata["fileNumber"][0]
    eventID = graphInfo.ndata["eventNumber"][0]
    
    part_keys = [
        "part_theta",    # 0
        "part_phi",      # 1
        "part_m",        # 2
        "part_pid",      # 3
        "part_id",       # 4
        "part_p",        # 5
        "part_p_t",      # 6
        "gen_status",    # 7
        "part_parent",   # 8
        "batch_id"       # 9
    ]
    partInfo = {key: part_true[:, i] for i, key in enumerate(part_keys)}
    
    particle_number_nomap = graphInfo.ndata["particle_number_nomap"]  # particle index

    unique_labels, counts = torch.unique(labels, return_counts=True)
    numHits_tracks = {int(label): int(count) for label, count in sorted(zip(unique_labels, counts), key=lambda x: x[0])}
    
    unique_particles, counts = torch.unique(particle_number_nomap, return_counts=True)
    numHits_particle = {int(p): int(c) for p, c in zip(unique_particles, counts)}
    
    
    # number of silicon and drift hits per particle 
    hit_type = graphInfo.ndata["hit_type"]
    type_hits_particle = {}
    for particle in unique_particles:
        mask = particle_number_nomap == particle
        num_siliconHits = torch.sum(hit_type[mask] == 1).item()
        num_driftHits = torch.sum(hit_type[mask] == 0).item()

        type_hits_particle[int(particle.item())] = {
            "silicon_hits": num_siliconHits,
            "drift_hits": num_driftHits
        }
    
    # dictionary:
    # - each entry is a particle and the content is the number of hits that belong to that particle in each cluster
    particle_label_counts = {}
    for p in unique_particles:
        mask_p = particle_number_nomap == p
        counts_dict = {}
        for l in unique_labels:
            mask_label = labels == l
            count = torch.sum(mask_p & mask_label).item()
            counts_dict[int(l)] = count
        particle_label_counts[int(p)] = counts_dict        
    
    # efficiency and purity 
    efficiency = {}
    purity = {}
    for p in unique_particles:
        efficiency_p = {}
        purity_p = {}
        for l in unique_labels:
            hits_in_label = particle_label_counts[int(p)][int(l)]
            efficiency_p[int(l)] = hits_in_label / numHits_particle[int(p)] if numHits_particle[int(p)] > 0 else 0.0
            purity_p[int(l)] = hits_in_label / numHits_tracks[int(l)] if numHits_tracks[int(l)] > 0 else 0.0
        efficiency[int(p)] = efficiency_p
        purity[int(p)] = purity_p
        
    tracks_dict = {}

    for l in unique_labels:
        track_particles = []
        track_purities = []
        track_efficiencies = []

        for p in unique_particles:
            eff_p_dict = efficiency.get(int(p))
            pur_p_dict = purity.get(int(p))

           
            if eff_p_dict is None or pur_p_dict is None:
                continue

            if int(l) not in eff_p_dict or int(l) not in pur_p_dict:
                continue

            eff = eff_p_dict[int(l)]
            pur = pur_p_dict[int(l)]

            track_particles.append(int(p))
            track_purities.append(pur)
            track_efficiencies.append(eff)

        tracks_dict[int(l)] = {
            "particle_index": track_particles,
            "efficiency": track_efficiencies,
            "purity": track_purities
        }
    
    # check if particle matches the cluster and check which clusters are not assigned
    particle_matches = {}
    labels_matched_set = set()

    for p in unique_particles:
        matched = False
        matched_eff = []
        matched_purity = []
        matched_labels = []
        
        for l in unique_labels:
            eff = efficiency[int(p)][int(l)]
            pur = purity[int(p)][int(l)]
            
            # if eff > 0.5 and pur > 0.5:
            if pur > 0.75:
                matched = True
                matched_eff.append(eff)
                matched_purity.append(pur)
                matched_labels.append(int(l))
                labels_matched_set.add(int(l))
        
        particle_matches[int(p)] = {
            "matched": matched,
            "track": matched_labels,
            "efficiency": matched_eff,
            "purity": matched_purity
        }

        
    # fakeTracks
    labels_not_matched = [int(l) for l in unique_labels if int(l) not in labels_matched_set]
    fakeTrackIndices = labels_not_matched
    
    siliconHits_fakeTracks = []
    driftHits_fakeTracks = []
    for idx, fakeTrack in enumerate(fakeTrackIndices):
        
        mask = labels == fakeTrack
        num_siliconHits = torch.sum(hit_type[mask] == 1).item()
        num_driftHits = torch.sum(hit_type[mask] == 0).item()
        siliconHits_fakeTracks.append(num_siliconHits)
        driftHits_fakeTracks.append(num_driftHits)
    
    
    # particle info - tracks matching
    particle_id = partInfo["part_id"] 
    part_theta = partInfo["part_theta"]
    part_pt = partInfo["part_p_t"]
    part_m = partInfo["part_m"]
    gen_status = partInfo["gen_status"]
    part_p = partInfo["part_p"]
    part_pid = partInfo["part_pid"]
    
    unique_particles_true = torch.unique(particle_id)
    for particle in unique_particles_true:
        
        mask = particle_id == particle
        
        if torch.any(mask):
            
            idx = torch.nonzero(mask, as_tuple=False)[0].item()
            
            pid = part_pid[mask].item()
            theta = part_theta[mask].item()
            gen_Status = gen_status[mask].item()
            p_val = float(part_p[idx].item())   
            m_val = float(part_m[idx].item())  
            pt_val = float(part_pt[idx].item())  
            energy = (p_val**2 + m_val**2) ** 0.5

        pid_int = int(particle.item())

        if pid_int in type_hits_particle:
            numSiliconHits = type_hits_particle[pid_int]["silicon_hits"]
            numDriftHits = type_hits_particle[pid_int]["drift_hits"]

        if pid_int in particle_matches:
            trackLabel = particle_matches[pid_int]["track"]
            hitEfficiency = particle_matches[pid_int]["efficiency"]
            hitPurity = particle_matches[pid_int]["purity"]

        deltaMC = 0
        # deltaMC = ...
        
        pids.append(pid)
        partIndices.append(pid_int)
        energies.append(energy)
        deltaMCs.append(deltaMC)
        pTs.append(pt_val)
        thetas.append(theta)
        genStatus.append(gen_Status)
        numSIhits.append(numSiliconHits)
        numCDChits.append(numDriftHits)
        trackLabels.append(trackLabel)
        hitEfficiencies.append(hitEfficiency)
        hitPurities.append(hitPurity)
        fileIDs.append(fileID)
        eventIDs.append(eventID)
    
    if -1 in particle_number_nomap:
        pids.append(-1)
        partIndices.append(-1)
        energies.append(-1)
        deltaMCs.append(-1)
        pTs.append(-1)
        thetas.append(-1)
        genStatus.append(-1)
        numSIhits.append(type_hits_particle[-1]["silicon_hits"])
        numCDChits.append(type_hits_particle[-1]["drift_hits"])
        trackLabels.append(particle_matches[-1]["track"])
        hitEfficiencies.append(particle_matches[-1]["efficiency"])
        hitPurities.append(particle_matches[-1]["purity"])
        fileIDs.append(fileID)
        eventIDs.append(eventID)

    # print("\n" + "="*80)
    # print(f"CLUSTER SUMMARY  |  File: {int(fileID)}  Event: {int(eventID)}")
    # print("="*80)

    # for l in unique_labels:
    #     cluster_id = int(l)
    #     cluster_data = tracks_dict[cluster_id]
        
    #     print(f"\n  Cluster {cluster_id:>4d}  ({numHits_tracks[cluster_id]} hits total)")
    #     print(f"  {'Particle':>10}  {'Momentum':>10}  {'Purity':>8}  {'Efficiency':>10}")
    #     print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
        
    #     # Collect particles with non-zero contribution to this cluster
    #     contributing = [
    #         (p, pur, eff)
    #         for p, pur, eff in zip(
    #             cluster_data["particle_index"],
    #             cluster_data["purity"],
    #             cluster_data["efficiency"],
    #         )
    #         if pur > 0 or eff > 0
    #     ]
        
    #     if not contributing:
    #         print(f"  {'(none)':>10}")
    #         continue
        
    #     # Sort by purity descending so the dominant particle comes first
    #     contributing.sort(key=lambda x: x[1], reverse=True)
        
    #     for p_idx, pur, eff in contributing:
    #         # Look up momentum for this particle
    #         mask = particle_id == p_idx
    #         if torch.any(mask):
    #             idx = torch.nonzero(mask, as_tuple=False)[0].item()
    #             mom = float(part_p[idx].item())
    #         else:
    #             mom = float("nan")   # background / noise particle (-1)
            
    #         matched_marker = " ✓" if pur > 0.75 else "  "
    #         print(f"  {p_idx:>10d}  {mom:>10.4f}  {pur:>8.4f}  {eff:>10.4f}{matched_marker}")
        
    #     is_fake = cluster_id in fakeTrackIndices
    #     if is_fake:
    #         si  = siliconHits_fakeTracks[fakeTrackIndices.index(cluster_id)]
    #         cdc = driftHits_fakeTracks [fakeTrackIndices.index(cluster_id)]
    #         print(f"  *** FAKE TRACK  (silicon={si}, drift={cdc}) ***")

    # print("\n" + "="*80 + "\n")

    # num_clusters = len(unique_labels)
    # num_real_particles = len([p for p in unique_particles if int(p.item()) != -1])
    # num_fake_tracks = len(fakeTrackIndices)
    # num_matched_particles = sum(1 for p in unique_particles if int(p.item()) != -1 and particle_matches[int(p.item())]["matched"])

    # print(f"\n  SUMMARY")
    # print(f"  {'-'*40}")
    # print(f"  Total clusters        : {num_clusters}")
    # print(f"  Fake tracks           : {num_fake_tracks}")
    # print(f"  Real particles        : {num_real_particles}")
    # print(f"  Matched particles     : {num_matched_particles}")
    # print(f"  Unmatched particles   : {num_real_particles - num_matched_particles}")

    # print("\n" + "="*80 + "\n")

    # sys.exit()
    
    return pids, partIndices, deltaMCs, energies, pTs, thetas, genStatus, numSIhits, numCDChits, trackLabels, hitEfficiencies, hitPurities, fakeTrackIndices, siliconHits_fakeTracks, driftHits_fakeTracks, tracks_dict, fileIDs, eventIDs

def generate_tracks_dataframe(
    fileIDs, 
    eventIDs, 
    pids,
    partIndices,
    deltaMCs,
    energies,
    pTs,
    thetas,
    genStatus,
    numSIhits,
    numCDChits,
    trackLabels,
    hitEfficiencies,
    hitPurities,
    fakeTrackIndices=None,
    siliconHits_fakeTracks=None,
    driftHits_fakeTracks=None,
    tracks_dict = None
):
    """
    Create a pandas DataFrame from the outputs of match_tracks().
    Only uses the lists/tensors returned by match_tracks().

    Parameters
    ----------
    deltaMCs, energies, thetas, genStatus, numSIhits, numCDChits,
    trackLabels, hitEfficiencies, hitPurities : list or torch.Tensor
        Outputs from match_tracks().
    fakeTrackIndices : list or torch.Tensor, optional
        Track indices not matched to any particle.

    Returns
    -------
    pd.DataFrame
        Summary table of particles and tracks.
    """

    # Helper to convert tensors to numpy arrays
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    # Main DataFrame
    df_dict = {
        "fileID" : [to_numpy(fileIDs[0])] * len(eventIDs),
        "eventID" : [to_numpy(eventIDs[0])] * len(eventIDs),
        "partIndex" : to_numpy(partIndices),
        "pid" : to_numpy(pids),
        "energy": to_numpy(energies),
        "pT": to_numpy(pTs),
        "deltaMC": to_numpy(deltaMCs),
        "theta": to_numpy(thetas),
        "genStatus": to_numpy(genStatus),
        "numSIhits": to_numpy(numSIhits),
        "numCDChits": to_numpy(numCDChits),
        "trackLabel": to_numpy(trackLabels),
        "hitEfficiency": to_numpy(hitEfficiencies),
        "hitPurity": to_numpy(hitPurities),
    }

    df = pd.DataFrame(df_dict)

    particle_index_list = []
    efficiency_list = []
    purity_list = []

    for l in fakeTrackIndices:
        
        track_info = tracks_dict.get(int(l))
        
        particle_index_list.append(track_info["particle_index"])
        efficiency_list.append(track_info["efficiency"])
        purity_list.append(track_info["purity"])
        
    if fakeTrackIndices is not None and len(fakeTrackIndices) > 0:
        
        fake_df = pd.DataFrame({
            "fileID" :       [to_numpy(fileIDs[0])] * len(fakeTrackIndices),
            "eventID" :      [to_numpy(eventIDs[0])] * len(fakeTrackIndices),
            "partIndex" :    to_numpy(particle_index_list),
            "pid" :          [None] * len(fakeTrackIndices),
            "energy":        [None] * len(fakeTrackIndices),
            "pT":            [None] * len(fakeTrackIndices),
            "deltaMC":       [None] * len(fakeTrackIndices),
            "theta":         [None] * len(fakeTrackIndices),
            "genStatus":     [None] * len(fakeTrackIndices),
            "numSIhits":     to_numpy(siliconHits_fakeTracks),
            "numCDChits":    to_numpy(driftHits_fakeTracks),
            "trackLabel":    to_numpy(fakeTrackIndices),
            "hitEfficiency": to_numpy(efficiency_list),
            "hitPurity":     to_numpy(purity_list),
        }) 
        df = pd.concat([df, fake_df], ignore_index=True)

    return df

def dataframe_position_labels(labels, dic, X, betas):
    
    graphInfo = dic["graph"]
    
    tensor_map = {
    "fileID": graphInfo.ndata["fileNumber"],
    "eventID": graphInfo.ndata["eventNumber"],
    "pos_x": graphInfo.ndata["pos_hits_xyz"][:, 0],
    "pos_y": graphInfo.ndata["pos_hits_xyz"][:, 1],
    "pos_z": graphInfo.ndata["pos_hits_xyz"][:, 2],
    "hit_type": graphInfo.ndata["hit_type"],
    "clusterID": labels,
    "particle_number": graphInfo.ndata["particle_number_nomap"],
    "particle_number_original": graphInfo.ndata["particle_number_nomap_original"],
    "beta": betas,
    }

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    df_dict = {key: to_numpy(tensor) for key, tensor in tensor_map.items()}

    X_np = to_numpy(X)

    # Add embedding columns dynamically
    for i in range(X_np.shape[1]):
        df_dict[f"emb{i}"] = X_np[:, i]

    # Optional: flatten any (N,1) arrays
    for k, v in df_dict.items():
        if hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] == 1:
            df_dict[k] = v.ravel()

    df = pd.DataFrame(df_dict)
    
    return df

def store_at_batch_end(
    path_save,
    df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + "IDEAtracking.pt"
    )
    if predict:
        df_batch = pd.concat(df_batch)
        df_batch.to_pickle(path_save_)
    
def store_at_batch_end_hits(
    path_save,
    df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + "IDEAtracking_hits.pt"
    )
    if predict:
        df_batch = pd.concat(df_batch)
        df_batch.to_pickle(path_save_)

def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.025, td=0.1):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """

    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
        
        # calculate indices_codpoints again
        indices_condpoints = find_condpoints(betas, unassigned, tbeta)
    return clustering


def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    device = betas.device
    mask_unassigned = torch.zeros(n_points).to(device)
    mask_unassigned[unassigned] = True
    select_condpoints = mask_unassigned.to(bool) * select_condpoints
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints


# """
# Custom Density Peak Clustering (DPC) — no external DPC library required.
 
# Implements the algorithm from:
#     Rodriguez & Laio, "Clustering by fast search and find of density peaks"
#     Science 344, 1492 (2014).
 
# Drop-in replacements for DPC, DPC_custom, and DPC_custom_CLD from the original
# inference script.
# """
 
# import numpy as np
# import torch
 
 
# # ---------------------------------------------------------------------------
# # Core DPC primitives
# # ---------------------------------------------------------------------------
 
# def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
#     """Pairwise Euclidean distance matrix  (N x N)."""
#     diff = X[:, None, :] - X[None, :, :]      # (N, N, D)
#     D = np.sqrt((diff ** 2).sum(axis=-1))      # (N, N)
#     return D
 
 
# def local_density_gaussian(D: np.ndarray, d_c: float) -> np.ndarray:
#     """
#     Gaussian-kernel local density (rho) for each point.
 
#     rho_i = sum_{j != i} exp(-(d_ij / d_c)^2)
#     """
#     N = D.shape[0]
#     mask = D < d_c                                       # (N, N)
#     np.fill_diagonal(mask, False)                        # exclude self
#     rho = np.sum(np.exp(-(D / d_c) ** 2) * mask, axis=1)
#     return rho
 
 
# def local_density_energy_gaussian(
#     D: np.ndarray,
#     d_c: float,
#     energies: np.ndarray,
# ) -> np.ndarray:
#     """
#     Energy-weighted Gaussian-kernel local density.
 
#     rho_i = sum_{j: d_ij < d_c} E_j * exp(-(d_ij / d_c)^2)
#     """
#     mask = D < d_c                                       # (N, N)
#     np.fill_diagonal(mask, False)
#     weights = energies[None, :] * np.exp(-(D / d_c) ** 2)   # (N, N)
#     rho = np.sum(weights * mask, axis=1)
#     return rho
 
 
# def distance_to_higher_density(
#     D: np.ndarray,
#     rho: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     For each point i compute:
#         delta_i   = min distance to any j with rho_j > rho_i
#         nearest_i = index of that nearest higher-density neighbour
 
#     The global density maximum gets delta = max(delta of all others).
#     """
#     N = D.shape[0]
#     order = np.argsort(-rho)          # descending density order
#     delta = np.full(N, np.inf)
#     nearest = np.full(N, -1, dtype=int)
 
#     for rank, i in enumerate(order):
#         # candidates: all j that come earlier in order (rho_j >= rho_i)
#         candidates = order[:rank]
#         if len(candidates) == 0:
#             # global maximum — delta set after the loop
#             continue
#         dists_to_candidates = D[i, candidates]
#         best = np.argmin(dists_to_candidates)
#         delta[i] = dists_to_candidates[best]
#         nearest[i] = candidates[best]
 
#     # Global density max gets delta = largest finite delta
#     global_max = order[0]
#     finite_deltas = delta[np.isfinite(delta)]
#     delta[global_max] = float(np.max(finite_deltas)) if len(finite_deltas) else 0.0
 
#     return delta, nearest
 
 
# def find_cluster_centers(
#     rho: np.ndarray,
#     delta: np.ndarray,
#     rho_min: float,
#     delta_min: float,
# ) -> np.ndarray:
#     """
#     Return indices of cluster centres:
#         points where rho >= rho_min  AND  delta >= delta_min.
#     """
#     mask = (rho >= rho_min) & (delta >= delta_min)
#     return np.where(mask)[0]
 
 
# def assign_cluster_ids(
#     rho: np.ndarray,
#     nearest: np.ndarray,
#     centers: np.ndarray,
# ) -> np.ndarray:
#     """
#     Assign every point to the cluster of its nearest higher-density neighbour,
#     propagating from centres outward (follows the chain toward density peaks).
#     """
#     N = len(rho)
#     ids = np.full(N, -1, dtype=int)
 
#     # Assign centres their own cluster index
#     for cluster_idx, c in enumerate(centers):
#         ids[c] = cluster_idx
 
#     # Process points in decreasing density order; each non-centre inherits
#     # its nearest higher-density neighbour's cluster assignment.
#     order = np.argsort(-rho)
#     for i in order:
#         if ids[i] == -1 and nearest[i] != -1:
#             ids[i] = ids[nearest[i]]
 
#     return ids
 
 
# # ---------------------------------------------------------------------------
# # Core radial-threshold mask: points close enough to their centre are "core"
# # ---------------------------------------------------------------------------
 
# def _apply_core_mask(
#     ids: np.ndarray,
#     centers: np.ndarray,
#     D: np.ndarray,
#     radius: float,
# ) -> np.ndarray:
#     """
#     Return labels array where label = cluster_index+1 for core points and
#     0 (noise) for points farther than *radius* from their assigned centre.
#     """
#     D_no_nan = D.copy()
#     D_no_nan[np.isnan(D_no_nan)] = 0.0
 
#     core_ids = np.full(len(ids), -1, dtype=int)   # -1 → noise (label 0)
#     for cluster_idx, c in enumerate(centers):
#         in_cluster = np.where(ids == cluster_idx)[0]
#         close = in_cluster[D_no_nan[in_cluster, c] < radius]
#         core_ids[close] = cluster_idx
 
#     return core_ids
 
 
# # ---------------------------------------------------------------------------
# # Public API — drop-in replacements
# # ---------------------------------------------------------------------------
 
# def DPC(X: torch.Tensor, device: torch.device) -> torch.Tensor:
#     """
#     Standard DPC (unweighted density).
 
#     Hyperparameters match the original function:
#         d_c=0.20, rho_min=2, delta_min=0.2, core_radius=0.3
#     """
#     d_c = 0.20
#     rho_min = 2.0
#     delta_min = 0.2
#     core_radius = 0.3
 
#     X_np = X.detach().cpu().numpy()
#     D = compute_distance_matrix(X_np)
#     rho = local_density_gaussian(D, d_c)
#     delta, nearest = distance_to_higher_density(D, rho)
#     centers = find_cluster_centers(rho, delta, rho_min, delta_min)
#     ids = assign_cluster_ids(rho, nearest, centers)
#     core_ids = _apply_core_mask(ids, centers, D, core_radius)
 
#     labels = torch.tensor(core_ids, dtype=torch.long, device=device) + 1
#     return labels
 
 
# def DPC_custom(
#     X: torch.Tensor,
#     energies: torch.Tensor,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Energy-weighted DPC.

#     Args:
#         X:        (N, D) float tensor of embedding coordinates
#         energies: (N,)   float tensor of hit energies
#         device:   target device for the returned label tensor
#     """
#     d_c = 0.1
#     rho_min = 0.05
#     delta_min = 0.4
#     core_radius = 1.0

#     X_np = X.detach().cpu().numpy()
#     energies_np = energies.view(-1).detach().cpu().numpy()

#     D = compute_distance_matrix(X_np)
#     rho = local_density_energy_gaussian(D, d_c, energies_np)
#     delta, nearest = distance_to_higher_density(D, rho)
#     centers = find_cluster_centers(rho, delta, rho_min, delta_min)
#     ids = assign_cluster_ids(rho, nearest, centers)
#     core_ids = _apply_core_mask(ids, centers, D, core_radius)

#     labels = torch.tensor(core_ids, dtype=torch.long, device=device) + 1
#     return labels
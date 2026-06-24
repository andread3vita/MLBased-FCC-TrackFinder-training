import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum, scatter_min, scatter_max
from sklearn.preprocessing import StandardScaler
import time
import sys

def get_number_hits(part_idx):
    number_of_hits = scatter_sum(torch.ones_like(part_idx), part_idx.long(), dim=0)
    return number_of_hits[1:].view(-1)

def find_cluster_id(hit_particle_link):
    
    unique_list_particles = list(np.unique(hit_particle_link))
    if np.sum(np.array(unique_list_particles) == -1) > 0:
        
        non_noise_idx = torch.where(hit_particle_link != -1)[0]  #
        noise_idx = torch.where(hit_particle_link == -1)[0]  #
        unique_list_particles1 = torch.unique(hit_particle_link)[1:]
        cluster_id_ = torch.searchsorted(
            unique_list_particles1, hit_particle_link[non_noise_idx], right=False
        )
        cluster_id_small = 1.0 * cluster_id_ + 1
        cluster_id = hit_particle_link.clone()
        cluster_id[non_noise_idx] = cluster_id_small
        cluster_id[noise_idx] = 0

    else:
        unique_list_particles1 = torch.unique(hit_particle_link)
        cluster_id = torch.searchsorted(
            unique_list_particles1, hit_particle_link, right=False
        )
        
        cluster_id = cluster_id + 1 
    return cluster_id, unique_list_particles

def create_inputs_from_table(output, get_vtx):
    
    graph_empty = False
    
    number_hits = np.int32(np.sum(output["mask"][0]))
    number_part = np.int32(np.sum(output["mask"][1]))
    isProducedBySecondary = np.int32(output["mask"][2,0:number_hits])
    hit_particle_link = torch.tensor(output["hits_labels"][0, 0:number_hits])
    features_hits = torch.permute(torch.tensor(output["hits_features"][:, 0:number_hits]), (1, 0))
    
    hit_type = features_hits[:, 9].clone()
    
    hit_type_one_hot = torch.nn.functional.one_hot(hit_type.long(), num_classes=2)
    if get_vtx:
        hit_type_one_hot = hit_type_one_hot
        features_hits = features_hits
        hit_particle_link = hit_particle_link
        isProducedBySecondary = isProducedBySecondary
    else:
        mask_DC = hit_type == 0
        hit_type_one_hot = hit_type_one_hot[mask_DC]
        features_hits = features_hits[mask_DC]
        hit_particle_link = hit_particle_link[mask_DC]
        hit_type = hit_type[mask_DC]
        isProducedBySecondary = isProducedBySecondary[mask_DC]

    unique_list_particles = list(np.unique(hit_particle_link))
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
    features_particles = torch.permute( torch.tensor(output["particle_features"][:, 0:number_part]),(1, 0))
    y_data_graph = features_particles
    
    y_pid = features_particles[:, 4]
    mask_particles = check_unique_particles(unique_list_particles, y_pid)
    y_data_graph = features_particles[mask_particles]
   
    if features_particles.shape[0] >= torch.sum(mask_particles).item():

        cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
        unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)

        features_particles = torch.permute( torch.tensor(output["particle_features"][:, 0:number_part]),(1, 0))
        y_pid = features_particles[:, 4]
        mask_particles = check_unique_particles(unique_list_particles, y_pid)
        y_data_graph = features_particles[mask_particles]
        
        assert len(y_data_graph) == len(unique_list_particles)
    else:
        graph_empty = True
    
    if graph_empty:
        return [None]
    else:
        result = [
            y_data_graph,
            hit_type_one_hot,
            cluster_id,
            hit_particle_link,
            features_hits,
            hit_type,
            isProducedBySecondary
        ]
        return result

def check_unique_particles(unique_list_particles, y_id):
    mask = torch.zeros_like(y_id)
    for i in range(0, len(unique_list_particles)):
        id_u = unique_list_particles[i]
        if torch.sum(y_id == id_u) > 0:
            mask = mask + (y_id == id_u)
    return mask.to(bool)

def create_graph_tracking_global(output, fileID, eventID, get_vtx=False, vector=False, overlay=False):
    
    graph_empty = False
    result = create_inputs_from_table(output, get_vtx)
    
    if len(result) == 1:
        graph_empty = True
    else:
        (
            y_data_graph,
            hit_type_one_hot,
            cluster_id,
            hit_particle_link,
            features_hits,
            hit_type,
            isProducedBySecondary
            
        ) = result
        
        if not overlay:
            
            remove_lowEnergyParticle = False
            remove_secondary = False
            flag_secondary = True
            
            if remove_lowEnergyParticle:

                # REMOVE LOOPERS
                # Remove loopers from the list of hits and the list of particles
                mask_not_lowEnergy, mask_particles = remove_lowEnergyParticles(
                    hit_particle_link, y_data_graph, features_hits[:, 3:6], cluster_id
                )
                hit_type_one_hot = hit_type_one_hot[mask_not_lowEnergy]
                cluster_id = cluster_id[mask_not_lowEnergy]
                hit_particle_link = hit_particle_link[mask_not_lowEnergy]
                features_hits = features_hits[mask_not_lowEnergy]
                hit_type = hit_type[mask_not_lowEnergy]
                
                y_data_graph = y_data_graph[mask_particles]
                
                # Compute the cluster id
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)    
            
            if remove_secondary:

                # REMOVE SECONDARY
                # Remove loopers from the list of hits and the list of particles
                mask_not_garbage, mask_particles = create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, 3)
                hit_type_one_hot = hit_type_one_hot[mask_garbage]
                cluster_id = cluster_id[mask_garbage]
                hit_particle_link = hit_particle_link[mask_garbage]
                features_hits = features_hits[mask_garbage]
                hit_type = hit_type[mask_garbage]
                
                y_data_graph = y_data_graph[mask_particles]
                
                # Compute the cluster id
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)    
            
            if flag_secondary:
                
                # FLAG SECONDARY
                original_particle_link = hit_particle_link.clone()
                mask_not_garbage, mask_particles = create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, 3)
                hit_particle_link[~mask_not_garbage] = -1
                y_data_graph = y_data_graph[mask_particles]
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)   
            
        else:
            
            mask_not_loopers, mask_particles = remove_loopers_overlay(
                hit_particle_link, y_data_graph, features_hits[:, 3:6], cluster_id
            )

            hit_type_one_hot = hit_type_one_hot[mask_not_loopers]
            cluster_id = cluster_id[mask_not_loopers]
            hit_particle_link = hit_particle_link[mask_not_loopers]
            features_hits = features_hits[mask_not_loopers]
            hit_type = hit_type[mask_not_loopers]
            y_data_graph = y_data_graph[mask_particles]

            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
            
            mask_loopers, mask_particles = create_noise_label(
            hit_particle_link, y_data_graph, cluster_id, True, features_hits[:,-1]
            )
            hit_particle_link[mask_loopers] = -1
            y_data_graph = y_data_graph[mask_particles]
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

        if hit_type_one_hot.shape[0] > 0:
            
            mask_dc = hit_type == 0
            mask_vtx = hit_type == 1
            number_of_vtx = torch.sum(mask_vtx)
            number_of_dc = torch.sum(mask_dc)
            g = dgl.DGLGraph()
            if vector:
                g.add_nodes(number_of_vtx + number_of_dc)
            else:
                g.add_nodes(number_of_vtx + number_of_dc * 2)

            left_right_pos = features_hits[:, 3:9][mask_dc]
            left_post = left_right_pos[:, 0:3]
            right_post = left_right_pos[:, 3:]
            vector_like_data = vector
            
            isProducedBySecondary = torch.tensor(isProducedBySecondary)
            
            if get_vtx:
                if vector_like_data:
                    particle_number = torch.cat(
                        (cluster_id[mask_vtx], cluster_id[mask_dc]), dim=0
                    )
                    particle_number_nomap = torch.cat(
                        (
                            hit_particle_link[mask_vtx],
                            hit_particle_link[mask_dc],
                        ),
                        dim=0,
                    )
                    
                    particle_number_nomap_original = torch.cat(
                        (
                            original_particle_link[mask_vtx],
                            original_particle_link[mask_dc],
                        ),
                        dim=0,
                    )
                    
                    pos_xyz = torch.cat(
                        (features_hits[:, 0:3][mask_vtx], left_post), dim=0
                    )

                    # pos_xyz = torch.cat(
                    #     (features_hits[:, 0:3][mask_vtx], features_hits[:, 0:3][mask_dc]), dim=0
                    # )

                    # is_overlay = torch.cat(
                    #     (features_hits[:,-1][mask_vtx].view(-1), features_hits[:,-1][mask_dc].view(-1)), dim=0
                    # )

                    vector_data = torch.cat(
                        (0 * features_hits[:, 0:3][mask_vtx], right_post - left_post),
                        dim=0,
                    )

                    hit_type_all = torch.cat(
                        (hit_type[mask_vtx], hit_type[mask_dc]), dim=0
                    )

                    cellid = torch.cat(
                        (
                            features_hits[:, -1][mask_vtx].view(-1, 1),
                            features_hits[:, -1][mask_dc].view(-1, 1),
                        ),
                        dim=0,
                    )

                    produced_from_secondary_ = torch.cat(
                        (
                            isProducedBySecondary[mask_vtx].view(-1, 1),
                            isProducedBySecondary[mask_dc].view(-1, 1),
                        ),
                        dim=0,
                    )
                    
                else:

                    particle_number = torch.cat(
                        (
                            cluster_id[mask_vtx],
                            cluster_id[mask_dc],
                            cluster_id[mask_dc],
                        ),
                        dim=0,
                    )
                    particle_number_nomap = torch.cat(
                        (
                            hit_particle_link[mask_vtx],
                            hit_particle_link[mask_dc],
                            hit_particle_link[mask_dc],
                        ),
                        dim=0,
                    )
                    
                    particle_number_nomap_original = torch.cat(
                        (
                            original_particle_link[mask_vtx],
                            original_particle_link[mask_dc],
                            original_particle_link[mask_dc],
                        ),
                        dim=0,
                    )
                
                    pos_xyz = torch.cat(
                        (features_hits[:, 0:3][mask_vtx], left_post, right_post), dim=0
                    )
                    hit_type_all = torch.cat(
                        (hit_type[mask_vtx], hit_type[mask_dc], hit_type[mask_dc]),
                        dim=0,
                    )
                    cellid = torch.cat(
                        (
                            features_hits[:, -1][mask_vtx].view(-1, 1),
                            features_hits[:, -1][mask_dc].view(-1, 1),
                            features_hits[:, -1][mask_dc].view(-1, 1),
                        ),
                        dim=0,
                    )
                    produced_from_secondary_ = torch.cat(
                        (
                            isProducedBySecondary[mask_vtx].view(-1, 1),
                            isProducedBySecondary[mask_dc].view(-1, 1),
                            isProducedBySecondary[mask_dc].view(-1, 1),
                        ),
                        dim=0,
                    )
            
            else:

                particle_number = torch.cat((cluster_id, cluster_id), dim=0)
                particle_number_nomap = torch.cat(
                    (hit_particle_link, hit_particle_link), dim=0
                )
                particle_number_nomap_original = torch.cat(
                    (original_particle_link, original_particle_link), dim=0
                )
                pos_xyz = torch.cat((left_post, right_post), dim=0)
                hit_type_all = torch.cat((hit_type, hit_type), dim=0)
                
            if vector_like_data:
                g.ndata["vector"] = vector_data
            
            g.ndata["fileNumber"] = torch.tensor([fileID] * len(hit_type_all))
            g.ndata["eventNumber"] = torch.tensor([eventID] * len(hit_type_all))
            g.ndata["hit_type"] = hit_type_all
            g.ndata["particle_number"] = particle_number.to(dtype=torch.int64)              # clusterID
            g.ndata["particle_number_nomap"] = particle_number_nomap                        # original particle number with -1 for noise (not mapped to clusterID)
            g.ndata["particle_number_nomap_original"] = particle_number_nomap_original      # original particle number (not mapped to clusterID)
            g.ndata["pos_hits_xyz"] = pos_xyz
            g.ndata["cellid"] = cellid
            # g.ndata["is_overlay"] = is_overlay
            g.ndata["isSecondary"] = produced_from_secondary_
            
            if len(y_data_graph) < 1:
                graph_empty = True
                
            if features_hits.shape[0] < 10:
                graph_empty = True
        else:
            graph_empty = True
            
    if graph_empty:
        g = 0
        y_data_graph = 0
   
    return [g, y_data_graph], graph_empty

def remove_loopers_overlay(hit_particle_link, y, coord, cluster_id):
    unique_p_numbers = torch.unique(hit_particle_link)
    cluster_id_unique = torch.unique(cluster_id)
    
    # remove particles with a couple hits
    number_of_hits = get_number_hits(cluster_id)
    mask_hits = number_of_hits < 5

    mask_all = mask_hits.view(-1)
    list_remove = unique_p_numbers[mask_all.view(-1)]
    
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask
    else:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    return ~mask.to(bool), ~mask_particles.to(bool)

def remove_lowEnergyParticles(hit_particle_link, y, coord, cluster_id):
    
    unique_p_numbers = torch.unique(hit_particle_link)
    cluster_id_unique = torch.unique(cluster_id)
    
    min_x = scatter_min(coord[:, 0], cluster_id.long() - 1)[0]
    min_z = scatter_min(coord[:, 2], cluster_id.long() - 1)[0]
    min_y = scatter_min(coord[:, 1], cluster_id.long() - 1)[0]
    max_x = scatter_max(coord[:, 0], cluster_id.long() - 1)[0]
    max_z = scatter_max(coord[:, 2], cluster_id.long() - 1)[0]
    max_y = scatter_max(coord[:, 1], cluster_id.long() - 1)[0]
    diff_x = torch.abs(max_x - min_x)
    diff_z = torch.abs(max_z - min_z)
    diff_y = torch.abs(max_y - min_y)
    
    mask_x = diff_x > 1600
    mask_z = diff_z > 2800
    mask_y = diff_y > 1600
    
    mask_p = mask_x + mask_z + mask_y
    
    # remove particles with a couple hits
    number_of_hits = get_number_hits(cluster_id)
    mask_hits = number_of_hits < 5

    mask_all = mask_hits.view(-1) + mask_p.view(-1)
    list_remove = unique_p_numbers[mask_all.view(-1)]
    
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask
    else:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    return ~mask.to(bool), ~mask_particles.to(bool)

def create_noise_label(hit_particle_link, y, cluster_id, overlay=False,overlay_flag=None):
    """
    Created a label to each node in the graph to determine if it is noise 
    Hits are considered as noise if:
    - They belong to an MC that left no more than 4 hits (mask_hits)
    - The particle has p below x, currently it is set to 0 so not condition on this case (mask_p)
    - The hit is overlaid background
    #TODO overlay hits could leave a track (there can be more than a couple hits for a given particle, for now we don't ask to reconstruc these but it might make our alg worse)

    Args:
        hit_particle_link (torch Tensor): particle the nodes belong to
        y (torch Tensor): particle features
        cluster_id (torch Tensor): particle the node belongs to from 1,N (no gaps)
        overlay (bool): is there background overlay in the data
        overlay_flag (torch Tensor): which hits are background
    Returns:
        mask (torch bool Tensor): which hits are noise
        mask_particles: which particles should be removed 
    """
    unique_p_numbers = torch.unique(hit_particle_link)

    number_of_overlay = scatter_sum(overlay_flag.view(-1), cluster_id.long(), dim=0)[1:].view(-1)
    mask_overlay = number_of_overlay>0
    mask_all =  mask_overlay.view(-1)

    list_remove = unique_p_numbers[mask_all.view(-1)]

    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask
    else:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    return mask.to(bool), ~mask_particles.to(bool)

def create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, minNumHits):
    """
    Create masks for hits and particles to remove noise hits from secondary particles.

    Args:
        hit_particle_link (torch.Tensor): Tensor of particle IDs for each hit, shape (N_hits,)
        isProducedBySecondary (torch.Tensor or np.ndarray): Boolean/0-1 array indicating secondary hits

    Returns:
        mask_hits (torch.BoolTensor): True for hits that are noise (to remove)
        mask_particles (torch.BoolTensor): True for particles to keep (signal)
    """
    
    unique_p_numbers = torch.unique(hit_particle_link)    
    number_of_hits = get_number_hits(cluster_id)
    mask_hits = number_of_hits < minNumHits
    list_remove = unique_p_numbers[mask_hits.view(-1)]
    
    mask_noise_hit = (torch.from_numpy(isProducedBySecondary) == 1)
    for p in unique_p_numbers:
        hits_of_p = hit_particle_link == p
        if mask_noise_hit[hits_of_p].all():
            list_remove = torch.cat([list_remove, p.view(1)])
        

    if len(list_remove) > 0:
        mask = mask_noise_hit
        
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask
    else:
        mask = mask_noise_hit
        
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    
    return ~mask.to(bool), ~mask_particles.to(bool)

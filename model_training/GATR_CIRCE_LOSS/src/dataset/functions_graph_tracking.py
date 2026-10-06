import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum, scatter_min, scatter_max
from sklearn.preprocessing import StandardScaler
import time
import sys

from src.utils.detector_features import (
    DEFAULT_LAYERS_PER_SUPERLAYER,
    build_detector_scalars,
)

def get_number_hits(part_idx):
    number_of_hits = scatter_sum(torch.ones_like(part_idx), part_idx.long(), dim=0)
    return number_of_hits[1:].view(-1)


def _signal_particle_ids(hit_particle_link):
    """Return sorted original particle IDs, excluding the noise sentinel."""
    hit_particle_link = torch.as_tensor(hit_particle_link, dtype=torch.int64)
    return torch.unique(
        hit_particle_link[hit_particle_link != -1], sorted=True
    )


def find_cluster_id(hit_particle_link):
    """Map original particle IDs to consecutive labels; zero is always noise."""
    hit_particle_link = torch.as_tensor(hit_particle_link, dtype=torch.int64)
    signal_ids = _signal_particle_ids(hit_particle_link)
    cluster_id = torch.zeros_like(hit_particle_link)
    signal_mask = hit_particle_link != -1
    if signal_ids.numel() > 0:
        cluster_id[signal_mask] = (
            torch.searchsorted(signal_ids, hit_particle_link[signal_mask]) + 1
        )
    return cluster_id, signal_ids


def _aligned_particle_features(
    features_particles, signal_ids, file_id=None, event_id=None,
    allow_empty=False, tolerate_malformed=False,
):
    """Select and order truth rows to match the mapped cluster-ID ordering.

    In the legacy path every particle referenced by a signal hit must occur
    exactly once in the particle table, otherwise the event is skipped. The
    shared comparison path is deliberately tolerant: CIRCE does not reject an
    event because optional particle metadata is incomplete, so GATr keeps the
    event and retains every available metadata row for a referenced particle.
    """
    signal_ids = torch.as_tensor(signal_ids, dtype=torch.int64)
    if signal_ids.numel() == 0:
        return features_particles[:0] if allow_empty else None
    if features_particles.ndim != 2 or features_particles.shape[1] <= 4:
        if tolerate_malformed:
            print(
                "WARNING: preserving shared event with no usable particle-ID "
                f"column (file={file_id}, event={event_id}); particle metadata "
                "will be empty",
                flush=True,
            )
            if features_particles.ndim == 2:
                return features_particles[:0]
            return torch.empty((0, 0), dtype=torch.float32)
        print(
            "WARNING: skipping malformed event with no particle-ID column "
            f"(file={file_id}, event={event_id})",
            flush=True,
        )
        return None

    raw_particle_ids = features_particles[:, 4]
    valid_ids = torch.isfinite(raw_particle_ids) & (
        raw_particle_ids == torch.round(raw_particle_ids)
    )
    if not bool(valid_ids.all()):
        if tolerate_malformed:
            print(
                "WARNING: preserving shared event with non-finite or "
                "non-integral truth particle IDs "
                f"(file={file_id}, event={event_id}); invalid metadata rows "
                "will be ignored",
                flush=True,
            )
            features_particles = features_particles[valid_ids]
            raw_particle_ids = raw_particle_ids[valid_ids]
            if features_particles.shape[0] == 0:
                return features_particles
        else:
            print(
                "WARNING: skipping event with non-finite or non-integral truth "
                f"particle IDs (file={file_id}, event={event_id})",
                flush=True,
            )
            return None

    particle_ids = raw_particle_ids.to(torch.int64)
    table_ids, table_counts = torch.unique(
        particle_ids, sorted=True, return_counts=True
    )
    if table_ids.numel() == 0:
        if tolerate_malformed:
            print(
                "WARNING: preserving shared event with signal hits but no "
                f"truth particle metadata (file={file_id}, event={event_id})",
                flush=True,
            )
            return features_particles[:0]
        print(
            "WARNING: skipping event with signal hits but no truth particles "
            f"(file={file_id}, event={event_id})",
            flush=True,
        )
        return None
    positions = torch.searchsorted(table_ids, signal_ids)
    safe_positions = positions.clamp(max=table_ids.numel() - 1)
    present = (positions < table_ids.numel()) & (
        table_ids[safe_positions] == signal_ids
    )
    match_counts = torch.zeros_like(signal_ids)
    match_counts[present] = table_counts[safe_positions[present]]
    valid_matches = match_counts == 1
    if not bool(valid_matches.all()):
        missing_ids = signal_ids[match_counts == 0].tolist()
        duplicate_ids = signal_ids[match_counts > 1].tolist()
        if tolerate_malformed:
            print(
                "WARNING: preserving shared event with inconsistent hit/truth "
                f"particle IDs: missing={missing_ids}, "
                f"duplicates={duplicate_ids}, file={file_id}, event={event_id}; "
                "available particle metadata rows will be retained",
                flush=True,
            )
            # CIRCE builds its particle-info mapping from every stored row and
            # naturally tolerates missing or repeated IDs. Preserve the same
            # available rows here; the shared launcher disables the auxiliary
            # helix regression, so these rows are validation metadata only.
            return features_particles[torch.isin(particle_ids, signal_ids)]
        print(
            "WARNING: skipping event with inconsistent hit/truth particle IDs: "
            f"missing={missing_ids}, duplicates={duplicate_ids}, "
            f"file={file_id}, event={event_id}",
            flush=True,
        )
        return None

    sorted_particle_ids, particle_order = torch.sort(particle_ids)
    row_positions = torch.searchsorted(sorted_particle_ids, signal_ids)
    return features_particles[particle_order[row_positions]]

def create_inputs_from_table(
    output, get_vtx, overlay=False, file_id=None, event_id=None,
    allow_empty_truth=False, preserve_nonfinite_hits=False,
):

    graph_empty = False

    number_hits = np.int32(np.sum(output["mask"][0]))
    number_part = np.int32(np.sum(output["mask"][1]))

    isProducedBySecondary = np.int32(
        output["mask"][2, 0:number_hits]
    )

    # The overlay label is data, while ``graph_config.overlay`` controls how
    # that label is used. Always retain the real flag in the graph even when
    # overlay filtering is disabled. Older non-overlay configs without this
    # sixth mask row remain usable, but enabling filtering requires the label.
    if output["mask"].shape[0] > 5:
        is_overlay = np.int32(
            output["mask"][5, 0:number_hits]
        )
    elif overlay:
        raise ValueError(
            "graph_config.overlay is enabled, but the data config does not "
            "include the per-hit 'overlay' field as mask row 5"
        )
    else:
        is_overlay = np.zeros(number_hits, dtype=np.int32)

    # Particle links are discrete identifiers, not floating-point model inputs.
    # Keep them integer-valued independently of the source Parquet dtype.
    hit_particle_link = torch.as_tensor(
        output["hits_labels"][0, 0:number_hits], dtype=torch.int64
    )

    features_hits = torch.permute(
        torch.tensor(
            output["hits_features"][:, 0:number_hits]
        ),
        (1, 0),
    )

    # The legacy GATr loader removes malformed hits. The shared comparison path
    # preserves them because CIRCE feeds them forward and lets the synchronized
    # non-finite-batch policy skip the complete batch on every DDP rank.
    finite_hits = torch.isfinite(features_hits).all(dim=1) & torch.isfinite(
        hit_particle_link
    )
    if not bool(finite_hits.all()) and not preserve_nonfinite_hits:
        removed_hits = int((~finite_hits).sum())
        print(
            "WARNING: dropping "
            f"{removed_hits}/{len(finite_hits)} non-finite hits from "
            f"file={file_id}, event={event_id}",
            flush=True,
        )
        finite_hits_numpy = finite_hits.cpu().numpy()
        features_hits = features_hits[finite_hits]
        hit_particle_link = hit_particle_link[finite_hits]
        isProducedBySecondary = isProducedBySecondary[finite_hits_numpy]
        is_overlay = is_overlay[finite_hits_numpy]

    if features_hits.shape[0] == 0:
        return [None]

    hit_type = features_hits[:, 9].clone()

    if preserve_nonfinite_hits:
        # This tensor is only an intermediate graph-construction aid. CIRCE
        # selects hit_type 0/1 and leaves any malformed type out of its feature
        # view, so avoid failing in one_hot before the same masks are applied.
        hit_type_one_hot = torch.zeros(
            hit_type.shape[0], 2, dtype=torch.int64, device=hit_type.device
        )
        valid_hit_type = (hit_type == 0) | (hit_type == 1)
        hit_type_one_hot[valid_hit_type] = torch.nn.functional.one_hot(
            hit_type[valid_hit_type].long(), num_classes=2
        )
    else:
        hit_type_one_hot = torch.nn.functional.one_hot(
            hit_type.long(),
            num_classes=2,
        )

    if get_vtx:
        hit_type_one_hot = hit_type_one_hot
        features_hits = features_hits
        hit_particle_link = hit_particle_link
        isProducedBySecondary = isProducedBySecondary
        is_overlay = is_overlay

    else:
        mask_DC = hit_type == 0

        hit_type_one_hot = hit_type_one_hot[mask_DC]
        features_hits = features_hits[mask_DC]
        hit_particle_link = hit_particle_link[mask_DC]
        hit_type = hit_type[mask_DC]

        isProducedBySecondary = isProducedBySecondary[mask_DC]
        is_overlay = is_overlay[mask_DC]

    features_particles = torch.permute(
        torch.tensor(
            output["particle_features"][:, 0:number_part]
        ),
        (1, 0),
    )

    cluster_id, signal_ids = find_cluster_id(hit_particle_link)
    y_data_graph = _aligned_particle_features(
        features_particles,
        signal_ids,
        file_id=file_id,
        event_id=event_id,
        allow_empty=allow_empty_truth,
        tolerate_malformed=allow_empty_truth,
    )
    if y_data_graph is None:
        return [None]

    result = [
        y_data_graph,
        hit_type_one_hot,
        cluster_id,
        hit_particle_link,
        features_hits,
        hit_type,
        isProducedBySecondary,
        is_overlay,
    ]

    return result

def check_unique_particles(unique_list_particles, y_id):
    # Vectorized membership avoids a Python loop over every truth particle for
    # every event during data loading.
    return torch.isin(y_id, unique_list_particles).to(bool)

def create_graph_tracking_global(
    output,
    fileID,
    eventID,
    get_vtx=False,
    vector=False,
    overlay=False,
    layers_per_superlayer=DEFAULT_LAYERS_PER_SUPERLAYER,
    keep_all_events=False,
):
    
    graph_empty = False
    result = create_inputs_from_table(
        output,
        get_vtx,
        overlay=overlay,
        file_id=fileID,
        event_id=eventID,
        allow_empty_truth=keep_all_events,
        preserve_nonfinite_hits=keep_all_events,
    )
    
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
            isProducedBySecondary,
            isOverlay
            
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
                
                if not keep_all_events:
                    y_data_graph = y_data_graph[mask_particles]
                
                # Compute the cluster id
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)    
            
            if remove_secondary:

                # REMOVE SECONDARY
                # Remove loopers from the list of hits and the list of particles
                mask_not_garbage, mask_particles = create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, 3)
                hit_type_one_hot = hit_type_one_hot[mask_not_garbage]
                cluster_id = cluster_id[mask_not_garbage]
                hit_particle_link = hit_particle_link[mask_not_garbage]
                features_hits = features_hits[mask_not_garbage]
                hit_type = hit_type[mask_not_garbage]
                numpy_keep = mask_not_garbage.cpu().numpy()
                isProducedBySecondary = isProducedBySecondary[numpy_keep]
                isOverlay = isOverlay[numpy_keep]
                
                if not keep_all_events:
                    y_data_graph = y_data_graph[mask_particles]
                
                # Compute the cluster id
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)    
            
            if flag_secondary:
                
                # FLAG SECONDARY
                original_particle_link = hit_particle_link.clone()
                mask_not_garbage, mask_particles = create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, 3)
                hit_particle_link[~mask_not_garbage] = -1
                if not keep_all_events:
                    y_data_graph = y_data_graph[mask_particles]
                cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

        else:

            remove_lowEnergyParticle = False
            remove_secondary = False
            flag_secondary_and_overlay = True

            if flag_secondary_and_overlay:

                # FLAG SECONDARY
                original_particle_link = hit_particle_link.clone()
                mask_not_garbage, mask_particles = create_garbage_label_overlay(hit_particle_link, isProducedBySecondary, isOverlay, cluster_id, 3)
                hit_particle_link[~mask_not_garbage] = -1
                if not keep_all_events:
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
            detector_scalars = build_detector_scalars(
                features_hits[:, 10:], layers_per_superlayer
            )
            vector_like_data = vector
            
            isProducedBySecondary = torch.tensor(isProducedBySecondary)
            isOverlay = torch.tensor(isOverlay)

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
                    
                    # Treat the drift ambiguity symmetrically: the midpoint is
                    # the measured wire position and the displacement encodes
                    # both left/right candidates without privileging either.
                    dc_midpoint = 0.5 * (left_post + right_post)
                    pos_xyz = torch.cat(
                        (features_hits[:, 0:3][mask_vtx], dc_midpoint), dim=0
                    )

                    is_overlay = torch.cat(
                        (
                            isOverlay[mask_vtx].view(-1, 1),
                            isOverlay[mask_dc].view(-1, 1),
                        ),
                        dim=0,
                    )

                    vector_data = torch.cat(
                        (
                            0 * features_hits[:, 0:3][mask_vtx],
                            0.5 * (right_post - left_post),
                        ),
                        dim=0,
                    )

                    scalar_data = torch.cat(
                        (
                            detector_scalars[mask_vtx],
                            detector_scalars[mask_dc],
                        ),
                        dim=0,
                    )

                    hit_type_all = torch.cat(
                        (hit_type[mask_vtx], hit_type[mask_dc]), dim=0
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
                    scalar_data = torch.cat(
                        (
                            detector_scalars[mask_vtx],
                            detector_scalars[mask_dc],
                            detector_scalars[mask_dc],
                        ),
                        dim=0,
                    )
                    hit_type_all = torch.cat(
                        (hit_type[mask_vtx], hit_type[mask_dc], hit_type[mask_dc]),
                        dim=0,
                    )

                    is_overlay = torch.cat(
                        (
                            isOverlay[mask_vtx].view(-1, 1),
                            isOverlay[mask_dc].view(-1, 1),
                            isOverlay[mask_dc].view(-1, 1)
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
                scalar_data = torch.cat(
                    (detector_scalars, detector_scalars), dim=0
                )
                
            if vector_like_data:
                g.ndata["vector"] = vector_data
            if scalar_data.shape[1] > 0:
                g.ndata["scalar_features"] = scalar_data
            
            g.ndata["fileNumber"] = torch.tensor([fileID] * len(hit_type_all))
            g.ndata["eventNumber"] = torch.tensor([eventID] * len(hit_type_all))
            g.ndata["hit_type"] = hit_type_all
            g.ndata["particle_number"] = particle_number.to(dtype=torch.int64)              # clusterID
            g.ndata["particle_number_nomap"] = particle_number_nomap                        # original particle number with -1 for noise (not mapped to clusterID)
            g.ndata["particle_number_nomap_original"] = particle_number_nomap_original      # original particle number (not mapped to clusterID)
            g.ndata["pos_hits_xyz"] = pos_xyz
            g.ndata["is_overlay"] = is_overlay
            g.ndata["isSecondary"] = produced_from_secondary_
            
            if len(y_data_graph) < 1 and not keep_all_events:
                graph_empty = True
                
            if features_hits.shape[0] < 10 and not keep_all_events:
                graph_empty = True
        else:
            graph_empty = True

        # Re-establish the same invariant after secondary/overlay filtering:
        # mapped cluster k+1 and truth row k refer to the same original ID.
        if not graph_empty:
            cluster_id, signal_ids = find_cluster_id(hit_particle_link)
            y_data_graph = _aligned_particle_features(
                y_data_graph,
                signal_ids,
                file_id=fileID,
                event_id=eventID,
                allow_empty=keep_all_events,
                tolerate_malformed=keep_all_events,
            )
            if y_data_graph is None:
                graph_empty = True
            
    if graph_empty:
        g = 0
        y_data_graph = 0
   
    return [g, y_data_graph], graph_empty

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

def _create_garbage_masks(hit_particle_link, flagged_hits, minNumHits):
    """Build hit and truth-row masks in canonical signal-particle ID order."""
    hit_particle_link = torch.as_tensor(hit_particle_link, dtype=torch.int64)
    flagged_hits = torch.as_tensor(flagged_hits, dtype=torch.bool)
    if flagged_hits.shape != hit_particle_link.shape:
        raise ValueError("flagged_hits and hit_particle_link must have identical shapes")

    signal_ids = _signal_particle_ids(hit_particle_link)
    if signal_ids.numel() == 0:
        return ~flagged_hits, torch.zeros(0, dtype=torch.bool)

    signal_links = hit_particle_link[hit_particle_link != -1]
    counted_ids, hit_counts = torch.unique(
        signal_links, sorted=True, return_counts=True
    )
    if not torch.equal(counted_ids, signal_ids):
        raise RuntimeError("internal particle-ID counting inconsistency")

    signal_mask = hit_particle_link != -1
    signal_inverse = torch.searchsorted(signal_ids, hit_particle_link[signal_mask])
    flagged_counts = scatter_sum(
        flagged_hits[signal_mask].to(torch.int64),
        signal_inverse,
        dim=0,
        dim_size=signal_ids.numel(),
    )
    remove_particle = (hit_counts < minNumHits) | (flagged_counts == hit_counts)

    remove_ids = signal_ids[remove_particle]
    remove_hits = flagged_hits.clone()
    if remove_ids.numel() > 0:
        remove_hits |= torch.isin(hit_particle_link, remove_ids)
    return ~remove_hits, ~remove_particle


def create_garbage_label(hit_particle_link, isProducedBySecondary, cluster_id, minNumHits):
    """
    Create masks for hits and particles to remove noise hits from secondary particles.

    Args:
        hit_particle_link (torch.Tensor): Tensor of particle IDs for each hit, shape (N_hits,)
        isProducedBySecondary (torch.Tensor or np.ndarray): Boolean/0-1 array indicating secondary hits

    Returns:
        mask_hits (torch.BoolTensor): True for hits to retain
        mask_particles (torch.BoolTensor): True for particles to keep (signal)
    """
    
    del cluster_id  # Counts are derived from original IDs, excluding noise.
    secondary_hits = torch.as_tensor(isProducedBySecondary) == 1
    return _create_garbage_masks(
        hit_particle_link, secondary_hits, minNumHits
    )

def create_garbage_label_overlay(
    hit_particle_link,
    isProducedBySecondary,
    isOverlay,
    cluster_id,
    minNumHits,
):
    """
    Create masks for hits and particles to remove noise hits from secondary particles.

    Args:
        hit_particle_link (torch.Tensor): Tensor of particle IDs for each hit, shape (N_hits,)
        isProducedBySecondary (torch.Tensor or np.ndarray): Boolean/0-1 array indicating secondary hits
        isOverlay (torch.Tensor or np.ndarray): Boolean/0-1 array indicating overlaied hits

    Returns:
        mask_hits (torch.BoolTensor): True for hits to retain
        mask_particles (torch.BoolTensor): True for particles to keep (signal)
    """
    
    del cluster_id  # Counts are derived from original IDs, excluding noise.
    mask_noise_overlay_hit = (
        (torch.as_tensor(isProducedBySecondary) == 1)
        | (torch.as_tensor(isOverlay) == 1)
    )
    return _create_garbage_masks(
        hit_particle_link, mask_noise_overlay_hit, minNumHits
    )

#!/usr/bin/env python
"""
Convert edm4hep digitized ROOT files to Parquet for local analysis with Polars.
Run this ON lxplus where podio/edm4hep are available.

Extracts all geometric info needed for C-GATr (CGA representation):
  - Drift chamber: wire position, drift distance (circle radius), wire direction
  - Vertex/Silicon: hit positions (points)
  - MC truth: particle-to-hit links, particle properties

Usage:
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
    pip install --user pyarrow
    python edm4hep_to_parquet.py --input_dir data_raw_train/idea_v3_1_nobackground/Pythia/Zcard \
                                  --output_dir data_parquet_train
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from podio import root_io


def extract_dc_hits(event, metadata):
    """Extract drift chamber hits with full geometric info for CGA circles."""
    dc_links = event.get("DCH_DigiSimAssociationCollection")
    dc_digis = event.get("DCH_DigiCollection")

    hits = {
        # Truth hit position
        "hit_x": [], "hit_y": [], "hit_z": [],
        # Truth momentum at hit
        "hit_px": [], "hit_py": [], "hit_pz": [],
        # Wire geometry (defines the circle axis)
        "wire_x": [], "wire_y": [], "wire_z": [],
        "wire_azimuthal_angle": [], "wire_stereo_angle": [],
        # Drift distance = circle radius
        "drift_distance": [],
        # Left-right ambiguity positions (precomputed)
        "left_x": [], "left_y": [], "left_z": [],
        "right_x": [], "right_y": [], "right_z": [],
        # Hit metadata
        "edep": [], "time": [], "path_length": [],
        # "cell_id": [], 
        "cluster_count": [],
        "produced_by_secondary": [],
        # MC link
        "mc_index": [],
    }

    import dd4hep as dd4hepModule
    from ROOT import dd4hep
    cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)

    for idx, dc_link in enumerate(dc_links):
        sim_hit = dc_link.getTo()
        digi_hit = dc_digis[idx]

        # Truth position
        pos = sim_hit.getPosition()
        hits["hit_x"].append(pos.x)
        hits["hit_y"].append(pos.y)
        hits["hit_z"].append(pos.z)

        # Truth momentum
        mom = sim_hit.getMomentum()
        hits["hit_px"].append(mom.x)
        hits["hit_py"].append(mom.y)
        hits["hit_pz"].append(mom.z)

        # Wire geometry from digi
        wire_pos = digi_hit.getPosition()
        hits["wire_x"].append(wire_pos[0])
        hits["wire_y"].append(wire_pos[1])
        hits["wire_z"].append(wire_pos[2])

        azimuthal = digi_hit.getWireAzimuthalAngle()
        stereo = digi_hit.getWireStereoAngle()
        hits["wire_azimuthal_angle"].append(azimuthal)
        hits["wire_stereo_angle"].append(stereo)

        drift_dist = digi_hit.getDistanceToWire()
        hits["drift_distance"].append(drift_dist)

        # Compute left-right positions
        d_x = np.sin(stereo) * np.sin(azimuthal)
        d_y = -(np.sin(stereo) * np.cos(azimuthal))
        d_z = np.cos(stereo)

        z_prime = np.array([d_x, d_y, d_z])
        z_prime /= np.linalg.norm(z_prime)
        x_prime = np.array([1.0, 0.0, -d_x / d_z])
        x_prime /= np.linalg.norm(x_prime)
        y_prime = np.cross(z_prime, x_prime)
        y_prime /= np.linalg.norm(y_prime)

        w = np.array([wire_pos[0], wire_pos[1], wire_pos[2]])
        left = x_prime * (-drift_dist) + w
        right = x_prime * drift_dist + w

        hits["left_x"].append(left[0])
        hits["left_y"].append(left[1])
        hits["left_z"].append(left[2])
        hits["right_x"].append(right[0])
        hits["right_y"].append(right[1])
        hits["right_z"].append(right[2])

        # Metadata
        hits["edep"].append(sim_hit.getEDep())
        hits["time"].append(sim_hit.getTime())
        hits["path_length"].append(sim_hit.getPathLength())

        # cell_id = sim_hit.getCellID()
        # hits["cell_id"].append(cell_id)
        hits["cluster_count"].append(digi_hit.getNClusters())
        hits["produced_by_secondary"].append(int(sim_hit.isProducedBySecondary()))

        # MC particle link
        mc = sim_hit.getParticle()
        hits["mc_index"].append(mc.getObjectID().index)

    return hits


def extract_vtx_silicon_hits(event):
    """Extract vertex and silicon wrapper hits (point primitives)."""
    collections = [
        ("VTXBSimDigiLinks", "vtx_barrel"),
        ("VTXDSimDigiLinks", "vtx_endcap"),
        ("SiWrBSimDigiLinks", "siwr_barrel"),
        ("SiWrDSimDigiLinks", "siwr_endcap"),
    ]

    hits = {
        "hit_x": [], "hit_y": [], "hit_z": [],
        "hit_px": [], "hit_py": [], "hit_pz": [],
        "edep": [], "time": [], "path_length": [],
        # "cell_id": [],
        "produced_by_secondary": [],
        "mc_index": [],
        "sub_detector": [],
    }

    for coll_name, det_label in collections:
        links = event.get(coll_name)
        for link in links:
            digi_hit = link.getFrom()
            sim_hit = link.getTo()

            pos = digi_hit.getPosition()
            hits["hit_x"].append(pos.x)
            hits["hit_y"].append(pos.y)
            hits["hit_z"].append(pos.z)

            mom = sim_hit.getMomentum()
            hits["hit_px"].append(mom.x)
            hits["hit_py"].append(mom.y)
            hits["hit_pz"].append(mom.z)

            hits["edep"].append(digi_hit.getEDep())
            hits["time"].append(digi_hit.getTime())
            hits["path_length"].append(sim_hit.getPathLength())
            # hits["cell_id"].append(sim_hit.getCellID())
            hits["produced_by_secondary"].append(int(sim_hit.isProducedBySecondary()))

            mc = sim_hit.getParticle()
            hits["mc_index"].append(mc.getObjectID().index)
            hits["sub_detector"].append(det_label)

    return hits


def extract_mc_particles(event):
    """Extract MC particle truth info."""
    mc_coll = event.get("MCParticles")
    particles = {
        "mc_index": [], "pdg": [], "charge": [], "mass": [],
        "px": [], "py": [], "pz": [],
        "p": [], "pt": [], "theta": [], "phi": [],
        "vx": [], "vy": [], "vz": [],
        "gen_status": [],
        "parent_index": [],
        "decayed_in_tracker": [],
    }

    for j, part in enumerate(mc_coll):
        mom = part.getMomentum()
        p = math.sqrt(mom.x**2 + mom.y**2 + mom.z**2)
        pt = math.sqrt(mom.x**2 + mom.y**2)
        theta = math.acos(mom.z / p) if p > 0 else 0.0
        phi = math.atan2(mom.y, mom.x) if p > 0 else 0.0

        particles["mc_index"].append(j)
        particles["pdg"].append(part.getPDG())
        particles["charge"].append(part.getCharge())
        particles["mass"].append(part.getMass())
        particles["px"].append(mom.x)
        particles["py"].append(mom.y)
        particles["pz"].append(mom.z)
        particles["p"].append(p)
        particles["pt"].append(pt)
        particles["theta"].append(theta)
        particles["phi"].append(phi)

        vtx = part.getVertex()
        particles["vx"].append(vtx.x)
        particles["vy"].append(vtx.y)
        particles["vz"].append(vtx.z)

        particles["gen_status"].append(part.getGeneratorStatus())
        particles["decayed_in_tracker"].append(int(part.isDecayedInTracker()))

        parents = part.getParents()
        if len(parents) > 0:
            particles["parent_index"].append(parents[0].getObjectID().index)
        else:
            particles["parent_index"].append(-1)

    return particles


def dicts_to_arrow_table(dicts):
    """Convert dict of lists to a PyArrow table."""
    arrays = {}
    for k, v in dicts.items():
        if isinstance(v[0], str) if len(v) > 0 else False:
            arrays[k] = pa.array(v, type=pa.string())
        elif isinstance(v[0], int) if len(v) > 0 else False:
            arrays[k] = pa.array(v, type=pa.int64())
        else:
            arrays[k] = pa.array(v, type=pa.float32())
    return pa.table(arrays)


def process_file(input_path, output_dir, seed, split):
    """Process one edm4hep digitized ROOT file into Parquet tables."""
    print(f"Processing: {input_path}")
    reader = root_io.Reader(input_path)
    metadata = reader.get("metadata")[0]

    all_dc = []
    all_vtx = []
    all_mc = []

    for event_id, event in enumerate(reader.get("events")):
        dc = extract_dc_hits(event, metadata)
        vtx = extract_vtx_silicon_hits(event)
        mc = extract_mc_particles(event)

        n_dc = len(dc["hit_x"])
        n_vtx = len(vtx["hit_x"])
        n_mc = len(mc["mc_index"])

        # Add event/file identifiers
        dc["event_id"] = [event_id] * n_dc
        dc["seed"] = [seed] * n_dc
        dc["hit_type"] = [0] * n_dc  # 0 = drift chamber

        vtx["event_id"] = [event_id] * n_vtx
        vtx["seed"] = [seed] * n_vtx
        vtx["hit_type"] = [1] * n_vtx  # 1 = vertex/silicon

        mc["event_id"] = [event_id] * n_mc
        mc["seed"] = [seed] * n_mc

        if n_dc > 0:
            all_dc.append(dc)
        if n_vtx > 0:
            all_vtx.append(vtx)
        if n_mc > 0:
            all_mc.append(mc)

    # Merge all events for this file
    def merge_dicts(dict_list):
        if not dict_list:
            return {}
        merged = {k: [] for k in dict_list[0]}
        for d in dict_list:
            for k in merged:
                merged[k].extend(d[k])
        return merged

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    dc_merged = merge_dicts(all_dc)
    if dc_merged:
        dc_table = dicts_to_arrow_table(dc_merged)
        pq.write_table(dc_table, os.path.join(seed_dir, f"dc_hits_{split}.parquet"))
        print(f"  DC hits: {len(dc_merged['hit_x'])} rows")

    vtx_merged = merge_dicts(all_vtx)
    if vtx_merged:
        vtx_table = dicts_to_arrow_table(vtx_merged)
        pq.write_table(vtx_table, os.path.join(seed_dir, f"vtx_hits_{split}.parquet"))
        print(f"  VTX/Si hits: {len(vtx_merged['hit_x'])} rows")

    mc_merged = merge_dicts(all_mc)
    if mc_merged:
        mc_table = dicts_to_arrow_table(mc_merged)
        pq.write_table(mc_table, os.path.join(seed_dir, f"mc_particles_{split}.parquet"))
        print(f"  MC particles: {len(mc_merged['mc_index'])} rows")


def main():
    parser = argparse.ArgumentParser(description="Convert edm4hep to Parquet")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir",
                     help="Legacy mode: directory containing "
                          "seed_*/digi_edm4hep/*.root files. Seed numbers "
                          "are parsed from the path.")
    src.add_argument("--input_file",
                     help="Single ROOT file to convert. Pair with --seed "
                          "to control the output seed_N/ directory name. "
                          "Designed to be invoked under xargs -P / GNU "
                          "parallel against a flat digi/ directory, "
                          "avoiding any intermediate symlink staging.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Output seed number when using --input_file. "
                             "Required with --input_file.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for Parquet files")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file is not None:
        if args.seed is None:
            sys.exit("ERROR: --input_file requires --seed.")
        if not os.path.isfile(args.input_file):
            sys.exit(f"ERROR: not a file: {args.input_file}")
        process_file(args.input_file, args.output_dir, args.seed, args.split)
        print(f"Done. seed_{args.seed} written under {args.output_dir}/")
        return

    pattern = os.path.join(args.input_dir, "seed_*", "digi_edm4hep", "*.root")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} files to process")

    for f in files:
        parts = f.split(os.sep)
        seed_part = [p for p in parts if p.startswith("seed_")]
        seed = int(seed_part[0].split("_")[1]) if seed_part else 0
        process_file(f, args.output_dir, seed, args.split)

    print(f"\nDone. Parquet files written to {args.output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Convert digitized edm4hep ROOT files into per-seed Parquet shards.

Run this on a host where podio/edm4hep are available (e.g. lxplus with
key4hep set up). The Parquet output is consumed locally by
`model_training/src/dataset/parquet_dataset.py` (Polars-backed dataset)
which feeds CGATr training and evaluation.

Three table families are produced per seed (under `<output_dir>/seed_<N>/`):

* `dc_hits_<split>.parquet`        — drift chamber hits with full circle
  geometry (wire position, wire direction angles, drift distance).
  Together with the wire direction, the drift radius defines the IPNS
  circle that CGATr ingests via its outer-product hit encoder.
* `vtx_hits_<split>.parquet`       — vertex / silicon-wrapper hits as points.
* `mc_particles_<split>.parquet`   — MC particle properties used for
  truth-matching, pT/eta breakdowns, and clustering metrics.

Usage:
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
    pip install --user pyarrow
    python edm4hep_to_parquet.py \\
        --input_dir  <raw_root_dir> \\
        --output_dir <parquet_out_dir> \\
        --split      train

The expected layout for `--input_dir` is
`<input_dir>/seed_<N>/digi_edm4hep/*.root`, mirroring the upstream
condor pipeline.
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
        "cell_id": [], "cluster_count": [],
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

        cell_id = sim_hit.getCellID()
        hits["cell_id"].append(cell_id)
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
        "cell_id": [],
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
            hits["cell_id"].append(sim_hit.getCellID())
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


UNSIGNED_COLUMNS = {"cell_id"}  # EDM4hep cellID is uint64; values with the top
                                # bit set overflow pyarrow's signed int64 path


def dicts_to_arrow_table(dicts):
    """Convert dict of lists to a PyArrow table."""
    arrays = {}
    for k, v in dicts.items():
        if k in UNSIGNED_COLUMNS:
            arrays[k] = pa.array(v, type=pa.uint64())
        elif isinstance(v[0], str) if len(v) > 0 else False:
            arrays[k] = pa.array(v, type=pa.string())
        elif isinstance(v[0], int) if len(v) > 0 else False:
            arrays[k] = pa.array(v, type=pa.int64())
        else:
            arrays[k] = pa.array(v, type=pa.float32())
    return pa.table(arrays)


FLUSH_EVERY = 20  # events per row-group flush


def process_file(input_path, output_dir, seed, split):
    """Process one file, streaming row groups to disk every FLUSH_EVERY events.

    Memory-efficient streaming conversion using pq.ParquetWriter. Keeps only
    FLUSH_EVERY events buffered in memory and writes through temporary files,
    preventing out-of-memory errors on large samples (such as keepAllParticles)
    and avoiding truncated output on interrupted jobs.
    """
    print(f"Processing: {input_path}", flush=True)
    reader = root_io.Reader(input_path)
    metadata = reader.get("metadata")[0]

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    names = ("dc_hits", "vtx_hits", "mc_particles")
    finals = {n: os.path.join(seed_dir, f"{n}_{split}.parquet") for n in names}
    tmps = {n: finals[n] + ".tmp" for n in names}
    writers = {n: None for n in names}
    schemas = {n: None for n in names}
    buf = {n: [] for n in names}
    rows = {n: 0 for n in names}

    def flush():
        for n in names:
            if not buf[n]:
                continue
            merged = {k: [] for k in buf[n][0]}
            for d in buf[n]:
                for k in merged:
                    merged[k].extend(d[k])
            table = dicts_to_arrow_table(merged)
            if writers[n] is None:
                schemas[n] = table.schema
                writers[n] = pq.ParquetWriter(tmps[n], schemas[n])
            writers[n].write_table(table.cast(schemas[n]))
            rows[n] += table.num_rows
            buf[n] = []

    for event_id, event in enumerate(reader.get("events")):
        dc = extract_dc_hits(event, metadata)
        vtx = extract_vtx_silicon_hits(event)
        mc = extract_mc_particles(event)
        for n, d, hit_type in (("dc_hits", dc, 0),
                               ("vtx_hits", vtx, 1),
                               ("mc_particles", mc, None)):
            key0 = "mc_index" if n == "mc_particles" else "hit_x"
            cnt = len(d[key0])
            if cnt == 0:
                continue
            d["event_id"] = [event_id] * cnt
            d["seed"] = [seed] * cnt
            if hit_type is not None:
                d["hit_type"] = [hit_type] * cnt
            buf[n].append(d)
        if (event_id + 1) % FLUSH_EVERY == 0:
            flush()
        if (event_id + 1) % 100 == 0:
            print(f"  ... {event_id + 1} events", flush=True)
    flush()
    for n in names:
        if writers[n] is not None:
            writers[n].close()
            os.replace(tmps[n], finals[n])
    print(f"  DC hits: {rows['dc_hits']} rows; VTX/Si hits: {rows['vtx_hits']} rows; "
          f"MC particles: {rows['mc_particles']} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert edm4hep to Parquet")
    parser.add_argument("--input_dir", required=True,
                        help="Directory with seed_*/digi_edm4hep/*.root files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for Parquet files")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "seed_*", "digi_edm4hep", "*.root")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} files to process")
    os.makedirs(args.output_dir, exist_ok=True)

    for f in files:
        # Extract seed number from path
        parts = f.split(os.sep)
        seed_part = [p for p in parts if p.startswith("seed_")]
        seed = int(seed_part[0].split("_")[1]) if seed_part else 0
        process_file(f, args.output_dir, seed, args.split)

    print(f"\nDone. Parquet files written to {args.output_dir}/")


if __name__ == "__main__":
    main()

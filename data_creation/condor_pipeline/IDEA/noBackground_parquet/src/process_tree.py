#!/usr/bin/env python3

"""Convert digitized IDEA EDM4hep to the shared CIRCE/GATr Parquet schema.

The extraction follows ``CIRCE/src/dataset/edm4hep_to_parquet.py``: PODIO
collections are read directly, drift geometry is derived from the digitized
hit, and MC relations come from the simulated hit.  Unlike the CIRCE utility,
the result remains one row per event with jagged columns. It stores both the
raw drift-circle parameters used by CIRCE and the left/right ambiguity points
used by GATr; model-specific representations are built only in their loaders.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from podio import root_io


HIT_FIELDS = (
    "hit_x_true", "hit_y_true", "hit_z_true",
    "hit_type", "hit_EDep", "hit_time", "hit_pathLength",
    "hit_particle_index", "hit_px", "hit_py", "hit_pz",
    "hit_x", "hit_y", "hit_z",
    "leftPosition_x", "leftPosition_y", "leftPosition_z",
    "rightPosition_x", "rightPosition_y", "rightPosition_z",
    "produced_by_secondary", "overlay", "cluster_count",
    "superLayer", "layer", "phi", "stereo",
    "drift_distance", "wire_azimuthal_angle", "wire_stereo_angle",
)

PARTICLE_FIELDS = (
    "part_p", "part_p_t", "part_theta", "part_phi", "part_m",
    "part_pid", "part_id", "gen_status", "part_parent",
    "part_vertex_x", "part_vertex_y", "part_vertex_z",
)

VECTOR_FIELDS = HIT_FIELDS + PARTICLE_FIELDS
PLANAR_ASSOCIATIONS = (
    "VTXDSimDigiLinks",
    "VTXBSimDigiLinks",
    "SiWrDSimDigiLinks",
    "SiWrBSimDigiLinks",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create event-wise GATR Parquet directly from digitized EDM4hep."
    )
    parser.add_argument("input_file", help="digitized EDM4hep ROOT input")
    parser.add_argument("output_file", help="output path ending in .parquet")
    parser.add_argument(
        "--file-number", type=int,
        help="dataset file identifier; inferred from Graphs_N when omitted",
    )
    parser.add_argument(
        "--row-group-size", type=int, default=25,
        help="events per Parquet row group (default: 25)",
    )
    parser.add_argument(
        "--compression", choices=("zstd", "snappy", "none"), default="zstd",
    )
    parser.add_argument("--compression-level", type=int, default=1)
    return parser.parse_args()


def infer_file_number(path):
    match = re.search(r"Graphs_(\d+)", Path(path).name)
    if match is None:
        raise ValueError(
            "Cannot infer file number from output filename; pass --file-number"
        )
    return int(match.group(1))


def new_event():
    return {name: [] for name in VECTOR_FIELDS}


def collection(event, name):
    try:
        return event.get(name)
    except Exception as error:
        raise RuntimeError(f"Required EDM4hep collection is missing: {name}") from error


def get_overlay_flag(sim_hit):
    """Return the overlay flag where supported by the EDM4hep hit type."""
    return int(sim_hit.isOverlay()) if hasattr(sim_hit, "isOverlay") else 0


def make_dch_decoder(metadata):
    # The Key4hep Python bindings expose BitFieldCoder through ROOT.dd4hep.
    # Keep this import local so ordinary static inspection does not initialize ROOT.
    import dd4hep as _dd4hep_module  # noqa: F401 - registers dd4hep in ROOT
    from ROOT import dd4hep

    encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    return dd4hep.BitFieldCoder(encoding)


def drift_ambiguity_positions(wire_position, drift_distance, azimuthal, stereo):
    """Return symmetric positions at +/- drift distance from the wire."""
    wire_direction = np.array(
        [
            np.sin(stereo) * np.sin(azimuthal),
            -np.sin(stereo) * np.cos(azimuthal),
            np.cos(stereo),
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(wire_direction)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError(f"Invalid wire direction for azimuthal={azimuthal}, stereo={stereo}")
    wire_direction /= norm

    # Match the historical IDEA convention when it is well-conditioned.  The
    # fallback is a stable perpendicular for the rare direction with dz~0.
    if abs(wire_direction[2]) > 1.0e-12:
        radial = np.array(
            [1.0, 0.0, -wire_direction[0] / wire_direction[2]],
            dtype=np.float64,
        )
    else:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        radial = np.cross(reference, wire_direction)
    radial_norm = np.linalg.norm(radial)
    if not np.isfinite(radial_norm) or radial_norm < 1.0e-12:
        raise ValueError("Could not construct the direction perpendicular to the wire")
    radial /= radial_norm

    wire_position = np.asarray(wire_position, dtype=np.float64)
    left = wire_position - float(drift_distance) * radial
    right = wire_position + float(drift_distance) * radial
    return left, right


def extract_drift_hits(event, metadata, values, hit_mc_indices):
    links = collection(event, "DCH_DigiSimAssociationCollection")
    digi_hits = collection(event, "DCH_DigiCollection")
    if len(links) != len(digi_hits):
        raise RuntimeError(
            "DCH association/digitized-hit size mismatch: "
            f"{len(links)} associations versus {len(digi_hits)} hits"
        )
    decoder = make_dch_decoder(metadata)

    for index, link in enumerate(links):
        sim_hit = link.getTo()
        digi_hit = digi_hits[index]
        true_position = sim_hit.getPosition()
        momentum = sim_hit.getMomentum()
        wire = digi_hit.getPosition()
        wire_position = np.array([wire[0], wire[1], wire[2]], dtype=np.float64)
        drift_distance = float(digi_hit.getDistanceToWire())
        azimuthal = float(digi_hit.getWireAzimuthalAngle())
        wire_stereo = float(digi_hit.getWireStereoAngle())
        left, right = drift_ambiguity_positions(
            wire_position, drift_distance, azimuthal, wire_stereo
        )

        cell_id = sim_hit.getCellID()
        particle_index = int(sim_hit.getParticle().getObjectID().index)
        hit_mc_indices.add(particle_index)

        values["hit_x_true"].append(true_position.x)
        values["hit_y_true"].append(true_position.y)
        values["hit_z_true"].append(true_position.z)
        values["hit_px"].append(momentum.x)
        values["hit_py"].append(momentum.y)
        values["hit_pz"].append(momentum.z)
        values["hit_x"].append(wire_position[0])
        values["hit_y"].append(wire_position[1])
        values["hit_z"].append(wire_position[2])
        values["leftPosition_x"].append(left[0])
        values["leftPosition_y"].append(left[1])
        values["leftPosition_z"].append(left[2])
        values["rightPosition_x"].append(right[0])
        values["rightPosition_y"].append(right[1])
        values["rightPosition_z"].append(right[2])
        values["hit_EDep"].append(digi_hit.getEDep())
        values["hit_time"].append(digi_hit.getTime())
        values["hit_pathLength"].append(sim_hit.getPathLength())
        values["cluster_count"].append(digi_hit.getNClusters())
        values["produced_by_secondary"].append(int(sim_hit.isProducedBySecondary()))
        values["overlay"].append(get_overlay_flag(sim_hit))
        values["hit_particle_index"].append(particle_index)
        values["hit_type"].append(0)
        values["superLayer"].append(decoder.get(cell_id, "superlayer"))
        values["layer"].append(decoder.get(cell_id, "layer"))
        values["phi"].append(decoder.get(cell_id, "nphi"))
        values["stereo"].append(decoder.get(cell_id, "stereosign"))
        values["drift_distance"].append(drift_distance)
        values["wire_azimuthal_angle"].append(azimuthal)
        values["wire_stereo_angle"].append(wire_stereo)


def extract_planar_hits(event, values, hit_mc_indices):
    for collection_name in PLANAR_ASSOCIATIONS:
        for link in collection(event, collection_name):
            digi_hit = link.getFrom()
            sim_hit = link.getTo()
            measured_position = digi_hit.getPosition()
            true_position = sim_hit.getPosition()
            momentum = sim_hit.getMomentum()
            particle_index = int(sim_hit.getParticle().getObjectID().index)
            hit_mc_indices.add(particle_index)

            values["hit_x_true"].append(true_position.x)
            values["hit_y_true"].append(true_position.y)
            values["hit_z_true"].append(true_position.z)
            values["hit_px"].append(momentum.x)
            values["hit_py"].append(momentum.y)
            values["hit_pz"].append(momentum.z)
            values["hit_x"].append(measured_position.x)
            values["hit_y"].append(measured_position.y)
            values["hit_z"].append(measured_position.z)
            for name in (
                "leftPosition_x", "leftPosition_y", "leftPosition_z",
                "rightPosition_x", "rightPosition_y", "rightPosition_z",
            ):
                values[name].append(0)
            values["hit_EDep"].append(digi_hit.getEDep())
            values["hit_time"].append(digi_hit.getTime())
            values["hit_pathLength"].append(sim_hit.getPathLength())
            values["cluster_count"].append(0)
            values["produced_by_secondary"].append(int(sim_hit.isProducedBySecondary()))
            values["overlay"].append(get_overlay_flag(sim_hit))
            values["hit_particle_index"].append(particle_index)
            values["hit_type"].append(1)
            values["superLayer"].append(0)
            values["layer"].append(0)
            values["phi"].append(0)
            values["stereo"].append(0)
            values["drift_distance"].append(0)
            values["wire_azimuthal_angle"].append(0)
            values["wire_stereo_angle"].append(0)


def extract_particles(event, values, hit_mc_indices):
    for particle in collection(event, "MCParticles"):
        particle_index = int(particle.getObjectID().index)
        if particle_index not in hit_mc_indices:
            continue
        momentum = particle.getMomentum()
        momentum_magnitude = math.sqrt(
            momentum.x * momentum.x + momentum.y * momentum.y + momentum.z * momentum.z
        )
        transverse_momentum = math.hypot(momentum.x, momentum.y)
        theta = (
            math.acos(np.clip(momentum.z / momentum_magnitude, -1.0, 1.0))
            if momentum_magnitude > 0
            else 0.0
        )
        phi = math.atan2(momentum.y, momentum.x) if momentum_magnitude > 0 else 0.0
        parents = particle.getParents()
        parent_index = int(parents[0].getObjectID().index) if len(parents) else -1
        vertex = particle.getVertex()

        values["part_p"].append(momentum_magnitude)
        values["part_p_t"].append(transverse_momentum)
        values["part_theta"].append(theta)
        values["part_phi"].append(phi)
        values["part_m"].append(particle.getMass())
        values["part_pid"].append(particle.getPDG())
        values["part_id"].append(particle_index)
        values["gen_status"].append(particle.getGeneratorStatus())
        values["part_parent"].append(parent_index)
        values["part_vertex_x"].append(vertex.x)
        values["part_vertex_y"].append(vertex.y)
        values["part_vertex_z"].append(vertex.z)


def validate_event(values, event_number):
    n_hits = len(values["hit_type"])
    n_particles = len(values["part_id"])
    bad_hit_fields = [name for name in HIT_FIELDS if len(values[name]) != n_hits]
    bad_particle_fields = [
        name for name in PARTICLE_FIELDS if len(values[name]) != n_particles
    ]
    if bad_hit_fields or bad_particle_fields:
        raise RuntimeError(
            f"Event {event_number} has inconsistent column lengths; "
            f"hit fields={bad_hit_fields}, particle fields={bad_particle_fields}"
        )
    missing_particles = set(map(int, values["hit_particle_index"])) - set(
        map(int, values["part_id"])
    )
    if missing_particles:
        raise RuntimeError(
            f"Event {event_number} has hits linked to unstored particles: "
            f"{sorted(missing_particles)[:10]}"
        )
    return n_hits, n_particles


def empty_columns():
    columns = {"event_number": [], "n_hit": [], "n_part": []}
    columns.update({name: [] for name in VECTOR_FIELDS})
    return columns


def append_event(columns, values, event_number, n_hits, n_particles):
    columns["event_number"].append(np.int32(event_number))
    columns["n_hit"].append(np.int32(n_hits))
    columns["n_part"].append(np.int32(n_particles))
    for name in VECTOR_FIELDS:
        columns[name].append(np.asarray(values[name], dtype=np.float32))


def make_arrow_table(columns, file_number):
    events = ak.Array(columns)
    for name in ("event_number", "n_hit", "n_part"):
        events = ak.with_field(events, ak.values_astype(events[name], np.int32), name)
    for name in VECTOR_FIELDS:
        events = ak.with_field(events, ak.values_astype(events[name], np.float32), name)
    events = ak.with_field(
        events, np.full(len(events), file_number, dtype=np.int32), "file_number"
    )
    events = ak.with_field(
        events, np.full(len(events), 1, dtype=np.int32), "shared_schema_version"
    )
    return ak.to_arrow_table(events, extensionarray=True)


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.suffix != ".parquet":
        raise ValueError(f"Output must end in .parquet: {output_path}")
    if args.row_group_size <= 0:
        raise ValueError("--row-group-size must be positive")
    file_number = (
        args.file_number
        if args.file_number is not None
        else infer_file_number(output_path)
    )

    reader = root_io.Reader(str(input_path))
    metadata = reader.get("metadata")[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    compression = None if args.compression == "none" else args.compression
    compression_level = args.compression_level if compression == "zstd" else None
    columns = empty_columns()
    writer = None
    total_events = 0
    total_hits = 0
    total_particles = 0
    try:
        for event_number, event in enumerate(reader.get("events")):
            values = new_event()
            hit_mc_indices = set()
            extract_drift_hits(event, metadata, values, hit_mc_indices)
            extract_planar_hits(event, values, hit_mc_indices)
            extract_particles(event, values, hit_mc_indices)
            n_hits, n_particles = validate_event(values, event_number)
            append_event(columns, values, event_number, n_hits, n_particles)
            total_events += 1
            total_hits += n_hits
            total_particles += n_particles
            print(
                f"Event {event_number}: {n_hits} hits, {n_particles} linked particles",
                flush=True,
            )

            if len(columns["event_number"]) == args.row_group_size:
                table = make_arrow_table(columns, file_number)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        table.schema,
                        compression=compression,
                        compression_level=compression_level,
                        version="2.6",
                    )
                writer.write_table(table, row_group_size=len(table))
                columns = empty_columns()

        if columns["event_number"]:
            table = make_arrow_table(columns, file_number)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression=compression,
                    compression_level=compression_level,
                    version="2.6",
                )
            writer.write_table(table, row_group_size=len(table))

        if writer is None:
            raise RuntimeError(f"No events found in {input_path}")
        writer.close()
        writer = None
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    os.replace(temporary_path, output_path)
    print(
        f"Wrote {total_events} events, {total_hits} hits and {total_particles} linked "
        f"particles to {output_path} (file_number={file_number}, "
        f"row_group_size={args.row_group_size})",
        flush=True,
    )


if __name__ == "__main__":
    main()

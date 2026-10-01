#!/usr/bin/env python3
"""Print YAML inputs and derived detector scalars from digitized hits."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from podio import root_io
except ImportError as error:
    raise SystemExit(
        "Could not import podio. Source the Key4hep stack before running this script."
    ) from error


DEFAULT_INPUT = (
    "/eos/experiment/fcc/ee/simulation/key4hep_2026_07_29/91GeV/"
    "IDEA_o1_v4/Zuds/digi/output_IDEA_DIGI_1_train.root"
)

# This must match inputs.hits_features.vars in the detector data config.
FEATURE_NAMES = (
    "hit_x",
    "hit_y",
    "hit_z",
    "leftPosition_x",
    "leftPosition_y",
    "leftPosition_z",
    "rightPosition_x",
    "rightPosition_y",
    "rightPosition_z",
    "hit_type",
    "hit_time",
    "layer",
    "superLayer",
    "stereo",
    "cluster_count",
)

PLANAR_DIGI_COLLECTIONS = (
    "VTXDDigis",
    "VTXBDigis",
    "SiWrDDigis",
    "SiWrBDigis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one event from an IDEA digitized EDM4hep ROOT file and print "
            "the 15 float32 hit values selected by "
            "config_tracking_parquet_detector.yaml. "
            "Every value is obtained from a digitized hit."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"digitized EDM4hep ROOT file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="zero-based event index (default: 0)",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        help="print at most this many hits; by default all hits are printed",
    )
    parser.add_argument(
        "--layers-per-superlayer",
        type=int,
        nargs=14,
        default=[8] * 14,
        metavar="N",
        help="number of layers in each of the 14 superlayers",
    )
    staging = parser.add_mutually_exclusive_group()
    staging.add_argument(
        "--stage-local",
        action="store_true",
        help="copy the ROOT file to a temporary local directory before reading",
    )
    staging.add_argument(
        "--no-stage-local",
        action="store_true",
        help="read directly even for /eos paths (PODIO may fail through EOS FUSE)",
    )
    return parser.parse_args()


def validate_yaml_order() -> None:
    """Fail if the colocated training YAML no longer matches FEATURE_NAMES."""
    config_path = (
        Path(__file__).resolve().parent
        / "config_files"
        / "config_tracking_parquet_detector.yaml"
    )
    if not config_path.is_file():
        return

    names = []
    in_hits_features = False
    in_vars = False
    variable_pattern = re.compile(r"^\s{6}-\s*\[\s*([^,\]]+)")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("  hits_features:"):
            in_hits_features = True
            continue
        if not in_hits_features:
            continue
        if line.startswith("    vars:"):
            in_vars = True
            continue
        if in_vars:
            match = variable_pattern.match(line)
            if match:
                names.append(match.group(1).strip())
            elif line.startswith("  ") and not line.startswith("      ") and line.strip():
                break

    if tuple(names) != FEATURE_NAMES:
        raise RuntimeError(
            f"{config_path} defines hits_features={tuple(names)}, but this extractor "
            f"implements {FEATURE_NAMES}"
        )


def collection(event, name: str):
    try:
        return event.get(name)
    except Exception as error:
        raise RuntimeError(f"Required EDM4hep collection is missing: {name}") from error


def xyz(vector) -> tuple[float, float, float]:
    """Return coordinates from either an EDM4hep vector or an indexable vector."""
    try:
        return float(vector.x), float(vector.y), float(vector.z)
    except AttributeError:
        return float(vector[0]), float(vector[1]), float(vector[2])


def make_dch_decoder(metadata):
    # Importing dd4hep registers its Python bindings in ROOT.
    import dd4hep as _dd4hep_module  # noqa: F401
    from ROOT import dd4hep

    encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    return dd4hep.BitFieldCoder(encoding)


def drift_ambiguity_positions(
    wire_position: tuple[float, float, float],
    drift_distance: float,
    azimuthal: float,
    stereo_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the left/right position construction used for Parquet production."""
    wire_direction = np.array(
        [
            np.sin(stereo_angle) * np.sin(azimuthal),
            -np.sin(stereo_angle) * np.cos(azimuthal),
            np.cos(stereo_angle),
        ],
        dtype=np.float64,
    )
    direction_norm = np.linalg.norm(wire_direction)
    if not np.isfinite(direction_norm) or direction_norm < 1.0e-12:
        raise ValueError(
            f"Invalid wire direction for azimuthal={azimuthal}, stereo={stereo_angle}"
        )
    wire_direction /= direction_norm

    if abs(wire_direction[2]) > 1.0e-12:
        radial = np.array(
            [1.0, 0.0, -wire_direction[0] / wire_direction[2]],
            dtype=np.float64,
        )
    else:
        radial = np.cross(
            np.array([0.0, 0.0, 1.0], dtype=np.float64), wire_direction
        )
    radial_norm = np.linalg.norm(radial)
    if not np.isfinite(radial_norm) or radial_norm < 1.0e-12:
        raise ValueError("Could not construct a direction perpendicular to the wire")
    radial /= radial_norm

    wire = np.asarray(wire_position, dtype=np.float64)
    left = wire - float(drift_distance) * radial
    right = wire + float(drift_distance) * radial
    return left, right


def float32_row(values) -> list[float]:
    """Match the float32 conversion performed when the Parquet file is written."""
    row = np.asarray(values, dtype=np.float32)
    if row.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            f"Expected {len(FEATURE_NAMES)} input values, received shape {row.shape}"
        )
    return row.tolist()


def extract_event_inputs(event, metadata):
    """Yield digitized-only (source, float32 input row) pairs."""
    digi_hits = collection(event, "DCH_DigiCollection")
    decoder = make_dch_decoder(metadata)

    # Keep the established ordering: drift-chamber hits first.
    for digi_hit in digi_hits:
        wire_position = xyz(digi_hit.getPosition())
        left, right = drift_ambiguity_positions(
            wire_position,
            float(digi_hit.getDistanceToWire()),
            float(digi_hit.getWireAzimuthalAngle()),
            float(digi_hit.getWireStereoAngle()),
        )
        cell_id = digi_hit.getCellID()
        yield "DCH", float32_row(
            (
                *wire_position,
                *left,
                *right,
                0,
                digi_hit.getTime(),
                decoder.get(cell_id, "layer"),
                decoder.get(cell_id, "superlayer"),
                decoder.get(cell_id, "stereosign"),
                digi_hit.getNClusters(),
            )
        )

    # The four planar digitized-hit collections follow in this fixed order.
    for collection_name in PLANAR_DIGI_COLLECTIONS:
        for digi_hit in collection(event, collection_name):
            measured_position = xyz(digi_hit.getPosition())
            yield collection_name, float32_row(
                (
                    *measured_position,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    digi_hit.getTime(),
                    0,
                    0,
                    0,
                    0,
                )
            )


def read_event(reader, event_index: int):
    if event_index < 0:
        raise ValueError("--event-index must be non-negative")
    for index, event in enumerate(reader.get("events")):
        if index == event_index:
            return event
    raise IndexError(f"Event index {event_index} is outside the ROOT file")


@contextmanager
def readable_input_path(input_path: Path, stage_local: bool):
    """Optionally stage an EOS file locally to avoid ROOT/PODIO FUSE failures."""
    if not stage_local:
        yield input_path
        return
    with tempfile.TemporaryDirectory(prefix="ggtf-root-input-") as temporary_dir:
        staged_path = Path(temporary_dir) / input_path.name
        print(f"Staging {input_path} to {staged_path}", file=sys.stderr, flush=True)
        shutil.copy2(input_path, staged_path)
        yield staged_path


def main() -> int:
    args = parse_args()
    if args.max_hits is not None and args.max_hits <= 0:
        raise ValueError("--max-hits must be positive")
    if len(args.layers_per_superlayer) != 14 or any(
        value <= 0 for value in args.layers_per_superlayer
    ):
        raise ValueError(
            "--layers-per-superlayer requires exactly 14 positive integers"
        )

    validate_yaml_order()
    input_path = Path(args.input_file)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    is_eos_path = input_path.is_absolute() and input_path.parts[:2] == ("/", "eos")
    stage_local = args.stage_local or (is_eos_path and not args.no_stage_local)
    with readable_input_path(input_path, stage_local) as reader_path:
        reader = root_io.Reader(str(reader_path))
        metadata = reader.get("metadata")[0]
        event = read_event(reader, args.event_index)
        rows = list(extract_event_inputs(event, metadata))

    print(f"# input_file: {input_path}")
    print(f"# event_index: {args.event_index}")
    print(f"# number_of_hits: {len(rows)}")
    print(
        "# representation: digitized-only YAML hits_features before graph "
        "reordering and GATr embedding"
    )
    print(f"# feature_order: {list(FEATURE_NAMES)}")
    offsets = np.concatenate(
        ([0], np.cumsum(np.asarray(args.layers_per_superlayer[:-1], dtype=np.int64)))
    )
    print(f"# layers_per_superlayer: {args.layers_per_superlayer}")
    print("# model_detector_feature_order: ['hit_time', 'global_layer', 'stereo', 'cluster_count']")
    number_to_print = len(rows) if args.max_hits is None else min(args.max_hits, len(rows))
    for index, (source, row) in enumerate(rows[:number_to_print]):
        layer = int(row[11])
        superlayer = int(row[12])
        if not 0 <= superlayer < 14:
            raise ValueError(f"Invalid superLayer {superlayer} in hit {index}")
        if not 0 <= layer < args.layers_per_superlayer[superlayer]:
            raise ValueError(
                f"Invalid layer {layer} for superLayer {superlayer} in hit {index}"
            )
        detector_scalars = [
            row[10],
            float(offsets[superlayer] + layer),
            row[13],
            row[14],
        ]
        print(
            f"hit[{index}] source={source} yaml_input={row} "
            f"model_detector_scalars={detector_scalars}"
        )
    if number_to_print != len(rows):
        print(f"# omitted_hits: {len(rows) - number_to_print}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

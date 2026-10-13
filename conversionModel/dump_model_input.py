#!/usr/bin/env python3
"""Dump the GGTFTrackFinder model input of one event from an EDM4hep ROOT file to CSV.

Reproduces the feature building of k4RecTracker/Tracking/components/GGTFTrackFinder.cpp
(appendPlanarHits + appendWireHits), so the CSV can be fed to `validate_onnx.py -i`.

Each row is one hit with 7 float32 features `x,y,z,planar_flag,u,v,w`:
  - planar hit : (x, y, z, 1, 0, 0, 0)
  - wire hit   : midpoint of the two drift-ambiguity solutions, 0,
                 half of the left-to-right vector  (right - left) / 2
Rows are ordered as in the C++ algorithm: all planar collections (in the given
order), then all wire collections.

Requires a key4hep environment (podio + edm4hep python bindings):
    source /cvmfs/sw.hsf.org/key4hep/setup.sh
"""

import argparse
import math
import sys

import numpy as np
from podio.reading import get_reader

DEFAULT_PLANAR = ["VTXBDigis", "VTXDDigis", "SiWrBDigis", "SiWrDDigis"]
DEFAULT_WIRE = ["DCH_DigiCollection"]
HEADER = "x,y,z,planar_flag,u,v,w"


def planar_features(collection):
    rows = []
    for hit in collection:
        p = hit.getPosition()
        rows.append(np.array([p.x, p.y, p.z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return rows


def wire_features(collection):
    rows = []
    skipped = 0
    for hit in collection:
        w = hit.getPosition()
        wire = np.array([w.x, w.y, w.z], dtype=np.float64)
        d = hit.getDistanceToWire()
        phi = hit.getWireAzimuthalAngle()
        stereo = hit.getWireStereoAngle()
        if not (math.isfinite(d) and math.isfinite(phi) and math.isfinite(stereo)):
            skipped += 1
            continue

        # Local wire frame, as in GGTFTrackFinder::appendWireHits (double precision).
        dx = math.sin(stereo) * math.sin(phi)
        dy = -math.sin(stereo) * math.cos(phi)
        dz = math.cos(stereo)
        z_prime = np.array([dx, dy, dz])
        z_prime /= np.linalg.norm(z_prime)
        x_prime = np.array([1.0, 0.0, -dx / dz])
        x_prime /= np.linalg.norm(x_prime)

        # Left/right solutions lie along x' at -d / +d; cast to float32 before
        # assembling the features, exactly like the C++ code.
        left = (wire - d * x_prime).astype(np.float32)
        right = (wire + d * x_prime).astype(np.float32)
        half = np.float32(0.5)
        mid = half * (left + right)
        vec = half * (right - left)
        rows.append(np.concatenate([mid, np.zeros(1, dtype=np.float32), vec]).astype(np.float32))
    return rows, skipped


def get_collection(frame, name):
    if name not in frame.getAvailableCollections():
        sys.exit(f"Collection '{name}' not found in event. Available: {sorted(frame.getAvailableCollections())}")
    return frame.get(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="EDM4hep ROOT file containing the digitized hits")
    parser.add_argument("-o", "--output", default="model_input_event0.csv", help="output CSV file")
    parser.add_argument("-e", "--event", type=int, default=0, help="event index in the file (default: 0)")
    parser.add_argument("--planar", nargs="+", default=DEFAULT_PLANAR,
                        help=f"TrackerHitPlane collections (default: {' '.join(DEFAULT_PLANAR)})")
    parser.add_argument("--wire", nargs="+", default=DEFAULT_WIRE,
                        help=f"SenseWireHit collections (default: {' '.join(DEFAULT_WIRE)})")
    args = parser.parse_args()

    reader = get_reader(args.input)
    events = reader.get("events")
    if not 0 <= args.event < len(events):
        sys.exit(f"Event {args.event} out of range: file has {len(events)} events")
    frame = events[args.event]

    rows = []
    n_planar = 0
    for name in args.planar:
        r = planar_features(get_collection(frame, name))
        n_planar += len(r)
        rows += r
    n_wire = 0
    n_skipped = 0
    for name in args.wire:
        r, s = wire_features(get_collection(frame, name))
        n_wire += len(r)
        n_skipped += s
        rows += r

    features = np.stack(rows) if rows else np.zeros((0, 7), dtype=np.float32)
    # %.9g round-trips float32 exactly.
    np.savetxt(args.output, features, fmt="%.9g", delimiter=",", header=HEADER, comments="")

    print(f"Event {args.event}: {n_planar} planar + {n_wire} wire hits -> {features.shape} written to {args.output}")
    if n_skipped:
        print(f"Skipped {n_skipped} wire hits with non-finite geometry values")


if __name__ == "__main__":
    main()

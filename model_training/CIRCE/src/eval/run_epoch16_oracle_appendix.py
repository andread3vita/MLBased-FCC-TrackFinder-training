"""Run a separately labelled, non-promotable truth-position helix appendix."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from src.eval.run_epoch16_cluster_sweep import (
    _assert_campaign_current,
    _run_point,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mc-signal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not (args.output / "SWEEP_COMPLETE").exists():
        raise RuntimeError("promotable sweep is not complete")
    args.campaign = json.loads(
        (args.output / "campaign_manifest.json").read_text()
    )
    args.provenance_sha256 = args.campaign["campaign_sha256"]
    args.dataset_manifest = Path(
        args.campaign["inputs"]["dataset_manifest"]["path"]
    )
    if (
        Path(args.campaign["inputs"]["forward_cache"]["path"])
        != (args.cache / "forward_hits.parquet").resolve()
        or Path(args.campaign["inputs"]["mc_signal"]["path"])
        != args.mc_signal.resolve()
    ):
        raise RuntimeError("oracle inputs do not match the pinned campaign")
    lock_handle = (args.output / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another sweep process holds the campaign lock") from exc
    _assert_campaign_current(args)

    source_path = args.output / "refinement_sources.json"
    sources = json.loads(source_path.read_text())
    if len(sources) > 6:
        indices = {
            round(index * (len(sources) - 1) / 5) for index in range(6)
        }
        sources = [sources[index] for index in sorted(indices)]
    appendix_sources = []
    for source in sources:
        appendix_sources.append(source)
        for tolerance in (0.02, 0.05, 0.1):
            _run_point(
                args,
                "oracle_helix",
                source["clusterer"],
                source["tbeta"],
                source["td"],
                min_hits=4,
                helix_tol=tolerance,
            )
    (args.output / "oracle_helix/README.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (args.output / "oracle_helix/README.json").write_text(json.dumps({
        "promotable": False,
        "reason": (
            "The helix fit uses hit_x/y/z truth positions unavailable to a "
            "deployable drift-chamber tracker."
        ),
        "sources": appendix_sources,
        "tolerances": [0.02, 0.05, 0.1],
    }, indent=2) + "\n")
    _assert_campaign_current(args)
    (args.output / "ORACLE_APPENDIX_COMPLETE").write_text(json.dumps({
        "campaign_sha256": args.provenance_sha256,
        "points": len(sources) * 3,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

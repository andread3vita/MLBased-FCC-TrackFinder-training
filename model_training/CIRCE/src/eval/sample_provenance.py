"""Resolve which evaluation sample sits behind a cache, and say so in captions.

Plot captions used to hardcode "Source: seed 181, 500 events" as a literal
string in four plotting scripts. That was true of the campaign they were first
written for and silently false for every render since: the confirm campaign's
plots claimed 500 events from seed 181 while showing 5,000 events from seeds
191-200, understating the statistics tenfold and naming the wrong seeds
(FINDINGS.md M86).

Captions now derive the sample from the campaign manifest that the forward pass
writes, so a caption cannot disagree with the data it sits under. If no manifest
is reachable the caption says the sample is unrecorded rather than inventing one.
"""
from __future__ import annotations

import json
from pathlib import Path

_MANIFEST_RELATIVE = Path("emb") / "manifest.json"
_UNKNOWN = "sample not recorded in a campaign manifest"


def find_manifest(path: str | Path) -> Path | None:
    """Nearest ``emb/manifest.json`` at or above ``path``.

    Caches live at ``<campaign>/<operating_point>/cache.parquet`` and the
    manifest at ``<campaign>/emb/manifest.json``, so the search walks upward and
    stops at the first hit, which keeps a sibling campaign from answering for
    this one.
    """
    start = Path(path).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        manifest = candidate / _MANIFEST_RELATIVE
        if manifest.is_file():
            return manifest
    return None


def describe_sample(path: str | Path, default: str = _UNKNOWN) -> str:
    """Caption fragment for one cache, e.g. ``seeds 191-200 · 5,000 events``."""
    manifest = find_manifest(path)
    if manifest is None:
        return default
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return default

    parts = []
    seeds = data.get("seeds")
    if seeds:
        # Single-seed campaigns are recorded as "181-181"; say "seed 181".
        low, _, high = str(seeds).partition("-")
        parts.append(f"seed {low}" if low == high else f"seeds {seeds}")
    n_events = data.get("n_events")
    if n_events:
        parts.append(f"{int(n_events):,} events")
    return " · ".join(parts) if parts else default


def describe_samples(paths, default: str = _UNKNOWN) -> str:
    """One caption for several curves, naming any disagreement between them.

    Overlays mix campaigns (the PGA arm is scored separately from the CGA one),
    so a single "Source:" line is only honest when every curve agrees.
    """
    seen: list[str] = []
    for path in paths:
        described = describe_sample(path, default=default)
        if described not in seen:
            seen.append(described)
    if not seen:
        return default
    if len(seen) == 1:
        return seen[0]
    return "mixed samples per curve (" + "; ".join(seen) + ")"

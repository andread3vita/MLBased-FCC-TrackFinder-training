"""Interactive validation-event visualizations shared by CIRCE and GATr."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shared_training.tracking_metrics import greedy_cluster


def validation_scatter_figure(points, labels, colour_title, title, axis_titles):
    """Create an interactive 2D or 3D one-trace-per-particle figure."""
    import plotly.graph_objects as go

    points = np.asarray(points)
    labels = np.asarray(labels, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(
            "Validation plotting requires an array with two or three columns; "
            f"received shape {points.shape}."
        )
    if points.shape[0] != labels.shape[0]:
        raise ValueError(
            "Validation plotting received different numbers of points and labels: "
            f"{points.shape[0]} and {labels.shape[0]}."
        )

    traces = []
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        label_text = "unassigned" if label < 0 else str(int(label))
        hover = (
            f"{colour_title}: {label_text}<br>"
            f"{axis_titles[0]}=%{{x:.3f}}<br>"
            f"{axis_titles[1]}=%{{y:.3f}}"
        )
        if points.shape[1] == 2:
            trace = go.Scatter(
                x=points[mask, 0], y=points[mask, 1], mode="markers",
                name=label_text, marker={"size": 5},
                hovertemplate=hover + "<extra></extra>",
            )
        else:
            trace = go.Scatter3d(
                x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                mode="markers", name=label_text, marker={"size": 3},
                hovertemplate=(
                    hover
                    + f"<br>{axis_titles[2]}=%{{z:.3f}}<extra></extra>"
                ),
            )
        traces.append(trace)

    figure = go.Figure(traces)
    layout = {
        "title": title,
        "legend": {"title": {"text": colour_title}, "itemsizing": "constant"},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 45},
    }
    if points.shape[1] == 2:
        layout.update(
            xaxis={"title": axis_titles[0]},
            yaxis={"title": axis_titles[1]},
        )
    else:
        layout["scene"] = {
            "xaxis_title": axis_titles[0],
            "yaxis_title": axis_titles[1],
            "zaxis_title": axis_titles[2],
        }
    figure.update_layout(**layout)
    return figure


def embedding_plot_coordinates(coords):
    """Return direct 2D/3D coordinates or deterministic three-component PCA."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            "Embedding visualization requires at least two dimensions; "
            f"received shape {coords.shape}."
        )
    embedding_dim = coords.shape[1]
    if embedding_dim == 2:
        return coords, ("embedding 0", "embedding 1"), "2D embedding"
    if embedding_dim == 3:
        return (
            coords,
            ("embedding 0", "embedding 1", "embedding 2"),
            "3D embedding",
        )

    centered = coords - np.mean(coords, axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:3].copy()
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    projected = centered @ components.T
    if projected.shape[1] < 3:
        projected = np.pad(
            projected, ((0, 0), (0, 3 - projected.shape[1])), mode="constant"
        )
    return projected, ("PC1", "PC2", "PC3"), f"PCA of {embedding_dim}D embedding"


def validation_event_media(
    event,
    working_point,
    *,
    rejected_seed_policy,
    output_dir=None,
    include_embedding=True,
):
    """Build and optionally save interactive media for one validation event."""
    reco_labels = greedy_cluster(
        event["beta"],
        event["coords"],
        float(working_point["tbeta"]),
        float(working_point["td"]),
        int(working_point["min_hits"]),
        rejected_seed_policy=rejected_seed_policy,
    )
    figures = {
        "plots/validation_event_0/hits_by_mc_particle": validation_scatter_figure(
            event["positions"], event["mc_particle_id"], "MC particle index",
            "Validation event 0: hits coloured by MC particle index",
            ("x", "y", "z"),
        ),
        "plots/validation_event_0/hits_by_reconstructed_particle": (
            validation_scatter_figure(
                event["positions"], reco_labels, "Reconstructed particle index",
                "Validation event 0: hits coloured by reconstructed index",
                ("x", "y", "z"),
            )
        ),
    }
    if include_embedding:
        points, axes, description = embedding_plot_coordinates(event["coords"])
        figures.update({
            "plots/validation_event_0/embedding_by_mc_particle": (
                validation_scatter_figure(
                    points, event["mc_particle_id"], "MC particle index",
                    "Validation event 0: embedding coloured by MC particle "
                    f"index ({description})",
                    axes,
                )
            ),
            "plots/validation_event_0/embedding_by_reconstructed_particle": (
                validation_scatter_figure(
                    points, reco_labels, "Reconstructed particle index",
                    "Validation event 0: embedding coloured by reconstructed "
                    f"index ({description})",
                    axes,
                )
            ),
        })

    html_payloads = {
        key: figure.to_html(full_html=False, include_plotlyjs="cdn")
        for key, figure in figures.items()
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for key, html in html_payloads.items():
            relative_key = key[len("plots/"):] if key.startswith("plots/") else key
            filename = relative_key.replace("/", "_") + ".html"
            (output / filename).write_text(html)
    return html_payloads

#!/usr/bin/env python3
"""Export the exact geometry-only training GATR model from a checkpoint to ONNX.

The input always contains seven geometric columns.
The model core matches ``src.models.Gatr_withModifications.ExampleWrapper`` and
therefore uses the exact vendored GATr v1.4.2 implementation used in training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import onnx
import torch
from torch import nn
from torch.backends import opt_einsum as torch_opt_einsum

from event_features import load_real_event_features

TRAINING_ROOT = Path(__file__).resolve().parents[1] / "model_training" / "GATR"
GATR_ROOT = TRAINING_ROOT / "src" / "gatr_v142"
if not (GATR_ROOT / "gatr" / "__init__.py").is_file():
    raise ImportError(f"Vendored GATr v1.4.2 package not found at {GATR_ROOT}")
sys.path.insert(0, str(GATR_ROOT))
sys.path.insert(0, str(TRAINING_ROOT))

import gatr
from gatr.interface import embed_point, embed_scalar, embed_translation
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.nets.gatr import GATr
from gatr.utils.einsum import enable_cached_einsum
from src.utils.parser_args import parser as training_parser


GEOMETRIC_MODEL_FEATURE_COUNT = 7


def assert_vendored_gatr() -> None:
    actual = Path(gatr.__file__).resolve()
    expected = (GATR_ROOT / "gatr").resolve()
    if expected not in actual.parents or gatr.__version__ != "1.4.2":
        raise ImportError(
            "Conversion must use the vendored GATr v1.4.2 at "
            f"{expected}; imported {gatr.__version__} from {actual}"
        )


assert_vendored_gatr()


def configure_export_einsum() -> None:
    """Select the symbolic-shape-safe einsum path used by export and comparison."""
    enable_cached_einsum(False)
    # torch.einsum otherwise delegates three-or-more-operand contractions back
    # to Python opt_einsum, which converts symbolic dimensions to concrete ints
    # and silently specializes the exported hit dimension to the example size.
    torch_opt_einsum.enabled = False


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_state(checkpoint: Mapping, weights: str) -> tuple[Mapping, str]:
    ema = checkpoint.get("ema_state_dict")
    has_ema = isinstance(ema, Mapping) and bool(ema)
    if weights == "ema" and not has_ema:
        raise ValueError("The checkpoint does not contain ema_state_dict")
    if weights == "ema" or (weights == "auto" and has_ema):
        return ema, "ema"
    return checkpoint.get("state_dict", checkpoint), "raw"


def strip_lightning_prefix(state: Mapping) -> dict[str, torch.Tensor]:
    tensor_state = {key: value for key, value in state.items() if torch.is_tensor(value)}
    marker = "clustering.weight"
    matching = [key for key in tensor_state if key.endswith(marker)]
    if len(matching) != 1:
        raise ValueError(f"Expected one key ending in {marker!r}, found {matching}")
    prefix = matching[0][:-len(marker)]
    return {
        key[len(prefix):] if prefix and key.startswith(prefix) else key: value
        for key, value in tensor_state.items()
    }


def make_training_args(checkpoint: Mapping):
    args = training_parser.parse_args([])
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    vars(args).update(hyper_parameters)
    if bool(hyper_parameters.get("use_detector_features", False)) or int(
        hyper_parameters.get("detector_scalar_dim", 0)
    ) != 0:
        raise ValueError(
            "Detector-feature checkpoints are not supported. Train a new "
            "geometry-only checkpoint with the current training code."
        )
    vars(args).pop("use_detector_features", None)
    vars(args).pop("detector_scalar_dim", None)
    vars(args).pop("detector_feature_names", None)
    return args


class ExactTrainingExportModel(nn.Module):
    """The inference part of the training model, without Lightning or DGL."""

    def __init__(self, args):
        super().__init__()
        self.position_scale = float(getattr(args, "position_scale", 1000.0))
        self.embedding_dim = int(args.clustering_space_dim)
        self.input_feature_count = GEOMETRIC_MODEL_FEATURE_COUNT
        self.register_buffer(
            "_ggtf_input_feature_count",
            torch.tensor(self.input_feature_count, dtype=torch.int64),
            persistent=False,
        )
        self._use_helix_proxy = float(getattr(args, "helix_loss_weight", 0.0)) > 0.0

        self.gatr = GATr(
            in_mv_channels=1, out_mv_channels=1,
            hidden_mv_channels=int(getattr(args, "hidden_mv_channels", 16)),
            in_s_channels=None, out_s_channels=None,
            hidden_s_channels=int(getattr(args, "hidden_s_channels", 64)),
            num_blocks=int(getattr(args, "gatr_blocks", 10)),
            attention=SelfAttentionConfig(), mlp=MLPConfig(),
            checkpoint=None,
        )
        self.clustering = nn.Linear(16, self.embedding_dim, bias=False)
        self.beta = nn.Linear(16, 1)
        self.helix_proxy = nn.Linear(16, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pos = features[:, 0:3] / self.position_scale
        hit_type = features[:, 3:4]
        vector = features[:, 4:7] / self.position_scale
        multivectors = (
            embed_point(pos) + embed_scalar(hit_type) + embed_translation(vector)
        ).unsqueeze(1)
        # The training wrapper supplies one data-derived join reference per
        # event.  GATr's default ``join_reference="data"`` assumes an explicit
        # batch dimension and, for this [num_hits, 1, 16] tensor, would average
        # over the singleton channel dimension instead of over the hits.
        join_reference = multivectors.mean(dim=0, keepdim=True).expand_as(multivectors)
        # One tensor is one event, so the block-diagonal graph mask is the
        # identity event block and can be omitted for ONNX export.
        embedded, _ = self.gatr(
            multivectors,
            scalars=None,
            attention_mask=None,
            join_reference=join_reference,
        )
        latent = embedded[:, 0, :]
        return torch.cat(
            (self.clustering(latent), self.beta(latent),
             self.helix_proxy(latent) if self._use_helix_proxy else latent.new_zeros((latent.shape[0], 4))),
            dim=1,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--weights-source",
        "--weights",
        dest="weights_source",
        choices=("ema", "raw"),
        default="ema",
        help="checkpoint weights to export (default: EMA weights used in validation)",
    )
    p.add_argument("--num-hits", type=int, default=16)
    p.add_argument("--skip-runtime-check", action="store_true")
    verification = p.add_mutually_exclusive_group(required=True)
    verification.add_argument(
        "--verify-parquet",
        type=Path,
        help="require numerical agreement on a real event before succeeding",
    )
    verification.add_argument(
        "--skip-numerical-check",
        action="store_true",
        help="export without a real-event numerical check (unsafe for deployment)",
    )
    p.add_argument("--event-index", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    cli = p.parse_args()

    if cli.num_hits < 1:
        raise ValueError("--num-hits must be positive")
    if cli.event_index < 0:
        raise ValueError("--event-index must be non-negative")
    if cli.output.suffix.lower() != ".onnx":
        raise ValueError("output must have the .onnx suffix")
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(cli.checkpoint)

    checkpoint = torch_load(cli.checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint is not a dictionary")
    selected, source = select_state(checkpoint, cli.weights_source)
    state = strip_lightning_prefix(selected)

    training_args = make_training_args(checkpoint)
    # Dynamic ONNX dimensions are represented by unhashable SymInts. GATr's
    # cached contraction-path lookup cannot use those as cache keys, while its
    # documented non-cached einsum implementation is export-compatible and
    # mathematically identical.
    configure_export_einsum()
    model = ExactTrainingExportModel(training_args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. Missing keys: {missing}; "
            f"unexpected keys: {unexpected}"
        )

    model.eval()
    dummy = torch.zeros(cli.num_hits, model.input_feature_count, dtype=torch.float32)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        # GATr v1.4.2 lazily materializes immutable algebra and join kernels.
        # Populate those caches eagerly because torch.load and tensor mutation
        # used to construct them cannot be captured inside a Dynamo graph.
        model(dummy)

        from torch.export import Dim

        # Use the Dynamo core directly: the public compatibility wrapper tries
        # to down-convert this graph and fails on its function attributes.
        from torch.onnx._internal.exporter import _core

        onnx_program = _core.export(
            model,
            args=(dummy,),
            input_names=["features"],
            output_names=["predictions"],
            dynamic_shapes={
                # GATr contains a few shape-dependent reshapes. Giving the
                # exporter a finite, realistic range avoids a PyTorch 2.6
                # symbolic-shape constraint while retaining variable events.
                "features": {0: Dim("num_hits", min=1, max=100000)},
            },
        )
        # Saving the ONNXProgram directly avoids PyTorch's compatibility
        # down-conversion pass, which is currently broken for this graph.
        onnx_program.save(str(cli.output))

    exported = onnx.load(str(cli.output))
    onnx.checker.check_model(exported)
    print(f"Exported exact {source} training weights to {cli.output}")
    print(f"Input: features[num_hits, {model.input_feature_count}]")
    print("Model input: geometry only; detector features are unsupported")
    print("Output: predictions[num_hits, embedding_dim + 5]")

    if not cli.skip_runtime_check:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(cli.output), providers=["CPUExecutionProvider"]
        )
        output = session.run(["predictions"], {"features": dummy.numpy()})[0]
        expected = model.embedding_dim + 5
        if output.shape != (cli.num_hits, expected):
            raise RuntimeError(
                f"ONNX Runtime returned {output.shape}, expected "
                f"{(cli.num_hits, expected)}"
            )
        print(f"ONNX Runtime check passed: output shape={output.shape}")

    if cli.verify_parquet is not None:
        import onnxruntime as ort

        features = load_real_event_features(cli.verify_parquet, cli.event_index)
        with torch.inference_mode():
            checkpoint_output = model(features).cpu()
        session = ort.InferenceSession(
            str(cli.output), providers=["CPUExecutionProvider"]
        )
        onnx_output = torch.from_numpy(
            session.run(None, {"features": features.numpy()})[0]
        )
        difference = torch.abs(checkpoint_output - onnx_output)
        maximum = float(difference.max())
        mean = float(difference.mean())
        close = torch.allclose(
            checkpoint_output,
            onnx_output,
            rtol=cli.rtol,
            atol=cli.atol,
        )
        print(
            "Real-event numerical check: "
            f"event={cli.event_index}, hits={features.shape[0]}, "
            f"max_abs_diff={maximum:.8g}, mean_abs_diff={mean:.8g}, "
            f"close={close} (rtol={cli.rtol:g}, atol={cli.atol:g})"
        )
        if not close:
            raise RuntimeError(
                "ONNX output differs from the checkpoint-derived local GATr "
                "model on the real event; refusing to report a verified export"
            )


if __name__ == "__main__":
    main()

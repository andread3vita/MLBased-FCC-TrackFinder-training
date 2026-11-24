import torch

from src.models.Gatr_withModifications import ExampleWrapper, GATR_BACKEND
from src.utils.detector_features import (
    DETECTOR_FEATURE_NAMES,
    DETECTOR_SOURCE_FEATURE_NAMES,
)


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs) -> None:
        super().__init__()
        self.mod = ExampleWrapper(args, dev=dev, **kwargs)

    def forward(self, g):
        return self.mod(g)


def get_model(data_config, args, dev, **kwargs):
    # pf_features_dims = len(data_config.input_dicts['pf_features'])
    # num_classes = len(data_config.label_value)
    print("Model options: ", kwargs)
    print("GATr backend:", GATR_BACKEND, flush=True)
    hit_features = tuple(data_config.input_dicts.get("hits_features", ()))
    use_detector_features = bool(getattr(args, "use_detector_features", False))
    if use_detector_features:
        configured_detector_features = hit_features[
            -len(DETECTOR_SOURCE_FEATURE_NAMES) :
        ]
        if configured_detector_features != DETECTOR_SOURCE_FEATURE_NAMES:
            raise ValueError(
                "--use-detector-features requires a data config whose "
                "hits_features list ends with "
                f"{DETECTOR_SOURCE_FEATURE_NAMES}; found "
                f"{configured_detector_features}"
            )
        args.detector_scalar_dim = len(DETECTOR_FEATURE_NAMES)
        print(
            "Model input: seven geometric values plus computed detector scalars "
            f"{DETECTOR_FEATURE_NAMES}",
            flush=True,
        )
    else:
        configured_detector_features = tuple(
            name for name in hit_features if name in DETECTOR_SOURCE_FEATURE_NAMES
        )
        if configured_detector_features:
            raise ValueError(
                "Geometry-only mode received a detector-feature data config; "
                "use config_tracking_parquet.yaml or enable "
                "--use-detector-features"
            )
        args.detector_scalar_dim = 0
        print("Model input: seven geometry-only features", flush=True)
    model = GraphTransformerNetWrapper(args, dev, **kwargs)

    model_info = {
        "gatr_backend": GATR_BACKEND,
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):

    return torch.nn.MSELoss()

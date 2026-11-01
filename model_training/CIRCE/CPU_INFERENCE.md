# Optional fused CPU inference

The CPU/FP32 path replaces each CGA geometric product's two dense `einsum`
operations with one registered LibTorch operator. It evaluates the nonzero
Cayley-table terms directly, in parallel over multivectors, without allocating
the `(..., 32, 32)` intermediate. This is what **fused C++** means here; the rest
of the network still runs through PyTorch/LibTorch. Weights and precision stay
unchanged. Floating-point reduction order can change, so outputs are not
promised to be bit-identical to the original dense implementation.

Preparation also expands shared K/V heads as views so CPU SDPA can select its
flash implementation, and resolves the fixed grade ranges before tracing the
existing LayerNorm calculation. Everything is opt-in, on a disposable model
after loading its checkpoint. The original training and CUDA paths stay as-is.

## Build and use

From `model_training/CIRCE`, with the model's PyTorch environment active:

```bash
python tools/build_cpu_gp.py --native
CIRCE_CPU_LIBRARY="$PWD/build/libcirce_cpu.so" python -m pytest -q src/cgatr/tests/test_cpu_inference.py
```

The builder needs a Linux C++17 compiler with OpenMP. `--native` targets the
build host's CPU; omit it for a generic build or rebuild on the deployment host.
Build against the same PyTorch/LibTorch version and C++ ABI used for inference.
No extension is built or loaded during normal training.

After loading the original EMA checkpoint into a CPU/FP32 evaluation model:

```python
import torch
from torch import nn
from src.cgatr.cpu_inference import prepare_cpu_inference

model = prepare_cpu_inference(model.cpu().eval(), "build/libcirce_cpu.so")

class SingleEvent(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not model.invariant_output_head:
            raise ValueError("This wrapper requires the invariant scalar head")
        self.model = model

    def forward(self, features):
        mv, scalars = self.model.embed(features)
        _, output = self.model.cgatr(mv, scalars=scalars, attention_mask=None)
        return output  # clustering coordinates and the raw beta logit

# features is a real complete event's existing CIRCE float32 [N, 10] input,
# containing both vertex and drift hits. No truth features enter the model.
runner = SingleEvent(model).eval()
with torch.inference_mode():
    traced = torch.jit.trace(runner, (features,), check_trace=False, strict=False)
    traced = torch.jit.freeze(traced.eval())
    torch.testing.assert_close(traced(features), runner(features), atol=1e-4, rtol=1e-4)
    traced.save("circe_cpu.pt")
```

Check the trace on additional event sizes and vertex-only/drift-only inputs
before deployment. Keep the original checkpoint for training; the prepared
module is inference-only. The custom product supports equal-shaped CPU/FP32
inputs ending in 32 blades, including noncontiguous tensors.

Load the operator **before** loading the TorchScript model. In Python, use
`torch.ops.load_library("build/libcirce_cpu.so")` before `torch.jit.load`.
For an existing Linux LibTorch executable, no runner rewrite is needed:

```bash
LD_PRELOAD="$PWD/build/libcirce_cpu.so" ./inference circe_cpu.pt
```

## Measurements

Xeon Gold 6130, four distinct physical cores, PyTorch/LibTorch 2.8.0, FP32,
8 blocks / 12 multivector channels / 64 scalar channels, 759,822 parameters.
The 1,000 fixed events were sampled uniformly within evaluation seeds 981–990
(100 per seed, RNG 20261031), without a hit cap: mean 3,959 hits, maximum 13,410.
Three warmup events were excluded; two full timed passes were averaged per
event. Confidence intervals resample events within source seeds.

| Implementation | Mean native pipeline (95% CI) | p95 | Reconstruction efficiency | Unmatched candidates |
|---|---:|---:|---:|---:|
| Existing C++ / optimized TorchScript | 6.169 s (5.937–6.423) | 13.485 s | 95.741% | 6.974% |
| Fused C++ prototype / frozen TorchScript | 2.340 s (2.236–2.450) | 6.044 s | 95.733% | 6.974% |

Both rows already use expanded-head CPU attention. The fused path is 2.64x
faster with a 0.009 percentage-point efficiency difference and passes the
existing reconstruction gates, including paired confidence checks. The native
pipeline includes geometry, the network, greedy clustering and fragment merging;
it excludes I/O and detector/EDM feature extraction. Efficiency uses the same
fixed cuts (beta 0.6, clustering distance 0.1, merge distance 0.1) and merged
one-to-one matching for generation-status 0+1 targets with more than three hits.

The actual PR library/export was also checked against that prototype: raw
predictions and raw/merged labels match bit for bit on all 50 diagnostic events.
Checkpoint trace/reload checks include a 13,410-hit event. A full 10,000-event
timing run of the PR backend is queued separately.

The matched comparison below uses the same 50 diagnostic events on cores
1,3,5,7 (the 1,000-event benchmark above uses 0,2,4,6), three warmup events and
one timed pass. It measures geometry plus network, excluding clustering and I/O.
This diagnostic mean is not the 1,000-event population estimate above.

| Implementation | Mean geometry + network |
|---|---:|
| PyTorch eager, stock attention | 13.468 s |
| Existing C++ / optimized TorchScript, expanded-head CPU SDPA | 6.495 s |
| This PR's fused C++ / frozen TorchScript, expanded-head CPU SDPA | 2.696 s |

The C++ executable runs TorchScript through LibTorch. Changing the language
alone does not explain these gains: CPU attention dispatch and the fused
geometric product both contribute.

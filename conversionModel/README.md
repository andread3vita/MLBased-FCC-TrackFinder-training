# GATr checkpoint → ONNX conversion

Converts a GATR_CIRCE_LOSS Light­ning checkpoint (`.ckpt`) into a standalone
ONNX model for GGTFTrackFinder-style inference, without needing the training
stack (no DGL, no Lightning).

## Environment

```bash
conda create -n model-conversion python=3.10
conda activate model-conversion
pip install torch==2.6.0 onnx==1.17.0 onnxruntime==1.20.1 einops xformers==0.0.29.post3 opt-einsum
```

(No GPU is needed; everything runs on CPU.)

## Usage

```bash
conda activate model-conversion
cd conversionModel
python convert_to_onnx.py -w model.ckpt -o model.onnx
python validate_onnx.py -w model.ckpt -o model.onnx                 # seeded random 30x7 input
python validate_onnx.py -w model.ckpt -o model.onnx -i model_input_event0.csv   # real event
```

## Precision recipes

| variant | command flags | what it does | measured event time (IDEA digi, ~4k hits, 4 intra-op threads, CPU) |
|---|---|---|---|
| **full fp32 (default)** | (none) | **everything in fp32** | **~3.6 s** |
| fp32 attention block | `--compute-dtype float64 --attention-dtype float32_block` | whole attention sub-block in fp32, rest fp64 | ~4.3 s |
| fp32 attention core | `--compute-dtype float64 --attention-dtype float32` | softmax core in fp32, rest fp64; attention ops fused by ORT | ~9 s |
| fp32 attention core, no fusion | `--compute-dtype float64 --attention-dtype float32_einsum` | same region as above but einsum-form ONNX ops | ~11 s |
| float64 (reference) | `--compute-dtype float64` | whole model in float64 | ~15 s |

Running with defaults exports a checkpoint in full-fp32:

```bash
python convert_to_onnx.py -w model.ckpt -o model_fp32full.onnx
```

Use `--compute-dtype float64` to produce the float64 reference model for
validation A/B against a new checkpoint; `--compute-dtype float32` implies the
whole compute stack (embedding, layernorms, MLPs, attention, heads) runs in
float32. The script prints a smoke check
`max |ORT - eager|` right after export: expect ~1e-5–3e-4 for float32
variants, ~1e-6 for float64.

Both modalities are validated against the float64 export of the *same* checkpoint
on real events (13-event track building bit-identical for fp32_block and
full fp32 on the validation sample; embedding shift max ~5e-3 for full fp32).
Always re-run your efficiency/fake-rate comparison before adopting a
precision-reduced model in production.

Notes:

* The float32 variants have no float64-fused-kernel issue; in float64 mode
  ORT has no fused MatMul kernel and everything runs through the generic
  einsum path.
* `--weights-source` (`ema` default, or `raw`) selects which checkpoint
  weights to convert, unchanged.

## Creating the input CSV from an EDM4hep file

`dump_model_input.py` reads the digitized hits of one event from an EDM4hep
ROOT file and writes the `[N, 7]` model input (`x,y,z,planar_flag,u,v,w`) with
the same encoding and hit ordering as `GGTFTrackFinder.cpp` (planar collections
first, then wire collections). It needs the podio/EDM4hep Python bindings, so
run it in a key4hep environment (not the conda one):

```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
python dump_model_input.py -i digitized_events.root -o model_input_event0.csv -e 0
```

Options: `-e` selects the event index (default 0); `--planar` and `--wire`
override the collection names (defaults: `VTXBDigis VTXDDigis SiWrBDigis SiWrDDigis`
and `DCH_DigiCollection`, as in `runTestTrackFinder.py`).


## Fast FP32 export (`convert_to_onnx_fast.py`)

Same input/output contract (`hits [N,7]` -> `output [N,9]`), built from
`model_training/GATR_EXPERIMENT/src/models/onnx_export.py` (folded/fused
linears, outer-product + GEMM geometric product and join, affine embedding, multi-query
attention). Works for GATR_EXPERIMENT and GATR_CIRCE_LOSS checkpoints; uses
the same conversion environment as above, nothing extra.

```bash
python convert_to_onnx_fast.py -w model.ckpt -o model_fast.onnx -i model_input_event0.csv
python convert_to_onnx_fast.py -w model.ckpt -o model_mha.onnx --attention ort-mha
python benchmark_onnx.py model_fp32full.onnx model_fast.onnx -i model_input_event0.csv --threads 4
```

| `--attention` | ONNX | notes |
|---|---|---|
| `blocked` (default) | standard | Loop over 256-row query blocks, every query sees all keys; bounded memory for large events |
| `ort-mha` | ONNX Runtime only (`com.microsoft.MultiHeadAttention`) | run with `session.set_denormal_as_zero=1` |
| `dense` | standard | plain MatMul/Softmax/MatMul |

`blocked`/`dense` also add guards (|Q|,|K|,|V| < 1e-12 -> 0, logits floored
at row max - 30) unless `--no-guards` is given. The converter prints the max
difference to the checkpoint's float64 eager model (with the FP32 eager
difference for scale) and the ONNX time per event, and writes a `.json`
sidecar with the settings. `benchmark_onnx.py` times several models on the
same events (`--flush-denormals` sets `session.set_denormal_as_zero=1`,
`--profile` writes ONNX Runtime profiles).

On the development machine (AMD EPYC 7302, 4 pinned threads, random-weight
10/16/64 model, synthetic 4,000-hit event, shared and loaded host): current
FP32 export 9.1 s, `blocked` ~3.6 s, `ort-mha` ~3.1-3.5 s (flush denormals
on). About 2.2 s of this is the attention core itself (10 blocks x 8 heads x
N^2). Exact rewrites behind it: the geometric product and join are an outer
product + GEMM (a gathered sparse form was memory bound, ~1.7 s); wide
equivariant linears use blade-sparse batched GEMMs (the 9 basis maps touch
24 of 256 entries per 16x16 block, 1.5-2x faster than the dense fold); Q/K
are no longer zero-padded to the value width (64 instead of 80 features in
the score MatMul). Re-measure with a trained
checkpoint and real events, and re-check efficiency/fake rate, before use;
subnormal effects in particular depend on the trained weights.

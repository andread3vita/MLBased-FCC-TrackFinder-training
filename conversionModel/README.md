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


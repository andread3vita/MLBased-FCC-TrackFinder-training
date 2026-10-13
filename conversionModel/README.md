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
cd efficientConversion
python convert_to_onnx.py -w model.ckpt -o model.onnx
python validate_onnx.py -w model.ckpt -o model.onnx                 # seeded random 30x7 input
python validate_onnx.py -w model.ckpt -o model.onnx -i model_input_event0.csv   # real event
```

## Creating the input CSV from an EDM4hep file

`dump_model_input.py` reads the digitized hits of one event from an EDM4hep
ROOT file and writes the `[N, 7]` model input (`x,y,z,planar_flag,u,v,w`) with
the same encoding and hit ordering as `GGTFTrackFinder.cpp` (planar collections
first, then wire collections). It needs the podio/EDM4hep Python bindings, so
run it in a key4hep environment (not the conda one):

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
python dump_model_input.py -i digitized_events.root -o model_input_event0.csv -e 0
```

Options: `-e` selects the event index (default 0); `--planar` and `--wire`
override the collection names (defaults: `VTXBDigis VTXDDigis SiWrBDigis SiWrDDigis`
and `DCH_DigiCollection`, as in `runTestTrackFinder.py`).


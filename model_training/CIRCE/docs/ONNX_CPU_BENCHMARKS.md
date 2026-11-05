# CIRCE ONNX CPU inference (2026-11-05)

Export an existing checkpoint once, then run its ONNX file using NumPy and stock ONNX Runtime. Training and the original checkpoint are unchanged. No custom C++ operator, LibTorch, `torch.compile`, quantization or thread pinning is required at deployment.

## Run

From the repository root, activate the usual CIRCE training environment for export:

```bash
cd model_training/CIRCE
python -m pip install onnx==1.21.0 onnxruntime==1.21.1
python tools/export_onnx.py /path/model.ckpt --data-dir /path/idea_parquet --output circe.onnx
python tools/onnx_infer.py circe.onnx circe.example.npy --threads 4 --output prediction.npy
```

`--data-dir` reads one example event (seed 981, index 0 by default); it does not retrain or fit weights. Alternatively supply `--example-features event.npy`. Export prefers checkpoint EMA weights, otherwise Lightning `state_dict` keys prefixed with `model.`. A JSON sidecar records checkpoint/model hashes, architecture, weight source and numerical settings. Only trusted checkpoints should be loaded (`weights_only=False` is explicit).

For the optional faster, ONNX Runtime specific variant:

```bash
python tools/export_onnx.py /path/model.ckpt --data-dir /path/idea_parquet --output circe_tuned.onnx --runtime-tuned
python tools/onnx_infer.py circe_tuned.onnx circe_tuned.example.npy --threads 4 --runtime-tuned --output prediction.npy
```

The inference example imports only NumPy and ONNX Runtime. Export additionally uses PyTorch and the existing CIRCE model/data dependencies. The commands run from `model_training/CIRCE` because existing model initialization reads `cga_utils` tables relative to that directory. Session initialization is outside the example's reported forward time; its single cold forward is not a replacement for the warmed multi-event benchmark below.

## Model contract and implementation

The supported checkpoints use 32-blade conformal `sphere_circle` encoding, no time column, an invariant scalar output head and shared K/V attention. Input is one nonempty event, contiguous **FP32 `[N,10]`** measured features in dataset order: `x,y,z,hit_type,wire_x,wire_y,wire_z,drift_distance,alpha,stereo_angle`. Positions/radii use the dataset's millimetres, angles radians; hit type is 0 for vertex or 1 for drift chamber. There is no hit-count truncation. Geometry embedding is included in the graph and the hit count/detector mixture are dynamic. For the two tested checkpoints the output is **`[N,5]`**: four condensation coordinates followed by a **raw beta logit**. Apply sigmoid to that last column before the existing clustering; clustering and EDM feature extraction are outside this graph.

Both export paths fold the fixed equivariant linear basis into ordinary dense FP32 weights. The 32-component Cl(4,1) geometric product uses a faithful 4x4 complex matrix representation encoded with real ONNX MatMul operations, verified against all 1,024 blade pairs. These algebraic rewrites preserve the mathematical operations but change FP32 accumulation order. The larger dense serialized weights are a tradeoff: about 26.5 MB for 6/8/96 and 71.9 MB for 8/12/64, despite their smaller trained parameter counts.

**Standard** (default): opset 17, standard ONNX operators only. A Loop evaluates attention in blocks of 256 query rows against **all** keys; it reduces score scratch memory without windowing or pruning attention. Graph guards zero constants and Q/K/V components with absolute value below `1e-12`, and floor logits at `row_max - 30` before softmax. This avoids expensive subnormal arithmetic under normal runtime settings. It is an explicit numerical approximation: at 14,683 keys the added unnormalized softmax mass is bounded by `14683 * exp(-30) = 1.374e-9`, but that is not an end-to-end reconstruction guarantee.

**Runtime-tuned**: an alternative branch from the folded matrix export, replacing attention with stock `com.microsoft.MultiHeadAttention` (scale 1 because Q/K are already scaled). ONNX Runtime selects its CPU attention implementation. Also set `SessionOptions.add_session_config_entry("session.set_denormal_as_zero", "1")`, which flushes subnormal floating-point values to zero. This branch does not use the standard path's graph guards/Loop. It needs ONNX Runtime with the built-in contrib operator and the explicit session setting; it is not a runtime-neutral ONNX file. Neither path is claimed to be bit-identical to original PyTorch. Validate reconstruction for every newly exported checkpoint.

Four intra-op threads, one inter-op thread, sequential execution and the default graph optimization level were used for **both** paths; these common settings are not counted as a runtime-tuned speedup. Benchmark affinity was used only for measurement isolation and is absent from production code. The tested ORT processes imported no PyTorch and loaded no OpenMP library. This does not establish the build settings of another deployment.

## Matched 1,000-event confirmation

CPU: Intel Xeon Gold 6130, four physical cores per process, FP32 CPUExecutionProvider, **ONNX Runtime 1.21.1**. Fixed sample: 100 events from each evaluation seed 981–990; all hits retained. Three excluded warmups followed by two full timed passes. Each event's two times are averaged before computing the displayed distribution. Timing includes geometry + network on preloaded features, excluding I/O, feature extraction, model/session loading and clustering. GPU training continued in the background. Standard and tuned used separate four-core groups on the same socket; the same group was reused for both architectures within a format. Shared-host/load effects remain possible.

| Model (blocks/MV/scalar) | Params | Export | Mean s | Median s | Min s | Max s | P95 s | 95% CI of mean s |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 8/12/64, fully trained | 759,822 | standard | 3.657 | 2.674 | 0.099 | 31.580 | 9.889 | 3.470–3.858 |
| 8/12/64, fully trained | 759,822 | tuned | 2.991 | 2.349 | 0.094 | 20.309 | 7.799 | 2.855–3.134 |
| 6/8/96, step 100k | 693,102 | standard | 2.120 | 1.530 | 0.057 | 18.563 | 5.817 | 2.010–2.237 |
| 6/8/96, step 100k | 693,102 | tuned | 1.614 | 1.225 | 0.051 | 10.815 | 4.273 | 1.537–1.694 |

The 6/8/96 snapshot is still in full training, at step 100,000 (approximately epoch 2.67). The 8/12/64 reference completed epoch 16 on 490k training events. These are frozen checkpoint comparisons; the accuracy difference between models is **not** an equally trained architecture comparison. “Teacher” in historical case names identifies the older reference checkpoint, not a distillation requirement; the current full training started from scratch.

The 1,000-event sample spans 216–13,410 hits. The smallest event from the complete 10,000-event evaluation population (seed 984/event 68, 216 hits) is included. The largest (984/448, 14,683 hits) is not, so it was measured separately with two passes:

| Model | Export | 216-hit event mean s | 14,683-hit event mean s |
|---|---|---:|---:|
| 8/12/64 | standard | 0.100 | 38.051 |
| 8/12/64 | tuned | 0.097 | 24.106 |
| 6/8/96 | standard | 0.059 | 25.851 |
| 6/8/96 | tuned | 0.055 | 13.378 |

These are timings of the population's **size extremes**, not measured latency extrema across all 10,000 events. They are not included as extra observations in the 1,000-event mean.

## Reconstruction check on the same 1,000 events

Paired clustering reuses the reference thresholds for each checkpoint: current 6/8/96 uses beta 0.001/distance 0.2; previous 8/12/64 uses beta 0.6/distance 0.1; both use merge distance 0.1. The metrics below are merged one-to-one efficiency for charged targets with generator status 0 or 1, pT > 0.1 GeV, 15° < theta < 165°, and more than three hits (`merged/gen01/n3`) and the merged unmatched-candidate fraction (`merged/cand3`). All four exports passed the recorded reconstruction gates, including paired bootstrap checks whose lower 95% efficiency-difference bounds exceed -0.5 percentage points. Gate passing does not establish zero accuracy loss. Small floating-point changes can change clustering near a threshold.

| Model | Export | Efficiency reference → ONNX % | Unmatched reference → ONNX % |
|---|---|---:|---:|
| 8/12/64 | standard | 95.7325 → 95.7413 | 6.9741 → 6.9558 |
| 8/12/64 | tuned | 95.7325 → 95.7281 | 6.9741 → 6.9441 |
| 6/8/96 | standard | 88.3150 → 88.3150 | 48.8830 → 48.8840 |
| 6/8/96 | tuned | 88.3150 → 88.3106 | 48.8830 → 48.8891 |

## Controlled progression from baseline PyTorch

A separate matched **50-event** screen uses the same events, four-core groups, ORT 1.21.1, three excluded warmups and one pass per stage. Baseline is original PyTorch 2.8.0 eager FP32 (original geometric product, feature embedding and shared-head attention), with inference mode, four intra-op/one inter-op threads and no compile/custom kernel. These stage means must not be substituted for the 1,000-event means above.

| Stage | 8/12/64 mean s | 6/8/96 mean s | What this measurement adds |
|---|---:|---:|---|
| Original PyTorch | 17.052 | 10.418 | Baseline |
| ONNX + matrix product + folded linears | 11.579 | 7.481 | Bundles conversion and algebra/linear rewrites; no isolated ONNX-only attribution |
| + blocked full attention | 8.986 | 5.993 | 256 query rows, all keys; numerical stabilization still absent |
| + portable graph guards | pending | 2.231 | Zero tiny constants/QKV and relative-logit floor; default export |
| Alternative: ORT built-in attention | pending | 6.355 | Branches from matrix/linear stage, without FTZ |
| + flush subnormals | pending | 1.736 | Same native graph, session.set_denormal_as_zero=1; runtime-tuned export |

For 6/8/96 on this 50-event screen, peak process RSS fell from 15,459.9 MiB in the untiled matrix/linear export to 407.9 MiB with blocking (452.4 MiB with the final guards). RSS includes runtime/model/input memory, not only attention scratch.

The standard and runtime-tuned rows follow different attention branches. Their gains are not additive. A bare ONNX export was not timed separately on this 50-event roster, so the first ONNX row cannot establish how much ONNX alone contributes. Earlier “13+ s PyTorch” values came from older runs and are not reused as a matched denominator here.

## Complete ONNX experiment ledger

Every completed ONNX timing run from this study is included below and in [onnx_cpu_benchmarks.csv](onnx_cpu_benchmarks.csv), including unsuccessful optimization candidates. Three-event probes deliberately use the sample's small/median/P95-size events, so their means do **not** estimate population latency. Fifty-event screens select candidates; the two-pass 1,000-event rows are confirmations. Two-event rows measure size extremes separately. Runtime versions, sample sizes and FTZ are shown to prevent mixing unlike comparisons. Earlier probes used other spare core groups and changing host load; use the controlled progression above for attribution.

Historical names: `baseline` = original dense GP export; `gather` = indexed Cayley contraction; `matrix` = matrix GP (`matrix_packed` used sparse sign/add projection maps, later `matrix_dense_packed` uses MatMul projections); `packed` = folded equivariant linears; `tileNNN` = query blocking; `native`/`mha` = built-in ORT attention; `ftz` = runtime flush-to-zero; `constants` = only tiny-constant cleanup; `clean` = earlier probability-tail cleanup; `scaled` = power-of-two attention rescaling; `guard18` = earlier numerical guard. These diagnostic variants are not additional production options. Exact model hashes, peak process RSS and original case identifiers are retained in the CSV.

| Case | Events × passes | ORT | FTZ | Mean s | Median s | Min s | Max s | P95 s | Reconstruction gate |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `comparison_current_standard_extremes` | 2 × 2 | 1.21.1 | no | 12.955 | 12.955 | 0.059 | 25.851 | 24.561 | not scored |
| `comparison_current_tuned_1000` | 1000 × 2 | 1.21.1 | yes | 1.614 | 1.225 | 0.051 | 10.815 | 4.273 | True |
| `comparison_current_tuned_extremes` | 2 × 2 | 1.21.1 | yes | 6.717 | 6.717 | 0.055 | 13.378 | 12.712 | not scored |
| `comparison_previous_standard_1000` | 1000 × 2 | 1.21.1 | no | 3.657 | 2.674 | 0.099 | 31.580 | 9.889 | True |
| `comparison_previous_standard_extremes` | 2 × 2 | 1.21.1 | no | 19.075 | 19.075 | 0.100 | 38.051 | 36.153 | not scored |
| `comparison_previous_tuned_1000` | 1000 × 2 | 1.21.1 | yes | 2.991 | 2.349 | 0.094 | 20.309 | 7.799 | True |
| `comparison_previous_tuned_extremes` | 2 × 2 | 1.21.1 | yes | 12.102 | 12.102 | 0.097 | 24.106 | 22.906 | not scored |
| `current96_mha_constants1e-12_quick` | 3 × 1 | 1.23.2 | no | 5.821 | 3.480 | 0.064 | 13.920 | 12.876 | not scored |
| `current96_mha_constants1e-20_quick` | 3 × 1 | 1.23.2 | no | 5.886 | 3.405 | 0.064 | 14.190 | 13.111 | not scored |
| `current96_native_ftz_1000` | 1000 × 2 | 1.23.2 | yes | 1.675 | 1.269 | 0.049 | 12.220 | 4.528 | True |
| `current96_native_ftz_quick` | 3 × 1 | 1.23.2 | yes | 1.729 | 1.137 | 0.043 | 4.009 | 3.722 | not scored |
| `current96_native_ftz_screen` | 50 × 2 | 1.23.2 | yes | 1.584 | 1.226 | 0.314 | 8.536 | 3.215 | True |
| `current96_native_quick` | 3 × 1 | 1.23.2 | no | 6.415 | 4.191 | 0.131 | 14.923 | 13.850 | not scored |
| `current96_standard_clean_quick` | 3 × 1 | 1.23.2 | no | 6.524 | 3.870 | 0.109 | 15.592 | 14.420 | not scored |
| `current96_standard_ftz_quick` | 3 × 1 | 1.23.2 | yes | 2.046 | 1.256 | 0.048 | 4.835 | 4.477 | not scored |
| `current96_standard_guard18_quick` | 3 × 1 | 1.23.2 | no | 10.069 | 5.383 | 0.186 | 24.639 | 22.714 | not scored |
| `current96_standard_quick` | 3 × 1 | 1.23.2 | no | 6.087 | 3.893 | 0.122 | 14.245 | 13.210 | not scored |
| `current96_standard_scaled_quick` | 3 × 1 | 1.23.2 | no | 6.597 | 3.625 | 0.126 | 16.041 | 14.799 | not scored |
| `current96_standard_screen` | 50 × 2 | 1.23.2 | no | 5.930 | 4.559 | 1.174 | 34.217 | 12.511 | True |
| `current96_standard_stable30_quick` | 3 × 1 | 1.23.2 | no | 2.762 | 1.878 | 0.103 | 6.305 | 5.862 | not scored |
| `current96_tile256_constants1e-12_quick` | 3 × 1 | 1.23.2 | no | 5.378 | 3.222 | 0.068 | 12.844 | 11.882 | not scored |
| `current96_tile256_constants1e-20_quick` | 3 × 1 | 1.23.2 | no | 5.263 | 3.140 | 0.068 | 12.581 | 11.637 | not scored |
| `portable_default_screen` | 50 × 2 | 1.23.2 | no | 2.315 | 1.708 | 0.433 | 15.918 | 5.009 | True |
| `portable_extremes` | 2 × 2 | 1.23.2 | no | 12.742 | 12.742 | 0.058 | 25.425 | 24.157 | not scored |
| `portable_ort121_1000` | 1000 × 2 | 1.21.1 | no | 2.120 | 1.530 | 0.057 | 18.563 | 5.817 | True |
| `portable_ort121_screen` | 50 × 2 | 1.21.1 | no | 2.300 | 1.657 | 0.429 | 15.365 | 4.945 | True |
| `portable_tile128_quick` | 3 × 1 | 1.23.2 | no | 2.647 | 1.769 | 0.060 | 6.111 | 5.676 | not scored |
| `portable_tile256_quick` | 3 × 1 | 1.23.2 | no | 2.618 | 1.735 | 0.061 | 6.059 | 5.627 | not scored |
| `portable_tile512_quick` | 3 × 1 | 1.23.2 | no | 2.725 | 1.558 | 0.058 | 6.558 | 6.058 | not scored |
| `portable_tile64_quick` | 3 × 1 | 1.23.2 | no | 2.874 | 1.886 | 0.056 | 6.679 | 6.199 | not scored |
| `pr_ablation_current96_blocked` | 50 × 1 | 1.21.1 | no | 5.993 | 4.616 | 1.088 | 34.990 | 12.624 | not scored |
| `pr_ablation_current96_matrix_linear` | 50 × 1 | 1.21.1 | no | 7.481 | 5.726 | 1.222 | 45.184 | 16.505 | not scored |
| `pr_ablation_current96_native` | 50 × 1 | 1.21.1 | no | 6.355 | 4.744 | 1.163 | 35.686 | 13.806 | not scored |
| `pr_ablation_current96_native_ftz` | 50 × 1 | 1.21.1 | yes | 1.736 | 1.321 | 0.345 | 9.280 | 3.643 | not scored |
| `pr_ablation_current96_standard` | 50 × 1 | 1.21.1 | no | 2.231 | 1.582 | 0.435 | 14.928 | 4.637 | not scored |
| `pr_ablation_teacher_blocked` | 50 × 1 | 1.21.1 | no | 8.986 | 6.645 | 1.216 | 58.895 | 20.761 | not scored |
| `pr_ablation_teacher_matrix_linear` | 50 × 1 | 1.21.1 | no | 11.579 | 8.506 | 1.535 | 74.387 | 25.659 | not scored |
| `teacher_baseline_tile256_quick` | 3 × 1 | 1.23.2 | no | 18.824 | 13.961 | 0.569 | 41.943 | 39.145 | not scored |
| `teacher_gather_tile256_quick` | 3 × 1 | 1.23.2 | no | 14.969 | 10.281 | 0.397 | 34.231 | 31.836 | not scored |
| `teacher_matrix_packed_mha_quick` | 3 × 1 | 1.23.2 | no | 12.149 | 8.097 | 0.245 | 28.105 | 26.104 | not scored |
| `teacher_matrix_packed_quick` | 3 × 1 | 1.23.2 | no | 14.757 | 9.545 | 0.257 | 34.470 | 31.977 | not scored |
| `teacher_matrix_packed_tile256_clean_quick` | 3 × 1 | 1.23.2 | no | 10.386 | 6.185 | 0.262 | 24.711 | 22.858 | not scored |
| `teacher_matrix_packed_tile256_ftz_quick` | 3 × 1 | 1.23.2 | yes | 5.195 | 3.879 | 0.201 | 11.505 | 10.742 | not scored |
| `teacher_matrix_packed_tile256_quick` | 3 × 1 | 1.23.2 | no | 12.322 | 7.886 | 0.288 | 28.792 | 26.701 | not scored |
| `teacher_matrix_packed_tile256_screen` | 50 × 1 | 1.23.2 | no | 11.237 | 8.429 | 1.953 | 64.392 | 23.911 | True |
| `teacher_matrix_tile256_quick` | 3 × 1 | 1.23.2 | no | 12.740 | 8.400 | 0.294 | 29.525 | 27.413 | not scored |
| `teacher_mha_constants1e-12_quick` | 3 × 1 | 1.23.2 | no | 11.771 | 7.430 | 0.116 | 27.768 | 25.734 | not scored |
| `teacher_mha_constants1e-20_quick` | 3 × 1 | 1.23.2 | no | 11.672 | 7.107 | 0.106 | 27.804 | 25.735 | not scored |
| `teacher_native_ftz_quick` | 3 × 1 | 1.23.2 | yes | 3.337 | 2.342 | 0.090 | 7.579 | 7.055 | not scored |
| `teacher_native_ftz_screen` | 50 × 2 | 1.23.2 | yes | 2.912 | 2.309 | 0.616 | 15.157 | 6.016 | True |
| `teacher_native_quick` | 3 × 1 | 1.23.2 | no | 11.309 | 6.900 | 0.116 | 26.912 | 24.911 | not scored |
| `teacher_standard_clean_quick` | 3 × 1 | 1.23.2 | no | 8.904 | 4.928 | 0.111 | 21.672 | 19.998 | not scored |
| `teacher_standard_clean_screen` | 50 × 2 | 1.23.2 | no | 7.569 | 5.370 | 0.991 | 51.200 | 17.430 | True |
| `teacher_standard_ftz_quick` | 3 × 1 | 1.23.2 | yes | 3.655 | 2.399 | 0.093 | 8.474 | 7.866 | not scored |
| `teacher_standard_quick` | 3 × 1 | 1.23.2 | no | 10.445 | 6.072 | 0.115 | 25.149 | 23.241 | not scored |
| `teacher_standard_scaled_quick` | 3 × 1 | 1.23.2 | no | 9.019 | 4.460 | 0.114 | 22.484 | 20.681 | not scored |
| `teacher_tile256_constants1e-12_quick` | 3 × 1 | 1.23.2 | no | 10.632 | 6.187 | 0.113 | 25.595 | 23.654 | not scored |
| `teacher_tile256_constants1e-20_quick` | 3 × 1 | 1.23.2 | no | 10.647 | 6.319 | 0.126 | 25.498 | 23.580 | not scored |

One earlier `teacher_matrix_tile256_screen` run was stopped after 9/50 events; its partial mean (10.270 s) is not treated as a completed result. No completed case is silently dropped because it was slow. Model-side numerical variants without reconstruction validation are exploratory, not accepted alternatives.

## Validation and provenance

The focused ONNX tests cover every GP blade pair, broadcast products, linear scalar/bias paths, shared K/V attention, dynamic lengths and partial tiles, and the declared tiny-value guard. Run:

```bash
python -m pytest src/cgatr/tests/test_onnx_inference.py -q
```

Production exports from this PR are additionally compared with the benchmarked artifacts on real events (216, 1,884 and 13,410 hits), reordered hits, vertex-only input and DC-only input. All 24 checks passed bit for bit. A separate structural comparison also confirmed identical operators, connections, attributes and constant tensors for each exported graph, ignoring names and provenance metadata.

- current96/standard: 6 cases, maximum absolute difference 0; bitwise equal on all cases: True.
- current96/tuned: 6 cases, maximum absolute difference 0; bitwise equal on all cases: True.
- teacher/standard: 6 cases, maximum absolute difference 0; bitwise equal on all cases: True.
- teacher/tuned: 6 cases, maximum absolute difference 0; bitwise equal on all cases: True.

Checkpoint and matched 1,000-event model SHA-256 digests:

- current checkpoint: `16f41978efcfdefccd50bc981e8f64466cfce295ea38c6ae5e8eac72f8975643`
- previous checkpoint: `35330d576ed877924744474bd1e4fab95dd34a46afd90375e0674c5630fdb719`
- current_standard benchmarked ONNX: `1f42dd90b51015c23eecb2f531dc69034b2aae215dc116c3762ec242f16657d1`
- previous_standard benchmarked ONNX: `8d2947e02a3246307d51082acc5614c1c454ef9938332b9df3ead9c0ab9822e4`
- previous_tuned benchmarked ONNX: `9cdb42cc90dac05227167d444961f333eb2a6c2b67393c3d68d7ca85839e095a`
- current_tuned benchmarked ONNX: `5d2d729df02281b0c1c281f81392fdbe48a668ebe63250e94052b07b0acf0fe9`
- Sample manifest: `e13c4d9cfeac26331008f1fcd63bc45da5c0ff830319698c0a9a224b9198d2a5`. Sample IDs are reproducible: use one `np.random.default_rng(20261031)` and, for each seed 981 through 990 in order, sort its `rng.choice(500, 100, replace=False)` event IDs. Event IDs range from 0 to 499. Select the 50-event screen by sorted `choice(1000, 50, replace=False)` indices from a fresh generator with the same seed.

The new export files have different binary hashes because graph names/provenance metadata differ. Structural graph equality and numerical parity with the benchmarked artifacts were checked separately. Trained checkpoints, full feature arrays and generated ONNX binaries are intentionally not committed.

Compatibility: opset 17, tested ORT 1.21.1 and earlier exploratory ORT 1.23.2 runs. Actual downstream CERN runtime/input contract is unconfirmed. The inspected [Key4hep constraints](https://github.com/key4hep/key4hep-spack/blob/4370a3980b749429ddc889ef589f326af2465058/environments/key4hep-common/packages.yaml) require ORT >=1.18, but package constraints are not proof of an installed version. The portable standard-operator path avoids the tuned path's contrib-operator requirement; the downstream runtime still needs support for the standard graph, including Loop.

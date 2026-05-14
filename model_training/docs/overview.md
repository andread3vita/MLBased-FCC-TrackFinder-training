# CGATr-Tracking

## 1. The model

```
input per hit (10 floats)
    ┌ pos_xyz [3]               (detector position)
    ├ hit_type [1]              (0=VTX silicon, 1=DC drift chamber)
    ├ wire_xyz [3]              (DC only — wire position)
    ├ drift [1]                 (DC only — drift radius)
    └ azimuthal, stereo [2]     (DC only — wire orientation)
                ▼
   ┌─ BatchNorm1d(pos_xyz)     normalise positions to ~unit scale
   ├─ BatchNorm1d(wire_xyz)    same for wires
   └─ BatchNorm1d(drift)       same for drift
                ▼
   ┌── if VTX: embed_point(pos_xyz)         → 32-dim CGA multivector
   └── if DC:  embed_circle_ipns(...)       → 32-dim CGA multivector
                ▼
   + embed_scalar(hit_type)                 → still 32-dim
                ▼
   reshape to (N, 1, 32)   — 1 multivector channel per hit
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ CGATr backbone — 10 transformer blocks, identical structure each:│
│                                                                  │
│   ┌─ EquivariantSelfAttention                                    │
│   │     • Q,K,V are 16-mv-ch + 64-scalar-ch each                 │
│   │     • CGA-equivariant inner product as attention metric      │
│   │     • BlockDiagonalMask (xformers) keeps events separate     │
│   │     • residual                                               │
│   └─ EquivariantMLP (geometric product + scalar gating)          │
│         • residual                                               │
└──────────────────────────────────────────────────────────────────┘
                ▼
   output (N, 1, 32)  — final CGA multivector per hit
                ▼
   ┌─ clustering head: nn.Linear(32, 5, bias=False)  → 5-D coord
   └─ beta head:       nn.Linear(32, 1) + sigmoid    → scalar in [0,1]
                ▼
   final output (N, 6) = [coord_1..5, beta]
```

Both heads are non-equivariant on purpose: the outputs (cluster
coordinate and beta probability) are E(3)-invariant scalars, so they
project the equivariant 32-MV down with a plain linear.

### Extras

**`embed_scalar(hit_type)`**: a CGA multivector has 32 slots labelled by
"grade" (scalar/vector/bivector/...). `embed_point` and
`embed_circle_ipns` only write to the geometric grades (vector / trivector)
and leave the scalar slot at 0. `embed_scalar(hit_type)` puts the integer
0/1 into that scalar slot, where `0 = VTX` (silicon vertex hit) and
`1 = DC` (drift chamber hit) — just a categorical detector-subsystem tag.
Adding the two gives one multivector that carries the geometric position
**and** a plain "VTX vs DC" tag, so the attention can look up modality
without us baking it into the geometry. After the input stage, hit `i`'s
32-D multivector looks like:

```
multivector[i] = [
    hit_type_i,          ← scalar slot (grade 0): 0.0 (VTX) or 1.0 (DC)
    geometry_i (vec),    ← grade 1: vector components from embed_point / embed_circle_ipns
    geometry_i (biv),    ← grade 2: bivector components
    geometry_i (tri),    ← grade 3: trivector components (this is where circles live)
    ...                  ← higher grades: 0 at input, populated by the transformer blocks
]
```

(Code: `src/cgatr/interface/scalar.py:6-21` — literally
`torch.cat([scalar, zeros(31)], dim=-1)`.)

**Scalar gating** (the activation inside the equivariant MLP): you can't
just apply ReLU to a multivector — that would break equivariance,
because rotating the input would change which components get clipped.
Instead the network reads the **scalar (grade-0) component as a gate**
and multiplies the *whole* multivector by `gelu(scalar_component)`.
The gate is rotation-invariant (it's a scalar), so the gated multivector
rotates the same way as the input — equivariance preserved.
(Code: `src/cgatr/layers/mlp/nonlinearities.py:30-36` —
`gates = mv[..., [0]]; out = gated_gelu(mv, gates=gates)`.)

**Counts**
- 10 transformer blocks
- 16 multivector channels + 64 scalar channels per hidden layer
- 32-dim CGA Cl(4,1) multivectors throughout (vs PGA's 16-dim)
- ~925k trainable parameters
- 5-D clustering coords + 1-D beta = 6-D output per hit

---

## 2. The loss

The total training loss is

```
L = attr_w * L_V_att   + repul_w * L_V_rep
  + L_beta_sig         + L_beta_noise
  + var_w * L_var
  + beta_suppress_w * L_beta_suppress
```

with the defaults below (`src/train_cgatr_parquet.py`).

> **Condensation point (CP).** For each truth track, its **condensation
> point** is the one signal hit with the highest β. It plays two roles:
> in training it's the attractor (L_V_att pulls same-track hits toward it;
> L_V_rep pushes other tracks away from it); in inference it's the seed
> for the greedy clusterer. The other real hits of the same track are
> the **non-CP signal hits** — `L_beta_suppress` keeps their β low so
> each track has exactly one CP (one seed), not several competing ones.

| term | weight (default) | what it does (1 line) |
|---|---:|---|
| `L_V_att`  (attractive) | `attr_weight = 1.0` | pulls every signal hit toward its track's CP in 5-D embedding |
| `L_V_rep`  (repulsive)  | `repul_weight = 1.0` | pushes hits of *other* tracks at least 1 unit away from each CP |
| `L_beta_sig` (signal beta) | (in-loss, no separate w) | wants at least one hit per truth track to have high β — guarantees a CP exists |
| `L_beta_noise` (noise beta) | (in-loss, no separate w) | wants noise hits (not on any truth track) to have low β — they should never become a CP |
| `L_var` (within-cluster spread) | `var_weight = 0.3` | mean ‖x_i − x_CP‖² per truth cluster — shrinks tail hits, fights elongation |
| `L_beta_suppress` | `beta_suppress_weight = 0.1` | penalises high β on **non-CP signal hits** (real track hits that aren't the chosen seed) → keeps exactly one CP per track and prevents the greedy clusterer from picking duplicate seeds |

One more knob:

| knob | value | what it does |
|---|---|---|
| `qmin` (charge floor) | `0.1` | charge `q = arctanh(β)² + qmin`; ensures even low-β hits contribute a baseline gradient |

Recipe summary:
- **v35** = `attr=1.0, repul=1.0, var=0.3, beta_suppress=0.1`

---

## 3. What beta is used for

Beta is the model's "is this hit a track-defining condensation point?"
score per hit. It feeds into both training and inference:

**At training** the beta head is supervised by three loss terms and also
weights the embedding losses through a derived "charge" `q`:

| usage | how |
|---|---|
| `q = arctanh(β)² + qmin` | charge for OC; high-β hits (CPs) get strong attractive/repulsive pull |
| `L_beta_sig` | wants ≥ 1 high-β hit per truth track — guarantees the track has a CP |
| `L_beta_noise` | wants noise hits to have low β so they're never picked as a seed |
| `L_beta_suppress` | penalises high β on non-CP signal hits (other real hits of the same track) → exactly **one** CP (seed) per truth track |

**At inference** beta drives the greedy clustering loop directly:

```
sort signal hits by beta descending          ← beta picks the order
for each hit s in order:
    if beta(s) < tbeta:    stop              ← beta picks where to stop
    if s already claimed:  continue          ← skip; s already belongs to an earlier cluster
    assign s and every unclaimed hit         ← create a new cluster around s:
        within td (5-D Euclidean) → new cluster
```

The "create a new cluster" step is the one that does the actual
clustering work: take the seed `s`, compute Euclidean distances in the
5-D embedding from `s` to every still-unclaimed hit, and bring into the
new cluster every hit closer than `td`. All those hits are then marked
"claimed" so subsequent iterations skip them. The cluster's label is
just `s`'s index, and any hit that's never claimed by any seed ends up
with label `-1` = noise.

Two knobs (v35 values):
- **`tbeta`** = "how confident must a hit be to spawn a cluster?" — `0.10`. Because the list is β-sorted descending, the moment we see `β(s) < tbeta` no later hit will pass the test either, so we `stop` (not `skip`).
- **`td`** = "how big is the ball around each seed?" (radius in 5-D embedding units) — `0.20`, `0.25` (strict-best after the OP sweep). Small `td` → many tight clusters; large `td` → fewer, possibly-merged clusters.

Code: `src/eval_sweep_v33.py:128-141` (`get_clustering_greedy` — the
function lives in the v33 eval module but is the inference clustering
used by every model from v33 onward, including v35).

---

## 4. The metrics

### 4a. Vocabulary

For each truth track in an event:

| term | what it counts |
|---|---|
| **track signal hits** (`n_hits_signal`) | how many of this truth track's hits ended up in the **signal subset** — the hits we actually fed to the clusterer. The denominator for efficiency. |
| **total truth hits** (`n_hits_total`) | how many detector hits this truth track produced in total (signal + secondaries / curls). Used by the reconstructable cuts (e.g. IDEA requires `n_hits_total > 10`). |
| **best cluster** | the predicted cluster that contains the most hits of this truth track. Found by `Counter(predicted_labels[track_hits]).most_common(1)[0]`. |
| **`best_match`** | the count of hits the truth track and its best cluster share — the **overlap** size. Numerator for both efficiency and purity. |
| **`cluster_size`** | total hits in the best cluster (its own + intruders from other tracks or noise). The denominator for purity. |

For an event-level / dataset-level metric:

| term | what it counts |
|---|---|
| **`n_reconstructable`** | how many truth tracks pass the analysis cut (the **denominator** of the FCC tracking-efficiency curve). For IDEA: `n_hits_total > 10 AND 15° < θ < 165° AND gen_status ∈ {0,1} AND \|charge\| > 0` — i.e. "primary charged particle inside the tracker acceptance with enough hits to be findable". |
| **`n_matched`** | among those reconstructable tracks, how many we successfully reconstructed (`matched = True`). The **numerator** of the tracking-efficiency curve. |

(Code: `src/eval_fcc_metrics_v36.py::per_track_records` for per-track
fields and lines 237–258 for the reconstructable cut definitions.)

### 4b. The metric formulas

| metric | one-line |
|---|---|
| **Efficiency** (per track) | `best_match / n_hits_signal` — how complete: of the hits we should have grabbed, how many did we put in one cluster together. |
| **Purity** (per track) | `best_match / cluster_size` — how clean: of the hits in the cluster, how many actually belong to this track. |
| **Loose match** (per track) | `purity ≥ 0.75` — there's at least one cluster that's mostly this track's hits, regardless of completeness. |
| **Strict-T match** (per track) | `purity ≥ 0.75 AND efficiency ≥ T` — clean *and* captures ≥ T·100% of the track. |
| **Tracking efficiency** (per pT bin, the FCC y-axis) | `n_matched / n_reconstructable` — fraction of reconstructable tracks we matched. |

*Loose is degenerate at small `td`.

---

## 5. Metric summary

v35 evaluated with greedy clustering at two operating points:

| operating point | loose | strict50 | strict90 | strict99 |
|---|---:|---:|---:|---:|
| v35 + greedy native (td=0.20) | 90.63% | 84.29% | 76.54% | 38.83% |
| **v35 + greedy best (td=0.25)** | 90.04% | **84.72%** | **77.82%** | **43.03%** |

`td=0.25` is the strict-metric optimum — gains 0.4 pp on strict50,
1.3 pp on strict90, 4.2 pp on strict99 over the native value.

---

## 6. Plots

All plots live under `model_training/eval_results/...` and are
referenced here with paths relative to this cheatsheet.

> Note: the plots below were generated as part of a side-by-side
> comparison study and so include curves for v36-EF and HDBSCAN
> alongside v35. We're keeping them as-is for now and not re-plotting.
> When reading, focus on the v35 series:
>
> - **blue square** = v35 + greedy native (td=0.20)
> - **orange square** = v35 + greedy best (td=0.25)
> - **green square** = v35 + HDBSCAN (an inference-only alternative)

### 6.1 Loose vs strict at a single threshold

`final_comparison_idea_strict50.png` — left panel is the FCC slide-24
loose metric, right panel is strict50. Same 6 series. Notice how the
green curve (v36-EF + greedy native) is **highest** on loose at low pT
but **lowest** on strict50 — the metric pathology in one figure.

![loose vs strict50](../eval_results/final_comparison/final_comparison_idea_strict50.png)

### 6.2 Strict-threshold sweep grid

`final_comparison_idea_grid.png` — match rate vs pT, one panel per
strict threshold T ∈ {0.50, 0.75, 0.85, 0.90, 0.95, 0.99, 1.00}. The
ranking flips between strict50 and strict99: HDBSCAN is competitive
when the bar is loose-completeness, but collapses when the bar becomes
"reconstruct (almost) the entire track."

![strict-threshold grid](../eval_results/final_comparison/final_comparison_idea_grid.png)

### 6.3 Strict90

`final_comparison_idea_strict90.png` — when you require the cluster
to capture ≥ 90% of the truth track, v35 + greedy at td=0.25 (orange)
is the clear winner. HDBSCAN (purple/brown) and v36-EF native (red)
all sit ~7-15 pp below.

![strict90](../eval_results/final_comparison/final_comparison_idea_strict90.png)

### 6.4 Strict99 — near-perfect reconstruction

`final_comparison_idea_strict99.png` — at this threshold HDBSCAN
collapses to ~10% match while v35 + greedy at td=0.25 holds ~50–60%
in the high-pT plateau. Greedy "wins" the precision regime.

![strict99](../eval_results/final_comparison/final_comparison_idea_strict99.png)

### 6.5 The metric pathology — loose vs strict as `td` → 0

`metrics_vs_td.png` from the v36-EF extreme `td` sweep. Loose match
climbs toward 1.0 as `td → 0` (every signal hit becomes its own
singleton cluster, trivially passing purity ≥ 0.75) while strict50
collapses to ~0%. **Single best slide for "why we report strict at all."**

![td pathology](../eval_results/v36ef_op_sweep_extreme/metrics_vs_td.png)

### 6.6 Clustering algorithm ablation (greedy / DBSCAN / HDBSCAN)

`eff_vs_pt_by_algo.png` from the v36-EF cache. Same forward pass, only
the clusterer changes. HDBSCAN (green) wins strict50 at low pT;
greedy (blue) wins loose. Same plot exists for v35 cache.

![algo ablation v36-EF](../eval_results/algo_ablation_v36ef/eff_vs_pt_by_algo.png)

### 6.7 (Reference) original 2-panel comparison

`final_comparison_idea.png` — kept for backwards compatibility; same
content as 6.1 but predates the threshold sweep.

![original comparison](../eval_results/final_comparison/final_comparison_idea.png)

---

## 7. Merge tracks by MC particle IDs

Given the existing per-cluster table (one row per predicted cluster,
with `matched_mc_idx` = dominant truth particle id, `purity` = its
fraction of the cluster), we associate cluster *c* to its
`matched_mc_idx = m` iff `purity(c) ≥ T_assoc` and `m ≠ 0`. All
clusters associated to the same *m* merge into one super-cluster
*S_m*. The merged efficiency / purity for truth track *m* are then

```
merged_overlap        = Σ_{c ∈ S_m} c.best_match
merged_cluster_size   = Σ_{c ∈ S_m} c.cluster_size
merged_efficiency_m   = merged_overlap / m.n_hits_signal
merged_purity_m       = merged_overlap / merged_cluster_size   (≥ T_assoc by construction)
```

Tracks with no associated cluster stay unmatched. Original clusters
that never got associated (purity below `T_assoc` or noise) remain in
the per-cluster table and still count as fakes — we do **not** absorb
them into something cleaner.


### 7.1 Results on the IDEA cut

`n_clusters` is the size of the predicted-cluster table after merge
(super-clusters + unassociated leftovers).

#### greedy native (`td = 0.20`)

| variant | n_clusters | loose | strict50 | strict75 | strict90 | strict99 |
|---|---:|---:|---:|---:|---:|---:|
| **unmerged (deployment)** | 1,197,380 | 90.63 % | 84.29 % | 81.16 % | 76.54 % | **38.83 %** |
| oracle T=0.50 |   153,978 | 93.79 % | 91.26 % | 90.40 % | 89.03 % | 75.18 % |
| oracle T=0.65 |   179,157 | 95.36 % | 91.34 % | 90.20 % | 88.35 % | 72.26 % |
| oracle T=0.75 |   193,138 | 96.10 % | 91.01 % | 89.71 % | 87.59 % | **70.61 %** |

#### greedy best (`td = 0.25`)

| variant | n_clusters | loose | strict50 | strict75 | strict90 | strict99 |
|---|---:|---:|---:|---:|---:|---:|
| **unmerged (deployment)** | 964,910 | 90.04 % | 84.72 % | 81.84 % | 77.82 % | **43.03 %** |
| oracle T=0.50 | 146,976 | 93.23 % | 90.64 % | 89.95 % | 88.66 % | 76.76 % |
| oracle T=0.65 | 169,400 | 94.80 % | 90.74 % | 89.76 % | 88.06 % | 73.94 % |
| oracle T=0.75 | 182,464 | 95.53 % | 90.41 % | 89.27 % | 87.32 % | **72.34 %** |

#### HDBSCAN (`min_cluster_size = 5`)

| variant | n_clusters | loose | strict50 | strict75 | strict90 | strict99 |
|---|---:|---:|---:|---:|---:|---:|
| **unmerged (deployment)** | 195,916 | 89.82 % | 83.70 % | 78.37 % | 70.73 % | **12.35 %** |
| oracle T=0.50 |  91,978 | 90.43 % | 88.81 % | 84.25 % | 75.38 % | 12.81 % |
| oracle T=0.65 |  94,735 | 90.80 % | 88.77 % | 84.09 % | 75.21 % | 12.74 % |
| oracle T=0.75 |  96,583 | 90.96 % | 88.59 % | 83.89 % | 75.06 % | **12.71 %** |

### 7.2 Plots

`oracle_compare_strict50.png`, `..._strict75.png`, `..._strict90.png`,
`..._strict99.png`: per-pT 3-panel comparisons (one panel per
inference algorithm), solid = deployment, dashed = oracle merge.
`oracle_compare_dashboard.png` puts strict50 and strict99 in one
6-panel grid — easiest single-image view of the headroom story.

![oracle vs deployment dashboard](../eval_results/oracle_merge_compare/oracle_compare_dashboard.png)

![oracle strict99](../eval_results/oracle_merge_compare/oracle_compare_strict99.png)

---

## 8. User-greedy clusterer (self-seed variant)

### 8.1 Algorithm

`src/user_greedy.py` implements a self-seed variant of beta-greedy
clustering: a candidate CP may seed a new cluster only if it itself
is still unassigned, and the candidate-CP list is re-sorted after
every seed assignment. The reference variant in
`src/eval_sweep_v33.py::get_clustering_greedy` keeps the initial
β-descending order fixed and lets an absorbed CP re-seed at its
absorbed position, which can spawn duplicate clusters.

Both variants are dimension-agnostic (`np.linalg.norm(..., axis=-1)`)
so the same code runs on the full 5-D v35 embedding without
modification. Pipeline glue: `src/eval_user_greedy_sweep.py` (sweep
+ FCC cache materialisation), `src/user_greedy_comparison.py`
(vs reference), `src/user_greedy_oracle_compare.py` (oracle
headroom), `src/run_user_oracle_merge.py` (batch merger).

### 8.2 Parameter sweep (42 OPs, v35 forward cache, 4248 events)

`eval_results/v35_user_greedy_sweep/`:
- `sweep.csv`, `sweep.md` — full per-OP metrics
- `heatmaps.png` — 2×3 grid (loose / strict50 / strict75 / strict90 /
  strict99 / fake) over the (`tbeta`, `td`) grid

![user-greedy sweep heatmaps](../eval_results/v35_user_greedy_sweep/heatmaps.png)

`tbeta` is a near-flat axis at this embedding scale: rows for
`tbeta ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70}` at the same `td`
differ by ≤ 0.2 pp on every strict metric. Once a high-beta seed
has been picked, every unassigned hit inside `td` is absorbed
regardless of its own beta; the seed beta itself easily clears any
threshold ≤ 0.70 in the current embedding. The `td` axis drives the
operating point. Extended sweep at `tbeta = 0.60` (with a
`tbeta = 0.10` control at large `td` to re-verify flat-tbeta):

| td   | loose   | strict50 | strict90 | strict99 | fake    | n_clusters |
|-----:|--------:|---------:|---------:|---------:|--------:|-----------:|
| 0.05 | 93.7 %  | 77.2 %   | 60.4 %   | **7.8 %**  | 6.6 %   | 2.94 M |
| 0.10 | 92.2 %  | 82.0 %   | 70.7 %   | 24.0 %    | 7.8 %   | 1.69 M |
| 0.15 | 91.2 %  | 83.6 %   | 74.5 %   | 32.9 %    | 9.5 %   | 1.28 M |
| 0.20 | 90.7 %  | 84.5 %   | 76.6 %   | 38.8 %    | 11.2 %  | 0.99 M |
| 0.25 | 90.1 %  | 84.9 %   | 77.9 %   | 43.0 %    | 13.0 %  | 0.76 M |
| 0.30 | 89.5 %  | 85.0 %   | 78.7 %   | 46.2 %    | 15.1 %  | 0.53 M |
| 0.35 | 89.0 %  | 85.0 %   | 79.3 %   | 48.7 %    | 17.0 %  | 0.44 M |
| **0.40** | **88.6 %** | **85.0 %** | **79.7 %** | **50.6 %** | 18.7 %  | 0.37 M |
| 0.45 | 88.2 %  | 84.9 %   | 80.0 %   | 52.3 %    | 20.3 %  | 0.33 M |
| 0.50 | 87.8 %  | 84.8 %   | 80.2 %   | 53.7 %    | 21.5 %  | 0.30 M |
| 0.55 | 87.4 %  | 84.7 %   | 80.3 %   | 54.9 %    | 23.3 %  | 0.27 M |
| 0.60 | 87.0 %  | 84.5 %   | 80.3 %   | 56.0 %    | 24.5 %  | 0.24 M |
| 0.70 | 86.4 %  | 84.2 %   | 80.4 %   | 57.9 %    | 26.7 %  | 0.21 M |
| 0.85 | 85.4 %  | 83.6 %   | 80.1 %   | 59.8 %    | 29.1 %  | 0.18 M |
| **1.00** | 84.3 %  | 82.8 %   | 79.7 %   | **61.2 %** | **30.7 %**  | **0.15 M** |


### 8.3 Comparison vs reference inference algorithms

`eval_results/v35_user_greedy_compare/comparison_overall.md`:

| algorithm | n_clusters | loose | strict50 | strict75 | strict90 | strict99 | fake |
|---|---:|---:|---:|---:|---:|---:|---:|
| v35 greedy (`tbeta=0.10, td=0.20`) | 1,197,380 | 90.63 % | 84.29 % | 81.16 % | 76.54 % | 38.83 % | 10.11 % |
| v35 greedy (`tbeta=0.10, td=0.25`) | 964,910 | 90.04 % | 84.72 % | 81.84 % | 77.82 % | 43.03 % | 11.33 % |
| HDBSCAN (`mcs=5`) | 195,916 | 89.82 % | 83.70 % | 78.37 % | 70.73 % | 12.35 % | 11.26 % |
| user `tbeta=0.10, td=0.05` | 2,943,719 | 93.73 % | 77.21 % | 70.66 % | 60.43 % | 7.81 % | 6.57 % |
| user `tbeta=0.10, td=0.20` | 988,028 | 90.66 % | 84.47 % | 81.21 % | 76.56 % | 38.84 % | 11.24 % |
| user `tbeta=0.10, td=0.25` | 756,514 | 90.06 % | 84.87 % | 81.92 % | 77.85 % | 43.04 % | 13.05 % |
| user `tbeta=0.60, td=0.30` | 534,579 | 89.48 % | 85.01 % | 82.35 % | **78.72 %** | **46.24 %** | 15.12 % |
| user `tbeta=0.70, td=0.05` | 1,992,058 | 93.55 % | 77.20 % | 70.65 % | 60.43 % | 7.81 % | 7.94 % |

![user-greedy vs reference dashboard](../eval_results/v35_user_greedy_compare/user_vs_ref_dashboard.png)

### 8.4 Final IDEA tracking-efficiency plot

`docs/figures/final_idea_fcc.png` — three-panel per-pT comparison
(loose / strict50 / strict99) of the best OP per algorithm under the
IDEA reconstructable cut. Single-panel companions
(`final_idea_fcc_{loose,strict50,strict99}.png`) mirror the FCC
slide-24 layout.

Label convention: every line is named by its parameters only. "v35
greedy" rows use `src/eval_sweep_v33.py::get_clustering_greedy` on
the v35 forward pass (the `v33` is historical — the weights are
v35's). `v35 user-greedy` is `src/user_greedy.py`.

| algorithm | n_clusters | loose | strict50 | strict90 | strict99 |
|---|---:|---:|---:|---:|---:|
| v35 greedy (tbeta=0.10, td=0.20)         | 1,197,380 | 90.63 % | 84.29 % | 76.54 % | 38.83 % |
| v35 greedy (tbeta=0.10, td=0.25)         |   964,910 | 90.04 % | 84.72 % | 77.82 % | 43.03 % |
| v35 HDBSCAN (mcs=5, beta\u22650.10)      |   195,916 | 89.82 % | 83.70 % | 70.73 % | 12.35 % |
| v35 DBSCAN (eps=0.10, beta\u22650.10)    |   204,741 | 88.22 % | 84.75 % | 76.34 % |  7.04 % |
| v35 DBSCAN (eps=0.20, beta\u22650.10)    |   156,987 | 85.44 % | 83.59 % | 78.33 % |  9.68 % |
| v35 DBSCAN (eps=0.40, beta\u22650.10)    |    96,027 | 81.09 % | 80.44 % | 77.45 % | 11.95 % |
| v35 user-greedy (tbeta=0.60, td=0.30)    |   534,579 | 89.48 % | 85.01 % | 78.72 % | 46.24 % |
| v35 user-greedy (tbeta=0.60, td=0.40)    |   372,800 | 88.59 % | **85.05 %** | 79.69 % | 50.64 % |
| v35 user-greedy (tbeta=0.60, td=0.70)    |   202,311 | 86.42 % | 84.19 % | **80.35 %** | 57.87 % |
| v35 user-greedy (tbeta=0.60, td=1.00)    |   151,559 | 84.33 % | 82.80 % | 79.65 % | **61.24 %** |

![final IDEA 3-panel](figures/final_idea_fcc.png)

![final IDEA loose, FCC slide-24 style](figures/final_idea_fcc_loose.png)

![final IDEA strict50](figures/final_idea_fcc_strict50.png)

![final IDEA strict99](figures/final_idea_fcc_strict99.png)

Reproduction:

```bash
cd model_training

# 42-OP base sweep (~1.5 h)
~/.conda/envs/hotdog-ml/bin/python -m src.eval_user_greedy_sweep \
    --save_caches_for 0.10:0.20 0.10:0.25 0.10:0.05 0.70:0.05

# (0.60, 0.30) cache (~1 min)
~/.conda/envs/hotdog-ml/bin/python -m src.eval_user_greedy_sweep \
    --tbetas 0.60 --tds 0.30 --save_caches_for 0.60:0.30 \
    --out_dir eval_results/v35_user_greedy_sweep_tbeta060_td030

# extended large-td sweep at tbeta=0.60 (~10 min)
~/.conda/envs/hotdog-ml/bin/python -m src.eval_user_greedy_sweep \
    --tbetas 0.60 \
    --tds 0.35 0.40 0.45 0.55 0.60 0.70 0.85 1.00 \
    --save_caches_for 0.60:0.40 0.60:0.50 0.60:0.70 0.60:1.00 \
    --out_dir eval_results/v35_user_greedy_sweep_large_td

# flat-tbeta control at large td (~5 min)
~/.conda/envs/hotdog-ml/bin/python -m src.eval_user_greedy_sweep \
    --tbetas 0.10 --tds 0.40 0.60 0.80 \
    --out_dir eval_results/v35_user_greedy_sweep_large_td_ctrl

# oracle merge at T ∈ {0.50, 0.65, 0.75}
~/.conda/envs/hotdog-ml/bin/python -m src.run_user_oracle_merge

# DBSCAN baselines for the final IDEA plot
~/.conda/envs/hotdog-ml/bin/python -m src.recluster_dbscan \
    --cache_path eval_results/v35_forward_cache \
    --out_dir   eval_results/v35_fcc_dbscan_eps0.10 \
    --eps 0.10 --min_samples 3 --beta_prefilter 0.10
# (similarly eps=0.20, eps=0.40)

# tables + plots
~/.conda/envs/hotdog-ml/bin/python -m src.user_greedy_comparison
~/.conda/envs/hotdog-ml/bin/python -m src.user_greedy_oracle_compare
~/.conda/envs/hotdog-ml/bin/python -m src.final_idea_fcc_plot
```

---

## 9. New training settings — Lightning training, 4D, no hit cap, ONNX-compatible

### 9.1 What changed vs the original v35

| knob | v35 (original) | v35-Lightning | why |
|---|---|---|---|
| trainer | custom torch DDP loop | PyTorch Lightning 2.x | DDP plumbing, checkpointing, CSV+logger, deterministic resume |
| `embed_dim` | 5 | **4** | v34 PCA (`eval_results/v34_analysis_merged/phase_a_decision.md`): the useful subspace is rank-4; 5D was a free dim |
| per-event hit cap | `max_hits=3000` (truncation) | **none** | the 3000 cap landed at the median — ~46 % of events were truncated, ~12 % lost 70 %+ of their hits |
| batching | fixed `batch_size` per rank | **TokenBudgetBatchSampler** `max_tokens=24000` | memory bounded by total hits, not event count: packs ~8 median events / batch, largest events (~17 k hits) become singleton batches |
| ONNX | non-exportable (`xformers.BlockDiagonalMask`) | **single batched export, dynamic in `B` and `N`** | C++ deployment |

Event-size distribution across full train+val (200 seeds, 99,995 events):

| split (seeds) | min | median | p99 | p99.9 | p99.99 | max | > 20k | > 24k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train 1-180 | 62 | 2,768 | 10,389 | 13,854 | 16,039 | **17,745** | 0 | 0 |
| val 181-200 | 118 | 2,781 | 10,557 | 13,657 | 14,690 | 15,712 | 0 | 0 |

`max_tokens = 24000` covers every single event in the dataset with
headroom; the absolute max event becomes a singleton batch, the
median packs ~8 events per batch.

### 9.2 ONNX export surface

Single batched artifact handles `B = 1` and `B > 1` inference; both
`B` and `N` are exposed as dynamic axes.

* **inputs**: `features [B, N, 10]` (float32), `padding_mask [B, N]` (bool, `True` = real hit)
* **output**: `coords_and_beta [B, N, embed_dim + 1]` — last column is the beta logit (caller applies sigmoid); padded rows carry garbage and must be sliced with `padding_mask`.

Three training-only ops are replaced with deployment-safe equivalents:

1. `xformers.BlockDiagonalMask` → `torch.nn.functional.scaled_dot_product_attention(attn_mask=...)` built from `padding_mask`.
2. Stock `CGATr._construct_dual_reference` (averages over all leading dims, mixing events) → per-event masked mean.
3. ORT's decomposed SDPA produces `NaN` on fully-padded query rows; the wrapper OR-augments the attention mask with identity and `torch.where`-zeros padded outputs after every sub-block (NaN-safe).

### 9.3 Files

Training (`model_training/src/`):
- `train_cgatr_lightning_v35.py` — CLI + DDP setup + DataLoader (TokenBudgetBatchSampler or fixed-batch) + `_BatchSamplerEpochCallback` for per-epoch `set_epoch()` + EpochCSV writer + EMA + resume-from-ckpt.
- `cgatr_v35_lightning_module.py` — LightningModule. Imports `CGATrParquetModel` + `object_condensation_loss` from `train_cgatr_parquet.py` (no model drift). Manual `all_reduce` for `train_loss` in `on_validation_epoch_end` because sentinel / too-few-signal batches skip per-step logging on some ranks and Lightning's auto `on_epoch` sync would deadlock.

ONNX (`conversion_to_onnx/`):
- `onnx_export_v35.py` — `OnnxV35` wrapper + batched export script.
- `onnx_parity_v35.py` — per-event torch (CPU) vs ORT (CPU) parity check.
- `eval_fcc_metrics_v35_ort.py` — full FCC-style eval running on ORT for the forward.

CGA primitive fixes (`model_training/src/cgatr/`):
- `primitives/{linear,bilinear}.py` — einsum patterns use ellipsis leading dims (single-event and batched share the same kernel); removed a SymInt-tripping assert.
- `primitives/attention.py` — Python-int channel counts (eliminates `TracerWarning`); `attn_mask` forwarded to SDPA; `expand_pairwise` skipped on the SDPA path.
- `layers/attention/attention.py` — passes channel counts down as Python ints.

Hardened sampler (`model_training/src/dataset/parquet_dataset.py`):
- `TokenBudgetBatchSampler` with cached per-epoch batches, world-size truncation **before** per-rank slice (eliminates "one-rank-finishes-first" NCCL deadlocks), oversized-event singletons.

### 9.4 Reproduction

```bash
cd model_training
~/.conda/envs/hotdog-ml/bin/python -m src.train_cgatr_lightning_v35 \
    --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
    --train_seeds 1-180 --val_seeds 181-200 \
    --num_epochs 8 --num_devices 2 \
    --max_tokens 24000 --max_hits 0 --embed_dim 4 \
    --start_lr 1e-3 --warmup_epochs 1 \
    --output_dir checkpoints/cgatr_v35_lt_4d \
    --run_tag v35_lt_4d
```

ONNX export + verification:

```bash
cd model_training

# Export
~/.conda/envs/hotdog-ml/bin/python -m src.onnx_export_spike \
    --checkpoint checkpoints/cgatr_v35_lt_4d/cgatr_best.ckpt \
    --out        checkpoints/cgatr_v35_lt_4d/cgatr_v35.onnx \
    --embed_dim  4

# Per-event torch ↔ ORT parity
~/.conda/envs/hotdog-ml/bin/python -m src.onnx_eval_smoke \
    --checkpoint checkpoints/cgatr_v35_lt_4d/cgatr_best.ckpt \
    --onnx       checkpoints/cgatr_v35_lt_4d/cgatr_v35.onnx \
    --embed_dim  4 \
    --data_dir   /home/marko.cechovic/cgatr/data_parquet_train \
    --val_seeds  181-181 --max_events 50

# Full FCC-style eval running through ORT
~/.conda/envs/hotdog-ml/bin/python -m src.eval_fcc_metrics_ort \
    --onnx checkpoints/cgatr_v35_lt_4d/cgatr_v35.onnx \
    --embed_dim 4 \
    --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
    --eval_seeds 181-200 \
    --tbeta 0.60 --td 0.30 \
    --output_dir eval_results/v35_lt_4d_fcc_ort
```

---

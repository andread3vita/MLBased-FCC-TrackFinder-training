# CIRCE: Conformal Isotropic Reconstruction of Charged Elements (FCC Track Finding)

## What this is

Self-contained training and evaluation package for the **CIRCE** (Conformal Geometric Algebra $Cl(4,1)$) track-finding model on the FCC IDEA drift chamber.

## Quick start

```bash
# 1. Environment setup (PyTorch 2.2+, CUDA, PyTorch Lightning, torch-scatter)
bash setup_env.sh

# 2. Run training across 4 GPUs with champion Pareto configuration
DATA_DIR=/path/to/july2026-zuds-parquet OUT_DIR=checkpoints/circe_production NUM_DEVICES=4 bash run_train.sh

# 3. Full FCC benchmark evaluation on keepAll holdout (produces all benchmark plots)
N_GPUS=4 DATA_DIR=/path/to/eval-keepall bash run_eval.sh checkpoints/circe_production/last.ckpt
```

## Champion Objective & Hyperparameters

A systematic 5-way factorial ablation over 50,000 matched events established CIRCE's Pareto-optimal configuration:
- **Architecture:** $Cl(4,1)$ Conformal Geometric Algebra, $E(3)$ equivariant basis (20 linear maps, verified mirror residual $< 1.1 \times 10^{-15}$), 10 blocks, 16 multivector + 64 scalar channels, `embed_dim = 4`.
- **Drift Hit Representation:** Measured circle encoding (wire center, wire direction unit vector, drift radius), preserving spatial curvature and eliminating discrete left/right point ambiguity.
- **Loss:** Compact-support Kieseler hinge repulsion (`max(0, 1 - d)`), $q_\text{min} = 3.0$, $\text{attr\_weight} = 1.0$, $\text{repul\_weight} = 2.0$, $\beta_\text{suppress} = 0.1$, and $\text{var\_weight} = 0.3$.
- **Clustering Operating Point:** Greedy clustering at $t_\beta = 0.60, t_d = 0.10$.

## Benchmark Performance on `keepAllParticles` (50,000 events)

Evaluated on the full 100-seed `eval-keepall` holdout (1,672,188 targets) under standard benchmark definitions ($15^\circ < \theta < 165^\circ, p_\mathrm{T} > 0.1$ GeV at $t_\beta = 0.60, t_d = 0.10$):

| Metric | Selection / Protocol | Raw Unmerged | Fragment Merged ($t_\mathrm{m}=0.10$) | Benchmark Target | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Tracking Efficiency ($N_\mathrm{hits} > 10$, 1-to-1)** | Double Majority ($\ge 50\%$ purity, $\ge 50\%$ eff) | **97.75%** | **97.76%** | $> 90.0\%$ | **Exceeded (+7.76%)** |
| **Tracking Efficiency ($N_\mathrm{hits} > 10$, Majority)** | Standard IDEA tracks, Purity $> 75\%$ | **97.28%** | **97.05%** | $> 90.0\%$ | **Exceeded (+7.05%)** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$, 1-to-1)** | Inclusive recovery down to 4 hits, Double Majority | **96.73%** | **96.73%** | — | High inclusive recovery |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$, Majority)** | Inclusive recovery, Purity $> 75\%$ | **96.12%** | **95.86%** | — | High inclusive recovery |
| **Fake Rate ($N_\mathrm{hits} > 10$, 1-to-1 Hungarian)** | Unassigned candidates in 1-to-1 match ($N > 10$) | **6.38%** | **4.71%** | $< 8.0\%$ | **Exceeded (beats 8% target)** |
| **Fake Rate ($N_\mathrm{hits} > 10$, Majority)** | Spurious fakes without multi-track ($N > 10$) | **0.50%** | **0.41%** | — | Ultra-pure |
| **Fake Rate ($N_\mathrm{hits} > 3$, 1-to-1 Hungarian)** | Unassigned candidates in 1-to-1 match ($N > 3$) | **10.79%** | **8.18%** | — | Standard `min_hits=3` |
| **Fake Rate ($N_\mathrm{hits} > 3$, Majority)** | Spurious fakes without multi-track ($N > 3$) | **2.81%** | **2.48%** | — | CLD paper convention |
| **Multi-Track Merge Rate ($N_\mathrm{hits} > 10$)** | Clusters swallowing $\ge 2$ particles ($>75\%$ eff) | **7.23%** | **7.54%** | — | Clean separation |
| **Multi-Track Merge Rate ($N_\mathrm{hits} > 3$)** | Clusters swallowing $\ge 2$ particles ($>75\%$ eff) | **12.93%** | **13.40%** | — | Clean separation |
| **Candidates / Event ($N_\mathrm{hits} > 10$)** | Reconstructed high-purity tracks | **25.46** | **25.04** | — | Benchmark tracks |
| **Candidates / Event ($N_\mathrm{hits} > 3$)** | Inclusive track candidates | **31.86** | **30.93** | — | Normal multiplicity |

Benchmark plots are available in `plots/`:
- `plots/head_to_head_keepall_efficiency.png` (and `.pdf`): Tracking Efficiency vs $p_\mathrm{T}$ and Polar Angle $\theta$ comparing Raw Unmerged vs. Fragment Merged ($t_\mathrm{m}=0.10$).
- `plots/fcc_comprehensive_suite.png` (and `.pdf`): 4-panel comprehensive evaluation suite ($p_\mathrm{T}$ turn-on, angular coverage, hit multiplicity, and grouped side-by-side performance breakdown).

## Loss Formulation: CIRCE Champion Loss vs Upstream Baseline (GGTF)

To facilitate unifying the loss implementations across the collaboration, the table and formulas below detail the exact mathematical differences between CIRCE's champion loss and upstream GGTF.

### 1. General Formulation

Both models optimize an Object Condensation objective:

$$
\mathcal{L}_{\text{total}} = w_{\text{att}} \mathcal{L}_V^{\text{att}} + w_{\text{rep}} \mathcal{L}_V^{\text{rep}} + \mathcal{L}_\beta^{\text{sig}} + \mathcal{L}_\beta^{\text{noise}} + w_{\text{suppress}} \mathcal{L}_\beta^{\text{suppress}} + w_{\text{var}} \mathcal{L}_{\text{var}}
$$

where condensation charge is defined from predicted $\beta_i \in [0, 1]$:

$$
q_i = \mathrm{arctanh}^2(\beta_i) + q_{\mathrm{min}}
$$

### 2. Side-by-Side Comparison

| Component | Upstream GGTF | CIRCE Champion |
| :--- | :---: | :---: |
| **Repulsive Potential $V_\mathrm{rep}(d_{ik})$** | $\exp\left(-\frac{d_{ik}^2}{2}\right)$ *(Gaussian)* | $\max(0, 1 - d_{ik})$ *(Compact hinge)* |
| **Charge Floor $q_\mathrm{min}$** | $0.1$ | **$3.0$** |
| **$\beta$-Suppression Weight $w_\mathrm{suppress}$** | $0.0$ *(disabled)* | **$0.1$** |
| **Variance Regularizer $w_\mathrm{var}$** | $0.0$ *(disabled)* | **$0.3$** *(1-epoch warmup)* |
| **Loss Weights $(w_\mathrm{att}, w_\mathrm{rep})$** | $(1.0, 1.0)$ | **$(1.0, 2.0)$** |

### 3. Key Physical Insight: Coupling of Repulsion Geometry & $\beta$-Suppression

A central finding from our 100-seed matched ablation (Pilot E) explains why GGTF could run without $\beta$-suppression while CIRCE benefits from `w_suppress = 0.1`:

- **In GGTF**: The infinite-range Gaussian potential $\exp(-d^2/2)$ exerts continuous, global pushback across the entire event. Stray secondary high-$\beta$ hits are pushed away by all other tracks, providing an implicit soft regularization.
- **In CIRCE**: The compact hinge $\max(0, 1 - d)$ has zero gradient outside $d \ge 1.0$. Because IDEA drift chamber tracks are long helices (50 to 150 hits), distant hits on the same physical particle experience zero external repulsion.
- **Ablation Evidence ($\beta$-Suppression)**: Without explicit suppression (`w_suppress = 0.0`), multiple hits on the same track predict high $\beta$, causing latent track spread to balloon by $2.2\times$ ($0.060 \to 0.131$), cluster collision rate to spike to $59.5\%$, and tracking efficiency to drop by $-7.47\%$. Adding `w_suppress = 0.1` penalizes non-seed hits via the suppression loss:

$$
\mathcal{L}_\beta^{\mathrm{suppress}} = \frac{1}{N_{\mathrm{non}\text{-}\alpha}} \sum_{i \notin \{\alpha(k)\}} \beta_i
$$

  enforcing exactly one condensation seed per track and cleanly solving track fragmentation.
- **Ablation Evidence (Repulsion Weight $w_\mathrm{rep} = 2.0$)**: At nominal `w_rep = 1.0`, bulk separation is healthy (median nearest-neighbour distance $\sim 0.88$), but $10.5\%$ of tracks have their nearest condensation seed within $d < 0.11$ (and $25\%$ within $d < 0.46$) in dense, collimated jet cores. Increasing to `w_rep = 2.0` doubles the outward repulsive force at the compact hinge boundary ($\max(0, 1 - d)$), cleanly pushing collimated jet tracks past the $t_d = 0.10$ clustering threshold and reducing cluster merges from $19.5\%$ to $12.5\%$ without inflating intra-cluster spread.

## Data path

Pre-filled in both `run_train.sh` and `train.slurm`:
```
/eos/home-m/mcechovi/projects/cgatr/data_parquet_zqq_uds_v1
```
This directory must contain `seed_*/` subdirectories (seeds 1–1196).
Override at runtime if your mount point differs:
```bash
DATA_DIR=/your/path sbatch train.slurm
# or bare-metal:
DATA_DIR=/your/path NUM_DEVICES=4 bash run_train.sh
```

## Training

**SLURM (recommended for 4xH100):**
```bash
sbatch train.slurm
```

**Bare-metal (4 GPUs directly):**
```bash
NUM_DEVICES=4 bash run_train.sh
```

Full checkpoints are saved after every completed validation sweep as
`validation_epoch=E_step=S_pareto_f1=F_max_eff=M.ckpt`. The production wrapper
also enables periodic full safety checkpoints every 200 optimizer steps and
auto-resumes on SLURM requeue with `--resume_ckpt last`.

## Tunables

| Variable | Default | Description |
|---|---|---|
| `MAX_TOKENS` | 16000 | Packed-batch token budget (total hits/batch). Lower it if you hit GPU OOM; raise it (memory permitting) for better utilisation. Not a data cap — events larger than the budget are kept as singleton batches. |
| `CPU_THREADS` | 4 | OMP/MKL/POLARS thread count. `run_eval.sh` uses half this value per shard (intentional: shards run in parallel). |
| `GRAD_CKPT` | 0 | Set to 1 for gradient checkpointing (~30% slower, saves VRAM). Enable it if you want to push `MAX_TOKENS` beyond what your GPU memory allows. |
| `NUM_EPOCHS` | 100 | Training epochs |
| `PRECISION` | 32-true | PyTorch precision (`32-true`, `bf16-mixed`) |
| `LIMIT_VAL` | 0.15 | Fraction of validation batches per epoch |
| `WARMUP_EPOCHS` | 2 | LR warmup duration |
| `START_LR` | 3e-4 | Peak learning rate |

## Evaluation & Downstream Integration

### 1. Built-in End-to-End Pipeline
```bash
N_GPUS=4 DATA_DIR=/path/to/eval-keepall bash run_eval.sh 'checkpoints/circe_production/validation_epoch=E_step=S_pareto_f1=F_max_eff=M.ckpt'
```
The eval pipeline runs in 4 stages:
1. Sharded GPU forward pass (one shard per GPU) producing `forward_hits.parquet`
2. Merge shards and link MC truth via `build_mc_signal.py`
3. Greedy clustering + truth matching (standard benchmark + Hungarian 1-to-1) via `fcc_cache_parallel.py`
4. Publication plotting via `plot_fcc_metrics.py`

### 2. Interfacing CIRCE with Custom Evaluation Notebooks
CIRCE uses a **4-dimensional Euclidean condensation space** (`embed_dim = 4`), which provides the necessary degrees of freedom in conformal space to untangle dense jet cores:
- **Output Tensor Layout**: The model output has 5 columns: `[coord_0, coord_1, coord_2, coord_3, beta_logit]`.
- **Slicing**:
  ```python
  # Coordinates in R^4 Euclidean space
  coords = output[:, :4]
  # Condensation score in [0, 1]
  beta = torch.sigmoid(output[:, 4])
  ```
  Or via the built-in helper:
  ```python
  coords, beta_logits = model.split_output(output)
  beta = torch.sigmoid(beta_logits)
  ```
- **Clustering Distance Metric**: Euclidean distance in $\mathbb{R}^4$ is mathematically identical to $\mathbb{R}^3$:
  $$d(x_i, x_j) = \sqrt{\sum_{k=1}^4 (x_{i,k} - x_{j,k})^2}$$
  Both `torch.norm(X[unassigned] - X[seed], dim=-1)` and `scipy`/`numpy` handle 4D natively.
- **Andrea's `inference_oc_tracks.py` compatibility**:
  When calling `evaluate_efficiency_tracks(..., embedding_dim=4)` pass `embedding_dim=4` (to override the legacy default of 3).
- **Andrea's `evaluate_tracking_efficiency_pt.py` compatibility**:
  The script automatically queries `model.embedding_dim`, which returns `4` on CIRCE models.

## Results

- **Unmerged**: `eval_results/<tag>/fcc_unmerged/plots/eff_vs_pt_idea.png`
  and `fake_rate_summary.png`
- **Oracle-merged**: `eval_results/<tag>/fcc_oracle_T*/plots/eff_vs_pt_idea.png`


## Gradient checkpointing

Set `GRAD_CKPT=1` in `run_train.sh` (or pass `--grad_checkpoint` to `src/train.py`)
if you encounter OOM at high token budgets. This is ~30% slower
but saves large amounts of activation memory.

## Environment notes

- PyTorch 2.5.1 + CUDA 12.1 (`cu121`)
- `torch_scatter` must match the torch/CUDA wheel (see `setup_env.sh`)
- H100 requires NVIDIA driver >= CUDA 12.1
- `lightning >= 2.2` for DDP + SIGUSR1 requeue support

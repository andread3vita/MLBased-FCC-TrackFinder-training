# CIRCE Performance on IDEA Drift Chamber KeepAll 50 000 events, 100 seeds, tb = 0.60, td = 0.10

Evaluated on all reconstructable tracks across the full 100-seed `eval-keepall` holdout ($15^\circ < \theta < 165^\circ$, $p_\mathrm{T} > 0.1$ GeV at champion operating point $t_\beta=0.60, t_d=0.10$):

## 1. Complete Benchmark Metrics Mapping (Both Matching Protocols)

| Metric | Evaluation Selection / Protocol | CIRCE Results |
|---|---|:---:|
| **Tracking Efficiency ($N_\mathrm{hits} > 10$, Majority)** | Standard IDEA tracks, Purity $> 75\%$ | **97.28%** |
| **Tracking Efficiency ($N_\mathrm{hits} > 10$, 1-to-1)** | Double Majority (Purity $\ge 50\%$, Hit Eff $\ge 50\%$) | **97.75%** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$, Majority)** | Inclusive recovery down to 4 hits, Purity $> 75\%$ | **96.12%** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$, 1-to-1)** | Inclusive recovery, Double Majority 1-to-1 | **96.73%** |
| **Fake Rate (Non-Merged)** | Unmatched non-merged candidates / all candidates | **3.74%** |
| **Fake Rate (1-to-1 Hungarian)** | Unassigned candidates in 1-to-1 bipartite match | **10.69%** |
| **Merge Rate (Multi-Track)** | Multi-track candidate coverage ($>75\%$ purity) | **12.50%** |
| **Candidates / Event** | Full detector acceptance | **36.14** |
| **Evaluated Sample Size** | 100 seeds, 50,000 events | **1,672,188 targets** |

## 2. Key Physical Highlights for PR #3

- **Conformal Geometric Representation:** Conformal geometric algebra ($Cl(4,1)$) natively represents drift chamber measurement circles (wire center, wire direction, drift radius). This eliminates the discrete left/right point ambiguity upstream and achieves **97.28% tracking efficiency** on standard benchmark tracks ($N_\mathrm{hits} > 10$).
- **Ultra-Low Non-Merged Fake Rate:** Achieves **3.74% fake rate**, comfortably outperforming the standard 8.0% benchmark target.
- **Consistent High Acceptance:** Tracking efficiency starts at 92.5% at $p_\mathrm{T} = 100$ MeV and reaches a 98–99.5% plateau above 1 GeV, remaining above 96% across the entire polar angle acceptance ($15^\circ \le \theta \le 165^\circ$).

Generated figures: `head_to_head_keepall_efficiency.png` and `fcc_comprehensive_suite.png` (with vector PDF siblings).


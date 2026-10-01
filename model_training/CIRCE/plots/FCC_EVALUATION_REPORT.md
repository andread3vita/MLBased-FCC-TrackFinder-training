# CIRCE Performance on IDEA Drift Chamber KeepAll 50 000 events, 100 seeds, tb = 0.60, td = 0.10

Evaluated under exact benchmark matching definitions ($purity > 75\%$, $15^\circ < \theta < 165^\circ$, $p_\mathrm{T} > 0.1$ GeV at champion operating point $t_\beta=0.60, t_d=0.10$):

## 1. Benchmark Results Summary

| Metric | Selection / Condition | Value | Benchmark Target | Status |
|---|---|:---:|:---:|:---:|
| **Tracking Efficiency ($N_\mathrm{hits} > 10$)** | Standard IDEA benchmark tracks | **97.28%** | $> 90.0\%$ | **Exceeded (+7.28%)** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$)** | Inclusive track recovery down to 4 hits | **96.12%** | — | High inclusive recovery |
| **Fake Rate** | Unmatched non-merged candidates / all candidates | **3.74%** | $< 8.0\%$ | **Exceeded (2.1x lower)** |
| **Merge Rate** | Multi-track candidate coverage ($>75\%$ purity) | **12.50%** | — | Clean separation |
| **Reconstructed Candidates / Event** | Full detector acceptance | **36.14** | — | Normal multiplicity |
| **Evaluated Sample Size** | 100 seeds, 50,000 events | **1,672,188 targets** | — | Full statistics |

## 2. Key Physical Highlights for PR #3

- **Conformal Geometric Representation:** Conformal geometric algebra ($Cl(4,1)$) natively represents drift chamber measurement circles (wire center, wire direction, drift radius). This eliminates the discrete left/right point ambiguity upstream and achieves **97.28% tracking efficiency** on standard benchmark tracks ($N_\mathrm{hits} > 10$).
- **Ultra-Low Fake Rate:** Achieves **3.74% fake rate**, comfortably outperforming the standard 8.0% benchmark target.
- **Consistent High Acceptance:** Tracking efficiency starts at 92.5% at $p_\mathrm{T} = 100$ MeV and reaches a 98–99.5% plateau above 1 GeV, remaining above 96% across the entire polar angle acceptance ($15^\circ \le \theta \le 165^\circ$).

Generated figures: `head_to_head_keepall_efficiency.png` and `fcc_comprehensive_suite.png` (with vector PDF siblings).


# End-to-End Cycle Evaluation

## Overview

The complete cycle connects the **direct task** (DDPM: parameters → image) and the **inverse task** (Xception: image → parameters) into a closed loop. It provides a holistic assessment of how well the generative model preserves the physical information encoded in Hamiltonian parameters, evaluated through three complementary lenses: regression quality on recovered parameters, pixel-level image fidelity, and thermodynamic physical observables.

---

## Pipeline Diagram

```
Input parameters θ ∈ ℝ⁸  (physical units)
        │
        ▼
  ┌─────────────┐
  │    DDPM     │  params → normalised cond. → generates 40×40 image in [-1,1]
  └─────────────┘
        │
        ▼  centre crop → 39×39, rescale to [0,1]
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
  ┌──────────────────┐                                   ┌──────────────────────┐
  │  Image metrics   │                                   │  Physical metrics    │
  │  MSE, Var(MSE)   │  ← compare with original →       │  M, |M|, χ, Cv, E    │
  │  SSIM, FFT-MSE   │                                   │  (disk mask r=18.3)  │
  │  FFT-Corr        │                                   └──────────────────────┘
  └──────────────────┘
        │
        ▼  resize to 224×224, grayscale→RGB
  ┌─────────────┐
  │  Xception   │  image → predicted θ̂ ∈ ℝ⁸ (normalised) → inverse MinMax → physical units
  └─────────────┘
        │
        ▼
  Regression metrics: R², MAE, RMSE  (θ_input vs θ̂)
```

---

## Critical Preprocessing Rules

Correct scaler alignment is essential for the cycle to be consistent. Both models were trained with their own MinMaxScaler fitted **only on the internal training split**:

| Step | Model | Scaler action |
|---|---|---|
| `params_phys_to_ddpm_cond(y_phys)` | DDPM | Apply DDPM scaler (fitted on DDPM train split) |
| `ddpm_image_to_xception_input(img_40)` | — | Crop 40→39, [−1,1]→[0,1], resize 39→224, broadcast to RGB |
| `xception_pred_to_phys(y_scaled)` | Xception | Inverse apply Xception scaler (fitted on Xception train split) |

**The two scalers are not the same** because the internal dataset version (`dataset-spines-united-v2`, 169,671 samples) used for training differs slightly in split seeds from the full dataset. Mixing scalers corrupts the cycle.

---

## Evaluation Protocols

### A. Internal Dataset (ciclo_completo_resultadosxclusters)

- Source: test split of `dataset-spines-united-v2` (15% holdout, same seed as training)
- DDPM generates $K_{ens}$ samples per parameter point (ensemble evaluation)
- Regression metrics compare **input parameters** $\boldsymbol{\theta}$ vs **re-estimated parameters** $\hat{\boldsymbol{\theta}}$
- Image metrics compare **original $s_z$ image** vs **generated image** (after crop)
- Physical metrics computed separately for original (single-config estimator, regime a) and generated (ensemble estimator, regime b)
- Additional analysis: **per-cluster R² and MAE** using magnetic phase labels from the dataset

### B. External Dataset (ciclo_external_dataset)

- Source: `dataset-spines-complete` (218,256 samples — the full dataset from Paper 1)
- **Scalers are still fitted on the internal dataset** (no re-fitting on external data)
- Evaluates out-of-distribution generalisation of the full cycle
- Currently the external cycle shows degraded performance, indicating a distribution shift between datasets or scaler mismatch effects
- Work in progress

---

## Metrics Reference

### Regression Metrics (parameter recovery)

| Metric | Formula | Interpretation |
|---|---|---|
| $R^2$ | $1 - \frac{\sum(\theta - \hat\theta)^2}{\sum(\theta - \bar\theta)^2}$ | Variance explained; 1.0 = perfect |
| MAE | $\frac{1}{N}\sum|\theta - \hat\theta|$ | Mean absolute error in physical units |
| RMSE | $\sqrt{\frac{1}{N}\sum(\theta - \hat\theta)^2}$ | Root mean squared error |

### Image Metrics

See [07_metrics.md](07_metrics.md) for full formulas.

| Metric | Range | Good value |
|---|---|---|
| MSE | $[0, \infty)$ | Low |
| Var(MSE) | $[0, \infty)$ | Low |
| SSIM | $[-1, 1]$ | High (→1) |
| FFT-MSE | $[0, \infty)$ | Low |
| FFT-Corr | $[-1, 1]$ | High (→1) |

### Physical Metrics

| Observable | Symbol | Physical meaning |
|---|---|---|
| Average magnetisation | $M$ | Net spin polarisation; $\approx \pm 1$ = ferromagnetic, $\approx 0$ = chiral/AFM |
| Absolute magnetisation | $|M|$ | Orientation-invariant order parameter |
| Magnetic susceptibility | $\chi$ | Sensitivity of magnetisation to field fluctuations |
| Specific heat | $C_v$ | Energy fluctuation magnitude; peaks at phase transitions |
| Exchange energy density | $E$ | $s_z$-projected nearest-neighbour exchange energy |

---

## Per-Cluster Analysis

The internal cycle notebook additionally computes metrics broken down by **magnetic phase cluster** (using the `labels` array in the dataset). This allows identifying:
- Which phases the DDPM faithfully reproduces
- Which phases suffer most from degenerate generation
- How per-cluster image quality correlates with per-cluster regression accuracy

---

## Notebooks

| Notebook | Dataset | Status |
|---|---|---|
| [ciclo_completo_resultadosxclusters](../notebooks/cycle/ciclo_completo_resultadosxclusters.ipynb) | Internal test split | ✅ Complete |
| [ciclo_external_dataset](../notebooks/cycle/ciclo_external_dataset.ipynb) | External (218k) | 🔧 In progress |

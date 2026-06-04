# End-to-End Magnetic State Generation Framework

An end-to-end deep learning framework for **generating and inverting 2D magnetic domain images** of nanodots governed by the extended Heisenberg Hamiltonian.

This repository contains the code for two complementary papers:

| | Paper | Task | Status |
|---|---|---|---|
| 📄 | *Regression-Based Explainable DL for Estimating Hamiltonian Parameters from Magnetic Nanodot Images* | **Inverse:** image → parameters | Published preprint |
| 📝 | *Image-Driven Estimation of Magnetic Configurations in Nanodots* | **Direct:** parameters → image | In preparation |

---

## Overview

Magnetic nanodots exhibit rich topological spin textures (skyrmions, helical domains, labyrinthine patterns) governed by a competition of exchange, Dzyaloshinskii–Moriya (DMI), anisotropy, and Zeeman interactions. Both predicting these textures from parameters and recovering parameters from images are challenging inverse/direct problems due to the **degeneracy** of the magnetic energy landscape.

This project addresses both directions with a closed **cycle**:

```
θ (8 Hamiltonian parameters)
        │
        ▼
   ┌─────────┐
   │  DDPM   │  Direct task: conditional image generation
   └─────────┘
        │  generated image (39×39)
        ▼
   ┌──────────┐
   │ Xception │  Inverse task: parameter regression
   └──────────┘
        │
        ▼
      θ̂ (recovered parameters)
```

---

## Repository Structure

```
.
├── docs/                              # Theory documentation
│   ├── 01_hamiltonian.md              # Extended Heisenberg Hamiltonian & parameter decomposition
│   ├── 02_monte_carlo.md              # MH-MC simulation & dataset generation pipeline
│   ├── 03_inverse_problem.md          # Image→params regression, Xception, RAMs
│   ├── 04_direct_problem.md           # Params→image: CVAE & DDPM architectures
│   ├── 05_complete_cycle.md           # End-to-end cycle pipeline & evaluation
│   ├── 06_datasets.md                 # Kaggle datasets, splits, preprocessing
│   └── 07_metrics.md                  # All metrics: regression, image, physical
│
├── notebooks/
│   ├── inverse/
│   │   ├── XceptionFullDataBaseV3100.ipynb   # Xception inverse model training ✅
│   │   └── xception_inverse.md               # Notebook documentation
│   ├── generative/
│   │   ├── ddpm_spines_train.ipynb           # DDPM generative model training ✅
│   │   └── ddpm_train.md                     # Notebook documentation
│   └── cycle/
│       ├── ciclo_completo_resultadosxclusters.ipynb  # Cycle on internal dataset ✅
│       ├── ciclo_completo.md                          # Notebook documentation
│       ├── ciclo_external_dataset.ipynb               # Cycle on external dataset 🔧
│       └── ciclo_external.md                          # Notebook documentation
│
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Datasets

Both datasets are publicly available on Kaggle:

| Dataset | Slug | Samples | Used for |
|---|---|---|---|
| Internal | `carloscanamejoy/dataset-spines-united-v2` | 169,671 | Model training + internal cycle |
| External | `carloscanamejoy/dataset-spines-complete` | 218,256 | External cycle (generalisation test) |

Each sample: a 39×39 grayscale $s_z$ magnetisation image + 8 Hamiltonian parameters.  
See [docs/06_datasets.md](docs/06_datasets.md) for full details and preprocessing rules.

---

## Notebooks at a Glance

### 1. Xception Inverse Model — `notebooks/inverse/`

Trains an Xception CNN to predict 8 Hamiltonian parameters from a magnetisation image.

**Key results on test set:**

| Parameter | $R^2$ | MAE |
|---|---|---|
| $\tilde{K}_{DM}$ | 0.9498 | 0.049 meV |
| $\tilde{J}_2$ | 0.9146 | 0.030 meV |
| $T^{(0)}$ | 0.8430 | 0.890 K |

See [notebooks/inverse/xception_inverse.md](notebooks/inverse/xception_inverse.md).

### 2. DDPM Generative Model — `notebooks/generative/`

Trains a conditional DDPM (U-Net backbone, cosine schedule, EMA) to generate 39×39 magnetisation images conditioned on 8 Hamiltonian parameters. Uses PyTorch.

See [notebooks/generative/ddpm_train.md](notebooks/generative/ddpm_train.md).

### 3. Complete Cycle — Internal Dataset — `notebooks/cycle/`

Runs the full θ → DDPM → image → Xception → θ̂ pipeline on the internal test split. Evaluates with regression metrics (R², MAE), image metrics (MSE, SSIM, FFT-Corr), physical observables (M, χ, Cv, E), and per-magnetic-phase breakdowns.

See [notebooks/cycle/ciclo_completo.md](notebooks/cycle/ciclo_completo.md).

### 4. Complete Cycle — External Dataset — `notebooks/cycle/` 🔧

Same pipeline applied to the larger external dataset (218k). Currently under investigation for distribution shift issues.

See [notebooks/cycle/ciclo_external.md](notebooks/cycle/ciclo_external.md).

---

## Theoretical Background

| Document | Content |
|---|---|
| [01_hamiltonian.md](docs/01_hamiltonian.md) | Full Hamiltonian with all 8 parameters, physical ranges, emergent phases |
| [02_monte_carlo.md](docs/02_monte_carlo.md) | MH-MC algorithm, simulated annealing, image generation pipeline |
| [03_inverse_problem.md](docs/03_inverse_problem.md) | Regression formulation, Xception, RAMs interpretability, identifiability analysis |
| [04_direct_problem.md](docs/04_direct_problem.md) | CVAE (ELBO, KLD, β-scheduling) and DDPM (forward/reverse process) |
| [05_complete_cycle.md](docs/05_complete_cycle.md) | Full cycle pipeline, preprocessing rules, evaluation protocols |
| [06_datasets.md](docs/06_datasets.md) | Dataset descriptions, splits, MinMaxScaler alignment rules |
| [07_metrics.md](docs/07_metrics.md) | All metrics with formulas: regression, image, physical |

---

## Quick Start (Google Colab)

Each notebook is self-contained and designed to run on **Google Colab with GPU** (A100 recommended for DDPM training).

**Prerequisites:**
1. Upload your `kaggle.json` API key when prompted
2. Select **Runtime → Change runtime type → GPU**
3. Run cells top to bottom

**Dependency installation** is handled inside each notebook via `!pip install`.

For a local environment, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Hamiltonian Parameters at a Glance

The 8 free parameters of the extended Heisenberg Hamiltonian (with $\tilde{J}_1 = 1.0$ meV fixed):

| # | Symbol | Role | Range |
|---|---|---|---|
| 0 | $T^{(0)}$ [K] | Annealing temperature | 0.2–20.0 |
| 1 | $\tilde{J}_2$ [meV] | 2nd-shell exchange | −0.66–0.66 |
| 2 | $\tilde{J}_3$ [meV] | 3rd-shell exchange | −0.29–0.29 |
| 3 | $\tilde{J}_4$ [meV] | 4th-shell exchange | −0.23–0.23 |
| 4 | $\tilde{K}_{an1}$ [meV/atom] | Bulk anisotropy | 0.0–0.2 |
| 5 | $\tilde{K}_{anS}$ [meV/atom] | Surface anisotropy | 0.0–0.2 |
| 6 | $\tilde{H}_{ex}$ [meV/atom] | External Zeeman field | 0.0–0.05 |
| 7 | $\tilde{K}_{DM}$ [meV] | DMI strength | 0.0–1.2 |

---

## Authors

- J. Méndez-Rondón — AI-LAB/Signal Processing and Recognition Group, Universidad Nacional de Colombia
- C. Canamejoy-Alegría — Signal Processing and Recognition Group, Universidad Nacional de Colombia
- J. Agudelo-Giraldo — Departamento de Física y Matemáticas, Universidad Autónoma de Manizales
- J. Montes-Monsalve — Dirección Académica, Universidad Nacional de Colombia, Sede La Paz
- A. Álvarez-Meza — AI-LAB/Signal Processing and Recognition Group, Universidad Nacional de Colombia

**Funding:** Hermes 62642, Universidad Nacional de Colombia; Code 111908, Minciencias (951-2024).

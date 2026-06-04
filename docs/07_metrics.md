# Evaluation Metrics

Three families of metrics are used to evaluate the complete cycle: **regression metrics** (parameter recovery quality), **image metrics** (pixel/spectral fidelity), and **physical metrics** (thermodynamic observables on the nanodot disk).

---

## 1. Regression Metrics (Parameter Recovery)

Applied to compare input parameters $\boldsymbol{\theta}$ vs re-estimated parameters $\hat{\boldsymbol{\theta}}$ after the full cycle.

### Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{\sum_{n=1}^{N}(\theta_{n,p} - \hat\theta_{n,p})^2}{\sum_{n=1}^{N}(\theta_{n,p} - \bar\theta_p)^2}
$$

- $\bar\theta_p$: mean of the ground-truth values for parameter $p$
- Range: $(-\infty, 1]$; $R^2 = 1$ is perfect, $R^2 = 0$ equals predicting the mean, $R^2 < 0$ is worse than the mean
- **Note:** $R^2$ is statistically undefined when the target has near-zero variance (e.g., $\tilde{K}_{DM} \approx 0$ in the ferromagnetic cluster). Reported as N/A in those cases.

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N}\sum_{n=1}^{N}|\theta_{n,p} - \hat\theta_{n,p}|
$$

- Reported in the **original physical units** of each parameter (K for $T^{(0)}$, meV for exchange/DMI, meV/atom for anisotropy/field)
- When computed on Min-Max normalised targets $[0,1]$, MAE is dimensionless and directly comparable across parameters

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(\theta_{n,p} - \hat\theta_{n,p})^2}
$$

---

## 2. Image Metrics

Applied to compare original $s_z$ images vs DDPM-generated images (both cropped to 39×39). All metrics are restricted to the **circular disk mask** $\mathcal{M}$ (radius $r_d = 18.3$ px).

### MSE Variance (Var-MSE)

Measures spatial heterogeneity of per-pixel squared errors:

$$
\text{Var}_{\text{MSE}}(a, b) = \frac{1}{N}\sum_{i=1}^{N}\left[e_i^2 - \overline{e^2}\right]^2, \quad e_i = (a_i - b_i)^2
$$

- $e_i$: squared error at pixel $i$
- $\overline{e^2} = \frac{1}{N}\sum_i e_i$: mean squared error (MSE)
- High Var-MSE indicates spatially concentrated errors (e.g., on domain walls only)

### SSIM (Structural Similarity Index)

$$
\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}
$$

- Computed over local windows (Gaussian kernel)
- $\text{SSIM} \in [-1, 1]$; higher is better
- **Limitation for periodic textures:** phase-shifted configurations (same frequency, different phase) are energetically equivalent but penalised by SSIM. FFT metrics address this.

### Spectral MSE (FFT-MSE)

$$
\text{FFT-MSE}(a, b) = \frac{1}{N}\sum_{i=1}^{N}\left(\tilde{S}_a^{(i)} - \tilde{S}_b^{(i)}\right)^2, \quad \tilde{S}_x = \frac{|\mathcal{F}\{x\}|}{\max|\mathcal{F}\{x\}|}
$$

- $\mathcal{F}\{x\}$: 2D DFT of image $x$, zero-frequency shifted to centre
- $\tilde{S}_x \in [0,1]^{H \times W}$: normalised amplitude spectrum
- Captures differences in **spatial frequency content** regardless of phase shifts
- Low FFT-MSE: the generated image has the same periodicity and domain-wall spacing as the original

### Spectral Pearson Correlation (FFT-Corr)

$$
\text{FFT-Corr}(a, b) = \frac{\sum_{i=1}^{N}(\tilde{S}_a^{(i)} - \bar{S}_a)(\tilde{S}_b^{(i)} - \bar{S}_b)}{\sqrt{\sum_{i=1}^{N}(\tilde{S}_a^{(i)} - \bar{S}_a)^2}\;\sqrt{\sum_{i=1}^{N}(\tilde{S}_b^{(i)} - \bar{S}_b)^2}}
$$

- $\text{FFT-Corr} \in [-1, 1]$; higher is better
- Measures the linear correlation between the amplitude spectra of the two images
- Robust to differences in absolute spectral amplitude; captures spectral shape similarity

---

## 3. Physical Metrics

Computed within the circular disk mask $\mathcal{M} = \{(y,x) \in \mathbb{Z}^2 : (y - c_y)^2 + (x - c_x)^2 \leq r_d^2\}$ with $r_d = 18.3$ px and $N_\mathcal{M} = |\mathcal{M}| \approx 1051$ pixels (39×39 image).

Physical metrics are evaluated in two regimes depending on the image origin:

| Regime | Source | Ensemble |
|---|---|---|
| **(a) Single-config** | Simulated / original images | Not available; use spatial proxies |
| **(b) Ensemble** | Generated images ($K_{ens}$ samples per θ) | Direct thermodynamic formulas |

### Average Magnetisation $M$

**Regime (a):**
$$M = \frac{1}{N_\mathcal{M}}\sum_{i \in \mathcal{M}} s_z(i)$$

**Regime (b):**
$$\langle M \rangle = \frac{1}{K_{ens}}\sum_{k=1}^{K_{ens}}\left[\frac{1}{N_\mathcal{M}}\sum_{i \in \mathcal{M}} s_z^{(k)}(i)\right]$$

- $M \in [-1, 1]$: $M \approx \pm 1$ is ferromagnetic; $M \approx 0$ is chiral/AFM/skyrmion

### Absolute Magnetisation $|M|$

**Regime (a):**
$$|M| = \frac{1}{N_\mathcal{M}}\left|\sum_{i \in \mathcal{M}} s_z(i)\right|$$

**Regime (b):**
$$\langle |M| \rangle = \frac{1}{K_{ens}}\sum_{k=1}^{K_{ens}}\left|\frac{1}{N_\mathcal{M}}\sum_{i \in \mathcal{M}} s_z^{(k)}(i)\right|$$

Note: $\langle |M| \rangle \neq |\langle M \rangle|$ in general; the absolute value is taken **before** ensemble averaging.

### Magnetic Susceptibility $\chi$

**Regime (a) — spatial proxy** (static fluctuation–dissipation sum rule):
$$\chi = \frac{1}{T}\sum_{\mathbf{r}} G(\mathbf{r}), \quad G(\mathbf{r}) = \frac{1}{N_\mathcal{M}}\sum_{i \in \mathcal{M}} s_z(i)\,s_z(i+\mathbf{r}) - \bar{s}_z^2$$

$G(\mathbf{r})$ is the spatial connected correlation function, computed from the single configuration via the Wiener–Khinchin theorem.

**Regime (b) — ensemble:**
$$\chi = \frac{N_\mathcal{M}}{T}\left(\langle M^2 \rangle_{ens} - \langle M \rangle_{ens}^2\right)$$

### Exchange Energy Density $E$

**Regime (a):**
$$E = -\frac{1}{N_\mathcal{M}}\sum_{\substack{\langle i,j \rangle \\ i,j \in \mathcal{M}}} s_z(i)\,s_z(j)$$

**Regime (b):**
$$\langle E \rangle = -\frac{1}{K_{ens}}\sum_{k=1}^{K_{ens}}\left[\frac{1}{N_\mathcal{M}}\sum_{\substack{\langle i,j \rangle \\ i,j \in \mathcal{M}}} s_z^{(k)}(i)\,s_z^{(k)}(j)\right]$$

- $\langle i,j \rangle$: nearest-neighbour pairs (row/column directions), each pair counted once; boundary pairs excluded
- $E \approx -1$: fully aligned ferromagnetic; $E \approx 0$: fully disordered
- **Captures only the $S_z^i S_z^j$ component** of the full Heisenberg exchange; transverse components and further-neighbour terms are not recoverable from the scalar $s_z$ image

### Specific Heat $C_v$

**Regime (a) — Binder's subsystem fluctuation method:**
$$C_v = \frac{|\mathcal{M}|}{T^2}\left(\langle \varepsilon^2 \rangle_B - \langle \varepsilon \rangle_B^2\right)$$

where $\varepsilon_k$ is the local energy density of non-overlapping $\ell \times \ell$ blocks ($\ell = 5$, retaining blocks with $|B_k \cap \mathcal{M}| \geq 4$, giving $K \approx 50$ valid blocks).

**Regime (b) — ensemble:**
$$C_v = \frac{N_\mathcal{M}}{T^2}\left(\langle E^2 \rangle_{ens} - \langle E \rangle_{ens}^2\right)$$

---

## Circular Disk Mask Definition

$$
\mathcal{M} = \left\{(y, x) \in \mathbb{Z}^2 : (y - c_y)^2 + (x - c_x)^2 \leq r_d^2\right\}
$$

- $(c_y, c_x) = \left(\frac{H-1}{2}, \frac{W-1}{2}\right)$: image centre
- $r_d = 18.3$ px: physical nanodot radius
- $N_\mathcal{M} = |\mathcal{M}| \approx 1051$ for a $40 \times 40$ image

All physical observables and FFT metrics are evaluated **only within** $\mathcal{M}$, setting background pixels to zero.

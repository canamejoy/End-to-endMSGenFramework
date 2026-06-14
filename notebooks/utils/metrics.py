"""
metrics.py — Shared metric functions for generative evaluation.
All functions assume images in [-1, 1] range, shape (H, W).
"""
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.metrics import structural_similarity as skssim
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# Cluster-id → magnetic structure mapping (6 distinct phases)
STRUCTURE_MAP = {
    4:  "Helical",
    5:  "Helical",
    13: "Helical",
    6:  "Labyrinthine & Conical",
    14: "Labyrinthine & Conical",
    8:  "Bimeron",
    10: "Ferromagnetic",
    11: "Ferromagnetic",
    12: "Ferromagnetic",
    15: "Skyrmions",
    16: "Skyrmions",
    17: "Field-Saturated",
}

STRUCTURE_NAMES = [
    "Ferromagnetic",
    "Helical",
    "Labyrinthine & Conical",
    "Bimeron",
    "Skyrmions",
    "Field-Saturated",
]

STRUCTURE_COLORS = {
    "Ferromagnetic":          "#1f77b4",   # blue
    "Helical":                "#d62728",   # red
    "Labyrinthine & Conical": "#2ca02c",   # green
    "Bimeron":                "#9467bd",   # purple
    "Skyrmions":              "#8c564b",   # brown
    "Field-Saturated":        "#e377c2",   # pink
}


def get_structure_label(cluster_id):
    """Map a numeric cluster ID to its structure name string."""
    return STRUCTURE_MAP.get(int(cluster_id), f"Unknown({cluster_id})")


MODEL_COLORS = {
    "DDPM": "#2563EB",
    "CVAE-Xception": "#16A34A",
    "CVAE-ViT": "#DC2626",
}
PARAM_NAMES = ["T⁰", "J̃₂", "K̃DM", "H̃ex", "K̃anS", "K̃an1", "J̃₃", "J̃₄"]


def circular_mask(h=39, w=39):
    """Binary mask: True inside the inscribed circle of the h x w image."""
    cy, cx = h // 2, w // 2
    r = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


MASK = circular_mask()


def masked_mse(a, b, mask=MASK):
    return ((a - b) ** 2)[mask].mean()


def masked_bce(a, b, mask=MASK, eps=1e-7):
    a_ = (a[mask] + 1) / 2
    b_ = (b[mask] + 1) / 2
    return -np.mean(a_ * np.log(b_ + eps) + (1 - a_) * np.log(1 - b_ + eps))


def masked_ssim(a, b):
    return skssim(a, b, data_range=2.0)


def cosine_similarity_pair(z1, z2):
    n1 = np.linalg.norm(z1) + 1e-8
    n2 = np.linalg.norm(z2) + 1e-8
    return float(np.dot(z1 / n1, z2 / n2))


def cosine_similarity_batch(z1, z2):
    """Cosine similarity between paired feature vectors, shape (B, D) -> (B,)."""
    n1 = np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8
    return np.sum((z1 / n1) * (z2 / n2), axis=-1)


def magnetization(img, mask=MASK):
    """Mean sz over the disk mask. Range [-1, 1]."""
    return img[mask].mean()


def abs_magnetization(img, mask=MASK):
    return np.abs(img[mask].mean())


def cnn_correlation(img, mask=MASK):
    """Vectorized nearest-neighbor spin correlation within the mask."""
    shifts = [(0, 1), (1, 0)]
    total, count = 0.0, 0
    for dy, dx in shifts:
        a = img[:-dy or None, :-dx or None] if (dy > 0 or dx > 0) else img
        b = img[dy:, dx:]
        ma = mask[:-dy or None, :-dx or None] if (dy > 0 or dx > 0) else mask
        mb = mask[dy:, dx:]
        valid = ma & mb
        total += (a * b)[valid].sum()
        count += valid.sum()
    return total / count if count > 0 else 0.0


def structure_factor(img):
    """Compute 2D structure factor S(q) = |FFT(img)|^2 / N."""
    N = img.size
    ft = fftshift(fft2(img))
    return np.abs(ft) ** 2 / N


def azimuthal_average(sq_2d, n_bins=None):
    """Azimuthally average S(q) into radial bins. Returns (q_bins, sq_avg)."""
    h, w = sq_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    q_bins, sq_avg = [], []
    for r in range(1, max_r + 1):
        ring = sq_2d[R == r]
        if len(ring) > 0:
            q_bins.append(float(r))
            sq_avg.append(ring.mean())
    return np.array(q_bins), np.array(sq_avg)


def oz_fit(img, q_max_fraction=0.4):
    """
    Fit Ornstein-Zernike: 1/S(q) = a + b*q^2 in the low-q regime.
    Returns: chi_proxy = 1/a, xi = sqrt(b/a), r2 of the fit.
    Returns (nan, nan, nan) if the fit fails or is unphysical (a<=0 or b<=0).
    The fit is physically meaningful only in disordered regimes — always
    report r2 alongside chi and xi.
    """
    sq_2d = structure_factor(img)
    q, sq = azimuthal_average(sq_2d)
    q_max = q_max_fraction * q.max()
    mask_q = (q > 0) & (q <= q_max) & (sq > 0)
    q_fit, sq_fit = q[mask_q], sq[mask_q]
    if len(q_fit) < 4:
        return np.nan, np.nan, np.nan
    inv_sq = 1.0 / sq_fit
    try:
        popt, _ = curve_fit(lambda q, a, b: a + b * q ** 2, q_fit, inv_sq,
                            p0=[1.0, 1.0], maxfev=5000)
        a, b = popt
        if a <= 0 or b <= 0:
            return np.nan, np.nan, np.nan
        r2 = r2_score(inv_sq, a + b * q_fit ** 2)
        return 1.0 / a, np.sqrt(b / a), r2
    except Exception:
        return np.nan, np.nan, np.nan


def chi_ensemble(ensemble_imgs, mask=MASK, temperature=1.0):
    """
    Fluctuation-dissipation estimate of susceptibility from a K-sample ensemble.
    chi = N/T * Var(m), where m_k = mean sz over mask for sample k.
    """
    ms = np.array([img[mask].mean() for img in ensemble_imgs])
    N = int(mask.sum())
    return N * float(ms.var()) / temperature


def center_crop(img, size=39):
    """
    Crop a (40, 40) DDPM output to (size, size) using top-left crop.
    No interpolation — preserves original pixel values.
    img: numpy array of shape (H, W) or (H, W, C), H >= size, W >= size.
    """
    return img[:size, :size]


def normalize_metrics(scores_dict, reference_key="E0", worst_key="E2"):
    """
    Normalize raw metric scores for interpretability.

    Similarity metrics (ssim, cosine): divided by mean of reference condition (E0).
        Normalized E0 = 1.0. Degraded conditions < 1.0.

    Error metrics (mse, bce): divided by mean of worst condition (E2).
        Normalized E2 = 1.0. Reference E0 ≈ 0.0.

    Args:
        scores_dict: dict mapping condition keys (e.g. "E0","E1","E2","E3") to
                     sub-dicts {"mse": array, "bce": array, "ssim": array, "cosine": array}
        reference_key: condition key used as denominator for similarity metrics
        worst_key: condition key used as denominator for error metrics

    Returns:
        norm_dict: same structure as scores_dict with normalized values
        denominators: dict of the actual denominator values used
    """
    sim_metrics   = ["ssim", "cosine"]
    error_metrics = ["mse", "bce"]

    denom_sim   = {m: float(np.mean(scores_dict[reference_key][m])) for m in sim_metrics}
    denom_error = {m: float(np.mean(scores_dict[worst_key][m]))     for m in error_metrics}
    denominators = {**denom_sim, **denom_error}

    norm_dict = {}
    for cond, metric_scores in scores_dict.items():
        norm_dict[cond] = {}
        for m, vals in metric_scores.items():
            vals = np.asarray(vals)
            denom = denom_sim[m] if m in sim_metrics else denom_error[m]
            norm_dict[cond][m] = vals / denom if denom > 1e-9 else vals

    return norm_dict, denominators


def shift_image(img, px, axis=1):
    """Shift image by px pixels along axis using np.roll."""
    return np.roll(img, px, axis=axis)


def reflect_image(img):
    """Horizontal flip (left-right reflection)."""
    return img[:, ::-1]


def save_figure(fig, path_no_ext, dpi=300):
    """Save figure in both PNG (300 dpi) and SVG formats."""
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")


def apply_figure_style():
    """Global publication figure style — call once at the top of each notebook."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })

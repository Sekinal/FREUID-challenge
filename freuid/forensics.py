"""Hand-crafted statistical / noise forensics features for ID-document images.

Motivation (see agents_docs/04): the dataset's attacks are physical manipulation,
GenAI digital edits, and print-and-capture. Deep backbones overfit the *seen*
document types; hand-crafted forensic statistics encode a generator-agnostic
*physical/statistical prior* that can generalise to **unseen** types far better.
This module extracts such a feature bank using only numpy / scipy / PIL (no cv2,
no torch), so it is CPU-cheap, parallelisable, and interpretable.

Feature families (mapped to the attack families):
- metadata / compression  -> recompression & resave fingerprints
- colour / intensity stats -> global statistical anomalies
- noise residual + local block-variance inconsistency -> splicing / inpainting
- DCT statistics: Benford divergence, blockiness -> JPEG / double-compression
- radial power spectrum + peakiness -> GAN/diffusion up-sampling & moiré (recapture)
- ELA (error-level analysis) -> locally-edited regions

``extract_features(path)`` is defensive: any failure in a family yields NaNs for
that family rather than raising, so a single corrupt image never kills a batch.
``FEATURE_NAMES`` is the canonical ordered list; ``feature_vector(path)`` returns
a float64 vector in that exact order.
"""
from __future__ import annotations

import io as _io
import os
from pathlib import Path

import numpy as np

# Heavy-ish imports are module-level so multiprocessing workers pick them up once.
from PIL import Image, ImageFile
from scipy import fft as _sfft
from scipy import ndimage as _ndi
from scipy import stats as _stats

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Analysis crop side (native pixels, no resize -> preserves noise statistics).
CROP = 512
# Radial power-spectrum bins.
N_RADIAL = 16
# Benford expected first-significant-digit distribution.
_BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))


# --------------------------------------------------------------------------
# Canonical feature order
# --------------------------------------------------------------------------
def _build_names() -> list[str]:
    names: list[str] = []
    # metadata / compression
    names += ["meta_w", "meta_h", "meta_aspect", "meta_megapix",
              "meta_filebytes", "meta_log_filebytes", "meta_bits_per_pixel"]
    # colour / intensity (R,G,B,gray)
    for ch in ("r", "g", "b", "gray"):
        names += [f"col_{ch}_mean", f"col_{ch}_std", f"col_{ch}_skew", f"col_{ch}_kurt"]
    names += ["col_sat_mean", "col_sat_std", "col_frac_bright", "col_frac_dark"]
    # noise residual + local inconsistency
    names += ["res_std", "res_skew", "res_kurt",
              "res_tilevar_std", "res_tilevar_iqr", "res_tilevar_maxratio",
              "res_lap_std", "res_hf_energy"]
    # DCT / JPEG
    names += ["dct_benford_chi2", "dct_ac_std", "dct_ac_kurt",
              "dct_zero_frac", "jpeg_blockiness", "jpeg_grid_ratio"]
    # radial power spectrum
    names += [f"spec_radial_{i}" for i in range(N_RADIAL)]
    names += ["spec_slope", "spec_hf_lf_ratio", "spec_peak_count", "spec_peak_max"]
    # ELA
    names += ["ela_mean", "ela_std", "ela_max", "ela_p99", "ela_frac_high"]
    return names


FEATURE_NAMES: list[str] = _build_names()
N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _center_square(arr: np.ndarray, side: int) -> np.ndarray:
    """Center crop a square of up to ``side`` pixels (no resize)."""
    h, w = arr.shape[:2]
    s = min(side, h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return arr[top:top + s, left:left + s]


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    x = x.ravel()
    if x.size < 4:
        return (float("nan"),) * 4
    return (
        float(np.mean(x)),
        float(np.std(x)),
        float(_stats.skew(x, bias=False)) if x.std() > 1e-9 else 0.0,
        float(_stats.kurtosis(x, bias=False)) if x.std() > 1e-9 else 0.0,
    )


def _safe(d: dict, keys: list[str], values) -> None:
    for k, v in zip(keys, values):
        d[k] = float(v)


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------
def _f_meta(d: dict, img: Image.Image, path: str) -> None:
    w, h = img.size
    try:
        fb = os.path.getsize(path)
    except OSError:
        fb = float("nan")
    _safe(d, ["meta_w", "meta_h", "meta_aspect", "meta_megapix",
              "meta_filebytes", "meta_log_filebytes", "meta_bits_per_pixel"],
          [w, h, (w / h) if h else float("nan"), w * h / 1e6,
           fb, np.log10(fb + 1.0) if fb == fb else float("nan"),
           (fb * 8.0 / (w * h * 3.0)) if (h and w and fb == fb) else float("nan")])


def _f_color(d: dict, rgb: np.ndarray, gray: np.ndarray) -> None:
    for i, ch in enumerate(("r", "g", "b")):
        m = _moments(rgb[..., i])
        _safe(d, [f"col_{ch}_mean", f"col_{ch}_std", f"col_{ch}_skew", f"col_{ch}_kurt"], m)
    _safe(d, ["col_gray_mean", "col_gray_std", "col_gray_skew", "col_gray_kurt"], _moments(gray))
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    _safe(d, ["col_sat_mean", "col_sat_std"], [sat.mean(), sat.std()])
    _safe(d, ["col_frac_bright", "col_frac_dark"],
          [float((gray > 250).mean()), float((gray < 5).mean())])


def _f_residual(d: dict, gray: np.ndarray) -> None:
    blur = _ndi.gaussian_filter(gray, sigma=1.0)
    resid = gray - blur
    _, rstd, rskew, rkurt = _moments(resid)
    # tile (block) variance inconsistency -> splicing / inpainting signal
    t = 32
    h, w = gray.shape
    nh, nw = h // t, w // t
    if nh >= 2 and nw >= 2:
        tiles = resid[:nh * t, :nw * t].reshape(nh, t, nw, t)
        var = tiles.var(axis=(1, 3)).ravel()
        logv = np.log1p(var)
        q1, q3 = np.percentile(logv, [25, 75])
        tv_std = float(logv.std())
        tv_iqr = float(q3 - q1)
        med = float(np.median(var))
        tv_maxratio = float(var.max() / (med + 1e-9)) if med >= 0 else float("nan")
    else:
        tv_std = tv_iqr = tv_maxratio = float("nan")
    lap = _ndi.laplace(gray)
    _safe(d, ["res_std", "res_skew", "res_kurt",
              "res_tilevar_std", "res_tilevar_iqr", "res_tilevar_maxratio",
              "res_lap_std", "res_hf_energy"],
          [rstd, rskew, rkurt, tv_std, tv_iqr, tv_maxratio,
           float(lap.std()), float(np.mean(resid ** 2))])


def _f_dct(d: dict, gray: np.ndarray) -> None:
    h, w = gray.shape
    nh, nw = h // 8, w // 8
    if nh < 1 or nw < 1:
        _safe(d, ["dct_benford_chi2", "dct_ac_std", "dct_ac_kurt", "dct_zero_frac"],
              [float("nan")] * 4)
    else:
        blocks = gray[:nh * 8, :nw * 8].reshape(nh, 8, nw, 8).transpose(0, 2, 1, 3)
        coeffs = _sfft.dctn(blocks, axes=(2, 3), norm="ortho")
        ac = coeffs[..., :, :].reshape(-1, 64)[:, 1:]  # drop DC
        ac_flat = ac.ravel()
        # Benford divergence of first significant digit of quantised AC magnitudes
        q = np.abs(np.round(ac_flat))
        q = q[q >= 1.0]
        if q.size:
            fsd = (q / np.power(10.0, np.floor(np.log10(q)))).astype(int)
            fsd = np.clip(fsd, 1, 9)
            hist = np.bincount(fsd, minlength=10)[1:10].astype(float)
            hist /= hist.sum()
            chi2 = float(np.sum((hist - _BENFORD) ** 2 / (_BENFORD + 1e-9)))
        else:
            chi2 = float("nan")
        _safe(d, ["dct_benford_chi2", "dct_ac_std", "dct_ac_kurt", "dct_zero_frac"],
              [chi2, float(ac_flat.std()),
               float(_stats.kurtosis(ac_flat, bias=False)) if ac_flat.std() > 1e-9 else 0.0,
               float((np.abs(ac_flat) < 0.5).mean())])
    # blockiness: gradient energy on the 8-grid vs off-grid
    dh = np.abs(np.diff(gray, axis=0))
    dv = np.abs(np.diff(gray, axis=1))
    grid_h = dh[7::8].mean() if dh.shape[0] > 8 else float("nan")
    grid_v = dv[:, 7::8].mean() if dv.shape[1] > 8 else float("nan")
    off_h = dh.mean()
    off_v = dv.mean()
    blockiness = float(np.nanmean([grid_h, grid_v]))
    off = float(np.nanmean([off_h, off_v]))
    _safe(d, ["jpeg_blockiness", "jpeg_grid_ratio"],
          [blockiness, blockiness / (off + 1e-9)])


def _radial_profile(power: np.ndarray, nbins: int) -> np.ndarray:
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = r.max()
    idx = np.clip((r / (rmax + 1e-9) * nbins).astype(int), 0, nbins - 1)
    out = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(out, idx.ravel(), power.ravel())
    np.add.at(cnt, idx.ravel(), 1.0)
    return out / (cnt + 1e-9)


def _f_spectral(d: dict, gray: np.ndarray) -> None:
    g = gray - gray.mean()
    F = _sfft.fftshift(_sfft.fft2(g))
    power = np.log1p(np.abs(F) ** 2)
    prof = _radial_profile(power, N_RADIAL)
    for i in range(N_RADIAL):
        d[f"spec_radial_{i}"] = float(prof[i])
    # log-log slope of the profile (spectral decay)
    freqs = np.arange(1, N_RADIAL)
    lp = prof[1:]
    if np.all(lp > 0):
        slope = float(np.polyfit(np.log(freqs), np.log(lp), 1)[0])
    else:
        slope = float("nan")
    half = N_RADIAL // 2
    hf_lf = float(prof[half:].mean() / (prof[:half].mean() + 1e-9))
    # peakiness: profile minus its smoothed self -> moiré / up-sampling combs
    sm = _ndi.uniform_filter1d(prof, size=3)
    excess = prof - sm
    std = excess.std() + 1e-9
    peak_count = float((excess > 2.0 * std).sum())
    peak_max = float(excess.max() / std)
    _safe(d, ["spec_slope", "spec_hf_lf_ratio", "spec_peak_count", "spec_peak_max"],
          [slope, hf_lf, peak_count, peak_max])


def _f_ela(d: dict, crop_img: Image.Image) -> None:
    # ELA on the analysis crop only (not the full-res image) -> much cheaper and
    # keeps the error map aligned with the same region the other features use.
    buf = _io.BytesIO()
    crop_img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    a = np.asarray(crop_img, dtype=np.float64)
    b = np.asarray(re, dtype=np.float64)
    ela = np.abs(a - b)
    _safe(d, ["ela_mean", "ela_std", "ela_max", "ela_p99", "ela_frac_high"],
          [ela.mean(), ela.std(), ela.max(),
           float(np.percentile(ela, 99)), float((ela > 20).mean())])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_features(path: str | Path, crop: int = CROP) -> dict[str, float]:
    """Return the forensic feature dict for one image (never raises)."""
    path = str(path)
    d: dict[str, float] = {k: float("nan") for k in FEATURE_NAMES}
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return d
    try:
        _f_meta(d, img, path)
    except Exception:
        pass
    try:
        # Crop in PIL space *before* converting to float64 so we never materialise
        # a full-resolution float array (a 4000x3000 image would be ~288 MB).
        w, h = img.size
        s = min(crop, w, h)
        left, top = (w - s) // 2, (h - s) // 2
        crop_img = img.crop((left, top, left + s, top + s))
        rgb = np.asarray(crop_img, dtype=np.float64)
        gray = rgb @ np.array([0.299, 0.587, 0.114])
    except Exception:
        return d
    for fn in (
        lambda: _f_color(d, rgb, gray),
        lambda: _f_residual(d, gray),
        lambda: _f_dct(d, gray),
        lambda: _f_spectral(d, gray),
        lambda: _f_ela(d, crop_img),
    ):
        try:
            fn()
        except Exception:
            pass
    return d


def feature_vector(path: str | Path, crop: int = CROP) -> np.ndarray:
    """Return features as a float64 vector in ``FEATURE_NAMES`` order."""
    d = extract_features(path, crop=crop)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float64)

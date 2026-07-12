"""OCR-derived *semantic consistency* features for ID-document fraud detection.

Motivation (agents_docs/04): the fraud signal we can SEE is localized field
replacement (inpaint / crop-replace) that leaves the document internally
inconsistent — a birth date whose Latin and Arabic-Indic renderings disagree, a
chronology that no longer makes sense, or text whose rendering differs from the
template. Those are *semantic* violations, not fragile pixel artifacts, so they
should survive the digital->captured shift and transfer to UNSEEN document types
(the private set) far better than a backbone that memorizes per-template cues.

This module turns a PaddleOCR result (list of {text, score, box}) into a fixed
feature vector. It is OCR-engine-agnostic: feed it any list of
``(text, score, (x, y, w, h))`` tuples.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

import numpy as np

# ---------------------------------------------------------------------------
# Digit / date normalisation
# ---------------------------------------------------------------------------
# Arabic-Indic (٠-٩) and Extended/Persian (۰-۹) -> ASCII
_AR = "٠١٢٣٤٥٦٧٨٩"
_FA = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {ord(a): str(i) for i, a in enumerate(_AR)}
_DIGIT_MAP.update({ord(f): str(i) for i, f in enumerate(_FA)})


def to_ascii_digits(s: str) -> str:
    return s.translate(_DIGIT_MAP)


def has_arabic_indic(s: str) -> bool:
    return any(c in _AR or c in _FA for c in s)


_DATE_RE = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})")
_DATE_YMD_RE = re.compile(r"(\d{4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def parse_dates(text: str):
    """Return list of (y, m, d) parsed from a (digit-normalised) string."""
    t = to_ascii_digits(text)
    out = []
    for d, m, y in _DATE_RE.findall(t):
        out.append((int(y), int(m), int(d)))
    for y, m, d in _DATE_YMD_RE.findall(t):
        out.append((int(y), int(m), int(d)))
    return out


def years_in(text: str):
    return [int(y) for y in _YEAR_RE.findall(to_ascii_digits(text))]


def _valid_ymd(y, m, d) -> bool:
    try:
        date(y, m, d)
        return 1900 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Name / script anomalies
# ---------------------------------------------------------------------------
# "Normal" Latin-name accents. Anything Latin *outside* this set inside an
# otherwise-alphabetic token is a synth text-generator tell (e.g. "Röberts
# Grïffïths Smîth" seen on Mauritius fraud).
_OK_ACCENTS = set("àáâãäåçèéêëìíîïñòóôõöùúûüýÿœæ" "ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝŸŒÆ")
_WEIRD = set("ïîìöôòüûùäâàëêèÿ")  # diacritics rarely stacked on English names


def _is_latin_alpha_token(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return False
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    return latin >= 0.6 * len(letters)


def name_anomaly_stats(tokens):
    weird = 0
    total_letters = 0
    n_name_tok = 0
    for t in tokens:
        if not _is_latin_alpha_token(t):
            continue
        n_name_tok += 1
        for c in t:
            if c.isalpha():
                total_letters += 1
                o = ord(c)
                if o > 127 and c in _WEIRD:
                    weird += 1
    return {
        "name_weird_accent_ct": float(weird),
        "name_weird_accent_frac": weird / total_letters if total_letters else 0.0,
        "n_latin_name_tokens": float(n_name_tok),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # volume / OCR quality
    "n_tokens", "total_text_len", "mean_conf", "min_conf",
    "frac_conf_lt30", "frac_conf_lt50", "std_conf",
    # dates
    "n_latin_dates", "n_arabic_date_tokens", "n_valid_dates",
    "n_distinct_years", "year_span",
    # chronology / logic violations
    "chrono_violation", "implausible_age", "future_or_ancient",
    "n_date_pairs_bad_order",
    # cross-script agreement (the headline signal)
    "has_dual_script", "crossscript_year_mismatch_frac", "crossscript_any_mismatch",
    # name / rendering anomalies
    "name_weird_accent_ct", "name_weird_accent_frac", "n_latin_name_tokens",
    # geometry (inpainted field often has odd box height/baseline)
    "box_h_cv", "box_h_max_over_med",
]


def extract_features(ocr_items, img_wh=None) -> dict:
    """ocr_items: list of dict(text=str, score=float, box=(x,y,w,h))."""
    texts = [it["text"] for it in ocr_items if it.get("text")]
    confs = np.array([float(it.get("score", 0.0)) for it in ocr_items], dtype=np.float64)
    heights = np.array(
        [it["box"][3] for it in ocr_items if it.get("box") is not None], dtype=np.float64
    )
    f = {k: 0.0 for k in FEATURE_NAMES}

    f["n_tokens"] = float(len(texts))
    f["total_text_len"] = float(sum(len(t) for t in texts))
    if confs.size:
        f["mean_conf"] = float(confs.mean())
        f["min_conf"] = float(confs.min())
        f["std_conf"] = float(confs.std())
        f["frac_conf_lt30"] = float((confs < 0.30).mean())
        f["frac_conf_lt50"] = float((confs < 0.50).mean())

    # collect dates, split by the SCRIPT the token was recognised under.
    # If a token carries an explicit `script` tag (latin/arabic pass) we trust
    # it; otherwise we fall back to detecting Arabic-Indic digits in the glyphs.
    latin_dates, latin_years, arabic_years = [], [], []
    n_arabic_tok = 0
    for it in ocr_items:
        t = it.get("text") or ""
        if not t:
            continue
        ds = parse_dates(t)
        script = it.get("script")
        is_ar = (script == "arabic") if script else has_arabic_indic(t)
        if is_ar:
            yr = [y for (y, _, _) in ds] + years_in(t)
            if yr:
                n_arabic_tok += 1
            arabic_years += yr
        elif ds:
            latin_dates += ds
            latin_years += [y for (y, _, _) in ds]

    valid = [(y, m, d) for (y, m, d) in latin_dates if _valid_ymd(y, m, d)]
    f["n_latin_dates"] = float(len(latin_dates))
    f["n_arabic_date_tokens"] = float(n_arabic_tok)
    f["n_valid_dates"] = float(len(valid))
    allyears = sorted({y for (y, _, _) in valid})
    f["n_distinct_years"] = float(len(allyears))
    f["year_span"] = float(allyears[-1] - allyears[0]) if len(allyears) >= 2 else 0.0

    # chronology: smallest year ~ birth, must precede issue/expiry; ages plausible
    if len(valid) >= 2:
        ys = sorted(y for (y, _, _) in valid)
        birth = ys[0]
        latest = ys[-1]
        f["implausible_age"] = 1.0 if (latest - birth) > 90 or (latest - birth) < 15 else 0.0
        f["future_or_ancient"] = 1.0 if (latest > 2032 or birth < 1925) else 0.0
        # bad-order pairs among dates that look like (issue, expiry) — expiry>issue
        bad = 0
        pairs = 0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pairs += 1
        # coarse: if any two dates in same year-decade are reversed vs text order
        seq = [date(y, m, d) for (y, m, d) in valid]
        for a, b in zip(seq, seq[1:]):
            if a > b and (a - b).days > 366 * 20:
                bad += 1
        f["n_date_pairs_bad_order"] = float(bad)
        f["chrono_violation"] = 1.0 if (f["implausible_age"] or f["future_or_ancient"] or bad) else 0.0

    # cross-script year agreement
    if arabic_years and latin_years:
        f["has_dual_script"] = 1.0
        aset = set(arabic_years)
        miss = sum(1 for y in latin_years if y not in aset)
        f["crossscript_year_mismatch_frac"] = miss / len(latin_years)
        f["crossscript_any_mismatch"] = 1.0 if miss > 0 else 0.0

    f.update(name_anomaly_stats(texts))

    if heights.size >= 4:
        med = np.median(heights)
        f["box_h_cv"] = float(heights.std() / (heights.mean() + 1e-6))
        f["box_h_max_over_med"] = float(heights.max() / (med + 1e-6))

    return f

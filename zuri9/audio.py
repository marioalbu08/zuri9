"""Ce se aude: volumul în timp, vârfurile (râs, aplauze, voce ridicată), pauzele."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HOP = 0.01  # 10 ms


def envelope_db(a: np.ndarray, sr: int, hop: float = HOP) -> np.ndarray:
    """Volumul (RMS, dB) pe ferestre de `hop` secunde."""
    n = int(sr * hop)
    m = len(a) // n
    x = a[: m * n].reshape(m, n).astype(np.float64)
    rms = np.sqrt((x * x).mean(axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-9)).astype(np.float32)


def loud_moments(env: np.ndarray, hop: float = HOP, min_rise_db: float = 6.0,
                 min_len: float = 0.6) -> list[dict]:
    """Momentele în care sunetul urcă mult peste nivelul din jur.

    La un podcast sunt de obicei râsete, aplauze sau o voce ridicată: semne că
    acolo s-a întâmplat ceva. Nivelul din jur e mediana pe 60 s.
    """
    if len(env) == 0:
        return []
    short = uniform_filter1d(env, size=max(1, int(1.0 / hop)))
    step = max(1, int(1.0 / hop))
    coarse = short[::step]
    base = median_filter(coarse, size=61, mode="nearest")
    base = np.repeat(base, step)[: len(short)]
    above = short - base > min_rise_db
    out, i, n = [], 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if (j - i) * hop >= min_len:
                seg = short[i:j] - base[i:j]
                out.append({"s": round(i * hop, 2), "e": round(j * hop, 2),
                            "rise_db": round(float(seg.max()), 1)})
            i = j
        else:
            i += 1
    return out


def silence_mask(env: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    """True unde e liniște față de vorbirea din fișier."""
    if floor_db is None:
        speech = np.percentile(env, 90)
        floor_db = speech - 30
    return env < floor_db

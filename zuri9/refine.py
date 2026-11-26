"""Din momentul ales de Claude, lista exactă de bucăți tăiate din sursă (EDL).

Taie doar între cuvinte, scoate pauzele lungi (jump cut) și mută fiecare
capăt până unde chiar se termină sunetul, ca să nu se taie silabe.
"""
from __future__ import annotations

import numpy as np

from .audio import HOP


def _speech_floor(env: np.ndarray) -> float:
    return float(np.percentile(env, 90) - 25)


def _extend(env: np.ndarray, t: float, direction: int, floor: float, limit: float) -> float:
    """Mută `t` spre `direction` cât timp volumul e încă peste prag (max `limit` s)."""
    i = int(t / HOP)
    steps = int(limit / HOP)
    k = 0
    while 0 <= i + direction < len(env) and k < steps and env[i] > floor:
        i += direction
        k += 1
    return i * HOP


def build_edl(clip: dict, lines: list[dict], words: list[dict], env: np.ndarray,
              max_gap: float = 0.45, pad: float = 0.06, max_s: float | None = None,
              fps: float = 30.0) -> dict:
    if "pieces" in clip:
        pieces = clip["pieces"]
    else:  # forma veche: un interval, cu rânduri scoase din interior
        cut = set(clip.get("cut_lines", []))
        keep = [i for i in range(clip["start_line"], clip["end_line"] + 1) if i not in cut]
        pieces = [[i, i] for i in keep]
    idx = []
    for a, b in pieces:
        for li in range(a, b + 1):
            idx.extend(range(lines[li]["w0"], lines[li]["w1"] + 1))
    if not idx:
        return {"segments": [], "words": [], "duration": 0.0}

    # bucăți de cuvinte consecutive, rupte la pauze lungi sau la rândurile scoase
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1 or words[b]["s"] - words[a]["e"] > max_gap:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)

    floor = _speech_floor(env)
    segs = []
    for r in runs:
        s, e = words[r[0]]["s"], words[r[-1]]["e"]
        prev_e = words[r[0] - 1]["e"] if r[0] > 0 else 0.0
        next_s = words[r[-1] + 1]["s"] if r[-1] + 1 < len(words) else e + 1.0
        # whisper pune adesea finalul cuvântului prea devreme: ascultăm sunetul
        s2 = max(_extend(env, s, -1, floor, 0.25) - pad, (prev_e + s) / 2, 0.0)
        e2 = min(_extend(env, e, +1, floor, 0.35) + pad, (e + next_s) / 2)
        segs.append([round(s2, 3), round(e2, 3), r])

    # lipim bucățile despărțite de goluri foarte mici
    merged = [segs[0]]
    for s, e, r in segs[1:]:
        if 0 <= s - merged[-1][1] < 0.12:   # doar bucăți vecine în sursă
            merged[-1][1] = e
            merged[-1][2] = merged[-1][2] + r
        else:
            merged.append([s, e, r])

    out_words, segments, t_out = [], [], 0.0
    for s, e, r in merged:
        if max_s and t_out + (e - s) > max_s + 3:
            break
        # durata fiecărei bucăți = un număr întreg de cadre, ca sunetul să nu alunece
        e = round(s + max(1, round((e - s) * fps)) / fps, 6)
        for k in r:
            w = words[k]
            out_words.append({"w": w["w"], "s": round(t_out + max(w["s"], s) - s, 3),
                              "e": round(t_out + min(w["e"], e) - s, 3)})
        segments.append([s, e])
        t_out += e - s
    return {"segments": segments, "words": out_words, "duration": round(t_out, 3)}

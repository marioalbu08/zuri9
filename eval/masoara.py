"""Cât din ce au ales editorii găsește programul.

Un Short oficial e făcut din bucăți din episod (găsite după sunet de
`gaseste_shorts.py`). Pentru fiecare Short calculăm ce parte din secundele lui
din episod apar și într-unul din primele K clipuri propuse (acoperirea). Short-ul e
„găsit” dacă acoperirea trece de 30%.

Comparăm cu alegeri fără AI, de aceeași lungime: K ferestre la întâmplare și cele K
ferestre cu cele mai multe vârfuri de volum.

    python eval/masoara.py ADEVAR.json MOMENTE.json [--work FOLDER_ANALIZA] [--k 3 5 10]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zuri9 import audio  # noqa: E402

FOUND = 0.30


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(iv) -> float:
    return sum(b - a for a, b in iv)


def inter(x, y) -> float:
    tot, i, j = 0.0, 0, 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            tot += b - a
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def gt_shorts(truth: dict) -> list[dict]:
    out, seen = [], []
    for sh in truth["shorts"]:
        iv = union([[s["ep_s"], s["ep_e"]] for s in sh.get("segments", [])])
        if length(iv) < 5:
            continue
        # același clip urcat de două ori (alt titlu) nu se numără de două ori
        if any(inter(iv, o) > 0.6 * min(length(iv), length(o)) for o in seen):
            continue
        seen.append(iv)
        out.append({"short": sh["short"], "iv": iv})
    return out


def coverage(gts, cands, k):
    """Pentru fiecare Short: cea mai mare acoperire dată de unul din primele k clipuri."""
    res = []
    for g in gts:
        best = max((inter(g["iv"], c) / length(g["iv"]) for c in cands[:k]), default=0.0)
        res.append(float(best))
    return res


def random_windows(dur, k, lens, rng):
    cands = []
    while len(cands) < k:
        L = lens[len(cands) % len(lens)]
        s = rng.uniform(0, dur - L)
        if all(inter([[s, s + L]], c) == 0 for c in cands):
            cands.append([[s, s + L]])
    return cands


def loud_windows(env, dur, k, L):
    loud = audio.loud_moments(env)
    t = np.array([m["s"] for m in loud])
    w = np.array([m["rise_db"] for m in loud])
    starts = np.arange(0, dur - L, 5.0)
    score = np.array([w[(t >= s) & (t < s + L)].sum() for s in starts])
    cands = []
    for i in np.argsort(-score, kind="stable"):
        s = starts[i]
        if all(inter([[s, s + L]], c) == 0 for c in cands):
            cands.append([[s, s + L]])
        if len(cands) >= k:
            break
    return cands


def evaluate(truth: dict, picks: dict, env=None, ks=(3, 5, 10), trials: int = 500) -> dict:
    gts = gt_shorts(truth)
    cands = [union(c.get("spans") or [[c["s"], c["e"]]]) for c in picks["clips"]]
    lens = [max(10.0, length(c)) for c in cands] or [45.0]
    dur = picks["lines"][-1]["e"]
    res = {"episode": truth["episode"], "shorts": len(gts),
           "gt_seconds": [round(length(g["iv"]), 1) for g in gts], "by_k": {}}
    rng = random.Random(7)
    for k in ks:
        cov = coverage(gts, cands, k)
        row = {"found": int(sum(c >= FOUND for c in cov)), "coverage": [round(c, 2) for c in cov]}
        rf = [sum(c >= FOUND for c in coverage(gts, random_windows(dur, k, lens, rng), k))
              for _ in range(trials)]
        row["random_found"] = round(float(np.mean(rf)), 2)
        if env is not None:
            L = float(np.median(lens))
            row["loud_found"] = int(sum(c >= FOUND for c in coverage(gts, loud_windows(env, dur, k, L), k)))
        res["by_k"][str(k)] = row
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("picks")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--work", default=None, help="folderul .analiza al episodului (pentru volum)")
    a = ap.parse_args()
    truth = json.load(open(a.truth, encoding="utf-8"))
    picks = json.load(open(a.picks, encoding="utf-8"))
    env = np.load(os.path.join(a.work, "env.npy")) if a.work else None
    print(json.dumps(evaluate(truth, picks, env, a.k), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

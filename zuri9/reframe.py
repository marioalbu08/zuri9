"""Încadrarea 9:16 dintr-un video orizontal: pe cine ținem în cadru și cum mișcăm camera.

1. Fețele (YuNet), de 10 ori pe secundă, legate în urme pe fiecare plan.
2. Cine vorbește: gura care se mișcă atunci când se aude vorbă (sunet + imagine).
3. Camera virtuală, ca un operator: stă pe loc cât omul nu iese din zona moartă,
   altfel urmărește lin; la schimbarea vorbitorului departe, taie, nu panoramează.
   Pe tăieturile de plan din sursă sare imediat.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .media import iter_frames, probe

MODEL = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")
AN_FPS = 10.0
AN_W = 640


def _detector(w: int, h: int):
    d = cv2.FaceDetectorYN.create(MODEL, "", (w, h), 0.6, 0.3, 50)
    d.setInputSize((w, h))
    return d


def _shot_cuts(grays: list[np.ndarray], thr: float = 0.35) -> list[int]:
    """Indicii cadrelor unde începe un plan nou (diferența de histogramă)."""
    cuts = [0]
    prev = None
    for i, g in enumerate(grays):
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel()
        h /= h.sum() + 1e-9
        if prev is not None:
            d = 0.5 * np.abs(h - prev).sum()
            if d > thr and i - cuts[-1] >= 3:
                cuts.append(i)
        prev = h
    return cuts


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def _mouth_motion(prev: np.ndarray, cur: np.ndarray, box) -> float:
    x, y, w, h = [int(v) for v in box]
    y0, y1 = y + int(0.55 * h), y + int(1.05 * h)
    x0, x1 = x + int(0.2 * w), x + int(0.8 * w)
    H, W = cur.shape
    x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0
    a = cv2.resize(prev[y0:y1, x0:x1], (32, 24)).astype(np.float32)
    b = cv2.resize(cur[y0:y1, x0:x1], (32, 24)).astype(np.float32)
    # scădem mișcarea întregii fețe (capul care se mișcă nu înseamnă vorbă)
    d = np.abs(b - a)
    return float(d.mean() - 0.5 * abs(b.mean() - a.mean()))


def analyze_segment(path: str, s: float, e: float, speaking) -> dict:
    """Fețe, urme și planuri pentru o bucată din sursă. `speaking(t)` -> bool."""
    times, grays, dets = [], [], []
    det = None
    for t, fr in iter_frames(path, AN_FPS, AN_W, start=s, end=e):
        if det is None:
            det = _detector(fr.shape[1], fr.shape[0])
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        _, faces = det.detect(fr)
        boxes = [] if faces is None else [tuple(f[:4]) + (float(f[14]),) for f in faces]
        times.append(t)
        grays.append(g)
        dets.append(boxes)
    if not times:
        return {"times": [], "cuts": [0], "tracks": [], "W": AN_W, "H": 360}
    H, W = grays[0].shape
    cuts = _shot_cuts(grays)

    # urme: legăm fețele de la un cadru la altul, doar în același plan
    tracks: list[dict] = []
    bounds = cuts + [len(times)]
    for si in range(len(cuts)):
        a, b = bounds[si], bounds[si + 1]
        live: list[dict] = []
        for i in range(a, b):
            used = set()
            for box in sorted(dets[i], key=lambda x: -x[2] * x[3]):
                best, bi = 0.2, None
                for k, tr in enumerate(live):
                    if k in used or tr["last"] < i - 5:
                        continue
                    o = _iou(tr["boxes"][-1], box[:4])
                    if o > best:
                        best, bi = o, k
                if bi is None:
                    tr = {"shot": si, "frames": [], "boxes": [], "motion": [], "last": i}
                    live.append(tr)
                    tracks.append(tr)
                    bi = len(live) - 1
                tr = live[bi]
                used.add(bi)
                m = _mouth_motion(grays[i - 1], grays[i], box[:4]) if i > a else 0.0
                tr["frames"].append(i)
                tr["boxes"].append(box[:4])
                tr["motion"].append(m)
                tr["last"] = i
    for tr in tracks:
        del tr["last"]
        sp = np.array([speaking(times[i]) for i in tr["frames"]], bool)
        mo = np.array(tr["motion"])
        area = np.mean([b[2] * b[3] for b in tr["boxes"]]) / (W * H)
        tr["area"] = float(area)
        tr["talk"] = float(mo[sp].mean() - mo[~sp].mean()) if sp.any() and (~sp).any() \
            else float(mo.mean()) if len(mo) else 0.0
    return {"times": times, "cuts": cuts, "tracks": tracks, "W": W, "H": H}


def _choose_targets(an: dict, hold: float = 1.5) -> np.ndarray:
    """Centrul x (0..1) pe care îl vrem în cadru, la fiecare cadru analizat; NaN = nimeni."""
    n = len(an["times"])
    W = an["W"]
    target = np.full(n, np.nan)
    bounds = an["cuts"] + [n]
    win = int(2.0 * AN_FPS)
    for si in range(len(an["cuts"])):
        a, b = bounds[si], bounds[si + 1]
        trs = [t for t in an["tracks"] if t["shot"] == si and len(t["frames"]) >= 3]
        if not trs:
            continue
        # cine vorbește în fiecare moment: mișcarea gurii pe o fereastră de 2 s,
        # cu un mic avantaj pentru fața mare (gros-planul e de obicei vorbitorul)
        score = np.full((len(trs), b - a), -np.inf)
        for k, tr in enumerate(trs):
            mo = np.zeros(b - a)
            have = np.zeros(b - a)
            for f, m in zip(tr["frames"], tr["motion"]):
                mo[f - a] = m
                have[f - a] = 1
            sm = np.convolve(mo, np.ones(win), "same") / np.maximum(np.convolve(have, np.ones(win), "same"), 1)
            present = np.convolve(have, np.ones(5), "same") > 0
            score[k] = np.where(present, sm + 2.0 * np.sqrt(tr["area"]), -np.inf)
        cur, since = None, 0
        min_hold = int(hold * AN_FPS)
        for j in range(b - a):
            best = int(np.argmax(score[:, j]))
            if not np.isfinite(score[best, j]):
                best = cur if cur is not None and np.isfinite(score[cur, j]) else None
            if cur is None or (best is not None and best != cur and since >= min_hold
                               and score[best, j] > score[cur, j] + 0.3):
                if best != cur:
                    since = 0
                cur = best
            since += 1
            if cur is None:
                continue
            tr = trs[cur]
            # cea mai apropiată detecție a urmei
            fi = np.searchsorted(tr["frames"], a + j)
            fi = min(max(fi, 0), len(tr["frames"]) - 1)
            x, y, w, h = tr["boxes"][fi]
            target[a + j] = (x + w / 2) / W
    return target


def camera_path(an: dict, src_w: int, src_h: int, out_aspect: float = 9 / 16,
                dead: float = 0.06) -> tuple[np.ndarray, np.ndarray, int]:
    """Poziția stângă a ferestrei de decupare (pixeli sursă) la fiecare cadru analizat."""
    n = len(an["times"])
    cw = int(round(src_h * out_aspect / 2)) * 2
    cw = min(cw, src_w)
    half = cw / 2 / src_w  # jumătate de fereastră, ca fracție din lățime
    tgt = _choose_targets(an) if n else np.array([])
    xs = np.full(n, 0.5)
    bounds = an["cuts"] + [n]
    for si in range(len(an["cuts"])):
        a, b = bounds[si], bounds[si + 1]
        seg = tgt[a:b].copy()
        ok = np.isfinite(seg)
        if not ok.any():
            xs[a:b] = xs[a - 1] if a > 0 else 0.5
            continue
        # golurile (fața nedetectată) le umplem cu ultima poziție știută
        idx = np.where(ok, np.arange(len(seg)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.argmax(ok)
        seg = np.where(ok, seg, seg[idx])
        seg[:first] = seg[first]
        # operatorul: o poziție ținută cât timp ținta stă în zona moartă;
        # salt (tăietură) când ținta se mută departe, altfel urmărire lină
        cam = np.empty_like(seg)
        pos = float(np.median(seg[: max(1, min(len(seg), int(AN_FPS)))]))
        for j, x in enumerate(seg):
            if abs(x - pos) > half:          # alt om, în altă parte a cadrului
                pos = x
            elif abs(x - pos) > dead:
                pos += (x - pos) * 0.12      # urmărire lină
            cam[j] = pos
        if b - a > 5:
            cam = _smooth_keep_jumps(cam, half)
        xs[a:b] = cam
    left = np.clip(xs * src_w - cw / 2, 0, src_w - cw)
    return np.array(an["times"]), left, cw


def _smooth_keep_jumps(x: np.ndarray, half: float) -> np.ndarray:
    jumps = np.where(np.abs(np.diff(x)) > half * 0.8)[0] + 1
    out = x.copy()
    for a, b in zip(np.r_[0, jumps], np.r_[jumps, len(x)]):
        if b - a > 3:
            out[a:b] = gaussian_filter1d(x[a:b], sigma=3, mode="nearest")
    return out


def plan_crops(path: str, segments: list[list[float]], speaking) -> list[dict]:
    """Pentru fiecare bucată din EDL: timpii analizați și poziția ferestrei."""
    info = probe(path)
    plans = []
    for s, e in segments:
        an = analyze_segment(path, s, e, speaking)
        times, left, cw = camera_path(an, info.width, info.height)
        plans.append({"s": s, "e": e, "times": times.tolist(), "left": left.tolist(),
                      "cw": cw, "cuts": [an["times"][c] for c in an["cuts"]] if an["times"] else [],
                      "tracks": len(an["tracks"])})
    return plans

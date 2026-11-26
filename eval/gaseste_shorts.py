"""Adevărul pentru măsurători: unde a tăiat editorul fiecare Short oficial din episod.

Fiecare Short e ascultat pe ferestre de 4 s; fiecare fereastră e căutată în tot
episodul (corelație pe atacurile din 12 benzi de frecvență, prin FFT). Ferestrele
care cad cu același decalaj formează o bucată din sursă. Muzica pusă peste de
editor nu strică potrivirea, pentru că vocea domină.

    python eval/gaseste_shorts.py EPISOD.mp4 SHORT1.webm SHORT2.webm ... > adevar.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zuri9.media import probe, read_audio  # noqa: E402

SR = 16000
HOP = 320          # 20 ms
NFFT = 1024
FR = SR / HOP      # 50 cadre pe secundă
EDGES = np.geomspace(120, 6000, 13)


def band_flux(a: np.ndarray) -> np.ndarray:
    """Atacurile pe benzi: creșterea energiei log, normalizată. [cadre, 12]"""
    win = np.hanning(NFFT).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    idx = [(np.searchsorted(freqs, lo), np.searchsorted(freqs, hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    n = max(0, (len(a) - NFFT) // HOP)
    out = np.zeros((n, len(idx)), np.float32)
    step = 20000
    for c0 in range(0, n, step):
        c1 = min(n, c0 + step)
        fr = np.lib.stride_tricks.as_strided(
            a[c0 * HOP:], shape=(c1 - c0, NFFT), strides=(a.strides[0] * HOP, a.strides[0]))
        sp = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
        for b, (i0, i1) in enumerate(idx):
            out[c0:c1, b] = np.log(sp[:, i0:i1].sum(axis=1) + 1e-8)
    d = np.diff(out, axis=0, prepend=out[:1])
    d = np.maximum(d, 0)
    d -= d.mean(axis=0)
    d /= d.std(axis=0) + 1e-6
    return d


def locate(ep_feat: np.ndarray, sh_feat: np.ndarray, win_s: float = 4.0, hop_s: float = 1.0):
    """Pentru fiecare fereastră din Short: (timp în Short, timp în episod, încredere)."""
    L = len(ep_feat)
    w = int(win_s * FR)
    nfft = 1 << int(np.ceil(np.log2(L + w)))
    EP = np.fft.rfft(ep_feat, n=nfft, axis=0)
    out = []
    for st in range(0, max(1, len(sh_feat) - w), int(hop_s * FR)):
        seg = sh_feat[st:st + w]
        if len(seg) < w:
            break
        S = np.fft.rfft(seg[::-1], n=nfft, axis=0)
        cc = np.fft.irfft(EP * S, n=nfft, axis=0).sum(axis=1)[w - 1: w - 1 + L - w]
        cc /= w * seg.shape[1]
        k = int(np.argmax(cc))
        peak = float(cc[k])
        # a doua cea mai bună potrivire, la peste 2 s distanță
        mask = np.ones(len(cc), bool)
        mask[max(0, k - int(2 * FR)): k + int(2 * FR)] = False
        second = float(cc[mask].max()) if mask.any() else 0.0
        out.append((st / FR, k / FR, peak, peak / max(second, 1e-6)))
    return out


def segments(matches, min_ratio: float = 1.6, tol: float = 0.08):
    """Ferestrele sigure, grupate pe decalaj constant -> bucăți [ep_s, ep_e, sh_s]."""
    good = [(ts, te) for ts, te, p, r in matches if r >= min_ratio]
    segs = []
    for ts, te in good:
        off = te - ts
        if segs and abs(segs[-1]["off"] - off) < tol and ts - segs[-1]["sh_e"] <= 2.5:
            segs[-1]["sh_e"] = ts + 4.0
            segs[-1]["n"] += 1
        else:
            segs.append({"off": off, "sh_s": ts, "sh_e": ts + 4.0, "n": 1})
    segs = [s for s in segs if s["n"] >= 2]
    return [{"ep_s": round(s["sh_s"] + s["off"], 2), "ep_e": round(s["sh_e"] + s["off"], 2),
             "sh_s": round(s["sh_s"], 2), "sh_e": round(s["sh_e"], 2)} for s in segs]


def main() -> None:
    ep, shorts = sys.argv[1], sys.argv[2:]
    print(f"[episod] {os.path.basename(ep)}", file=sys.stderr)
    ep_feat = band_flux(read_audio(ep, SR))
    res = {"episode": os.path.basename(ep), "shorts": []}
    for sh in shorts:
        info = probe(sh)
        m = locate(ep_feat, band_flux(read_audio(sh, SR)))
        segs = segments(m)
        sure = sum(1 for x in m if x[3] >= 1.6)
        entry = {"short": os.path.basename(sh), "duration": round(info.duration, 2),
                 "windows": len(m), "windows_sure": sure, "segments": segs}
        if segs:
            entry["ep_s"] = min(s["ep_s"] for s in segs)
            entry["ep_e"] = max(s["ep_e"] for s in segs)
        print(f"  {entry['short']}: {sure}/{len(m)} ferestre sigure, "
              f"{len(segs)} bucăți, {entry.get('ep_s')}–{entry.get('ep_e')}", file=sys.stderr)
        res["shorts"].append(entry)
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

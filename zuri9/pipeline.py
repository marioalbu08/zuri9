"""Tot drumul, de la materialul brut la clipurile verticale."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from . import audio, brain, captions, reframe, refine, render
from .media import load_json, probe, read_audio, save_json
from .transcribe import transcribe


def _speaking_fn(words: list[dict]):
    hop = audio.HOP
    if not words:
        return lambda t: False
    n = int(words[-1]["e"] / hop) + 2
    mask = np.zeros(n, bool)
    for w in words:
        mask[int(w["s"] / hop): int(w["e"] / hop) + 1] = True
    return lambda t: bool(mask[min(max(int(t / hop), 0), n - 1)])


def analyze(src: str, work: str, language: str | None = None, whisper: str = "medium") -> dict:
    """Sunetul și transcrierea, cu cache în `work`."""
    os.makedirs(work, exist_ok=True)
    info = probe(src)
    tr_path = os.path.join(work, f"transcript_{whisper}.json")
    env_path = os.path.join(work, "env.npy")
    if os.path.exists(tr_path) and os.path.exists(env_path):
        return {"info": info, "tr": load_json(tr_path), "env": np.load(env_path)}
    t = time.time()
    a = read_audio(src, 16000)
    env = audio.envelope_db(a, 16000)
    np.save(env_path, env)
    print(f"[sunet] {info.duration / 60:.1f} min citite în {time.time() - t:.0f} s")
    tr = transcribe(a, language=language, size=whisper)
    save_json(tr_path, tr)
    return {"info": info, "tr": tr, "env": env}


def run(src: str, out_dir: str, n: int = 5, candidates: int = 10, min_s: int = 20,
        max_s: int = 60, request: str = "", language: str | None = None,
        whisper: str = "medium", fps: float = 30.0, render_clips: bool = True) -> dict:
    base = os.path.splitext(os.path.basename(src))[0]
    work = os.path.join(out_dir, ".analiza", base)
    an = analyze(src, work, language, whisper)
    words, env = an["tr"]["words"], an["env"]
    loud = audio.loud_moments(env)

    # cheia include promptul: dacă schimbăm instrucțiunile, alegerea se refăce
    key = hashlib.sha1(f"{brain.MODEL}|{candidates}|{min_s}|{max_s}|{request}|{brain.SYSTEM}|"
                       f"{json.dumps(brain.SCHEMA, sort_keys=True)}".encode()).hexdigest()[:10]
    pick_path = os.path.join(work, f"momente_{key}.json")
    if os.path.exists(pick_path):
        picked = load_json(pick_path)
    else:
        t = time.time()
        picked = brain.pick_moments(words, loud, n=candidates, min_s=min_s, max_s=max_s,
                                    request=request)
        picked["seconds"] = round(time.time() - t, 1)
        save_json(pick_path, picked)
        print(f"[claude] {len(picked['clips'])} momente în {picked['seconds']} s, "
              f"{picked['cost_usd']} $ ({picked['via']})")
    lines = picked["lines"]
    speaking = _speaking_fn(words)

    results = []
    for i, clip in enumerate(picked["clips"][:n], 1):
        edl = refine.build_edl(clip, lines, words, env, max_s=max_s, fps=fps)
        if not edl["segments"]:
            continue
        name = f"{base}_clip{i:02d}"
        rec = {"name": name, "title": clip["title"], "hook": clip["hook"], "why": clip["why"],
               "kind": clip["kind"], "score": clip["score"], "src_start": clip["s"],
               "src_end": clip["e"], "duration": edl["duration"],
               "segments": edl["segments"], "cuts_inside": len(edl["segments"]) - 1}
        if render_clips:
            t = time.time()
            plans = reframe.plan_crops(src, edl["segments"], speaking)
            ass = os.path.join(work, name + ".ass")
            captions.write_ass(edl["words"], ass)
            r = render.render(src, edl, plans, ass, os.path.join(out_dir, name + ".mp4"), fps=fps)
            rec["render_s"] = round(time.time() - t, 1)
            rec["file"] = os.path.basename(r["path"])
            save_json(os.path.join(work, name + ".plan.json"), {"edl": edl, "plans": plans})
            print(f"[clip {i}] {clip['title']}: {edl['duration']:.0f} s, "
                  f"{rec['cuts_inside']} tăieturi, randat în {rec['render_s']} s")
        results.append(rec)
    report = {"source": os.path.basename(src), "picks_file": pick_path, "duration_min": round(an["info"].duration / 60, 1),
              "model": picked.get("model"), "cost_usd": picked.get("cost_usd"),
              "clips": results, "candidates": [
                  {k: c[k] for k in ("title", "s", "e", "score", "kind")} for c in picked["clips"]]}
    save_json(os.path.join(out_dir, f"{base}_clipuri.json"), report)
    _write_md(report, os.path.join(out_dir, f"{base}_clipuri.md"))
    return report


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _write_md(rep: dict, path: str) -> None:
    out = [f"# Clipurile din {rep['source']}", "",
           f"Sursa: {rep['duration_min']} min. Momentele alese de {rep['model']} "
           f"(cost {rep['cost_usd']} $).", ""]
    for c in rep["clips"]:
        out += [f"## {c['title']}", "",
                f"- **Fișier:** `{c.get('file', '(nerandat)')}`, {c['duration']:.0f} s, "
                f"{c['cuts_inside']} tăieturi în interior",
                f"- **Din sursă:** {_fmt(c['src_start'])} – {_fmt(c['src_end'])}",
                f"- **Începe cu:** „{c['hook']}”",
                f"- **De ce:** {c['why']} ({c['kind']}, scor {c['score']}/10)", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

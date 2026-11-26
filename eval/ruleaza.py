"""Rulează alegerea momentelor pe un set de episoade și o compară cu Shorts-urile oficiale.

    python eval/ruleaza.py --set reglaj          # episoadele pe care lucrăm
    python eval/ruleaza.py --set final           # o singură dată, cu codul înghețat

Datele (episoadele și Shorts-urile) stau în afara repo-ului, în ZURI9_DATE
(implicit ~/zuri9-date): episoade/<id>.*, shorts/<id>.*. Le descarcă `descarca.sh`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from zuri9 import pipeline  # noqa: E402
from masoara import evaluate  # noqa: E402

DATA = os.environ.get("ZURI9_DATE", os.path.join(os.path.expanduser("~"), "zuri9-date"))


def find(kind: str, vid: str) -> str:
    c = [p for p in glob.glob(os.path.join(DATA, kind, vid + ".*"))
         if not p.endswith((".json", ".part", ".ytdl"))]
    if not c:
        raise FileNotFoundError(f"{kind}/{vid}")
    return sorted(c, key=lambda p: (not p.endswith(".m4a"), p))[0]


def truth_for(vid: str, shorts: list[str]) -> dict:
    path = os.path.join(DATA, f"adevar_{vid}.json")
    if not os.path.exists(path):
        cmd = [sys.executable, os.path.join(HERE, "gaseste_shorts.py"), find("episoade", vid)]
        cmd += [find("shorts", s) for s in shorts]
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out.stdout)
    return json.load(open(path, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="reglaj")
    ap.add_argument("--candidati", type=int, default=10)
    ap.add_argument("--eticheta", default="", help="nume pentru fișierul de rezultate")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "set.json"), encoding="utf-8"))[a.set]
    out_dir = os.path.join(DATA, "out")
    rows, cost = [], 0.0
    for vid, meta in cfg.items():
        truth = truth_for(vid, meta["shorts"])
        rep = pipeline.run(find("episoade", vid), out_dir, n=0, candidates=a.candidati,
                           language="ro", render_clips=False)
        work = os.path.join(out_dir, ".analiza", vid)
        picks = json.load(open(rep["picks_file"], encoding="utf-8"))
        res = evaluate(truth, picks, np.load(os.path.join(work, "env.npy")))
        res["nr"] = meta["nr"]
        res["cost_usd"] = picks.get("cost_usd")
        cost += picks.get("cost_usd") or 0
        rows.append(res)
        k10 = res["by_k"]["10"]
        print(f"[{meta['nr']}] {res['shorts']} Shorts: găsite {k10['found']} din top 10 "
              f"(la întâmplare {k10['random_found']}, după volum {k10.get('loud_found')}); "
              f"acoperiri {k10['coverage']}")
    tot = {"set": a.set, "episodes": len(rows), "shorts": sum(r["shorts"] for r in rows),
           "cost_usd": round(cost, 3), "by_k": {}}
    for k in rows[0]["by_k"]:
        tot["by_k"][k] = {
            "claude": sum(r["by_k"][k]["found"] for r in rows),
            "random": round(sum(r["by_k"][k]["random_found"] for r in rows), 2),
            "loudness": sum(r["by_k"][k].get("loud_found", 0) for r in rows),
            "mean_coverage": round(float(np.mean([c for r in rows for c in r["by_k"][k]["coverage"]])), 3),
        }
    print(json.dumps(tot, ensure_ascii=False, indent=1))
    name = f"rezultate-{a.set}{('-' + a.eticheta) if a.eticheta else ''}.json"
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump({"total": tot, "episodes": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

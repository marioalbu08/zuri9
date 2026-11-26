"""python -m zuri9 episod.mp4 [-n 5] [--out clipuri] [--cere "..."]"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="zuri9", description="Material brut -> clipuri verticale.")
    ap.add_argument("src", help="fișierul video (orizontal sau vertical)")
    ap.add_argument("-n", type=int, default=5, help="câte clipuri randează (implicit 5)")
    ap.add_argument("--candidati", type=int, default=10, help="câte momente cere de la Claude")
    ap.add_argument("--out", default="clipuri", help="folderul pentru clipuri")
    ap.add_argument("--cere", default="", help="ce vrei, în cuvinte (de ex. „doar momentele amuzante”)")
    ap.add_argument("--min", type=int, default=20, help="durata minimă, în secunde")
    ap.add_argument("--max", type=int, default=60, help="durata maximă, în secunde")
    ap.add_argument("--limba", default=None, help="limba (ro, en...); implicit o ghicește")
    ap.add_argument("--whisper", default="medium", help="modelul whisper (small, medium...)")
    ap.add_argument("--fara-randare", action="store_true", help="doar alege momentele")
    a = ap.parse_args()
    run(a.src, a.out, n=a.n, candidates=a.candidati, min_s=a.min, max_s=a.max,
        request=a.cere, language=a.limba, whisper=a.whisper, render_clips=not a.fara_randare)


if __name__ == "__main__":
    main()

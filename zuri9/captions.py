"""Subtitrările arse în clip: câte 1-3 cuvinte, cuvântul spus acum colorat (ASS)."""
from __future__ import annotations


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def chunks(words: list[dict], max_words: int = 3, max_chars: int = 16,
           max_gap: float = 0.5) -> list[list[dict]]:
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words
                    or len(" ".join(x["w"] for x in cur + [w])) > max_chars
                    or w["s"] - cur[-1]["e"] > max_gap
                    or cur[-1]["w"][-1:] in ".?!,"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def write_ass(words: list[dict], path: str, w: int = 1080, h: int = 1920,
              font: str = "Arial Black", size: int = 86, y_frac: float = 0.70,
              highlight: str = "&H0000E5FF&") -> None:
    """`highlight` e în ordinea ASS (BGR): galben implicit."""
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    x, y = w // 2, int(h * y_frac)
    for ch in chunks(words):
        end_chunk = ch[-1]["e"] + 0.15
        for k, cw in enumerate(ch):
            s = cw["s"] if k else ch[0]["s"]
            e = ch[k + 1]["s"] if k + 1 < len(ch) else end_chunk
            if e - s < 0.04:
                continue
            parts = []
            for j, ww in enumerate(ch):
                t = _esc(ww["w"]).upper()
                parts.append(f"{{\\c{highlight}}}{t}{{\\c&H00FFFFFF&}}" if j == k else t)
            pop = "{\\fscx108\\fscy108\\t(0,80,\\fscx100\\fscy100)}" if k == 0 else ""
            lines.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Main,,0,0,0,,"
                         f"{{\\pos({x},{y})}}{pop}{' '.join(parts)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")

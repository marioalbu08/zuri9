"""Claude alege momentele: citește transcrierea (cu semnele din sunet) și, dacă e
nevoie, foi de cadre. Întoarce clipurile ca intervale de rânduri din transcriere.

Merge cu ANTHROPIC_API_KEY (SDK-ul oficial) sau, fără cheie, prin `claude -p`
(creditele Claude Code).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile

MODEL = os.environ.get("ZURI9_MODEL", "claude-opus-5-5")

SYSTEM = """Ești editor de clipuri scurte (TikTok, Reels, Shorts) pentru un podcast sau un interviu.
Primești transcrierea întreagă, pe rânduri numerotate, cu ora de început. Semnul
(volum +N dB) arată un vârf de sunet: de obicei râs, aplauze sau o voce ridicată.

Alege momentele care merită clip vertical de sine stătător:
- Primele 2-3 secunde agață: o afirmație tare, o întrebare, un conflict, o cifră, o emoție.
  Nu începe cu „și”, „deci”, „da, exact” sau cu un răspuns la o întrebare care nu se aude.
- Se înțelege fără restul episodului: cine nu a văzut episodul pricepe despre ce e vorba.
- Are un final: o concluzie, o poantă, o replică bună. Nu se oprește în mijlocul ideii.
- Durata vorbită: între {min_s} și {max_s} de secunde.
- Clipurile nu folosesc aceleași rânduri.

Un clip e o listă de bucăți (`pieces`), fiecare un interval de rânduri întregi, redate în
ordinea din listă. Ca editorii buni de podcast, poți:
- tăia din interiorul unui moment rândurile care lungesc (o paranteză, o repetare, o bâlbâială):
  atunci clipul are mai multe bucăți apropiate;
- lega fraze din locuri diferite ale episodului, dacă împreună spun o poveste mai bună
  (afirmația tare de la minutul 40 pusă la început, ca un cârlig, urmată de explicația de la minutul 12).
Fiecare bucată începe și se termină la o frază întreagă. Rezultatul trebuie să sune firesc.

Scrie titlul și cârligul în limba transcrierii. `why` e o propoziție: de ce ar merge clipul.
`score` (1-10) e cât de sigur ești că va prinde, comparat cu celelalte momente din episod."""

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pieces": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"from_line": {"type": "integer"},
                                       "to_line": {"type": "integer"}},
                        "required": ["from_line", "to_line"],
                        "additionalProperties": False}},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "why": {"type": "string"},
                    "kind": {"type": "string", "enum": [
                        "poveste", "opinie tare", "sfat", "amuzant", "emoție", "cifră", "conflict", "altceva"]},
                    "score": {"type": "integer"},
                },
                "required": ["pieces", "title", "hook", "why", "kind", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def phrase_lines(words: list[dict], max_words: int = 28, pause: float = 0.6) -> list[dict]:
    """Împarte cuvintele în rânduri: la pauze, la punct sau după `max_words`."""
    lines, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - w["e"]) if nxt else 99
        end_punct = w["w"][-1:] in ".?!…"
        if gap >= pause or (end_punct and len(cur) >= 6) or len(cur) >= max_words:
            lines.append({"w0": cur[0], "w1": cur[-1], "s": words[cur[0]]["s"],
                          "e": words[cur[-1]]["e"],
                          "text": " ".join(words[k]["w"] for k in cur)})
            cur = []
    return lines


def transcript_text(lines: list[dict], loud: list[dict]) -> str:
    out, j = [], 0
    for i, ln in enumerate(lines):
        mark = ""
        while j < len(loud) and loud[j]["s"] < ln["e"] + 0.5:
            if loud[j]["s"] >= ln["s"] - 0.5:
                mark = f" (volum +{loud[j]['rise_db']:.0f} dB)"
            j += 1
        m, s = divmod(int(ln["s"]), 60)
        h, m = divmod(m, 60)
        out.append(f"L{i} [{h}:{m:02d}:{s:02d}] {ln['text']}{mark}")
    return "\n".join(out)


def _have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _ask_sdk(system: str, user_text: str, images: list[str], effort: str) -> tuple[dict, float]:
    import anthropic

    client = anthropic.Anthropic()
    content = []
    for p in images:
        with open(p, "rb") as f:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(f.read()).decode()}})
    content.append({"type": "text", "text": user_text})
    # dacă un filtru de siguranță refuză (rar, la subiecte medicale sau penale),
    # serverul reia cererea pe alt model în loc să întoarcă nimic
    with client.beta.messages.stream(
        model=MODEL, max_tokens=32000, system=system,
        thinking={"type": "adaptive"}, output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude a refuzat cererea")
    text = next(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    cost = (u.input_tokens * 4 + u.output_tokens * 20) / 1e6
    return json.loads(text), cost


def _claude_exe() -> str | None:
    exe = shutil.which("claude")
    if exe and exe.lower().endswith((".cmd", ".bat")):
        # pe Windows, npm pune un .CMD; cmd.exe taie argumentele cu mai multe rânduri
        # (promptul de sistem), deci chemăm direct executabilul de dedesubt
        real = os.path.join(os.path.dirname(exe), "node_modules", "@anthropic-ai",
                            "claude-code", "bin", "claude.exe")
        if os.path.exists(real):
            return real
    return exe


def _ask_cli(system: str, user_text: str, images: list[str], effort: str) -> tuple[dict, float]:
    exe = _claude_exe()
    if not exe:
        raise RuntimeError("nici ANTHROPIC_API_KEY, nici CLI-ul claude")
    tools = ""
    if images:
        # fără cheie API, Claude Code citește imaginile cu unealta Read
        user_text = ("Citește întâi imaginile acestea, cu Read: " + ", ".join(images)
                     + "\n\n" + user_text)
        tools = "Read"
    cmd = [exe, "-p", "--tools", tools, "--no-session-persistence", "--strict-mcp-config",
           "--model", MODEL, "--effort", effort, "--output-format", "json",
           "--system-prompt", system, "--json-schema", json.dumps(SCHEMA)]
    if images:
        cmd += ["--allowedTools", "Read", "--add-dir", os.path.dirname(images[0])]
    # rulează din alt folder, ca să nu ia setările și hook-urile proiectului
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run(cmd, input=user_text, capture_output=True, text=True,
                           encoding="utf-8", timeout=1800, cwd=cwd)
    if p.returncode != 0 or not p.stdout.strip().startswith("{"):
        raise RuntimeError((p.stderr or p.stdout).strip()[:400] or "claude -p n-a răspuns")
    outer = json.loads(p.stdout)
    if outer.get("is_error"):
        raise RuntimeError(str(outer.get("result"))[:400])
    data = outer.get("structured_output") or json.loads(outer.get("result", "{}"))
    return data, float(outer.get("total_cost_usd") or 0)


def ask(system: str, user_text: str, images: list[str] | None = None,
        effort: str = "high") -> tuple[dict, float, str]:
    images = images or []
    if _have_api_key():
        d, c = _ask_sdk(system, user_text, images, effort)
        return d, c, "api"
    d, c = _ask_cli(system, user_text, images, effort)
    return d, c, "claude -p"


def pick_moments(words: list[dict], loud: list[dict], n: int = 8, min_s: int = 20,
                 max_s: int = 60, request: str = "", sheets: list[str] | None = None,
                 effort: str = "high") -> dict:
    lines = phrase_lines(words)
    text = transcript_text(lines, loud)
    system = SYSTEM.format(min_s=min_s, max_s=max_s)
    user = f"Alege {n} momente, de la cel mai bun la cel mai slab.\n"
    if request:
        user += f"\nCererea omului care montează (are prioritate): {request}\n"
    if sheets:
        user += ("\nImaginile sunt foi de cadre din video, cu ora pe fiecare cadru. "
                 "Folosește-le ca să vezi ce se întâmplă în imagine.\n")
    user += "\nTranscrierea:\n" + text
    data, cost, via = ask(system, user, sheets, effort)
    clips = []
    for c in data.get("clips", []):
        pieces = []
        for p in c.get("pieces", []):
            a, b = int(p["from_line"]), int(p["to_line"])
            if 0 <= a <= b < len(lines):
                pieces.append([a, b])
        if not pieces:
            continue
        spans = [[lines[a]["s"], lines[b]["e"]] for a, b in pieces]
        clips.append({**c, "pieces": pieces, "spans": spans,
                      "s": min(x[0] for x in spans), "e": max(x[1] for x in spans)})
    return {"clips": clips, "lines": lines, "cost_usd": round(cost, 4), "via": via,
            "model": MODEL}

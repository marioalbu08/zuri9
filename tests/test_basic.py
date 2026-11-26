"""Teste rapide, fără video și fără Claude."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zuri9 import audio, brain, captions, refine  # noqa: E402


def _words(spec):
    """spec: [(cuvânt, start, end)]"""
    return [{"w": w, "s": s, "e": e, "p": 1.0} for w, s, e in spec]


def _env(dur, speech):
    env = np.full(int(dur / audio.HOP), -70.0, np.float32)
    for s, e in speech:
        env[int(s / audio.HOP): int(e / audio.HOP)] = -20.0
    return env


def test_phrase_lines_split_on_pause():
    w = _words([("Salut", 0.0, 0.4), ("tuturor.", 0.45, 0.9), ("Azi", 2.0, 2.3), ("vorbim", 2.35, 2.8)])
    lines = brain.phrase_lines(w)
    assert len(lines) == 2
    assert lines[1]["text"] == "Azi vorbim"


def test_edl_jump_cut_and_no_word_clipped():
    w = _words([("Unu", 1.0, 1.4), ("doi", 1.45, 1.8), ("trei", 3.0, 3.4), ("patru", 3.45, 3.9)])
    lines = brain.phrase_lines(w)
    env = _env(6.0, [(1.0, 1.95), (3.0, 4.0)])  # „doi” și „patru” se aud mai mult decât zice whisper
    edl = refine.build_edl({"start_line": 0, "end_line": len(lines) - 1, "cut_lines": []},
                           lines, w, env, fps=30)
    assert len(edl["segments"]) == 2               # pauza de 1,2 s a fost scoasă
    (s1, e1), (s2, e2) = edl["segments"]
    assert s1 <= 1.0 and e1 >= 1.95 - 0.01          # coada lui „doi” păstrată
    assert e2 >= 4.0 - 0.01
    assert edl["duration"] < 4.0
    for a, b in edl["segments"]:                   # număr întreg de cadre
        assert abs((b - a) * 30 - round((b - a) * 30)) < 1e-3
    ts = [x["s"] for x in edl["words"]]
    assert ts == sorted(ts)


def test_cut_lines_removed():
    w = _words([("A", 0, 0.3), ("B.", 0.35, 0.6), ("paranteză", 1.5, 2.0), ("C", 3.0, 3.3), ("D", 3.35, 3.6)])
    lines = brain.phrase_lines(w, pause=0.6)
    edl = refine.build_edl({"start_line": 0, "end_line": len(lines) - 1, "cut_lines": [1]},
                           lines, w, _env(5, [(0, 0.6), (1.5, 2.0), (3.0, 3.6)]))
    assert "paranteză" not in [x["w"] for x in edl["words"]]


def test_pieces_out_of_order():
    # cârligul de la sfârșit pus primul, apoi începutul
    w = _words([("Început", 0, 0.5), ("aici.", 0.55, 1.0), ("Final", 5.0, 5.4), ("tare!", 5.45, 6.0)])
    lines = brain.phrase_lines(w)
    env = _env(8, [(0, 1.0), (5.0, 6.0)])
    edl = refine.build_edl({"pieces": [[1, 1], [0, 0]]}, lines, w, env)
    assert [x["w"] for x in edl["words"]] == ["Final", "tare!", "Început", "aici."]
    assert edl["segments"][0][0] > edl["segments"][1][1]   # bucățile nu se lipesc greșit
    assert len(edl["segments"]) == 2


def test_loud_moments_finds_laugh():
    env = np.full(12000, -30.0, np.float32)  # 120 s
    env[6000:6150] = -18.0                    # 1,5 s mult mai tare
    m = audio.loud_moments(env)
    assert len(m) == 1 and 59 < m[0]["s"] < 61


def test_ass_chunks(tmp_path):
    w = _words([("ăsta", 0, 0.3), ("e", 0.3, 0.4), ("un", 0.4, 0.5), ("test", 0.5, 0.9), ("lung.", 0.9, 1.3)])
    p = tmp_path / "x.ass"
    captions.write_ass(w, str(p))
    txt = p.read_text(encoding="utf-8")
    assert "ĂSTA" in txt and txt.count("Dialogue:") == 5

"""Citirea fișierelor video: metadate, sunet, cadre. Totul prin ffmpeg."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np


def ffmpeg_exe() -> str:
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # vine cu requirements.txt

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Info:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(path: str) -> Info:
    """Metadatele din `ffmpeg -i` (imageio-ffmpeg nu are ffprobe)."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = p.stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        raise RuntimeError(f"nu pot citi {path}: {err[-300:]}")
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    w = h = 0
    fps = 30.0
    v = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", err)
    if v:
        w, h = int(v[1]), int(v[2])
    f = re.search(r"([\d.]+) fps", err)
    if f:
        fps = float(f[1])
    # telefoanele scriu rotația în metadate; ffmpeg o aplică la decodare
    rot = re.search(r"rotate\s*:\s*(-?\d+)|rotation of (-?[\d.]+)", err)
    if rot:
        deg = abs(int(float(rot[1] or rot[2]))) % 180
        if deg == 90:
            w, h = h, w
    return Info(path, dur, w, h, fps, "Audio:" in err)


def read_audio(path: str, sr: int = 16000, start: float | None = None,
               end: float | None = None, channels: int = 1) -> np.ndarray:
    """Sunetul ca float32 (mono: [n], stereo: [n, 2])."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"] if start is None else ["-t", f"{end - start:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-300:])
    a = np.frombuffer(p.stdout, dtype=np.float32)
    return a.reshape(-1, channels) if channels > 1 else a


def iter_frames(path: str, fps: float, width: int, start: float | None = None,
                end: float | None = None, gray: bool = False):
    """Cadre la `fps` pe secundă, scalate la lățimea `width`.

    Întoarce (timp, cadru). Timpul e relativ la începutul fișierului.
    """
    info = probe(path)
    h = int(round(info.height * width / info.width / 2)) * 2
    pix = "gray" if gray else "bgr24"
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    t0 = 0.0
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
        t0 = start
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", f"{end - t0:.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale={width}:{h}:flags=area",
            "-f", "rawvideo", "-pix_fmt", pix, "-"]
    ch = 1 if gray else 3
    size = width * h * ch
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=size * 4)
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, width, ch) if ch == 3 \
                else np.frombuffer(buf, np.uint8).reshape(h, width)
            # fps filtrul pune cadrul i la timpul i/fps
            yield t0 + i / fps, frame
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

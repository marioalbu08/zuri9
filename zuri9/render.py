"""Randarea clipului vertical: bucățile din EDL, decupate după planul camerei,
cu subtitrări arse și sunetul adus la -14 LUFS (nivelul TikTok/Reels)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import wave

import cv2
import numpy as np

from .media import ffmpeg_exe, iter_frames, probe, read_audio

SR = 48000


def _encoder() -> list[str]:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "h264_nvenc" in p.stdout:
        # verificăm că placa chiar poate coda (driverul poate lipsi)
        t = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", "color=black:s=256x256:d=0.1", "-c:v", "h264_nvenc",
                            "-f", "null", "-"], capture_output=True)
        if t.returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def _audio(src: str, segments: list[list[float]], out_wav: str, fps: float,
           fade: float = 0.008) -> None:
    parts = []
    nf = int(fade * SR)
    ramp = np.linspace(0, 1, nf, dtype=np.float32)[:, None]
    for s, e in segments:
        a = read_audio(src, SR, s, e, channels=2)
        # exact cât cadrele video: n cadre * SR / fps
        want = int(round(round((e - s) * fps) * SR / fps))
        if len(a) < want:
            a = np.vstack([a, np.zeros((want - len(a), 2), np.float32)])
        a = a[:want].copy()
        if len(a) > 2 * nf:
            a[:nf] *= ramp
            a[-nf:] *= ramp[::-1]
        parts.append(a)
    x = np.vstack(parts) if parts else np.zeros((SR, 2), np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _loudnorm_args(wav: str, target: float = -14.0) -> str:
    """Prima trecere loudnorm: măsoară, ca a doua să aplice un câștig liniar."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", wav, "-af",
                        f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        return f"loudnorm=I={target}:TP=-1.5:LRA=11"
    j = json.loads(m.group(0))
    return (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={j['input_i']}:"
            f"measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
            f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")


def render(src: str, edl: dict, plans: list[dict], ass_path: str | None, out_path: str,
           fps: float = 30.0, out_w: int = 1080, out_h: int = 1920) -> dict:
    info = probe(src)
    vertical_src = info.width / max(1, info.height) < 0.7
    tmp = tempfile.mkdtemp(prefix="zuri9_")
    wav = os.path.join(tmp, "a.wav")
    _audio(src, edl["segments"], wav, fps)
    an = _loudnorm_args(wav)

    vf = []
    if ass_path:
        # ffmpeg pe Windows: calea din filtru cu / și cu „:” scăpat
        p = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
        vf.append(f"ass='{p}'")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
           "-i", "-", "-i", wav]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-af", an, "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), *_encoder(),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = 0
    try:
        for (s, e), plan in zip(edl["segments"], plans):
            n = int(round((e - s) * fps))
            pt = np.array(plan["times"]) if plan["times"] else np.array([s])
            pl = np.array(plan["left"]) if plan["left"] else np.array([(info.width - plan["cw"]) / 2])
            cw = plan["cw"]
            k = 0
            for t, fr in iter_frames(src, fps, info.width, start=s, end=e):
                if k >= n:
                    break
                if vertical_src:
                    out = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    # pe tăieturile de plan nu interpolăm între două poziții
                    i = int(np.searchsorted(pt, t, side="right") - 1)
                    i = min(max(i, 0), len(pt) - 1)
                    x = pl[i]
                    if i + 1 < len(pt) and abs(pl[i + 1] - pl[i]) < cw * 0.4:
                        f = (t - pt[i]) / max(1e-6, pt[i + 1] - pt[i])
                        x = pl[i] + (pl[i + 1] - pl[i]) * min(max(f, 0), 1)
                    x = int(round(x))
                    crop = fr[:, x:x + cw]
                    out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                proc.stdin.write(out.tobytes())
                k += 1
                frames += 1
            # dacă decodorul a dat mai puține cadre, repetăm ultimul (sunetul are lungimea exactă)
            while k < n and frames:
                proc.stdin.write(out.tobytes())
                k += 1
                frames += 1
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg: {err[-500:]}")
    return {"frames": frames, "duration": frames / fps, "path": out_path}

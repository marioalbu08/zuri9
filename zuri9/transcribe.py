"""Transcrierea cu timp pe cuvânt: faster-whisper, local, pe placa video dacă se poate."""
from __future__ import annotations

import glob
import os
import site
import sys
import time

import numpy as np

_dll_ready = False


def _cuda_dlls() -> None:
    """Pe Windows, cuBLAS și cuDNN din pachetele pip `nvidia-*` nu sunt în PATH."""
    global _dll_ready
    if _dll_ready or sys.platform != "win32":
        return
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    _dll_ready = True


def _model(size: str):
    from faster_whisper import WhisperModel

    _cuda_dlls()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            m = WhisperModel(size, device="cuda", compute_type="int8_float16")
            # verificare: dacă lipsesc bibliotecile CUDA, cade abia la primul apel
            list(m.transcribe(np.zeros(16000, np.float32), language="ro")[0])
            return m, "cuda"
    except Exception as e:  # noqa: BLE001
        print(f"[whisper] placa video nu merge ({str(e)[:120]}), folosesc procesorul")
    return WhisperModel(size, device="cpu", compute_type="int8"), "cpu"


def transcribe(audio16k: np.ndarray, language: str | None = None,
               size: str = "medium") -> dict:
    """Întoarce {"language", "words": [{w, s, e, p}], "segments": [{s, e, text}]}."""
    model, device = _model(size)
    t = time.time()
    if device == "cuda":
        # pe placa video, bucățile de vorbire (tăiate de VAD) merg câte 8 deodată
        from faster_whisper import BatchedInferencePipeline

        segs, info = BatchedInferencePipeline(model).transcribe(
            audio16k, language=language, word_timestamps=True, batch_size=8,
            beam_size=5, vad_parameters={"min_silence_duration_ms": 400})
    else:
        segs, info = model.transcribe(
            audio16k, language=language, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5, condition_on_previous_text=False,
        )
    words, segments = [], []
    for sg in segs:
        segments.append({"s": round(sg.start, 3), "e": round(sg.end, 3), "text": sg.text.strip()})
        for w in sg.words or []:
            txt = w.word.strip()
            if txt:
                words.append({"w": txt, "s": round(w.start, 3), "e": round(w.end, 3),
                              "p": round(w.probability, 3)})
    took = time.time() - t
    dur = len(audio16k) / 16000
    print(f"[whisper] {size} pe {device}: {dur / 60:.1f} min în {took:.0f} s")
    return {"language": info.language, "model": size, "device": device,
            "words": words, "segments": segments}

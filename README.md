# zuri9

**Dintr-un podcast, un interviu sau un vlog lung, clipuri verticale pentru TikTok, Reels și Shorts.**

zuri9 ascultă tot materialul, alege momentele care merită clip, taie doar între cuvinte, încadrează 9:16 pe cine vorbește și arde subtitrările.

## Cum

- **Ce se spune:** faster-whisper, cu timp pe fiecare cuvânt, local.
- **Ce merită:** Claude citește toată transcrierea, cu vârfurile de volum marcate (râs, aplauze), și alege momentele. Un clip poate lega fraze din locuri diferite, cum fac editorii.
- **Tăieturile:** doar între cuvinte; pauzele lungi scoase; capetele mutate până unde se termină sunetul.
- **Încadrarea:** fețele găsite cu YuNet; vorbitorul e cel a cărui gură se mișcă atunci când se aude vorbă; o cameră virtuală care stă pe loc și urmărește lin.
- **Clipul:** 1080×1920, subtitrări cuvânt cu cuvânt, sunetul la -14 LUFS.

Video-ul nu pleacă de pe calculator. La Claude ajunge doar textul transcrierii.

## Pornire

```
pip install -r requirements.txt
python -m zuri9 episod.mp4 -n 5 --out clipuri
python -m zuri9 episod.mp4 --cere "doar momentele amuzante" --max 45
```

Claude merge cu `ANTHROPIC_API_KEY` sau prin `claude -p` (Claude Code). Pentru whisper pe placă NVIDIA: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.

## Măsurat

Momentele alese sunt comparate cu Shorts-urile oficiale tăiate de editorii unui podcast românesc. Shorts-urile sunt găsite în episod după sunet (`eval/`).

Până acum e măsurat un singur episod (1 h 46 min, 3 Shorts oficiale). În primele 10 momente propuse sunt 2 din cele 3. Ferestrele alese la întâmplare găsesc în medie 0,22, iar cele după volum găsesc 1. E prea puțin pentru o concluzie; setul întreg are 10 episoade.

## Limite

- Alegerea se bazează pe ce se spune. Imaginea nu e încă citită.
- Fără fețe în cadru, decupajul rămâne la mijloc.

Detectorul de fețe e YuNet din [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (licență MIT).

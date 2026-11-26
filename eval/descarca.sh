#!/usr/bin/env bash
# Descarcă setul de test (episoadele și Shorts-urile din eval/set.json) în ZURI9_DATE.
# Doar sunetul, în afară de episodul 329, luat și cu imagine pentru randare.
# Cere yt-dlp și ffmpeg; video-urile rămân pe calculatorul tău, nu intră în repo.
set -euo pipefail
cd "$(dirname "$0")"
DATA="${ZURI9_DATE:-$HOME/zuri9-date}"
mkdir -p "$DATA/episoade" "$DATA/shorts"
read -r EP SH < <(python -c "
import json; c=json.load(open('set.json', encoding='utf-8'))
eps=[v for s in ('reglaj','final') for v in c[s]]
shs=[x for s in ('reglaj','final') for v in c[s].values() for x in v['shorts']]
print(','.join(eps), ','.join(shs))")
YT=(yt-dlp -q --no-warnings --js-runtimes node)
"${YT[@]}" -o "$DATA/episoade/%(id)s.%(ext)s" -f "ba[ext=m4a]/ba" ${EP//,/ }
"${YT[@]}" -o "$DATA/shorts/%(id)s.%(ext)s" -f "ba/b" ${SH//,/ }
"${YT[@]}" -o "$DATA/episoade/%(id)s.%(ext)s" -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]" \
  --merge-output-format mp4 LCpuIk-0GF0
echo "Gata: $DATA"

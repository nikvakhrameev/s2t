#!/bin/zsh
# Generates sample audio with macOS TTS: noisy silence on the edges, long pauses
# inside, fillers and glossary terms; exported to several containers on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."
out=tests/audio
mkdir -p $out

say -v Milena -o $out/ru_raw.aiff "Ну, в общем, мы вчера задеплоили новый сервис в кубернетес. [[slnc 4000]] И, э-э, постгрес начал тормозить, потому что, как бы, не хватило коннекшенов в пуле. [[slnc 3000]] Надо посмотреть метрики в графане."
say -v Samantha -o $out/en_raw.aiff "So, um, yesterday we deployed the new service to kubernetes. [[slnc 4000]] And, you know, postgres started to slow down because, like, the connection pool was exhausted. [[slnc 2500]] We should check the grafana dashboards."

noise="anoisesrc=d=3:c=pink:a=0.002"
for lang in ru en; do
  ffmpeg -y -loglevel error -f lavfi -i $noise -i $out/${lang}_raw.aiff -f lavfi -i $noise \
    -filter_complex "[0][1][2]concat=n=3:v=0:a=1" -ar 44100 $out/$lang.tmp.wav
done
ffmpeg -y -loglevel error -i $out/ru.tmp.wav -c:a libopus -b:a 32k $out/ru.ogg
ffmpeg -y -loglevel error -i $out/en.tmp.wav -c:a aac -b:a 96k $out/en.m4a
ffmpeg -y -loglevel error -f lavfi -i "anoisesrc=d=6:c=pink:a=0.003" $out/silence.mp3
rm $out/*.tmp.wav
ls -la $out

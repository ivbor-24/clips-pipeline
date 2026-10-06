# Выпуск версии

Для сопровождающих: как выпустить версию с готовыми Docker-образами.

## Порядок

1. **PR с номером версии:** `src/__init__.py`, `pyproject.toml`, `web/package.json`,
   `uv.lock` (`uv lock`), `web/package-lock.json`, раздел в `CHANGELOG.md`. Слить в `main`.
2. **Тег и черновик GitHub Release** `vX.Y.Z` на этом коммите: текст из `CHANGELOG.md`,
   ограничения. Черновик публике не виден — до публикации образов страница выпуска не пустая.
3. **Образы** — на машине с Docker, в чистом клоне на теге (сборка CUDA под все карты идёт
   часами, поэтому не в GitHub Actions):

   ```bash
   git clone https://github.com/ivbor-24/clips-pipeline.git release-build && cd release-build
   git checkout vX.Y.Z
   just publish-images            # собрать и проверить всё, собрать исходники — без публикации
   just publish-images --push     # то же и опубликовать
   ```

   Скрипт `scripts/publish_images.sh` обрабатывает образы **по одному**: веб-интерфейс,
   `openvino`, `cpu`, `cuda`. Каждый собирается (`ghcr.io/ivbor-24/clips-pipeline:X.Y.Z-<бэкенд>`,
   `ghcr.io/ivbor-24/clips-pipeline-web:X.Y.Z`, с метками версии и коммита), проверяется
   (версия в образе, `whisper-cli`, библиотеки CUDA, конфигурация nginx), с него снимается
   список пакетов; с `--push` он сразу публикуется вместе с подвижным тегом (`:openvino`, `:cpu`,
   `:cuda`, веб — `:latest`) и удаляется с диска. Так на диске одновременно один образ: пик —
   образ `cuda` (~8 ГБ) при распаковке, ~16 ГБ свободного места. Затем скачиваются исходники
   всех copyleft-пакетов (`scripts/image_sources.sh`, ~1,2 ГБ), упаковываются в
   `dist/release-X.Y.Z/clips-pipeline-X.Y.Z-sources.tar` и с `--push` прикладываются к
   выпуску (черновику).

   Скрипт из более новой версии можно запустить на старом теге: `--checkout DIR`.

   Нужен токен GitHub с `write:packages` в credential helper git'а: `docker login` идёт только
   на время публикации, токен не остаётся в `~/.docker/config.json`. Если у сети Docker нет
   интернета — `BUILD_NETWORK=host`; при нехватке памяти на сборку — `BUILD_JOBS=2`.
4. **Опубликовать черновик выпуска** — когда образы и архив исходников на месте.
5. **Перевести ветку `release`** на тег — с этого момента `setup.sh` у пользователей ветки
   `release` скачивает новые образы:

   ```bash
   git push origin vX.Y.Z^{commit}:refs/heads/release
   ```

## Переносимая сборка

Опубликованные образы собраны с `CPU_TARGET=portable` (любой x86-64 с AVX), а не под процессор
машины сборки. На GPU-бэкендах разницы нет: замер на Intel Arc B580 + Xeon E5-2630 v2,
25-минутная лекция — 530 с против 526 с у сборки под этот процессор, LLM 20,72 против 20,71
ток/с, те же 12 клипов. Распознавание на NVIDIA и CPU (faster-whisper) от этих флагов не
зависит. Кому нужен LLM на процессоре с AVX2 и быстрее — локальная сборка под свой процессор.

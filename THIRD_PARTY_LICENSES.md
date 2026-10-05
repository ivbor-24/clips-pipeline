# Сторонние лицензии (third-party licenses)

Справочный файл: какие сторонние компоненты использует или скачивает проект и под какими
лицензиями они распространяются. Это **не юридическая консультация**.

Сам проект распространяется под лицензией MIT ([LICENSE](LICENSE)); здесь перечислены
сторонние компоненты.

## Как обновлять этот файл

CI проверяет файл скриптом `scripts/check_licenses.py` (`just licenses`): у каждого пакета из
`uv.lock` (все extras: `cpu`, `cuda`, `openvino`, `test`, `dev`) есть строка в таблице Python-пакетов
(библиотеки NVIDIA — в абзаце об образе `cuda`), строк пакетов, которых в `uv.lock` больше нет,
не осталось, ни у одного пакета нет GPL-лицензии в любом варианте, а веб-пакеты — только под
разрешёнными лицензиями. Добавили зависимость — проверьте её лицензию и допишите строку; версии
в таблице справочные и не проверяются.

- **Пакеты Python** — список получен из окружения образов `openvino`/`cpu` через
  `importlib.metadata`: имя, версия, поле `License` и классификаторы лицензий
  (`dist.metadata` каждого пакета). Повторить в установленном окружении:
  ```python
  import importlib.metadata as md
  for d in md.distributions():
      m = d.metadata
      print(d.name, d.version, m.get("License"), m.get_all("Classifier"))
  ```
- **Пакеты веб-интерфейса** — из `web/package-lock.json` (поле `license` у каждого пакета;
  `prod`-зависимости попадают в собранный бандл, `dev` — только инструменты сборки).

## Модели

Модели **не хранятся в репозитории** — их скачивают `setup.sh` / `just prefetch-models`
по закреплённым ревизиям из `config/models.lock.yaml` (репозиторий Hugging Face + sha256).

| Модель | Назначение | Источник | Лицензия |
|---|---|---|---|
| Qwen3-14B (GGUF, Q4_K_M) | Скоринг, локальная LLM | `bartowski/Qwen_Qwen3-14B-GGUF` — квантизация `Qwen/Qwen3-14B` | Apache-2.0 |
| Whisper large-v3-turbo | Транскрибация | `ggerganov/whisper.cpp` (`ggml-large-v3-turbo.bin`) и `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (CTranslate2) | MIT (OpenAI); конвертации ggml (whisper.cpp) и CTranslate2 (faster-whisper) — тоже MIT |
| Whisper small | Транскрибация (резервная модель) | `ggerganov/whisper.cpp` (`ggml-small.bin`) и `Systran/faster-whisper-small` | MIT (OpenAI) |
| YuNet (`face_detection_yunet_2023mar.onnx`) | Детекция лиц (кроп 9:16) | `opencv/face_detection_yunet` (OpenCV Zoo) | MIT |
| `sentence-transformers/all-MiniLM-L6-v2` | Термины для эвристик (KeyBERT) | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| Gemma 4 E4B (GGUF, Q4_0) | Скоринг, только профиль `local_llm` (не по умолчанию) | `google/gemma-4-E4B-it-qat-q4_0-gguf` | Условия Google для Gemma — проверить перед использованием |

Прежний детектор лиц (`insightface` с моделями `buffalo_l`, разрешены только для
некоммерческого использования) заменён на YuNet (MIT).

## Программы в Docker-образе (не Python-пакеты)

Определено по `Dockerfile.backend`, `Dockerfile.frontend`, `scripts/install_whisper_cpp.sh`,
`scripts/install_llama_cpp_backend.sh`.

| Компонент | Откуда | Лицензия |
|---|---|---|
| ffmpeg | Debian (`python:3.11-slim`) | Сборка Debian включает GPL-компоненты, в т.ч. libx264, поэтому бинарник в целом — GPL-2.0-or-later |
| llama.cpp / ggml | через llama-cpp-python (собирается из исходников) | MIT |
| whisper.cpp | `ggml-org/whisper.cpp`, сборка скриптом `install_whisper_cpp.sh` | MIT |
| Mesa (`mesa-vulkan-drivers`) | Debian trixie-backports | MIT |
| Драйвер VA-API Intel и среда Quick Sync (`intel-media-va-driver`, `libmfx-gen1.2`) | Debian | MIT (Intel) |
| nginx | `nginx:alpine` (`Dockerfile.frontend`) | BSD-2-Clause |
| Python | `python:3.11-slim` | PSF-2.0 |
| База Debian | `python:3.11-slim` / `node:20-alpine` / `nginx:alpine` | Совокупность свободных лицензий Debian |
| Библиотеки CUDA и cuDNN (только образ `cuda`) | pip-пакеты `nvidia-*` (зависимости torch, extra `cuda` из `uv.lock`) | Проприетарная лицензия NVIDIA; распространение — на условиях NVIDIA |

Пакеты `nvidia-*` из `uv.lock` (extra `cuda`, ставятся вместе с torch): `nvidia-cublas-cu12`,
`nvidia-cuda-cupti-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`,
`nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-cufile-cu12`, `nvidia-curand-cu12`,
`nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusparselt-cu12`, `nvidia-nccl-cu12`,
`nvidia-nvjitlink-cu12`, `nvidia-nvtx-cu12`.

В образе также присутствуют служебные пакеты: `fonts-urw-base35` (шрифт субтитров
Nimbus Mono PS; насколько известно, AGPL-3.0 с исключением для шрифтов — проверить перед
публикацией образов: шрифт попадает в образ, а в клипы — только как пиксели субтитров),
`tini` (MIT), `clinfo`, `curl`, `ca-certificates`.

## Пакеты Python

Не из permissive-семейства (GPL/LGPL/AGPL/MPL, проприетарное, без лицензии) — отдельно:

- `certifi` 2026.7.22 — **MPL-2.0**
- `pathspec` 1.1.1 — **MPL-2.0**
- `tqdm` 4.70.1 — **MPL-2.0 AND MIT**

MPL-2.0 — слабый copyleft (обязательства только на изменённые файлы), но по критериям этого
файла вынесен отдельно.

Полная таблица (имя, версия, лицензия в SPDX; пустые поля дополнены из классификаторов,
неоднозначные отмечены):

| Пакет | Версия | Лицензия |
|---|---|---|
| aiosqlite | 0.22.1 | MIT |
| alembic | 1.20.0 | MIT |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anthropic | 1.11.0 | MIT |
| anyio | 4.15.1 | MIT |
| ast_serialize | 0.11.2 | MIT |
| av | 18.1.0 | BSD-3-Clause |
| backports-asyncio-runner | 1.2.0 | PSF-2.0 |
| black | 26.5.1 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| cffi | 2.1.1 | MIT-0 |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| coloredlogs | 15.0.1 | MIT |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| ctranslate2 | 4.8.2 | MIT |
| diskcache | 5.6.3 | Apache-2.0 |
| docstring-parser | 0.18.0 | MIT |
| ecdsa | 0.19.2 | MIT |
| exceptiongroup | 1.3.1 | MIT |
| fastapi | 0.141.1 | MIT |
| faster-whisper | 1.2.1 | MIT |
| filelock | 4.0.6 | MIT |
| flatbuffers | 25.12.19 | Apache-2.0 |
| fsspec | 2026.9.0 | BSD-3-Clause |
| greenlet | 3.5.6 | MIT AND PSF-2.0 |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.13.1 | BSD-3-Clause |
| httptools | 0.8.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| httpx2 | 2.13.1 | BSD-3-Clause |
| huggingface_hub | 1.33.0 | Apache-2.0 |
| humanfriendly | 10.0 | MIT |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| jinja2 | 3.1.6 | BSD-3-Clause |
| jiter | 0.17.0 | MIT |
| joblib | 1.6.0 | BSD-3-Clause |
| keybert | 0.9.0 | MIT |
| librt | 0.16.0 | MIT |
| llama_cpp_python | 0.3.34 | MIT |
| mako | 1.4.3 | MIT |
| markdown-it-py | 4.2.0 | MIT |
| markupsafe | 3.0.3 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT |
| mpmath | 1.3.0 | BSD-3-Clause |
| mypy | 2.3.1 | MIT |
| mypy_extensions | 1.1.0 | MIT |
| narwhals | 2.26.0 | MIT |
| networkx | 3.7 | BSD-3-Clause |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.30.0 | MIT |
| openai | 3.24.0 | Apache-2.0 |
| opencv-python | 5.0.0.93 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pathspec | 1.1.1 | MPL-2.0 |
| platformdirs | 4.12.1 | MIT |
| pluggy | 1.6.0 | MIT |
| protobuf | 7.36.2 | BSD-3-Clause |
| pyasn1 | 0.6.4 | BSD-2-Clause |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| pydantic-settings | 2.15.0 | MIT |
| pydantic_core | 2.46.5 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| pytest-asyncio | 1.4.0 | Apache-2.0 |
| python-dotenv | 1.2.3 | BSD-3-Clause |
| python-jose | 3.5.0 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| pytokens | 0.4.1 | MIT |
| pyyaml | 6.0.3 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| rich | 15.0.0 | MIT |
| rsa | 4.9.1 | Apache-2.0 |
| ruff | 0.16.9 | MIT |
| safetensors | 0.8.0 | Apache-2.0 |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| sentence-transformers | 6.1.0 | Apache-2.0 |
| setuptools | 84.0.0 | MIT |
| shellingham | 1.5.4 | ISC |
| six | 1.17.0 | MIT |
| sniffio | 1.3.1 | MIT OR Apache-2.0 |
| sqlalchemy | 2.1.1 | MIT |
| starlette | 1.7.0 | BSD-3-Clause |
| structlog | 26.1.0 | MIT OR Apache-2.0 |
| sympy | 1.14.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| tokenizers | 0.23.2 | Apache-2.0 |
| tomli | 2.4.1 | MIT |
| torch | 2.8.0+cpu | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 5.17.0 | Apache-2.0 |
| triton | 3.4.0 | MIT |
| truststore | 0.10.4 | MIT |
| typer | 0.27.2 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |
| uvloop | 0.22.1 | MIT OR Apache-2.0 |
| watchfiles | 1.3.0 | MIT |
| websockets | 17.1 | BSD-3-Clause |
| yt-dlp | 2026.8.19 | Unlicense |

## Веб-интерфейс

Сводка по лицензиям `prod`-зависимостей (число уникальных имён пакетов; всего 362 имени):

| Лицензия | Пакетов |
|---|---|
| MIT | 316 |
| ISC | 23 |
| BSD-3-Clause | 7 |
| Apache-2.0 | 4 |
| BSD-2-Clause | 4 |
| BlueOak-1.0.0 | 1 |
| OFL-1.1 | 1 |
| Python-2.0 | 1 |
| CC-BY-4.0 | 1 |
| Unlicense | 1 |
| 0BSD | 1 |
| (MIT OR CC0-1.0) | 1 |

Пакет `isexe` присутствует в двух версиях: ISC и BlueOak-1.0.0 (учтён в обеих строках выше
как один двойной случай).

`prod`-пакетов не из permissive-семейства или без лицензии **нет**. `dev`-зависимости
(включая `lightningcss` под MPL-2.0) в собранный бандл не попадают — это только инструменты
сборки и проверки.

## Распространение образов

- **ffmpeg (GPL-2.0-or-later):** готовые образы со сборкой ffmpeg из Debian при распространении
  подпадают под требования GPL. Сейчас образы собираются у пользователя
  (`./setup.sh` / `docker compose build`), а не публикуются готовыми — это снимает обязательства
  по предоставлению исходников соответствующих компонентов.
- **Библиотеки NVIDIA в образе `cuda`** (`nvidia-*`, CUDA и cuDNN) — проприетарные; их
  перераспространение — только на условиях NVIDIA.
- **Gemma 4 E4B** (профиль `local_llm`) — применяются условия Google для Gemma, проверить их
  перед включением профиля и использованием.
- **Лицензия самого проекта — MIT.** Зависимости не накладывают на код ограничений:
  ffmpeg вызывается отдельной программой, GPL-пакетов среди Python- и веб-зависимостей нет
  (это проверяет CI). Ограничения касаются только готовых образов: при их публикации —
  исходники GPL-компонентов ffmpeg и условия NVIDIA для образа `cuda`.

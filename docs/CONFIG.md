# ⚙️ Scientific Clips Pipeline — Configuration Guide

> **Версия:** 0.8.0  
> **Последнее обновление:** 2026-10-02

Это руководство описывает параметры конфигурации пайплайна. Актуальная схема всегда доступна в `src/config.py`, а боевой пример — в `config/config.yaml`.

---

## 📁 Структура конфигурационного файла

Конфигурация хранится в `config/config.yaml` и использует формат YAML. Все параметры разделены на логические секции.

### Полный пример конфигурации

```yaml
profiles_dir: "config/profiles"

input:
  default_source: ""
  cache_dir: "artifacts/cache"

preprocessing:
  audio_sample_rate: 16000
  audio_channels: 1
  video_max_height: 1080
  video_codec: "h264"
  video_crf: 23
  copy_compatible_video: true
  denoise: false
  denoise_method: "afftdn"

transcription:
  model: "large-v3-turbo"
  language: "auto"
  batch_size: 8
  compute_type: "float16"
  chunk_duration_min: 30
  engine: "faster_whisper"
  whisper_cpp_binary: "whisper-cli"
  whisper_cpp_model_path: ""
  whisper_cpp_threads: 8
  whisper_cpp_beam_size: 4
  whisper_cpp_max_context: 0

scoring:
  min_duration: 45
  max_duration: 90
  min_score_threshold: 0.45
  max_clips_per_video: 12
  term_extraction_method: "keybert"
  strategy: "llm_windows"
  window_sec: 150
  window_step_sec: 75
  windows_per_call: 3
  llm:
    enabled: true
    provider: "llama_cpp"
    model: "Qwen/Qwen3-14B"
    model_repo: "bartowski/Qwen_Qwen3-14B-GGUF"
    model_file: "Qwen_Qwen3-14B-Q4_K_M.gguf"
    model_path: "artifacts/cache/models/Qwen_Qwen3-14B-Q4_K_M.gguf"
    api_key: null
    api_base: null
    temperature: 0.2
    seed: 42
    max_tokens: 4000
    prompt_file: "config/prompts/segment_analysis_v1.txt"
    window_prompt_file: "config/prompts/segment_analysis_v3.txt"
    disable_thinking: true
    llm_weight: 0.4
    max_segments: null
    n_ctx: 8192
    n_gpu_layers: -1
    n_batch: 512
    top_p: 0.95
    chat_format: null
    download_if_missing: true
    min_tokens_per_sec: 10.0
    verbose: false

cropping:
  output_width: 1080
  output_height: 1920
  face_confidence_threshold: 0.6
  moving_average_window: 5
  sample_fps: 1
  shot_size_ratio_threshold: 1.35
  shot_center_jump_threshold: 0.5
  min_track_segment_sec: 2.0
  scene_cut_threshold: 10
  frame_decoder: "auto"
  min_reframe_share: 0.25

rendering:
  audio_loudnorm_i: -14
  audio_loudnorm_tp: -1.5
  audio_loudnorm_lra: 11
  padding_color: "black"
  subtitle_format: "srt"
  subtitle_style: "typewriter"
  subtitle_style_overrides: {}
  subtitle_fonts_dir: null
  subtitle_font: "Arial"
  subtitle_fontsize: 48
  subtitle_primary_color: "&H00FFFFFF"
  subtitle_outline_color: "&H00000000"
  subtitle_back_color: "&H80000000"
  video_codec: "h264"
  video_preset: "medium"
  video_encoder: "auto"
  render_timeout_sec: 300

chapters:
  enabled: false
  min_minutes: 5
  max_minutes: 7
  target_count: null
  prompt_file: "config/prompts/chapter_detection_v1.txt"
  output_format: "both"

broll:
  enabled: false
  max_suggestions: null
  prompt_file: "config/prompts/broll_suggestions_v1.txt"

cleanup:
  retention_days: 30
  max_upload_size_gb: 50

output:
  clips_dir: "output/clips"
  manifest_file: "output/manifest.json"
  review_file: "artifacts/review.json"
  isolated: true
  output_dir_template: "{video_name}_{timestamp}"
  timestamp_format: "%Y-%m-%d_%H-%M-%S"

logging:
  level: "INFO"
  log_file: "artifacts/logs/pipeline.log"
  json_format: true

gpu:
  backend: auto
```

---

## 🔍 Описание секций

### 1. Секция верхнего уровня — Общие настройки

Помимо секций ниже, у конфига есть ключи верхнего уровня:

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `work_dir` | str | "." | Рабочий каталог задачи (корень проекта для CLI; для задач из веб-интерфейса — `jobs/job_<id>/`) |
| `profiles_dir` | str | "config/profiles" | Каталог профилей пользователя (`load_profile`) |
| `hardware_dir` | str | "config/hardware" | Каталог аппаратных профилей (`with_hardware_profile`, выбирается переменной `PIPELINE_HARDWARE`) |

Количество клипов, их длительность и порог скора находятся в секции `scoring`, а не на верхнем уровне; флаги `dry_run`, `resume` и `force` — аргументы CLI (`just process`), а не ключи конфига.

---

### 1.1. Секция `input` — Входные данные

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `default_source` | str | "" | Путь к файлу или URL по умолчанию (если не передан в CLI) |
| `cache_dir` | str | "artifacts/cache" | Каталог кэша (транскрипция и модели) |

---

### 1.2. Секция `preprocessing` — Препроцессинг видео и аудио

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `video_max_height` | int | 1080 | Максимальная высота `prep.mp4`; меньшие исходники не растягиваются |
| `video_codec` / `video_crf` | str / int | "h264" / 23 | Параметры перекодирования, если оно нужно |
| `copy_compatible_video` | bool | true | Исходник 8-bit H.264 (yuv420p) не выше `video_max_height` ремуксится (`-c:v copy`) без перекодирования; `false` — всегда перекодировать |
| `audio_sample_rate` / `audio_channels` | int | 16000 / 1 | Формат `raw.wav` для транскрибации |
| `denoise` / `denoise_method` | bool / str | false / "afftdn" | Шумоподавление перед транскрибацией |

**Зачем ремукс:** 90-минутная 1080p-лекция перекодировалась в тот же 1080p ~45 мин
и теряла качество (5 → 1.8 Мбит/с). Ремукс занимает секунды.

---

### 2. Секция `transcription` — Настройки транскрипции

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | str | "large-v3-turbo" | Модель Whisper: tiny, base, small, medium, large-v3, large-v3-turbo |
| `language` | str | "auto" | Язык аудио: ru, en, auto (автодетект) |
| `batch_size` | int | 8 | Размер батча для транскрипции |
| `compute_type` | str | "float16" | Тип вычислений: float16, int8, float32 |
| `chunk_duration_min` | int | 30 | Длительность чанка в минутах |
| `engine` | str | "faster_whisper" | Движок: `faster_whisper` (CPU / NVIDIA CUDA) или `whisper_cpp` (whisper-cli; GPU Intel/AMD/NVIDIA через Vulkan). Если бинарь или модель whisper.cpp не найдены, задача падает с ошибкой, в которой сказано, как их поставить (`just install-whisper-cpp`, `just prefetch-models`). Тихого отката на faster-whisper нет: на GPU Intel и AMD он считал бы на CPU |
| `whisper_cpp_binary` | str | "whisper-cli" | Путь к `whisper-cli` или имя в `PATH` |
| `whisper_cpp_model_path` | str | "" | ggml-модель, например `ggml-large-v3-turbo.bin` |
| `whisper_cpp_threads` | int | 8 | CPU-потоки whisper-cli (`-t`) |
| `whisper_cpp_beam_size` | int | 4 | Beam size (`-bs`) |
| `whisper_cpp_max_context` | int | 0 | Сколько токенов текста переносить между 30-с окнами (`-mc`; -1 = максимум модели). **Держите 0:** с переносом длинный чанк деградирует в обрывки по 1–3 слова без пунктуации (в прогоне 2 — 51% сегментов) |

Кэш транскрипта (`artifacts/cache/transcription/`) ключуется по фактически
работавшему движку и, для whisper.cpp, по модели, `beam_size` и `max_context`.
При OOM faster-whisper перезагружает модель в `int8` и повторяет чанк один раз.

**Установка whisper.cpp:** `just install-whisper-cpp openvino` (или `cpu` /
`cuda`) собирает `whisper-cli` закреплённой версии в
`artifacts/cache/whisper-cpp-bin/`; `./install.sh openvino` делает это сам.
ggml-модель скачивает `just prefetch-models config/config.yaml intel_arc`
(в `whisper_cpp_model_path`, или в `artifacts/cache/models/ggml-<model>.bin`,
если путь пуст).

**Рекомендации для 12 GB VRAM:**
- Используйте `large-v3-turbo` вместо `large-v3` для экономии памяти
- Уменьшите `batch_size` до 4 или 2 при OOM ошибках
- Используйте `compute_type: float16` для GPU

**Модели Whisper (по возрастанию качества и размера):**
| Модель | VRAM | Скорость | Качество |
|--------|------|----------|----------|
| tiny | ~1 GB | Очень быстро | Низкое |
| base | ~1 GB | Быстро | Среднее |
| small | ~2 GB | Средне | Хорошее |
| medium | ~5 GB | Медленно | Очень хорошее |
| large-v3 | ~10 GB | Очень медленно | Отличное |
| large-v3-turbo | ~6 GB | Быстро | Отличное |

---

### 3. Секция `scoring` — Настройки семантического скоринга

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `min_duration` | int | 45 | Минимальная длительность клипа в секундах |
| `max_duration` | int | 90 | Максимальная длительность клипа в секундах |
| `min_score_threshold` | float | 0.45 | Минимальный score клипа. Скор клипа — **максимум** скоров его сегментов (сильный момент не размывается усреднением) |
| `max_clips_per_video` | int | 12 | Сколько клипов-кандидатов предложить; в ревью обычно оставляют 5–7 |
| `term_extraction_method` | str | "keybert" | Метод извлечения терминов: `keybert` (модель KeyBERT) или `statistical` (частоты слов, без модели); старое значение `yake` читается как `statistical` |
| `strategy` | str | "llm_windows" | Как выбираются клипы (по умолчанию — лучшая из измеренных, `llm_windows`; без LLM — откат на эвристики `segment_merge` с уведомлением): `segment_merge` — скоры фраз (эвристики + LLM) и склейка подряд до `max_duration`; `llm_windows` — LLM сама выбирает клип «от хука до законченной мысли» внутри окон (см. ниже) |
| `window_sec` / `window_step_sec` | float | 150 / 75 | `llm_windows`: длина окна, показываемого LLM, и шаг между окнами (окна перекрываются) |
| `windows_per_call` | int | 3 | `llm_windows`: окон в одном запросе к LLM |
| `llm.enabled` | bool | false | Включить LLM-анализ поверх эвристик |
| `llm.provider` | str | "openai" | Провайдер LLM: llama_cpp, openai, anthropic, qwen_local |
| `llm.llm_weight` | float | 0.4 | Вес LLM-score при смешивании с эвристиками (0 = только эвристики) |
| `llm.max_segments` | int | null | Сколько сегментов (топ по эвристике) отправлять в LLM; null = все |

Боевой `config/config.yaml` включает LLM: `llm.enabled: true`, `provider: llama_cpp`,
Qwen3-14B (`model_repo` `bartowski/Qwen_Qwen3-14B-GGUF`). Значения в таблицах — значения
модели по умолчанию (`src/config.py`).

**Стратегия `llm_windows`** (`src/window_scoring.py`, промпт `segment_analysis_v3.txt`; v2 — первая итерация):
1. Транскрипт режется на перекрывающиеся окна по границам фраз.
2. Для каждого окна LLM возвращает строку начала (хук, понятный без контекста),
   строку конца (мысль закончена), оценки `hook`, `complete`, `value` и название.
3. Скор клипа = 0.4·hook + 0.35·complete + 0.25·value. Клип подгоняется под
   `min_duration`..`max_duration` (хук сохраняется: длинный укорачивается с конца,
   короткий продлевается — такие помечаются тегами `trimmed_to_max` / `extended_to_min`).
4. Отбираются лучшие непересекающиеся клипы (не больше `max_clips_per_video`,
   не ниже `min_score_threshold`).
Эвристики в этой стратегии не используются; без LLM или при её сбое скоринг
откатывается на `segment_merge` только на эвристиках. Окна, ответы и все кандидаты —
в `artifacts/llm_analysis.json`.

**Эвристики скоринга (внутренние веса):**

1. **Term Density (30%)** — плотность научных терминов в сегменте
2. **Definition Markers (35%)** — наличие определений ("это", "является", "называется")
3. **Rhetorical Devices (15%)** — вопросы, эмфазы, вовлекающие конструкции
4. **Self-Containment (20%)** — низкий процент местоимений, полные мысли

### 3.1. Секция `scoring.llm` — LLM-анализ

Используется при `scoring.llm.enabled: true`.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `provider` | str | "openai" | `llama_cpp` для локальных GGUF-моделей, `openai`/`anthropic` для API, `qwen_local` для transformers |
| `model` | str | "gpt-4" | Имя модели для отображения/выбора (для `llama_cpp` — идентификатор HF-репозитория) |
| `api_key` | str | null | Ключ для `openai`/`anthropic` (в боевом конфиге хранят в `.env`, а не здесь) |
| `api_base` | str | null | Адрес API для `openai`: сервис с OpenAI-совместимым API (DeepSeek, OpenRouter, свой сервер); null — сам OpenAI |
| `prompt_file` | str | "config/prompts/segment_analysis_v1.txt" | Промпт `segment_merge` |
| `model_repo` | str | null | Hugging Face repo для скачивания GGUF (только llama_cpp) |
| `model_file` | str | null | Имя GGUF-файла внутри repo |
| `model_path` | str | null | Локальный путь к GGUF (держите его проектным, например `artifacts/cache/models/...`, а не абсолютным путём конкретной машины); если задан и файл существует — скачивание не требуется |
| `n_ctx` | int | 4096 | Размер контекста llama.cpp |
| `n_gpu_layers` | int | -1 | Число слоёв на GPU (-1 = все); 0 = CPU-only |
| `n_batch` | int | 512 | Размер батча для подачи токенов |
| `temperature` | float | 0.7 | Температура генерации |
| `window_prompt_file` | str | "config/prompts/segment_analysis_v3.txt" | Промпт стратегии `llm_windows` |
| `disable_thinking` | bool | false (в `config.yaml` — true) | Добавить `/no_think` (мягкий переключатель Qwen3): рассуждения съедали ~40% токенов |
| `seed` | int | null (в `config.yaml` — 42) | Фиксированный seed генерации (llama_cpp) для воспроизводимости |
| `max_tokens` | int | 1000 | Максимум генерируемых токенов на батч (в боевом `config.yaml` — 4000: сегменты идут в LLM батчами по 15, и с reasoning-моделями 1000 не хватает) |
| `min_tokens_per_sec` | float | 10.0 | Если **первый** вызов медленнее — бэкенд считается непригодным (например, тихий откат на CPU) и скоринг идёт на эвристиках; медленный последующий вызов только логируется |
| `top_p` | float | 0.95 | nucleus sampling |
| `chat_format` | str | null | Автоопределение из GGUF; можно задать явно, например, `qwen3` или `chatml` |
| `download_if_missing` | bool | true | Скачать модель, если она отсутствует локально |
| `verbose` | bool | false | Подробный вывод llama.cpp |

**Поведение при сбоях:** упавший батч пропускается (`llm_batch_failed` в логе),
остальные применяются; на эвристики откатываемся, только если упало больше половины
батчей или модель не загрузилась. Неверный API-ключ, недоступная модель (HTTP
401/403/404) или нет связи с API — сразу откат, остальные батчи не отправляются.
Сырые ответы LLM вместе с эвристическими скорами сохраняются в
`artifacts/llm_analysis.json` (отладочный артефакт).

**API-провайдеры.** Ключ — в `.env` (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`),
провайдер и модель обычно выбирает `setup.sh` (пишет их в `data/settings.yaml`).
Модели отличаются набором параметров, поэтому:
- `anthropic`: `temperature` не передаётся (текущие модели Claude и SDK его не
  принимают), лимит ответа — не меньше 16000 токенов: модель рассуждает перед
  ответом, и рассуждения входят в лимит;
- `openai`: сначала запрос с `temperature`, `max_tokens` и JSON-режимом; если
  модель отклоняет параметр (HTTP 400: рассуждающим моделям нужен
  `max_completion_tokens`, `temperature` — только по умолчанию; у некоторых
  совместимых сервисов нет JSON-режима), запрос повторяется без него, и до
  конца задачи этот параметр больше не отправляется.

**Рекомендации:**
- Увеличьте `min_score_threshold` до 0.65 для более строгого отбора
- Для боевого теста на Intel Arc B580 оставьте `provider: llama_cpp`, `n_gpu_layers: -1`
- Убедитесь, что `HF_TOKEN` экспортирован и вы приняли условия модели на Hugging Face
- Не хардкодьте `model_path` под домашнюю директорию конкретной машины — используйте
  путь внутри проекта (`artifacts/cache/models/...`). При отсутствии файла адаптер
  скачает модель по `model_repo`/`model_file` и сам создаст симлинк по `model_path`,
  так что повторные запуски (в т.ч. на другой машине с тем же чекаутом) переиспользуют
  уже скачанный файл вместо повторного скачивания
- `HF_HOME`/`TRANSFORMERS_CACHE` по умолчанию указывают на
  `artifacts/cache/...` и для `just process` (см. `justfile`), и внутри Docker-образа —
  кэш моделей общий между локальным запуском и контейнером, если смонтирован один и тот же `artifacts/` том

---

### 4. Секция `cropping` — Настройки кропа и детекции лица

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `output_width` | int | 1080 | Ширина выходного видео (вертикальное 9:16) |
| `output_height` | int | 1920 | Высота выходного видео |
| `face_confidence_threshold` | float | 0.6 | Порог уверенности детекции лица |
| `moving_average_window` | int | 5 | Окно сглаживания координат (moving average) |
| `sample_fps` | int | 1 | Частота сэмплирования кадров для детекции |
| `shot_size_ratio_threshold` | float | 1.35 | Скачок размера бокса лица больше этого — смена крупности/склейка, новый сегмент кропа |
| `shot_center_jump_threshold` | float | 0.5 | Скачок центра лица (в долях ширины бокса) — склейка/панорама |
| `min_track_segment_sec` | float | 2.0 | Более короткие сегменты (пропуск детекции, случайный бокс) поглощаются соседом |
| `scene_cut_threshold` | float | 10 | Порог детектора склеек ffmpeg `scdet` (0–100; 0 — выключить). Граница сегмента кропа переносится точно на склейку, иначе она попадает на ближайший сэмпл (до секунды позже), и на доли секунды виден кроп предыдущего плана. Склейки ищутся на 12,5 кадрах в секунду (точность — кадр) |
| `frame_decoder` | str | "auto" | Чем декодировать видео для детектора лиц: `auto` — GPU через VAAPI (Intel, AMD), если пробное декодирование этого видео прошло, иначе CPU; `software` — всегда CPU; `vaapi` — GPU, при сбое на клипе — CPU. Кадры для детектора сохраняются шириной 640 px. На Arc этап кропа вдвое быстрее (42 с вместо 83 с на 12 клипах эталонной лекции, на CPU — 61 с), сегменты кропа те же (границы в пределах 0,04 с, центр — 6 px); NVIDIA пока декодирует на CPU |
| `min_reframe_share` | float | 0.25 | Без склейки (спикер сдвинулся внутри плана) окно кропа переносится, только если лицо сместилось больше чем на эту долю ширины окна; иначе окно не дёргается |

Сегменты кропа строятся по **сырым** детекциям (сглаживание смазывало склейку и
сдвигало границу на ~2 с); окно каждого сегмента — полновысотная полоса 9:16 по
медианному центру лица. Первый сегмент всегда начинается со старта клипа.

**Детектор лиц** — YuNet (OpenCV, лицензия MIT), файл
`artifacts/cache/models/face_detection_yunet_2023mar.onnx` (~230 КБ, закреплён в
`config/models.lock.yaml`, скачивается `just prefetch-models`). Работает на CPU:
кадр уменьшается до 640 px по ширине, ~30 мс на кадр даже на старом Xeon, так
что видеокарта не нужна и VRAM он не занимает.

**Рекомендации:**
- Уменьшите `face_confidence_threshold` до 0.4 если лицо часто теряется

---

### 5. Секция `rendering` — Настройки рендеринга

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `video_codec` | str | "h264" | Видео кодек (h264, hevc, av1, copy; внутри маппится на ffmpeg encoder) |
| `video_preset` | str | "medium" | Пресет кодирования: ultrafast, fast, medium, slow, veryslow (для libx264 и Quick Sync) |
| `video_encoder` | str | "auto" | Чем кодировать клипы: `auto` — кодировщиком GPU, если пробное кодирование прошло, иначе libx264; `software` — всегда libx264 на CPU; `qsv` / `vaapi` / `nvenc` — только этим кодировщиком, а если он не работает, задача падает с причиной. См. «Кодирование на GPU» ниже |
| `audio_loudnorm_i` | int | -14 | Integrated loudness target (LUFS) |
| `audio_loudnorm_tp` | float | -1.5 | True peak limit (dBTP) |
| `audio_loudnorm_lra` | int | 11 | Loudness range |
| `padding_color` | str | "black" | Цвет подложки при паддинге |
| `subtitle_format` | str | "srt" | Файл субтитров рядом с клипом: srt или ass. В видео субтитры всегда прожигаются из сгенерированного ASS, чтобы применялся стиль |
| `subtitle_style` | str | "typewriter" | Пресет стиля: `typewriter` (по умолчанию) или старые `default`, `modern`, `minimal` (управляются ключами `subtitle_font`, `subtitle_fontsize`, `subtitle_*_color`) |
| `subtitle_style_overrides` | dict | {} | Поля поверх пресета `typewriter`, например `{box_opacity: 0.6, font_size: 60}`; неизвестный ключ — ошибка валидации |
| `subtitle_fonts_dir` | str | null | Доп. каталог шрифтов для libass (шрифты вне fontconfig) |
| `subtitle_font` | str | "Arial" | Шрифт субтитров |
| `subtitle_fontsize` | int | 48 | Размер шрифта |
| `subtitle_primary_color` | str | "&H00FFFFFF" | Основной цвет субтитров (ASS) |
| `subtitle_outline_color` | str | "&H00000000" | Цвет обводки (ASS) |
| `subtitle_back_color` | str | "&H80000000" | Цвет фона (ASS) |
| `render_timeout_sec` | int | 300 | Таймаут рендеринга одного клипа в секундах |

**Стиль субтитров** (`SubtitleStyle` в `src/config.py`, пресеты в `src/subtitles.py`):

| Поле | `typewriter` | Смысл |
|------|------|------|
| `font`, `font_size`, `bold` | Nimbus Mono PS, 64, да | Машинописный шрифт с кириллицей (Courier-подобный) |
| `text_color` | #FFFFFF | Цвет текста |
| `box`, `box_color`, `box_opacity`, `box_padding` | да, #4D4D4D, 0.75, 14 | Одна подложка на реплику (libass BorderStyle 4) |
| `outline`, `outline_color`, `shadow` | 0, #000000, 0 | Обводка и тень текста (при `box: false`) |
| `anchor` | bottom | Якорь блока: bottom, center, top |
| `margin_v`, `margin_h` | 0.25, 0.13 | Отступы в долях кадра: от якорного края и слева/справа. Держат текст вне зон интерфейса TikTok/Reels/Shorts (подпись и кнопки снизу и справа) |
| `max_lines`, `max_chars_per_line` | 3, 26 | Реплика — не больше 3 строк по 26 символов; длинный сегмент делится на несколько реплик |

Задел на выбор стиля: джоба может передать `rendering.subtitle_style` и
`rendering.subtitle_style_overrides` в `config_overrides` (секция `rendering` разрешена),
новые пресеты добавляются в `PRESETS` в `src/subtitles.py`.

**Шрифт:** Nimbus Mono PS входит в URW base35 (Debian/Ubuntu: `fonts-urw-base35`, есть в
Docker-образе). Если шрифт не найден, fontconfig подставит другой моноширинный. Перед
нативной установкой `scripts/check_system_deps.sh` проверяет, видит ли fontconfig
«Nimbus Mono PS» (`fc-list "Nimbus Mono PS"`), и предупреждает с командой установки
пакета (`fonts-urw-base35` / `gsfonts` / `urw-base35-fonts`), если не видит. Шрифты вне
fontconfig можно подключить через `subtitle_fonts_dir`.

**Кодирование на GPU** (`video_encoder: auto`). Обрезка, масштаб и субтитры
считаются на CPU, на GPU уходит только кодировщик: с libx264 он занимал большую часть
времени рендера. Перед первым клипом задачи рендер пробует закодировать полсекунды
чёрного кадра и берёт первый кодировщик, который справился:

1. NVENC (`h264_nvenc`) — если бэкенд `cuda`;
2. Quick Sync (`h264_qsv`) и VAAPI (`h264_vaapi`, по очереди на каждом `/dev/dri/renderD*`) —
   если в системе есть GPU-узлы `/dev/dri` (Intel; VAAPI работает и на AMD через Mesa);
3. libx264 на CPU.

Какой кодировщик выбран, видно в `job.log` (`video_encoder`), в `clip_*.meta.json` (`encoder`) и
на странице «Диагностика» (строка «Video encoding»). Если на машине с GPU-бэкендом ни один
кодировщик GPU не заработал, клипы кодируются на CPU, а страница задачи показывает
уведомление с причиной. Если кодировщик GPU прошёл пробу, но упал на настоящем клипе, этот
и остальные клипы задачи доделываются на CPU, тоже с уведомлением. Качество кодировщиков GPU
подобрано так, чтобы клипы не отличались от libx264 `-crf 23` (замер на эталонной лекции —
в CHANGELOG). `video_codec: h265` / `av1` выбирает `hevc_*` / `av1_*` тех же кодировщиков.

Что нужно: Docker-образ `openvino` уже содержит драйвер VA-API и среду Quick Sync, образ
`cuda` получает NVENC от NVIDIA Container Toolkit. При установке без Docker на Intel нужны
пакеты `intel-media-va-driver` и `libmfx-gen1.2` (Debian/Ubuntu; в Arch — `intel-media-driver`
и `vpl-gpu-rt`), на NVIDIA — драйвер с NVENC.

**Рендер продолжается по клипам.** ffmpeg пишет `clip_NNN.part.mp4` и переименовывает его, когда
клип готов; `clip_NNN.meta.json` записывается последним и хранит отпечаток входа клипа
(`render_key`: границы, кроп, субтитры, настройки рендера). Если задачу остановили посреди
рендера, при продолжении готовые клипы с тем же отпечатком не рендерятся заново. Смена стиля
субтитров или границ клипа меняет отпечаток, и такой клип рендерится снова; смена
кодировщика, таймаута или `subtitle_format` — нет. `--force` рендерит все клипы заново.

**Рекомендации:**
- Увеличьте `render_timeout_sec` для длинных клипов (с libx264 на слабом CPU — как в профиле `intel_arc`)

---

### 5.1. Секция `chapters` — Главы (draft-режим)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `enabled` | bool | false | Включить генерацию глав в draft-пайплайне (`run_chapters_pipeline`) |
| `min_minutes` | int | 5 | Минимальная длительность главы в минутах |
| `max_minutes` | int | 7 | Максимальная длительность главы в минутах |
| `target_count` | int | null | Точное число глав; `null` — на усмотрение модели (ориентир — `min_minutes`/`max_minutes`) |
| `prompt_file` | str | "config/prompts/chapter_detection_v1.txt" | Промпт детекции глав |
| `output_format` | str | "both" | Формат вывода: json, youtube, both |

---

### 5.2. Секция `broll` — Подсказки по B-roll

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `enabled` | bool | false | Включить генерацию advisory-подсказок по B-roll в draft-пайплайне |
| `max_suggestions` | int | null | Ограничение числа подсказок; `null` — на усмотрение модели |
| `prompt_file` | str | "config/prompts/broll_suggestions_v1.txt" | Промпт подсказок по B-roll |

---

### 6. Секция `output` — Настройки выходных данных

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `clips_dir` | str | "output/clips" | Директория для сохранения клипов |
| `manifest_file` | str | "output/manifest.json" | Путь к файлу манифеста |
| `review_file` | str | "artifacts/review.json" | Путь к файлу review.json |
| `isolated` | bool | true | Создавать отдельную директорию для каждого видео |
| `output_dir_template` | str | "{video_name}_{timestamp}" | Шаблон имени изолированной директории |
| `timestamp_format` | str | "%Y-%m-%d_%H-%M-%S" | Формет временной метки |

**Рекомендации:**
- `isolated: true` гарантирует, что несколько запусков не перезаписывают артефакты друг друга

---

### 7. Секция `cleanup` — Очистка

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `retention_days` | int | 30 | Через сколько дней после завершения задачи удалить её исходное видео (загрузку, копию в `artifacts/video` и аудио); клипы, артефакты и карточка остаются, продолжить задачу уже нельзя. `0` — не удалять |
| `max_upload_size_gb` | int | 50 | Только предупреждение в логе, если `uploads/` больше |

Очистку выполняет API: при старте и раз в 6 часов (`src/api/services/cleanup.py`), вручную —
`DELETE /api/v1/cleanup/`. Загрузки, на которые не ссылается ни одна задача, удаляются через
сутки. Задача целиком удаляется только кнопкой «Delete» (`POST /api/v1/jobs/{id}/delete`).

### 8. Секция `logging` — Логирование

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `level` | str | "INFO" | Уровень логирования |
| `log_file` | str | "artifacts/logs/pipeline.log" | Путь к файлу логов |
| `json_format` | bool | true | Структурированный JSON-формат логов |

Джобы, запущенные через API/веб (их выполняет воркер задач `src/worker.py`),
дополнительно пишут полный JSON-лог в `<work_dir>/artifacts/logs/job.log`
(только записи этой джобы).

### 9. Секция `gpu` — GPU backend

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `backend` | str | "auto" | Автоопределение: auto, cuda, openvino, rocm, cpu |
| `whisper_device_override` | str | null | Принудительное устройство Whisper |
| `whisper_compute_override` | str | null | Принудительный compute_type Whisper |

**Очистка кэша:**
```bash
just clean-cache  # Полная очистка
```

---

## 🔧 Профили конфигурации

Готовые профили лежат в `config/profiles/`: `local_llm.yaml`, `intel_arc.yaml`,
`low_vram.yaml`. Профиль применяется флагом `--profile` в CLI или ключом
`profile` в настройках джобы; на аппаратный профиль (`config/hardware/`) он
накладывается сверху. Три примера в конце раздела — шаблоны для собственных
профилей, их в репозитории нет.

### Профиль "Локальная LLM" (`config/profiles/local_llm.yaml`)

```yaml
scoring:
  llm:
    enabled: true
    provider: "llama_cpp"
    model_repo: "google/gemma-4-E4B-it-qat-q4_0-gguf"
    model_file: "gemma-4-E4B_q4_0-it.gguf"
    n_ctx: 4096
    n_gpu_layers: -1
```

Использование:
```bash
just process video.mp4 config/config.yaml False False False local_llm
```

### Профиль "Intel Arc (Vulkan) + CPU без AVX2" (`config/profiles/intel_arc.yaml`)

Только настройки железа для нативной установки через `./install.sh openvino`:
транскрибация на GPU через whisper.cpp/Vulkan и увеличенный таймаут рендера на
медленном CPU. Язык, стратегия скоринга и число кандидатов — из `config.yaml`.

```yaml
transcription:
  engine: "whisper_cpp"
  whisper_cpp_binary: "artifacts/cache/whisper-cpp-bin/whisper-cli"
  whisper_cpp_model_path: "artifacts/cache/models/ggml-large-v3-turbo.bin"
rendering:
  render_timeout_sec: 900
```

Использование: `--profile intel_arc` в CLI
(`just process video.mp4 config/config.yaml False False False intel_arc`)
или профиль `intel_arc` в форме создания джобы в веб-интерфейсе.

### Профиль "Минимум VRAM" (`config/profiles/low_vram.yaml`)

Для карт с ~8 ГБ VRAM: модель Whisper меньше, int8, меньше клипов:

```yaml
transcription:
  model: "small"
  compute_type: "int8"
  batch_size: 2

scoring:
  max_clips_per_video: 3

gpu:
  backend: "auto"
  whisper_compute_override: "int8"
```

### Закреплённые модели (`config/models.lock.yaml`)

Аналог `uv.lock` для весов моделей. Для каждой модели зафиксирован коммит
репозитория на Hugging Face; для одиночных файлов (GGUF, ggml) — ещё sha256
и размер.

- Скачивание (LLM, faster-whisper, `just prefetch-models`) идёт с
  закреплённого коммита, скачанный GGUF/ggml проверяется по sha256.
- `just prefetch-models` проверяет sha256 уже лежащих файлов один раз и пишет
  рядом `<файл>.sha256`; повторные запуски проверяют только размер и дату.
- Перед загрузкой GGUF в пайплайне сверяется только размер: несовпадение даёт
  предупреждение `model_size_mismatch` в логе.
- Модель, которой нет в файле, работает как раньше: последняя версия из
  репозитория, без проверки.

Сменить версию модели: взять новый коммит и sha256/размер файла со страницы
Files на huggingface.co, обновить запись и запустить `just prefetch-models`.

### Шаблон: быстрый старт

```yaml
transcription:
  model: "small"
  batch_size: 4
  compute_type: "float16"

rendering:
  video_preset: "fast"
```

### Шаблон: максимальное качество

```yaml
preprocessing:
  video_crf: 18

transcription:
  model: "large-v3-turbo"
  batch_size: 8
  compute_type: "float16"

scoring:
  min_score_threshold: 0.65
  term_extraction_method: "keybert"

cropping:
  face_confidence_threshold: 0.7

rendering:
  video_preset: "slow"
```

### Шаблон: batch-обработка (множество видео)

```yaml
scoring:
  max_clips_per_video: 3

rendering:
  video_preset: "medium"
  render_timeout_sec: 600
```

---

## 💾 Настройки пользователя (страница Settings)

В веб-интерфейсе есть страница **Settings** (`/config`). Раньше её редактор
перезаписывал сам `config/config.yaml` (теряя комментарии и мешая `git pull`);
теперь изменения пишутся только в отдельный файл **`data/settings.yaml`** —
то, что пользователь изменил относительно значений по умолчанию, вложенным
словарём, например:

```yaml
scoring:
  max_clips_per_video: 8
```

- Путь к файлу — переменная окружения `PIPELINE_USER_SETTINGS`, по умолчанию
  `data/settings.yaml` (относительно рабочего каталога: `/app` в Docker, корень
  проекта при нативной установке). `data/` bind-mount'ится и не отслеживается
  git. Файл создаётся атомарно (temp + rename) с правами `0600` — в нём может
  лежать API-ключ (`scoring.llm.api_key`).
- Отсутствующий или пустой файл означает «настроек пользователя нет». Битый
  или нечитаемый файл не останавливает API и воркер: они логируют
  предупреждение и используют базовый конфиг, а страница Settings показывает
  причину жёлтым баннером (поле `error` в `GET /config/`).
- Разрешены только те секции, которые может переопределять и задача
  (`ALLOWED_OVERRIDE_SECTIONS` в `src/user_settings.py`, общий список с
  `src/api/job_config.py`): `input`, `preprocessing`, `transcription`,
  `scoring`, `cropping`, `rendering`, `chapters`, `broll`, `cleanup`,
  `output`, `logging`. Опасные ключи (`work_dir`, `gpu`, `profiles_dir`)
  закрыты.

### Порядок слоёв конфигурации

Эффективный конфиг, с которым запускается задача, собирается так (одна функция
`src.user_settings.load_app_config` для API и воркера):

1. `config/config.yaml` — базовый конфиг;
2. аппаратный профиль (`with_hardware_profile`, выбран `PIPELINE_HARDWARE`);
3. настройки пользователя (`data/settings.yaml`);
4. профиль задачи (`profile` в настройках джобы);
5. переопределения задачи (JSON-поле `config_overrides`).

**CLI (`src/cli.py`) сознательно в этой цепочке не участвует**: он читает только
тот файл, который ему передали, поэтому прогон из командной строки
воспроизводим и не зависит от настроек, изменённых в веб-интерфейсе.

### Как сбросить настройки

- На странице Settings — кнопка **Reset** у поля (вернуть одно значение по
  умолчанию) или **Reset all** (сбросить всё).
- Вручную: удалить файл `data/settings.yaml` (или `POST /api/v1/config/reset`
  с пустым списком ключей). Отсутствующий файл = настройки по умолчанию.

---

## 🧪 Валидация конфигурации

Перед запуском проверьте корректность конфига:

```bash
just validate-config
```

Или напрямую:

```bash
python -m src.cli validate-config --config config/config.yaml
```

---

## 📝 Переопределение параметров через CLI

Некоторые параметры можно переопределить через командную строку:

```bash
# Полный прогон
just process video.mp4

# С кастомным конфигом
just process video.mp4 config/custom.yaml

# Dry-run (проверка без выполнения)
just process video.mp4 config/config.yaml True

# Resume (пропуск валидных этапов)
just process video.mp4 config/config.yaml False True

# Force (перезаписать всё)
just process video.mp4 config/config.yaml False False True
```

### Описание CLI-флагов

| Флаг | Описание |
|------|----------|
| `input` | Путь к видеофайлу или YouTube URL (обязательный) |
| `config` | Путь к YAML-конфигу (по умолчанию: `config/config.yaml`) |
| `dry_run` | Проверка конфигурации без выполнения пайплайна |
| `resume` | Пропускать этапы, если их output-артефакт существует и валиден |
| `force` | Игнорировать существующие артефакты, перезаписать всё |

### Приоритет флагов

- `force=True` имеет приоритет над `resume=True` (перезаписывает всё)
- `dry_run=True` отменяет выполнение всех этапов, только валидация

---

## 🆘 Troubleshooting конфигурации

### Проблема: Конфиг не валидируется

**Решение:**
1. Проверьте синтаксис YAML (отступы, двоеточия)
2. Убедитесь, что все обязательные секции присутствуют
3. Запустите `just validate-config` для детальной ошибки

### Проблема: OOM ошибки на GPU

**Решение:**
1. Уменьшите `batch_size` в секции `transcription`
2. Используйте `large-v3-turbo` вместо `large-v3`
3. Добавьте `compute_type: float16`

### Проблема: Слишком много/мало клипов

**Решение:**
1. Увеличьте/уменьшите `min_score_threshold`
2. Измените `max_clips_per_video`

---

## 📈 Мониторинг и логи

Логи содержат информацию о применённых настройках:

```json
{
  "level": "info",
  "event": "Pipeline started",
  "config_hash": "sha256:...",
  "timestamp": "2026-04-16T10:00:00Z"
}
```

Для просмотра в реальном времени:

```bash
tail -f artifacts/logs/pipeline.log | jq .
```

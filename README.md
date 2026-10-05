# Clips Pipeline

[![CI](https://github.com/ivbor-24/clips-pipeline/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/ivbor-24/clips-pipeline/actions/workflows/test-gpu.yml)

Длинные лекции, подкасты и интервью → короткие вертикальные клипы 9:16 с
субтитрами для Shorts, Reels и TikTok. Всё работает на вашем компьютере:
распознавание речи, выбор клипов языковой моделью, кадрирование по лицу
спикера, кодирование на GPU.

*Turns long lectures and podcasts into vertical 9:16 clips with subtitles,
locally: speech recognition, a local LLM that picks clips from a hook to a
finished thought, face-tracking crop and GPU encoding. The documentation is in
Russian.*

## Как это работает

1. **Распознавание речи** — whisper.cpp (Intel Arc, через Vulkan) или
   faster-whisper (NVIDIA, CPU), с таймкодами.
2. **Выбор клипов** — языковая модель (по умолчанию Qwen3-14B через llama.cpp,
   локально; можно подключить API-провайдера) читает расшифровку окнами и
   выбирает фрагменты от хука в первых секундах до законченной мысли, 45–90 с.
   На видео — 12 кандидатов, из которых вы оставляете лучшие.
3. **Кадрирование 9:16** — детектор лиц YuNet ведёт кадр за спикером, окно
   переключается ровно на склейках.
4. **Рендер** — субтитры в безопасной зоне платформ, нормализация звука,
   кодирование на GPU (Quick Sync, NVENC, VAAPI) или libx264.
5. **Отбор** — в веб-интерфейсе: «годен / нужна правка / брак», скачивание MP4 и
   SRT.

Пример: 95-минутная лекция на Intel Arc B580 обрабатывается целиком примерно за
20 минут.

## Требования

- Linux (официально Ubuntu; проверено также на CachyOS), Docker с Compose v2 и
  buildx.
- Видеокарта:
  - **Intel Arc** (Vulkan) — проверено на B580;
  - **NVIDIA** (CUDA, драйвер 525+, NVIDIA Container Toolkit) — образ
    собирается, на реальной карте не проверялся;
  - **без GPU** — работает, но локальная языковая модель очень медленная:
    лучше подключить API-провайдера.
- ~10 ГБ видеопамяти для Qwen3-14B, ~25 ГБ на диске, ~16 ГБ оперативной памяти
  на время первой сборки.

Что проверено, а что нет — [матрица поддержки](docs/ADMIN_GUIDE.md#10-матрица-поддержки).

## Установка

```bash
git clone https://github.com/ivbor-24/clips-pipeline.git && cd clips-pipeline
./setup.sh
```

`setup.sh` проверяет Docker, находит видеокарту, пишет `.env`, собирает образы
(15–40 минут в первый раз: llama.cpp компилируется под ваш процессор и GPU),
скачивает модели (~11 ГБ) и запускает веб-интерфейс. Подробности и разбор
ошибок — [docs/INSTALL.md](docs/INSTALL.md).

```bash
just up                  # запустить: http://127.0.0.1:8080
just stop                # остановить; задача в работе продолжится при следующем запуске
just status              # что запущено
git pull && ./setup.sh   # обновить; данные и настройки сохраняются
```

Без `just` то же самое делает `scripts/stack.sh up|stop|status`.

## Первая задача

Откройте http://127.0.0.1:8080, нажмите **New Job**, загрузите видео или
вставьте ссылку и дождитесь клипов. На странице **Review** отметьте, какие
клипы годятся, и скачайте их. Подробно — [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Документация

- [docs/INSTALL.md](docs/INSTALL.md) — установка, обновление, бэкап, «если что-то не так», установка без Docker
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — работа в веб-интерфейсе
- [docs/ADMIN_GUIDE.md](docs/ADMIN_GUIDE.md) — обслуживание, доступ из сети, данные, матрица поддержки
- [docs/CONFIG.md](docs/CONFIG.md) — все настройки пайплайна
- [docs/EVALUATION.md](docs/EVALUATION.md) — оценка качества клипов по своему эталонному набору

## Разработка

```bash
./install.sh openvino    # или cpu, cuda: окружение (uv), сборка llama.cpp, самопроверка GPU
just dev                 # API, воркер и веб-интерфейс разработки (http://127.0.0.1:5173)
just ci                  # lint (ruff, black, shellcheck, лицензии), uv.lock, все тесты
just smoke               # установка с нуля в Docker и одна задача (в свежем клоне)
```

- **Код:**
  - `src/` — этапы пайплайна (`ingestion`, `transcription`, `scoring` и
    `window_scoring`, `face_cropping`, `rendering`), `src/api/` — API на
    FastAPI, `src/worker.py` — воркер задач;
  - `web/` — веб-интерфейс (React, Vite);
  - `scripts/` — установка и служебные скрипты;
  - `config/` — настройки по умолчанию, профили, промпты.
- **Этапы общаются только через файлы задачи** (`artifacts/*.json`), поэтому
  остановленная задача продолжается с последнего готового этапа.
- **Стиль:** black (100 символов), ruff, аннотации типов, логи structlog с
  параметрами, а не f-строками, конфигурация — модели pydantic. Схема базы —
  только миграциями Alembic (`just db-revision "что меняется"`).
- **Новая зависимость** — строка с её лицензией в
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); lint это проверяет.
- **Выпуск версии и публикация образов** — [docs/RELEASING.md](docs/RELEASING.md).
- **CI** на каждый pull request: lint, все тесты и установка с нуля в Docker
  с одной задачей; при изменении сборки образа — ещё сборка образа для NVIDIA.
  Локально то же самое: `just ci`, `just smoke`, `just check-cuda-image`.

## Лицензия

[MIT](LICENSE). Присланные изменения принимаются под той же лицензией.

Сторонние компоненты (модели, зависимости, программы в Docker-образе) и их
лицензии перечислены в [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
Модели скачиваются у их авторов и распространяются на их условиях (Qwen3 —
Apache-2.0, whisper и YuNet — MIT).

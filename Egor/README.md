# Legal-ASR

Микросервисная система распознавания речи и идентификации спикеров в юридических консультациях. Работает в режиме, близком к реальному времени: каждый сегмент речи отправляется в обработку сразу после того, как Voice Activity Detection (VAD) обнаружил окончание реплики.

На вход поступает аудиопоток (mp3-файл, web-вход — в разработке), на выход — текст реплик с указанием говорящего и временным диапазоном вида:

```
[Lawyer] (chunk_3.4s_11.2s): добрый день, расскажите, что у вас случилось
[Client] (chunk_11.5s_18.0s): хочу расторгнуть договор долевого участия
```

Система передает транскрибцию для дальнейшей LLM-обработки: суммаризация, выделение фактов, создание подсказок юристу, предоставление релевантной информации из статей, кодексов. На текущем этапе реализованы и измерены ASR + диаризация по эталонным голосам.

---

## Архитектура

Три независимых сервиса общаются через Redis-очереди:

```
                       ┌────────────────────────┐
   audio source ──────►│       gateway          │── tasks:asr ─────►   asr_worker     ── results:asr ─────┐
   (mp3-файл)          │ Silero VAD + scheduler │                                                          │
                       │                        │── tasks:speaker ─►  speaker_worker  ── results:speaker ──┤
                       │  control:speaker  ────►│                                                          │
                       └────────────────────────┘                                                          │
                                  ▲                                                                        │
                                  │              OrderedTranscriptWriter ◄───────────────────────────────-─┘
                                  │                          │
                                  │                          ▼
                                  │             data/output/<id>_transcript.txt
                                  │             data/output/<id>_timings.json
                                  │             data/output/<id>_metrics.json
```

| Компонент | Назначение | Модель |
|-----------|------------|--------|
| **gateway** | Читает аудио, режет на речевые сегменты VAD'ом, отправляет чанки воркерам, собирает результаты в порядке seq_num | Silero VAD v5 |
| **asr_worker** | Распознаёт текст из аудио-чанка | GigaAM-v3 E2E CTC (русский) |
| **speaker_worker** | Сравнивает эмбеддинг чанка с эталонными голосами (cosine similarity), возвращает имя или `Unknown` | Pyannote `pyannote/embedding` |
| **webgateway** | FastAPI + WebSocket UI для записи эталонов и live-консультации (**в разработке**) | — |
| **redis** | Брокер очередей задач/результатов и канал управления | Redis 7 |

VAD работает в двух режимах: **batched** (для оффлайн-прогона по файлу — буферизуем всё, один проход `get_speech_timestamps`) и **streaming** (`VADIterator`, выдаёт сегменты по мере появления тишины — основной режим для оценки латентности и для будущей live-сессии).

Эталонные голоса — это `data/input/<consultation>/voices/<имя>.mp3`. При запуске симулятора `gateway` шлёт сообщение `control:speaker` с путём к нужной папке голосов, и speaker_worker пересоздаёт базу эмбеддингов под конкретную консультацию.

---

## Структура проекта

```
legal-asr/
├── docker-compose.yml           # базовый стек: redis + три воркера на CPU
├── docker-compose.override.yml  # dev-overlay: bind-mount исходников и кэша моделей
├── docker-compose.prod.yml      # prod-overlay: добавляет GPU-резервирование
├── README.md                    # этот файл
│
├── asr_service/                 # ASR-микросервис
│   ├── Dockerfile               # python:3.10-slim + torch + GigaAM
│   ├── requirements.txt
│   └── worker.py                # подписан на tasks:asr, кладёт в results:asr
│
├── speaker_service/             # Speaker ID микросервис
│   ├── Dockerfile               # python:3.10-slim + torch + pyannote.audio
│   ├── requirements.txt
│   └── worker.py                # подписан на tasks:speaker и control:speaker
│
├── gateway/                     # Логика подачи аудио в Redis
│   ├── audio_source.py          # Protocol: что должен уметь источник аудио (frames(), close())
│   ├── sources/
│   │   └── file_source.py       # FileAudioSource: читает mp3 через pydub, выдаёт PCM-кадры с заданным realtime-темпом
│   ├── vad.py                   # SileroSegmenter: batched и streaming режимы, нарезка на сегменты
│   ├── pipeline.py              # StreamingPipeline: 3 потока — продюсер сегментов, sender в Redis, collector результатов
│   ├── result_collector.py      # OrderedTranscriptWriter: буфер результатов, выдача строк строго по seq_num + сбор метрик латентности
│   ├── gateway_simulator.py     # CLI-запуск: прогон одной консультации, сохранение транскрипта + метрик
│   └── requirements.txt
│
├── webgateway/                  # Веб-интерфейс (в разработке)
│   ├── Dockerfile
│   ├── app.py                   
│   └── requirements.txt
│
├── metrics/                     # Оценка качества прогонов
│   ├── parse.py                 # парсеры ground-truth (text.txt) и предсказаний (transcript.txt)
│   ├── compute.py               # WER, CER, speaker accuracy через jiwer-выравнивание + сводка по таймингам
│   ├── evaluate.py              # CLI и функция evaluate(): печать отчёта и сохранение в JSON
│   └── requirements.txt
│
├── data/                        # Аудио и артефакты (целиком в .gitignore)
│   ├── input/
│   │   └── <consultation>/
│   │       ├── audio.mp3        # запись консультации
│   │       ├── text.txt         # эталонная разметка `[Speaker]: текст` для расчёта WER
│   │       └── voices/
│   │           ├── Lawyer.mp3   # эталон голоса юриста
│   │           └── Client.mp3   # эталон голоса клиента
│   ├── output/                  # *_transcript.txt, *_timings.json, *_metrics.json
│   └── cache/                   # кэш весов моделей (не пересобирать при rebuild)
│
└── logs/                        # построчные логи всех сервисов
```

---

## Запуск

### Подготовка
1. Установлен Docker Desktop (WSL2 backend на Windows).
2. В корне создать файл `.env` с токеном Hugging Face (нужен для pyannote):
   ```
   HF_TOKEN=hf_***
   ```
3. Положить материалы консультации в `data/input/<имя>/`.

### Прогон по mp3-файлу (CLI-симулятор)
```bash
docker compose up -d                            # поднять стек
CONSULTATION=consultation2 \
STREAMING_VAD=true \
REALTIME_FACTOR=1.0 \
py -3.10 -m gateway.gateway_simulator           # прогон + автоматический подсчёт метрик
```
- `REALTIME_FACTOR=0.0` — максимально быстро (для замеров),
- `REALTIME_FACTOR=1.0` — имитация реального темпа речи,
- `STREAMING_VAD=true` — режим VADIterator (как в live-сессии); `false` — batched.

Результаты появятся в `data/output/consultation2_transcript.txt`, `_timings.json`, `_metrics.json`.

### Live-консультация в браузере
(в разработке)


### Production (GPU-сервер)
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Текущие результаты

Прогон `consultation2` (записанная консультация ~13 мин 20 с, 2 спикера: юрист и клиент) в режиме streaming VAD при `REALTIME_FACTOR=1.0` (аудио подаётся в темпе реальной речи), на CPU:

**Качество:**

| Метрика | Значение |
|---------|----------|
| Word Error Rate (WER) | **7.57 %** |
| Character Error Rate (CER) | **3.59 %** |
| Speaker accuracy (word-level) | **93.44 %** |
| &nbsp;&nbsp;Lawyer | 96.94 % (1439 эталонных слов) |
| &nbsp;&nbsp;Client | 75.62 % (283 эталонных слова) |

**Производительность:**

| Метрика | Значение |
|---------|----------|
| Длина аудио | 800.4 с |
| Wall-clock | 1138.4 с (RTF 1.42) |
| ASR per-chunk: mean / max | 1008 мс / 2630 мс |
| Speaker per-chunk: mean / max | 305 мс / 1056 мс |
| End-to-end latency: mean / max | **1044 мс / 2689 мс** |

End-to-end latency = время от отправки чанка в Redis до записи строки в транскрипт. На live-сессии это и есть задержка, которую видит пользователь после окончания фразы. На CPU средняя латентность ~1 с — вписывается в цель «реального времени» для разговорного темпа.

Расчёт качества — `metrics/evaluate.py`: по словам с выравниванием через `jiwer.process_words`. Speaker accuracy считается на пересечении референса и гипотезы (только слова, которые ASR распознал верно или подменил — для удалений атрибуция спикера не определена). Перекос Lawyer/Client объясняется дисбалансом речи в этой консультации (юрист говорит ~5× больше клиента) и наличием коротких реплик клиента, для которых эмбеддинг менее устойчив.

---

## Дальнейшие планы

- Доработка веб-интерфейса до production-готового состояния
- Расширение базы голосов и оценка качества на большем количестве консультаций
- Интеграция других моделей и их сравнение. Перебор параметров
- Подключение LLM-постобработки для улучшения качества транскрибации
Подключение LLM-постобработки для определения спикера в случае Unknown (?)
- Перенос на GPU-сервер с замером ускорения и production-нагрузки (?)

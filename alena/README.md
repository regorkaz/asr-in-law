# Legal ASR Service

Локальный прототип модуля потокового распознавания речи для русскоязычных юридических консультаций.

Сервис принимает аудио с микрофона или из аудиофайла, обрабатывает его в режиме, близком к онлайн-сценарию, выполняет распознавание речи с помощью T-one, определяет роль говорящего (`LAWYER`, `CLIENT`, `UNKNOWN`) по предварительно записанному голосу юриста и сохраняет итоговую расшифровку в формате `transcript.json`.

Проект разрабатывается как ASR-часть более общей системы мультиагентного помощника в юридическом домене. Следующий NLP-модуль должен использовать полученную расшифровку для анализа консультации и генерации подсказок юристу.

---

## Основные возможности

- создание отдельной сессии консультации;
- запись или загрузка эталона голоса юриста;
- потоковая передача аудио через WebSocket;
- запуск консультации с микрофона;
- загрузка аудиофайла консультации и отправка его как stream;
- распознавание речи моделью T-one;
- определение говорящего;
- сохранение промежуточных и итоговых результатов;
- веб-интерфейс для базового взаимодействия с системой;
- расчёт метрик качества и производительности.

---

## Текущая структура проекта

```text
legal_asr_service/
├── __init__.py
├── audio_utils.py
├── cli.py
├── config.py
├── metrics/
│   ├── __init__.py
│   ├── compute.py
│   ├── evaluate.py
│   └── parse.py
├── schemas.py
├── server.py
├── session.py
├── speaker_id.py
├── static/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── streaming.py
├── timing.py
└── tone_asr.py
```

### Описание файлов

| Файл | Назначение |
|---|---|
| `audio_utils.py` | Утилиты для работы с аудио: чтение файлов, преобразование PCM16/float32, ресемплинг, приведение к mono, разбиение на чанки. |
| `cli.py` | Командный интерфейс для запуска сервера, создания сессий, записи голоса юриста, загрузки enrollment-аудио, stream-подачи файла или микрофона. |
| `config.py` | Основные настройки сервиса: частота дискретизации, параметры VAD, speaker-id, T-one, директории данных и вывода. |
| `schemas.py` | Pydantic-схемы ответов API и структуры transcript-документа: сессия, сегмент, событие WebSocket. |
| `server.py` | FastAPI-приложение: HTTP endpoints, WebSocket endpoint для аудио, выдача веб-интерфейса и transcript-файлов. |
| `session.py` | Управление сессиями: создание, восстановление, enrollment юриста, обработка аудио, финализация, сохранение состояния. |
| `speaker_id.py` | Извлечение эмбеддингов говорящего через SpeechBrain и сравнение текущего фрагмента с эталоном голоса юриста. |
| `streaming.py` | Основная потоковая логика: обработка входных PCM-фреймов, WebRTC VAD, накопление ASR-чанков, вызов T-one, назначение speaker labels. |
| `timing.py` | Сбор метрик производительности: время ASR, speaker-id, total processing, segment latency, stream lag и другие timing-показатели. |
| `tone_asr.py` | Обёртка над T-one `StreamingCTCPipeline`: загрузка модели, выбор decoder-а, настройка splitter-а, обработка чанков и финализация. |
| `static/index.html` | HTML-страница веб-интерфейса: создание сессии, запись голоса юриста, запуск консультации, live transcript, скачивание результата. |
| `static/app.js` | Логика веб-интерфейса: работа с микрофоном, MediaRecorder, WebSocket, загрузкой аудиофайлов, ресемплингом и отображением сегментов. |
| `static/styles.css` | Стили веб-интерфейса. |
| `metrics/parse.py` | Парсинг эталонных расшифровок и предсказанных transcript-файлов. |
| `metrics/compute.py` | Расчёт WER, CER, speaker attribution accuracy, confusion matrix и timing summary. |
| `metrics/evaluate.py` | CLI для расчёта метрик по одной консультации или по manifest-файлу. |

---

## Директории данных и результатов

Ожидаемая структура пользовательских данных:

```text
data/
├── consultation_1.txt
├── consultation_2.txt
├── ...
└── metrics_manifest.csv
```

Результаты сессий сохраняются в директорию `output/`:

```text
output/
└── <session_id>/
    ├── session_state.json
    ├── segments.jsonl
    ├── timings.json
    └── transcript.json
```

| Файл | Назначение |
|---|---|
| `session_state.json` | Служебное состояние сессии. |
| `segments.jsonl` | Построчная запись распознанных сегментов. |
| `timings.json` | Сырые timing-данные для расчёта производительности. |
| `transcript.json` | Итоговая структурированная расшифровка консультации. |

---

## Установка

### Системные зависимости Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-dev \
  build-essential \
  ffmpeg \
  portaudio19-dev \
  libsndfile1 \
  libasound2-dev
```

### Создание виртуальной среды

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

При первом запуске будут загружены веса T-one и SpeechBrain-модель для speaker embeddings.

---

## Запуск сервера

```bash
uvicorn legal_asr_service.server:app --host 0.0.0.0 --port 8000
```

Веб-интерфейс доступен по адресу:

```text
http://localhost:8000/ui
```

Важно: для работы микрофона в браузере страницу лучше открывать именно через `localhost`, а не через `0.0.0.0`.

---

## Основные команды CLI

### Создать сессию

```bash
python -m legal_asr_service.cli create-session
```

С threshold для speaker-id:

```bash
python -m legal_asr_service.cli create-session --threshold 0.72
```

### Загрузить эталон голоса юриста

```bash
python -m legal_asr_service.cli enroll \
  --session-id <SESSION_ID> \
  --file lawyer.wav
```

### Записать эталон голоса юриста с микрофона

```bash
python -m legal_asr_service.cli record-enroll \
  --session-id <SESSION_ID> \
  --seconds 45 \
  --output lawyer_enrollment.wav
```

### Подать аудиофайл как stream

```bash
python -m legal_asr_service.cli stream-file \
  --session-id <SESSION_ID> \
  --file consultation.wav
```

С другим размером блока отправки:

```bash
python -m legal_asr_service.cli stream-file \
  --session-id <SESSION_ID> \
  --file consultation.wav \
  --block-ms 300
```

### Запустить streaming с микрофона

```bash
python -m legal_asr_service.cli stream-mic \
  --session-id <SESSION_ID>
```

### Полный demo-сценарий

```bash
python -m legal_asr_service.cli demo \
  --lawyer-file lawyer.wav \
  --consultation-file consultation.wav
```

Команда создаёт сессию, загружает голос юриста, подаёт консультацию как stream и выводит итоговый transcript.

---

## Взаимодействие через веб-интерфейс

1. Запустить сервер.
2. Открыть `http://localhost:8000/ui`.
3. Создать сессию.
4. Записать или загрузить голос юриста.
5. Запустить консультацию с микрофона или загрузить аудиофайл консультации и отправить его как stream.
6. Наблюдать live transcript.
7. После финализации скачать `transcript.json`.

---

## Расчёт метрик

### Оценка одной консультации

```bash
python -m legal_asr_service.metrics.evaluate \
  --name consultation_1 \
  --ref data/consultation_1.txt \
  --pred output/<SESSION_ID>/transcript.json \
  --timings output/<SESSION_ID>/timings.json \
  --out output/<SESSION_ID>/metrics.json
```

### Оценка нескольких консультаций через manifest

Пример `data/metrics_manifest.csv`:

```csv
name,ref,pred,timings
consultation_1,data/consultation_1.txt,output/1_metrics/transcript.json,output/1_metrics/timings.json
consultation_2,data/consultation_2.txt,output/2_metrics/transcript.json,output/2_metrics/timings.json
consultation_3,data/consultation_3.txt,output/3_metrics/transcript.json,output/3_metrics/timings.json
```

Запуск:

```bash
python -m legal_asr_service.metrics.evaluate \
  --manifest data/metrics_manifest.csv \
  --out output/metrics_summary.json
```

---

## Основные метрики

### Качество распознавания

- `WER` — word error rate;
- `CER` — character error rate.

### Определение говорящего

- `speaker_accuracy` — точность speaker attribution на уровне слов;
- `speaker_coverage` — доля слов, по которым удалось провести сравнение speaker labels;
- `per_speaker` — точность отдельно для `LAWYER` и `CLIENT`;
- `confusion_matrix` — матрица ошибок определения говорящего.

### Производительность

- `overall_rtf` — wall-clock time / audio duration;
- `asr_processing_rtf` — суммарная ASR-обработка / audio duration;
- `speaker_processing_rtf` — суммарная speaker-id обработка / audio duration;
- `total_processing_rtf` — общая вычислительная стоимость / audio duration;
- `asr_segment_s` — ASR-время на итоговый сегмент;
- `speaker_segment_s` — speaker-id время на итоговый сегмент;
- `segment_latency_s` — задержка появления сегмента после конца аудиофрагмента;
- `availability_delay_s` — задержка появления полного сегмента после начала аудиофрагмента;
- `stream_lag_s` — отставание обработки от реального времени.

---

## Переменные окружения

Некоторые параметры можно задавать через переменные окружения.

```bash
LEGAL_ASR_OUTPUT_DIR=output
LEGAL_ASR_DATA_DIR=data
LEGAL_ASR_ENABLE_SPEAKER_ID=1
LEGAL_ASR_TONE_DECODER=beam_search
LEGAL_ASR_PERSIST_INTERVAL_SEC=5
```

Для экспериментов с T-one splitter:

```bash
LEGAL_ASR_TONE_MAX_PHRASE_DURATION_MS=15000
LEGAL_ASR_TONE_MIN_SILENCE_DURATION_MS=450
LEGAL_ASR_TONE_SILENCE_THRESHOLD=0.9
```

Пример запуска с параметрами:

```bash
LEGAL_ASR_ENABLE_SPEAKER_ID=1 \
LEGAL_ASR_TONE_DECODER=beam_search \
LEGAL_ASR_PERSIST_INTERVAL_SEC=5 \
uvicorn legal_asr_service.server:app --host 0.0.0.0 --port 8000
```

---

## Формат итогового transcript

Пример сегмента в `transcript.json`:

```json
{
  "segment_id": 1,
  "start_time": 1.2,
  "end_time": 4.8,
  "speaker": "LAWYER",
  "speaker_confidence": 0.91,
  "speaker_similarity": 0.84,
  "text": "добрый день расскажите что у вас случилось",
  "asr_confidence": null,
  "metadata": {
    "assigned_by": "overlap_match"
  }
}
```

---


## Дальнейшие планы

- Улучшить постобработку распознанного текста.
- Настроить сегментацию для уменьшения задержки выдачи полезного текста.
- Повысить устойчивость определения говорящего на коротких репликах клиента.
- Сравнить T-one и GigaAM в составе общего NLP-пайплайна.
- Проверить влияние качества ASR на генерацию подсказок юристу.

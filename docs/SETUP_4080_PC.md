# Перенос OCR-бенчмарка на ПК с RTX 4080

## Требования

| Компонент | Минимум | Рекомендация |
|---|---|---|
| OS | Windows 10/11 x64 | Windows 11 |
| Python | 3.11+ (64-bit, x86_64) | 3.11.x через py launcher |
| GPU | NVIDIA RTX 4080 (compute 8.9) | — |
| NVIDIA Driver | >= 525.60 | Последний Game Ready |
| CUDA Toolkit | Не нужен (pip ставит runtime) | — |
| RAM | 16 GB | 32 GB (MinerU жрёт ~4 GB VRAM + RAM) |
| Диск | ~15 GB свободно | SSD |

## Шаг 1: Установить базовые зависимости

### Python 3.11

Скачать с https://www.python.org/downloads/ — **обязательно** отметить:
- [x] Add Python to PATH
- [x] py launcher

Проверка:
```powershell
py -3.11 --version
python -c "import platform; print(platform.architecture()[0], platform.machine())"
# Ожидаем: 64bit AMD64
```

### Tesseract OCR

Скачать installer: https://github.com/UB-Mannheim/tesseract/wiki

При установке выбрать дополнительные языки:
- [x] Russian (rus)
- [x] English (eng) — обычно включён по умолчанию

**Важно**: добавить Tesseract в PATH, или запомнить путь установки
(обычно `C:\Program Files\Tesseract-OCR`).

Проверка:
```powershell
tesseract --version
tesseract --list-langs   # должны быть eng и rus
```

### Git

Если ещё не установлен: https://git-scm.com/download/win

### NVIDIA Driver

Убедиться что стоит свежий драйвер:
```powershell
nvidia-smi
# Должен показать RTX 4080, driver >= 525
```

## Шаг 2: Клонировать репозиторий

```powershell
cd C:\Projects   # или любая папка
git clone <URL-репозитория> img_2_pdf
cd img_2_pdf
```

Или скопировать папку с текущего ПК (без `.venv_latest_*`, `artifacts/`, `outputs/`).

## Шаг 3: Проверить окружение

```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process
.\scripts\bootstrap_new_pc.ps1
```

Скрипт проверит:
1. Python 3.11 — найден ли
2. GPU — определит RTX 4080, compute 8.9, покажет что PaddlePaddle-GPU и PyTorch-CUDA поддерживаются
3. Tesseract — найден ли, есть ли rus и eng language packs
4. Репозиторий — pyproject.toml на месте
5. Тестовый PDF — если указан

Ожидаемый вывод:
```
GPU:     NVIDIA GeForce RTX 4080 (compute 8.9)
  torch GPU  : YES
  paddle GPU : YES
```

## Шаг 4: Скопировать тестовый PDF

Скопировать файл `Imaging Edge Mobile_paddleocr_uvdoc.pdf` (или любой другой тестовый PDF)
на новый ПК. Например в `C:\TestData\`.

## Шаг 5: Запустить полный бенчмарк

### Все 7 движков, без языка (baseline):
```powershell
.\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "C:\TestData\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -SampleSize 3 -Dpi 300 `
  -OutputRoot ".\artifacts\ocr_gpu_nolang"
```

### Все движки кроме surya, язык rus:
```powershell
.\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "C:\TestData\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -SampleSize 3 -Dpi 300 -Lang "rus" `
  -Engines "pytesseract","ocrmypdf","pymupdf","paddleocr","mineru","chandra" `
  -OutputRoot ".\artifacts\ocr_gpu_rus"
```

### Все движки кроме surya, язык rus+eng:
```powershell
.\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "C:\TestData\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -SampleSize 3 -Dpi 300 -Lang "rus+eng" `
  -Engines "pytesseract","ocrmypdf","pymupdf","paddleocr","mineru","chandra" `
  -OutputRoot ".\artifacts\ocr_gpu_rus_eng"
```

> Surya не принимает `--lang` — он определяет язык автоматически.
> Его запускать отдельно без `-Lang`.

### Surya отдельно:
```powershell
.\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "C:\TestData\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -SampleSize 3 -Dpi 300 `
  -Engines "surya" `
  -OutputRoot ".\artifacts\ocr_gpu_surya"
```

## Шаг 6: Сгенерировать отчёты для сравнения

```powershell
py -3.11 scripts/compare_ocr_results.py --input-root .\artifacts\ocr_gpu_nolang
py -3.11 scripts/compare_ocr_results.py --input-root .\artifacts\ocr_gpu_rus
py -3.11 scripts/compare_ocr_results.py --input-root .\artifacts\ocr_gpu_rus_eng
py -3.11 scripts/compare_ocr_results.py --input-root .\artifacts\ocr_gpu_surya
```

Отчёты появятся в `.\outputs\` — открыть `ocr_comparison_report.html` в браузере.

## Что ожидать на RTX 4080

| Движок | Backend | GPU? | Ожидаемое время (3 стр.) |
|---|---|---|---|
| pytesseract | Tesseract | CPU | ~30-40с |
| ocrmypdf | Tesseract | CPU | ~500с (делает полный PDF) |
| pymupdf | Tesseract | CPU | ~35-40с |
| paddleocr | PaddlePaddle-GPU cu118 | **GPU** | ~10-20с |
| surya | PyTorch cu121 | **GPU** | ~30-60с (было 797с на CPU) |
| mineru | PyTorch cu121 | **GPU** | ~10-20с (было 56с на CPU) |
| chandra | PyTorch cu121 | **GPU** | ~15-30с |

## Параметры скрипта (справка)

| Параметр | Описание | По умолчанию |
|---|---|---|
| `-PdfPath` | Путь к тестовому PDF | Обязательный |
| `-SampleSize` | Кол-во страниц для теста | 1 |
| `-Dpi` | Разрешение рендеринга | 160 |
| `-Lang` | Язык OCR (`rus`, `rus+eng`, `en`) | Пусто (дефолт движка) |
| `-Engines` | Список движков через запятую | Все 7 |
| `-OutputRoot` | Папка для результатов | `artifacts\ocr_latest_matrix` |
| `-TesseractPath` | Путь к папке с tesseract.exe | Авто-поиск |
| `-Recreate` | Пересоздать venv с нуля | Нет |
| `-SkipEditableInstall` | Не ставить проект в editable mode | Нет |
| `-BootstrapPython` | Команда для вызова Python | `py` |
| `-BootstrapVersion` | Версия Python для py launcher | `3.11` |

## Troubleshooting

### PaddlePaddle-GPU не ставится
```
ERROR: Could not find a version that satisfies the requirement paddlepaddle-gpu
```
Скрипт автоматически откатится на CPU-версию. Если хочешь GPU — проверь:
- Python 3.11 64-bit
- Установка идёт через китайский index (скрипт делает это сам)

### torch не видит CUDA
```python
import torch; print(torch.cuda.is_available())  # False
```
- Проверь `nvidia-smi` — драйвер видит карту?
- Скрипт ставит `cu121` — для RTX 4080 это корректно

### Tesseract не найден
Если Tesseract установлен не в стандартный путь:
```powershell
.\scripts\benchmark_ocr_matrix.ps1 -TesseractPath "D:\Tools\Tesseract" ...
```

### Ошибка "execution policy"
```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process
```
Или запускать через:
```powershell
powershell -ExecutionPolicy Bypass -Command "& .\scripts\benchmark_ocr_matrix.ps1 ..."
```

### Не хватает памяти GPU
MinerU и Surya могут потребовать много VRAM. RTX 4080 (16 GB) должно хватить.
Если нет — запускать движки по одному через `-Engines`.

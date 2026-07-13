# Сравнение с ScanTailor Advanced

Дата анализа: 2026-07-10.

Upstream: [4lex4/scantailor-advanced](https://github.com/4lex4/scantailor-advanced),
ветка `master`, commit `3d1e74e6ace413733511086934a66f4e3f7a6027` от 2020-05-31.
Проверка `origin` показала, что это текущий HEAD единственной опубликованной ветки.

ScanTailor Advanced распространяется по GPL-3.0, а UniScan — по MIT. В UniScan не копировался
исходный код ScanTailor. Ниже зафиксированы наблюдаемые функции и архитектурные идеи, которые
можно независимо реализовать на основе открытых алгоритмов и собственного дизайна.

## Краткий вывод

Это не прямые конкуренты. ScanTailor Advanced — специализированный интерактивный постпроцессор
уже полученных сканов. Захват с камеры, OCR и сборка PDF у него намеренно вне области проекта.
UniScan закрывает путь `камера/изображения/PDF → коррекция границ → очистка → единый PDF`, но пока
существенно слабее в точной доводке книжных и архивных страниц.

Главные полезные пробелы UniScan:

1. адаптивная бинаризация с выбираемым методом и настраиваемыми параметрами;
2. явные области содержимого и страницы, единые поля и выравнивание по всему документу;
3. несколько автоматических dewarp-backends с control points для коррекции выбранной модели;
4. зоны изображений и заливки для смешанных текстово-графических страниц;
5. регулируемая очистка точек с визуализацией удаляемых объектов;
6. явные проекты и профили обработки, а не только аварийное autosave-состояние;
7. кэширование стадий и ограниченная параллельная обработка страниц;
8. миниатюры, поиск статистических выбросов и сортировка проблемных страниц.

## Сопоставление возможностей

| Область | ScanTailor Advanced | UniScan | Вывод |
|---|---|---|---|
| Получение страниц | Импорт сканов; захват и сборка PDF вне scope | Камера, файлы, папки, clipboard, drag-and-drop, многостраничный PDF | UniScan функциональнее |
| Поиск документа | Не является camera document scanner | OpenCV default, optional BYOM Office Lens, ручные углы | UniScan функциональнее для фотографий |
| Развороты | Auto/manual cutters, типы страниц, адаптация разреза к разным размерам, fill offcut | Автопоиск корешка с midpoint fallback и ручные инструменты | ScanTailor точнее и лучше управляется |
| Наклон | Отдельная стадия, ручной/автоматический режим, deviation и сортировка проблемных страниц | Otsu + `minAreaRect`, без оценки доверия и статистики по документу | ScanTailor зрелее; алгоритм UniScan уязвим к рамкам и крупным изображениям |
| Кривизна книги | Автоматическая и ручная цилиндрическая модель, редактируемые верхняя/нижняя линии и depth | Offline text-line dewarp, optional UVDoc и сохранённые control points | Automatic-first подход сохранён, но пользователь может исправить ошибку модели |
| Содержимое и поля | Content box, page box, auto/manual/original margins, выравнивание, guides, физические единицы | Content-box detection, A4/Letter layout, поля и выравнивание; без guides/zones | ScanTailor точнее для сложных проектов |
| Ч/б обработка | Otsu, Sauvola, Wolf, illumination normalization, Savitzky-Golay и morphology smoothing | Otsu/Sauvola/Wolf, selectable parameters, optional illumination correction | ScanTailor всё ещё богаче для архивной доводки |
| Шум | Регулируемый despeckle и отдельная визуализация удаленных точек | Safe component despeckle + NLMeans, без отдельного removed-speck preview | ScanTailor удобнее для ручного контроля |
| Смешанные страницы | B/W, color/grayscale, mixed; picture/fill zones; foreground/background split | Общий стиль применяется ко всей странице | Крупный пробел, но не P0 для обычных camera scans |
| Цвет | Color segmentation, posterization, отдельные параметры RGB | Контраст, яркость, grayscale, sharpen | ScanTailor богаче для DjVu/архивной подготовки |
| Пакетные настройки | Параметры на страницу и применение к выбранным страницам/диапазонам | Apply to all или selected в части GUI-команд | ScanTailor точнее |
| Проекты | Явное чтение/запись проекта со всеми стадиями, recent projects, default profiles | Crash-safe autosave/restore одной текущей сессии | Нужны New/Open/Save As и профили |
| Миниатюры | Многоколоночный undockable view, выбор качества/размера, отдельные caches, problem marks | Текстовый список; thumbnail уже хранится на диске, но не показывается | Можно улучшить без новых вычислений |
| Производительность | `QThreadPool` с регулируемым числом потоков, очередь отменяемых задач, stage caches | Bounded persistent stage cache, но один background job и последовательные страницы | ScanTailor лучше масштабируется по страницам |
| Экспорт | TIFF с выбором compression, отдельные foreground/background outputs; PDF вне scope | Atomic PDF и набор изображений PNG/JPEG/WebP/TIFF | UniScan удобнее как конечный PDF pipeline |
| Проверяемость | C++ unit tests, но upstream заморожен на Qt 5 и commit 2020 года | CI, growing regression suite, coverage gates, quality corpora, runtime diagnostics, portable build | UniScan современнее как поддерживаемый продукт |

## Что в ScanTailor сделано потенциально быстрее

Фактический общий benchmark не запускался: в текущем окружении отсутствуют CMake, Qt 5 и Boost,
а проекты выполняют разные наборы операций. Поэтому ниже — вывод из архитектуры, а не измеренные
цифры.

- Batch processing распределяет независимые страницы по `QThreadPool`; число потоков можно менять
  во время работы. UniScan сейчас последовательно проходит страницы внутри одного фонового job.
- Для каждой стадии есть `CacheDrivenTask`, поэтому переключение страниц и перестроение миниатюр не
  обязано повторять весь pipeline.
- Sauvola и Wolf считают локальные mean/variance через integral images за O(width × height), а
  бинарное изображение хранится плотно по битам.
- Thumbnail caches разделены по качеству и обновляются независимо от полноразмерного output.

При этом нельзя делать вывод «C++ всегда быстрее Python»: тяжелые операции UniScan выполняются
в нативных OpenCV и ONNX Runtime. Для одного camera frame detector UniScan решает более сложную
задачу, которой у ScanTailor нет. Реальный выигрыш ScanTailor ожидается прежде всего на повторной
интерактивной обработке и больших пакетах страниц.

## Что не стоит переносить сейчас

- Полный шестистадийный интерфейс усложнит короткий camera-to-PDF сценарий UniScan.
- Foreground/background split, color segmenter и DjVu-ориентированная posterization нужны только
  после появления реальных пользовательских примеров.
- Собственный raster engine, bit-packed image type и C++ task framework не нужны: OpenCV, NumPy и
  стандартный executor покрывают задачу дешевле.
- Портирование Qt UI или GPL-классов в MIT-репозиторий недопустимо. Реализуются только функции и
  опубликованные алгоритмы собственным кодом.

## Приоритет реализации

### P0 — удобство и быстрые победы

- Показать уже существующие disk-backed thumbnails вместо текстового списка.
- Добавить явные New / Open session / Save As и профиль обработки.
- Добавить cache key для processed preview: `page revision + settings`, чтобы изменение выбора не
  повторяло одинаковую обработку.
- Ввести метрики времени по стадиям и benchmark на 10/50/200 страниц до распараллеливания.

### P1 — качество документа

- Добавить `Global / Otsu / Adaptive Gaussian / Sauvola` как независимую реализацию через
  OpenCV/NumPy и опубликованные формулы; Wolf добавлять только если corpus покажет преимущество.
- Добавить content box, унификацию полей и выравнивание страниц с применением к выбранным/all.
- Добавить уровень connected-component despeckle и overlay удаляемых точек.
- Добавить quality flags: необычный skew, размер content box, fallback detector, DPI/aspect outlier.

### P2 — книги и смешанный контент

- Автоматический выбор между text-line, UVDoc и дополнительными permissive-licensed backends;
  control points остаются общей correction layer поверх выбранной модели.
- Picture/fill zones и отдельная обработка текстовой маски.
- Ограниченный worker pool с сохранением порядка, атомарностью output и лимитом памяти.

## Уже сделано по результатам анализа

- Preview теперь имеет режимы `Processed`, `Original` и `Compare`; один кадр занимает всю центральную
  область, а original-only не считает processed preview.
- Добавлены горячие клавиши для импорта, захвата, экспорта, обновления preview и операций над
  страницами.
- Boundary detection, deskew и dewarp разделены. Добавлены offline text-line dewarp, selectable
  Hough/hybrid/min-area deskew и отдельная диагностика геометрии в run report.
- Следующий GUI-шаг — thumbnail cards, для которых уже существует `CaptureEntry.thumbnail_image`.

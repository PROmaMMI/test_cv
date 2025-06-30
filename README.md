# Проект: Распознавание блюд в видео с YOLOv11

Этот репозиторий содержит код и инструкции для подготовки данных, обучения модели YOLOv11 для распознавания блюд в ресторане и выполнения предсказаний на видео.

---

## Структура проекта должна быть такой:

```
├── data/
│   ├── raw_videos/      # Оригинальные видеофайлы
│   ├── frames/          # Извлечённые кадры
│   ├── labels/          # Аннотации для кадров (YOLO формат)
│   └── classes.txt      # Список классов
├── dataset/             
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── extract_frames.py    # Скрипт для извлечения кадров из видео (создает data/frames/)
│   ├── split_dataset.py     # Скрипт для разбиения на train/val/test
│   └── augmentation.py      # Скрипт для аугментации train-набора 
├── requirements.txt         # Зависимости проекта
├── data.yaml                # Конфигурация датасета для YOLO
├── yolo11n.pt               # Предобученные веса YOLOv11-nano
└── runs/                    # Папка с результатами обучений и предсказаний

```

---

## Установка и подготовка окружения

1. Клонируйте репозиторий:

   ```bash
   git clone <URL репозитория>
   cd test_comp_vision
   ```
2. Создайте виртуальное окружение и установите зависимости:

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # для Linux/Mac
   # .\venv\Scripts\activate  # для Windows
   pip install -r requirements.txt
   ```
3. Создайте папку `data/raw_videos/` и поместите в нее ваши видео для извлечения кадров
---


## Шаг 1: Извлечение кадров из видео

Скрипт `src/extract_frames.py` извлекает кадры из всех видео из `data/raw_videos/` и помещает в `data/frames/`.

```bash
python src/extract_frames.py
```

---

## Шаг 2: Аннотирование кадров

Используем CVAT для разметки. Основная последовательность:

1. Запустите локальный CVAT (или используйте веб-версию).
2. Создайте новый проект и импортируйте папку `data/frames/`.
3. Аннотируйте объекты (bounding boxes + классы) с использованием функции трекинга.
4. Экспортируйте аннотации в формате YOLO и поместите файлы в `data/labels/`.


---

## Шаг 3: Подготовка датасета

Скрипт `src/split_dataset.py` формирует папки(70% - train, 15% - val, 15% - test):

* `dataset/images/{train,val,test}`
* `dataset/labels/{train,val,test}`

```bash
python src/split_dataset.py
```

---

## Шаг 4: Аугментация данных

Скрипт `src/augmentation.py` применяет аугментации Albumentations к train-набору.

```bash
python src/augmentation.py
```

---

## Шаг 5: Обучение модели

Используем YOLOv11:

```bash
yolo train \
  data=data.yaml \
  model=yolo11n.pt \
  imgsz=640 \
  batch=16 \
  epochs=50 \
```

Пример второй итерации (гиперпараметры изменены):

```bash
yolo train \
  data=data.yaml \
  model=runs/detect/train/weights/best.pt \
  imgsz=640 \
  batch=8 \
  epochs=10 \
```

---

## Шаг 6: Предсказание на видео

Для визуализации результатов на видео:

```bash
yolo predict \
  model=runs/detect/train1/weights/best.pt \
  source=data/raw_videos/{video_name} \
  conf=0.1 \
  save=True
```

---

## Полное воспроизведение

1. Установить зависимости
2. Запустить `extract_frames.py`
3. Аннотировать через CVAT и поместить в `data/labels/`
4. Запустить `split_dataset.py`
5. Запустить `augmentation.py`
6. Обучить модель `yolo train`
7. Запустить `yolo predict` для видео-примеров

---

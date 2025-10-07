#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# === НАСТРОЙКИ ===
DATA_DIR="/media/alex/6011659d-0cd0-4325-8e75-ac1515aeeb591/Projects/personal_files/PassportOCR/manual_dataset/"
PY="python3"
MAIN="main.py"

IMGSZ=1536
CONF=0.25
IOU=0.6
DEVICE=0

# Явный корень для датасета (можешь убрать, если не нужно)
DATASET_ROOT="/media/alex/6011659d-0cd0-4325-8e75-ac1515aeeb591/Projects/personal_files/PassportOCR/dataset_for_easyocr/"

LOG_DIR="./logs"
DONE_LIST="./processed_files.txt"
mkdir -p "$LOG_DIR"
touch "$DONE_LIST"
LOG_FILE="$LOG_DIR/build_dataset_$(date +'%Y%m%d_%H%M%S').log"

echo "== Старт: $(date) ==" | tee -a "$LOG_FILE"
echo "Каталог с исходниками: $DATA_DIR" | tee -a "$LOG_FILE"

total=0; done_cnt=0; skip_cnt=0; fail_cnt=0

# Собираем список файлов сразу из find (без промежуточных переменных с выражениями)
mapfile -d '' files < <(
  find "$DATA_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) -print0 \
  | sort -z
)

total=${#files[@]}
echo "Найдено изображений: $total" | tee -a "$LOG_FILE"

for f in "${files[@]}"; do
  if grep -Fxq "$f" "$DONE_LIST"; then
    ((skip_cnt++)) || true
    echo "[SKIP] Уже обработан: $f" | tee -a "$LOG_FILE"
    continue
  fi

  echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "[RUN ] $(date '+%F %T') -> $f" | tee -a "$LOG_FILE"

  if "$PY" "$MAIN" "$f" --imgsz "$IMGSZ" --conf "$CONF" --iou "$IOU" --device "$DEVICE" >>"$LOG_FILE" 2>&1; then
    echo "$f" >> "$DONE_LIST"
    ((done_cnt++)) || true
    echo "[ OK ] Успешно" | tee -a "$LOG_FILE"
  else
    ((fail_cnt++)) || true
    echo "[FAIL] Ошибка при обработке: $f (подробности в $LOG_FILE)" | tee -a "$LOG_FILE"
  fi
done

echo "== Готово: $(date) ==" | tee -a "$LOG_FILE"
echo "Всего: $total | Успешно: $done_cnt | Пропущено: $skip_cnt | Ошибок: $fail_cnt" | tee -a "$LOG_FILE"

#!/bin/bash
set -e

DEST_DIR="detection_models"
BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"

mkdir -p "$DEST_DIR"

declare -a MODELS=(
    "rtdetr-x.pt"
    "yolov8n.pt"
    "yolov8x.pt"
    "yolov8n-oiv7.pt"
    "yolov9e.pt"
    "yolo11x.pt"
    "yolo11n.pt"
)

echo "Downloading models to $DEST_DIR..."

for model in "${MODELS[@]}"; do
    if [ -f "$DEST_DIR/$model" ]; then
        echo "Example: $model already exists. Skipping."
    else
        echo "Downloading $model..."
        wget -q --show-progress -O "$DEST_DIR/$model" "$BASE_URL/$model"
    fi
done

echo "Download complete."

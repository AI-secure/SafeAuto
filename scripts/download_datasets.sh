#!/bin/bash
set -e

# Base output directory
DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Hugging Face raw file base URLs
BASE1="https://huggingface.co/datasets/javyduck/SafeAuto-DriveLM/resolve/main"
BASE2="https://huggingface.co/datasets/javyduck/SafeAuto-BDDX/resolve/main"

# Files to download from each repo
FILES1=("DriveLM_train.tar" "DriveLM_val.tar")
FILES2=("BDDX_Processed.tar" "BDDX_Test.tar")

# Download, extract, and clean up
download_and_extract() {
  local BASE_URL=$1
  shift
  local FILES=("$@")

  for file in "${FILES[@]}"; do
    local folder_name="$DATA_DIR/${file%.tar}"

    if [ -d "$folder_name" ]; then
      echo "✅ $folder_name already exists. Skipping."
      continue
    fi

    echo "📥 Downloading $file ..."
    wget -q --show-progress "$BASE_URL/$file" -O "$DATA_DIR/$file"

    echo "📦 Extracting $file ..."
    tar -xf "$DATA_DIR/$file" -C "$DATA_DIR"

    echo "🧹 Removing $file ..."
    rm "$DATA_DIR/$file"
  done
}

# Run both datasets
download_and_extract "$BASE1" "${FILES1[@]}"
download_and_extract "$BASE2" "${FILES2[@]}"

echo "✅ All datasets downloaded and extracted into $DATA_DIR/"

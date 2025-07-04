#!/bin/bash
set -e

# Create destination directories if they don't exist
mkdir -p checkpoints/
mkdir -p cache_dir/

# Function to clone if not already downloaded
download_repo() {
    local url="$1"
    local dest="$2"
    if [ ! -d "$dest" ]; then
        echo "Cloning $url into $dest ..."
        GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$dest"
        cd "$dest"
        git lfs pull
        cd -
    else
        echo "$dest already exists. Skipping download."
    fi
}

# Download Video-LLaVA model and projector checkpoint
download_repo https://huggingface.co/LanguageBind/Video-LLaVA-7B         checkpoints/Video-LLaVA-7B
download_repo https://huggingface.co/LanguageBind/Video-LLaVA-Pretrain-7B checkpoints/Video-LLaVA-Pretrain-7B

# Download LanguageBind encoders
download_repo https://huggingface.co/LanguageBind/LanguageBind_Video_merge cache_dir/LanguageBind_Video_merge
download_repo https://huggingface.co/LanguageBind/LanguageBind_Image       cache_dir/LanguageBind_Image

echo "✅ All models downloaded successfully."

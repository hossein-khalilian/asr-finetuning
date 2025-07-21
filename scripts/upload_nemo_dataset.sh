#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

input_path="$1"
folder_name=$(basename "$input_path")
base_name=$(echo "$folder_name" | awk -F "___" '{print $NF}')


repo_name="${base_name}-nemo"
clone_path="$HOME/.cache/datasets/$repo_name"

echo "Working with repo '$repo_name'..."

output=$(huggingface-cli repo create "$repo_name" --repo-type dataset --private 2>&1)
exit_code=$?

if [ $exit_code -ne 0 ]; then
  if echo "$output" | grep -q -i '409\|Conflict'; then
    echo "Repository '$repo_name' already exists. Skipping creation."
  else
    echo "Failed to create repo '$repo_name':"
    echo "$output"
    exit $exit_code
  fi
else
  echo "Repository '$repo_name' created successfully."
fi


if [ ! -d "$input_path/audio_files" ]; then
  echo "Directory '$input_path/audio_files' does not exist."
  exit 1
fi

if [ ! -d "$input_path/manifests" ]; then
  echo "Directory '$input_path/manifests' does not exist."
  exit 1
fi

# Create chunks directory
chunks_dir="$input_path/chunks"
mkdir -p "$chunks_dir"

# Compress audio_files into chunks of max 500MB
# We'll use tar + split: tar all audio_files, pipe to split, split into 500MB files
chunk_size=$((500 * 1024 * 1024))  # 500 MB in bytes

echo "Compressing 'audio_files' into chunks of 500MB max each in '$chunks_dir'..."

tar --sort=name \
    --mtime='UTC 2020-01-01' \
    --owner=0 --group=0 --numeric-owner \
    -cf - -C "$input_path" audio_files | \
    pigz -n | \
    pv -s $(du -sb "$input_path/audio_files" | awk '{print $1}') | \
    split -b "$chunk_size" - "$chunks_dir/audio_files.tar.gz.part_"

echo "Compression completed."


# Upload chunks and manifests to Hugging Face
echo "Uploading 'chunks/' and 'manifests/' to Hugging Face dataset repo..."

huggingface-cli upload "$repo_name" "$chunks_dir" "chunks/" --repo-type=dataset
huggingface-cli upload "$repo_name" "$input_path/manifests" "manifests/" --repo-type=dataset

echo "Upload completed."

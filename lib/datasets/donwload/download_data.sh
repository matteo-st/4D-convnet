#!/bin/bash

cd ~/4D-convnet/data

# Declare an associative array where keys are the desired filenames and values are the URLs
declare -A files=(
    ["SYNTHIA-SEQS-01-DAWN.rar"]="http://synthia-dataset.net/download/693/"
    ["SYNTHIA-SEQS-01-FALL.rar"]="http://synthia-dataset.net/download/760/"
    ["SYNTHIA-SEQS-01-FOG.rar"]="http://synthia-dataset.net/download/729/"
    ["SYNTHIA-SEQS-01-NIGHT.rar"]="http://synthia-dataset.net/download/749/"
    ["SYNTHIA-SEQS-01-SPRING.rar"]="http://synthia-dataset.net/download/743/"
    ["SYNTHIA-SEQS-01-SUMMER.rar"]="http://synthia-dataset.net/download/762/"
    ["SYNTHIA-SEQS-01-SUNSET.rar"]="http://synthia-dataset.net/download/694/"
    ["SYNTHIA-SEQS-01-WINTER.rar"]="http://synthia-dataset.net/download/764/"
    ["SYNTHIA-SEQS-01-WINTERNIGHT.rar"]="http://synthia-dataset.net/download/689/"
    # Add more files here
)

# Iterate over the associative array
for file in "${!files[@]}"; do
    url="${files[$file]}"
    echo "Downloading $file from $url"
    wget -O "$file" "$url"
    echo "$file download completed."
done

echo "All downloads completed."

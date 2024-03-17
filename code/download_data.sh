#!/bin/bash

# Declare an associative array where keys are the desired filenames and values are the URLs
declare -A files=(
    ["SYNTHIA-SEQS-01-DAWN"]="http://synthia-dataset.net/download/693/"
    ["SYNTHIA-SEQS-01-FALL"]="http://synthia-dataset.net/download/760/"
    ["SYNTHIA-SEQS-01-FOG"]="http://synthia-dataset.net/download/729/"
    ["SYNTHIA-SEQS-01-NIGHT"]="http://synthia-dataset.net/download/749/"
    ["SYNTHIA-SEQS-01-SPRING"]="http://synthia-dataset.net/download/743/"
    ["SYNTHIA-SEQS-01-SUMMER"]="http://synthia-dataset.net/download/762/"
    ["SYNTHIA-SEQS-01-SUNSET"]="http://synthia-dataset.net/download/694/"
    ["SYNTHIA-SEQS-01-WINTER"]="http://synthia-dataset.net/download/764/"
    ["SYNTHIA-SEQS-01-WINTERNIGHT"]="http://synthia-dataset.net/download/689/"
    # Add more files here
)

# Iterate over the associative array
for file in "${!files[@]}"; do
    url="${files[$file]}"
    echo "Downloading $file from $url"
    wget "$url" 
    echo "$file download completed."
done

echo "All downloads completed."

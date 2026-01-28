#!/bin/bash
# Download Push-T dataset from official source

set -e

echo "Downloading Push-T dataset..."
mkdir -p data
cd data

# Download dataset
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip

# Extract
echo "Extracting dataset..."
unzip pusht.zip
rm pusht.zip

echo "Dataset ready at data/pusht/"
echo "Total size: $(du -sh pusht | cut -f1)"

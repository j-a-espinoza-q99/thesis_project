#!/bin/bash
# =============================================================================
# Download Amazon Reviews 2023 and ESCI datasets from HuggingFace
# =============================================================================
set -e

echo "Downloading Amazon Reviews 2023 dataset..."
# Using HuggingFace datasets library (run in Python)
python -c "
from datasets import load_dataset
import os

domains = ['All_Beauty', 'Video_Games', 'Office_Products']
cache_dir = os.path.join('data', 'raw')

for domain in domains:
    print(f'Downloading metadata for {domain}...')
    load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'raw_meta_{domain}',
                 split='full', cache_dir=cache_dir, trust_remote_code=True)
    print(f'Downloading reviews for {domain}...')
    load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'raw_review_{domain}',
                 split='full', cache_dir=cache_dir, trust_remote_code=True)

print('Amazon Reviews 2023 download complete.')
"

echo "Downloading ESCI dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('tasksource/esci', split='test', cache_dir='data/raw')
print(f'ESCI test set has {len(ds)} examples.')
"

echo "Downloading Amazon-C4 dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('McAuley-Lab/Amazon-C4', split='test', cache_dir='data/raw')
print(f'Amazon-C4 test set has {len(ds)} examples.')
"

echo "All datasets downloaded."
#!/bin/bash
# =============================================================================
# BLAIR Baseline Verification Only
# =============================================================================

set -e
source ../venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

echo "============================================"
echo "BLAIR Baseline Verification"
echo "============================================"

# Reproduce BLAIR paper results exactly
cd ../blair_reference/blair

# 1. Download and prepare pretraining data
echo "[1] Preparing pretraining data..."
python sample_pretraining_data.py

# 2. Train BLAIR (if needed)
echo "[2] Training BLAIR-base..."
bash base.sh

# 3. Test checkpoint loading
echo "[3] Testing checkpoint loading..."
python test_load_checkpoints.py

# 4. Sequential recommendation evaluation
echo "[4] Sequential recommendation evaluation..."
cd ../seq_rec_results/dataset
python process_amazon_2023.py \
    --domain All_Beauty \
    --device cuda:0 \
    --plm hyp1231/blair-roberta-base

python process_amazon_2023.py \
    --domain Video_Games \
    --device cuda:0 \
    --plm hyp1231/blair-roberta-base

python process_amazon_2023.py \
    --domain Office_Products \
    --device cuda:0 \
    --plm hyp1231/blair-roberta-base

cd ..
python run.py -m UniSRec -d All_Beauty --gpu_id=0 --plm_name hyp1231/blair-roberta-base
python run.py -m UniSRec -d Video_Games --gpu_id=0 --plm_name hyp1231/blair-roberta-base
python run.py -m UniSRec -d Office_Products --gpu_id=0 --plm_name hyp1231/blair-roberta-base

# 5. Product search evaluation
echo "[5] Product search evaluation..."
cd ../product_search_results
python generate_emb.py \
    --dataset McAuley-Lab/Amazon-C4 \
    --plm_name hyp1231/blair-roberta-base \
    --feat_name blair-base

python eval_search.py \
    --dataset McAuley-Lab/Amazon-C4 \
    --suffix blair-baseCLS

# Also evaluate with other PLM baselines
for plm in "roberta-base" "princeton-nlp/sup-simcse-roberta-base"; do
    python generate_emb.py \
        --dataset McAuley-Lab/Amazon-C4 \
        --plm_name $plm \
        --feat_name $(echo $plm | tr '/' '_')

    python eval_search.py \
        --dataset McAuley-Lab/Amazon-C4 \
        --suffix $(echo $plm | tr '/' '_')CLS
done

echo ""
echo "BLAIR baseline verification complete!"
echo "Compare with paper results:"
echo "  - Sequential Rec: Table 4 in BLAIR paper"
echo "  - Product Search: Table 7 in BLAIR paper"
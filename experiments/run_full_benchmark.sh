#!/bin/bash
# =============================================================================
# Full Benchmark Pipeline
# Runs: BLAIR → DeepSeek → Claude/Voyage → Custom Model
# =============================================================================

set -e

# Load environment
source ../venv/bin/activate
source ../.env

echo "============================================"
echo "Full Thesis Benchmark Pipeline"
echo "============================================"
echo "Start time: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Set GPU parameters
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ---------------------------------------------------------------------------
# Step 1: Download and Prepare Data
# ---------------------------------------------------------------------------
echo "[1/5] Preparing data..."
python ../scripts/prepare_data.py \
    --domains All_Beauty Video_Games Office_Products \
    --output_dir ../data/processed

# ---------------------------------------------------------------------------
# Step 2: BLAIR Baseline Verification
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Running BLAIR baseline verification..."

echo "  [2a] Sequential Recommendation..."
python ../training/train_blair.py \
    --model hyp1231/blair-roberta-base \
    --task sequential_recommendation \
    --domains All_Beauty Video_Games Office_Products \
    --output_dir ../results/blair/seq_rec

echo "  [2b] Product Search (ESCI)..."
python ../scripts/eval_search.py \
    --model hyp1231/blair-roberta-base \
    --dataset esci \
    --output_dir ../results/blair/esci

echo "  [2c] Complex Product Search (Amazon-C4)..."
python ../scripts/eval_search.py \
    --model hyp1231/blair-roberta-base \
    --dataset McAuley-Lab/Amazon-C4 \
    --output_dir ../results/blair/amazon_c4

# ---------------------------------------------------------------------------
# Step 3: DeepSeek API Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Running DeepSeek API evaluation..."
python ../experiments/run_deepseek.py \
    --domains All_Beauty Video_Games Office_Products \
    --output_dir ../results/deepseek

# ---------------------------------------------------------------------------
# Step 4: Claude / Voyage AI Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Running Claude/Voyage AI evaluation..."
python ../experiments/run_claude.py \
    --domains All_Beauty Video_Games Office_Products \
    --output_dir ../results/claude_voyage

# ---------------------------------------------------------------------------
# Step 5: Custom Model Training & Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Training and evaluating custom model..."

# Train with all loss components
python ../training/train_custom.py \
    --config ../config/config.yaml \
    --domains All_Beauty Video_Games Office_Products \
    --use_wandb

# Run ablation studies
echo "  Running ablation studies..."
for ablate in "no_adv" "no_div" "no_pop" "no_aug"; do
    echo "    Ablation: $ablate"
    python ../training/train_custom.py \
        --config ../config/config.yaml \
        --ablation $ablate \
        --domains All_Beauty \
        --output_dir ../results/custom/ablation_$ablate
done

# ---------------------------------------------------------------------------
# Generate Final Report
# ---------------------------------------------------------------------------
echo ""
echo "Generating final comparison report..."
python ../evaluation/generate_report.py \
    --results_dir ../results \
    --output ../results/final_comparison_report.pdf

echo ""
echo "============================================"
echo "Benchmark Complete!"
echo "End time: $(date)"
echo "Results saved to: ../results/"
echo "============================================"
#!/bin/bash
"""
Example script for running FSDP training with different configurations.

This script demonstrates various FSDP training scenarios for the multimodal LLM.
"""

# Set common parameters
DATASET_PATH="/path/to/your/dataset"
MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Base"
BATCH_SIZE=2
MAX_SEQ_LEN=1024
EPOCHS=3
LR=1e-5
OUTPUT_DIR="./checkpoints_fsdp"

echo "=== FSDP Training Examples ==="

# Example 1: Single GPU (no FSDP)
echo "1. Single GPU training (no FSDP):"
echo "python train.py \\"
echo "  --dataset-path $DATASET_PATH \\"
echo "  --model-name $MODEL_NAME \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-sequence-length $MAX_SEQ_LEN \\"
echo "  --epochs $EPOCHS \\"
echo "  --lr $LR \\"
echo "  --output-dir ${OUTPUT_DIR}_single \\"
echo "  --devices 1 \\"
echo "  --precision bf16-mixed"
echo ""

# Example 2: Multi-GPU FSDP with FULL_SHARD
echo "2. Multi-GPU FSDP with FULL_SHARD (recommended for large models):"
echo "python train.py \\"
echo "  --dataset-path $DATASET_PATH \\"
echo "  --model-name $MODEL_NAME \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-sequence-length $MAX_SEQ_LEN \\"
echo "  --epochs $EPOCHS \\"
echo "  --lr $LR \\"
echo "  --output-dir ${OUTPUT_DIR}_full_shard \\"
echo "  --devices 4 \\"
echo "  --fsdp-sharding-strategy FULL_SHARD \\"
echo "  --fsdp-state-dict-type full \\"
echo "  --precision bf16-mixed"
echo ""

# Example 3: Multi-GPU FSDP with HYBRID_SHARD
echo "3. Multi-GPU FSDP with HYBRID_SHARD (good for NVLink systems):"
echo "python train.py \\"
echo "  --dataset-path $DATASET_PATH \\"
echo "  --model-name $MODEL_NAME \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-sequence-length $MAX_SEQ_LEN \\"
echo "  --epochs $EPOCHS \\"
echo "  --lr $LR \\"
echo "  --output-dir ${OUTPUT_DIR}_hybrid_shard \\"
echo "  --devices 8 \\"
echo "  --fsdp-sharding-strategy HYBRID_SHARD \\"
echo "  --fsdp-state-dict-type full \\"
echo "  --precision bf16-mixed"
echo ""

# Example 4: Multi-node FSDP training
echo "4. Multi-node FSDP training (2 nodes, 4 GPUs each):"
echo "python train.py \\"
echo "  --dataset-path $DATASET_PATH \\"
echo "  --model-name $MODEL_NAME \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-sequence-length $MAX_SEQ_LEN \\"
echo "  --epochs $EPOCHS \\"
echo "  --lr $LR \\"
echo "  --output-dir ${OUTPUT_DIR}_multinode \\"
echo "  --devices 4 \\"
echo "  --num-nodes 2 \\"
echo "  --fsdp-sharding-strategy FULL_SHARD \\"
echo "  --fsdp-state-dict-type full \\"
echo "  --precision bf16-mixed"
echo ""

# Example 5: FSDP with sharded checkpoints (for very large models)
echo "5. FSDP with sharded checkpoints (saves memory during checkpointing):"
echo "python train.py \\"
echo "  --dataset-path $DATASET_PATH \\"
echo "  --model-name $MODEL_NAME \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-sequence-length $MAX_SEQ_LEN \\"
echo "  --epochs $EPOCHS \\"
echo "  --lr $LR \\"
echo "  --output-dir ${OUTPUT_DIR}_sharded_ckpt \\"
echo "  --devices 4 \\"
echo "  --fsdp-sharding-strategy FULL_SHARD \\"
echo "  --fsdp-state-dict-type sharded \\"
echo "  --precision bf16-mixed"
echo ""

echo "=== FSDP Strategy Guide ==="
echo "FULL_SHARD: Shards parameters, gradients, and optimizer states (maximum memory savings)"
echo "SHARD_GRAD_OP: Shards gradients and optimizer states only"
echo "NO_SHARD: No sharding (equivalent to DDP)"
echo "HYBRID_SHARD: Shards within nodes, replicates across nodes (good for NVLink)"
echo ""
echo "State Dict Types:"
echo "full: Consolidates all shards on rank 0 for checkpointing (easier to load later)"
echo "sharded: Saves each shard separately (saves memory during checkpointing)"
echo ""
echo "Choose your configuration based on:"
echo "- Model size: Larger models benefit more from FULL_SHARD"
echo "- Hardware: HYBRID_SHARD works well with NVLink"
echo "- Memory constraints: Use sharded checkpoints for very large models"

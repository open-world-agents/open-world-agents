# FSDP Training for Multimodal LLM

This directory contains training scripts with FSDP (Fully Sharded Data Parallel) support for training large multimodal language models on desktop interaction data.

## Features

- **FSDP Integration**: Automatic FSDP configuration for multi-GPU and multi-node training
- **Auto-Wrap Policy**: Intelligent detection of transformer blocks for optimal sharding
- **Flexible Sharding**: Support for different FSDP sharding strategies
- **Memory Efficient**: Reduced memory footprint for large model training
- **Checkpoint Management**: Both full and sharded checkpoint saving options

## Quick Start

### Single GPU Training
```bash
python train.py \
  --dataset-path /path/to/dataset \
  --devices 1 \
  --batch-size 2 \
  --epochs 3
```

### Multi-GPU FSDP Training
```bash
python train.py \
  --dataset-path /path/to/dataset \
  --devices 4 \
  --fsdp-sharding-strategy FULL_SHARD \
  --batch-size 2 \
  --epochs 3
```

### Multi-Node Training
```bash
python train.py \
  --dataset-path /path/to/dataset \
  --devices 4 \
  --num-nodes 2 \
  --fsdp-sharding-strategy FULL_SHARD \
  --batch-size 2 \
  --epochs 3
```

## FSDP Configuration Options

### Sharding Strategies

- **FULL_SHARD**: Shards parameters, gradients, and optimizer states (maximum memory savings)
- **SHARD_GRAD_OP**: Shards gradients and optimizer states only
- **NO_SHARD**: No sharding (equivalent to DDP)
- **HYBRID_SHARD**: Shards within nodes, replicates across nodes (optimal for NVLink systems)

### State Dict Types

- **full**: Consolidates all shards on rank 0 for checkpointing (easier to load later)
- **sharded**: Saves each shard separately (saves memory during checkpointing)

## Auto-Wrap Policy

The training script automatically detects and wraps common transformer blocks:

- LlamaDecoderLayer (Llama models)
- PhiDecoderLayer (Phi models)
- Qwen2DecoderLayer (Qwen2 models)
- GemmaDecoderLayer (Gemma models)
- CLIPEncoderLayer (CLIP vision encoder)
- SiglipEncoderLayer (SigLIP vision encoder)

This ensures optimal memory distribution across GPUs while maintaining model performance.

## Memory Optimization Tips

1. **Use FULL_SHARD** for maximum memory savings with large models
2. **Use HYBRID_SHARD** on systems with NVLink for better communication efficiency
3. **Use sharded checkpoints** when training very large models to reduce checkpoint memory overhead
4. **Adjust batch size** based on available GPU memory after FSDP sharding

## Monitoring Training

The script provides detailed logging including:
- FSDP strategy configuration
- Auto-wrap policy details
- Training loss and metrics
- Checkpoint saving information

## Example Configurations

See `train_fsdp_example.sh` for complete example commands for different training scenarios.

## Requirements

- PyTorch with FSDP support
- Lightning Fabric
- Transformers library
- Multiple GPUs for FSDP benefits

## Troubleshooting

### Common Issues

1. **Out of Memory**: Try reducing batch size or using FULL_SHARD strategy
2. **Slow Training**: Check if NVLink is available and consider HYBRID_SHARD
3. **Checkpoint Loading**: Use full state dict type for easier checkpoint loading

### Performance Tips

- Enable mixed precision training with `--precision bf16-mixed`
- Use appropriate batch size for your GPU memory
- Monitor GPU utilization to ensure efficient resource usage

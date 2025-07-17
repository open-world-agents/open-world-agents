# Pretraining with FSLDataset

This directory contains scripts for pretraining vision-language models using FSLDataset.

## Quick Start

### Using Configuration File (Recommended)

```bash
# Single GPU
python pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml

# Multi-GPU with accelerate
accelerate launch pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml

# Multi-GPU with DeepSpeed
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml
```

### Configuration File

The `pretrain_training_config.yaml` file contains all necessary settings:

- **Dataset**: `/mnt/raid12/datasets/owa/data/super-hexagon-event`
- **Model**: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- **Learning Rate**: Cosine scheduler from 3e-4 to 3e-5
- **Warmup**: 2000 steps
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0
- **Optimizer**: AdamW (fused)
- **Sequence Length**: 1024 tokens

### Key Features

1. **FSLDataset Integration**: Efficient sequence packing with 3x training acceleration
2. **Distributed Training**: Full support for multi-GPU and multi-node training
3. **Memory Optimization**: Gradient checkpointing, bf16 precision
4. **Proper Logging**: Accelerator-aware logging for distributed environments
5. **Parameter Monitoring**: Automatic trainable parameter counting

### Customization

You can override any configuration parameter via command line:

```bash
python pretrain_vlm_fsl_dataset.py \
    --config pretrain_training_config.yaml \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 4 \
    --output_dir custom_output_dir
```

### Validation

You can validate the configuration by running with `--help` to see all parsed parameters:

```bash
python pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml --help
```

## Configuration Details

### Learning Rate Schedule

- **Type**: Cosine annealing
- **Initial LR**: 3e-4
- **Final LR**: 3e-5 (eta_min)
- **Warmup**: 2000 steps

### Batch Size Strategy

- **Per Device**: 2 (small for long sequences)
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 16 per GPU

### Memory Optimization

- **Precision**: bfloat16 + tf32
- **Gradient Checkpointing**: Enabled
- **Sequence Length**: 1024 (optimized for FSLDataset)

### Monitoring

- **Logging**: Every 100 steps
- **Evaluation**: Every 1000 steps
- **Checkpoints**: Every 1000 steps (keep 3)
- **Weights & Biases**: Enabled

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`
2. **Slow Training**: Increase `dataloader_num_workers` or check FSLDataset preparation
3. **Config Errors**: Use `--help` flag to validate configuration parsing

### Performance Tips

1. Use DeepSpeed ZeRO-3 for large models
2. Enable `dataloader_persistent_workers` for faster data loading
3. Monitor FSLDataset statistics for optimal sequence packing
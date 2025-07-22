# Pretraining with FSLDataset

This directory contains scripts for pretraining vision-language models using FSLDataset.

## Quick Start

```bash
# Multi-GPU with accelerate
accelerate launch --config_file=accelerate_configs/multi_gpu.yaml \
    pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml

# Multi-GPU with DeepSpeed
accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml
```

### Validation

You can validate the configuration by running with `--help` to see all parsed parameters:

```bash
python pretrain_vlm_fsl_dataset.py --config pretrain_training_config.yaml --help
```
# Pretraining with FSLDataset

This directory contains scripts for pretraining vision-language models using FSLDataset.

> Prerequisites: quick environment setup can be done following [devcontainer guide](https://github.com/MaumAI-Company/closed-world-agents/blob/main/.devcontainer/README.md)

## Quick Start

1. Launch video decoding server. See [video-decoding-server](../../../open-world-agents/projects/video-decoding-server/README.md) for more details.

2. Setup variables for video decoding server.
```sh
export VIDEO_DECODING_SERVER_URL="127.0.0.1:8000"
curl -m 1 -L -s -o /dev/null -w "%{http_code}" http://$VIDEO_DECODING_SERVER_URL/v2/health/ready | grep -q '^200$' && echo '✅ Server alive!' || echo '❌ Server down'
```

3. Start training.
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

### TODOs

- Test following
```
vuv pip install unsloth-zoo

from unsloth_zoo.gradient_checkpointing import patch_unsloth_gradient_checkpointing
patch_unsloth_gradient_checkpointing()
```

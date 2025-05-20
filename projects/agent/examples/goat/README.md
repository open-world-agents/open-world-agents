_TYPER_STANDARD_TRACEBACK=1

```sh
$ python 01_build_dataset.py --train-path /mnt/raid11/datasets/owa/mcaps/super-hexagon --test-path /mnt/raid11/datasets/owa/mcaps/super-hexagon-30s --output-path /mnt/raid11/datasets/owa/data/super-hexagon

$ python 02_eda.py visualize /mnt/raid11/datasets/owa/data/super-hexagon
$ python 02_eda.py stat /mnt/raid11/datasets/owa/data/super-hexagon

$ python 03_prepare_model.py prepare-model /mnt/raid11/datasets/owa/checkpoints/SmolVLM2-256M-Video-Instruct-expanded --model-id HuggingFaceTB/SmolVLM2-256M-Video-Instruct

```
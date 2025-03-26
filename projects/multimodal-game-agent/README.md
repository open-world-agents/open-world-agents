## Steps to train-to-eval your own multimodal game agent

**WARNING**: This `multimodal-game-agent` project is experimental.

**WARNING**: Currently, overall implementation is focused on playing single game: Super Hexagon. Many part of code is **hard-coded**(So you must read over whole code once, in now.) and has a game-specific implementation in many part. Please me cautious and be aware to modify various part of this code to make your own game agent to be run.

1. Prepare your data.
    ```sh
    # below line is example.
    $ DATA_DIR=/mnt/raid11/datasets/owa/mcaps/super_hexagon
    $ CHECKPOINT_DIR=/mnt/raid11/datasets/owa/

    $ ls $DATA_DIR
    your-data-1.mcap your-data-1.mkv your-data-2.mcap your-data-2.mkv
    ```
2. Generate `jsonl` file containing query. 
    ```sh
    python scripts/01_generate_queries.py / super_hexagon.jsonl
    ```
3. Check whether generated dataset has no problem. Be aware to maintain proper data distribution, without imbalance.
    ```sh
    python scripts/02_tokenize_queries.py eda super_hexagon.jsonl
    ```
4. Prepare token-expanded model.
    ```sh
    python scripts/02_tokenize_queries.py prepare $CHECKPOINT_DIR/SmolVLM2-500M-Video-Instruct-expanded --apply-semantic-init
    ```
5. Check your dataset in your eye.
    ```sh
    python scripts/02_tokenize_queries.py collator super_hexagon.jsonl --model-id $CHECKPOINT_DIR/SmolVLM2-500M-Video-Instruct-expanded
    ```
6. Train your model. Make sure & Find the settings when the loss goes below `0.1`.
    1. Using sbatch
        ```sh
        sbatch train.sbatch
        ```
    2. Using srun
        ```sh
        srun --gpus 8 --cpus-per-task 128 --nodes=1 accelerate launch scripts/03_train_agent.py --query_path super_hexagon.jsonl --model_id $CHECKPOINT_DIR/SmolVLM2-500M-Video-Instruct-expanded --output_dir $CHECKPOINT_DIR/checkpoints/super_hexagon/repeat_10-epoch_20-lr4e-5-seminit_arrow --repeat_n 10 --num_epochs 20 --learning_rate 4e-5
        ```
7. Offline-evaluate your model.
    ```sh
    srun --gpus 1 --cpus-per-task 8 --pty --partition a100 python scripts/04_offline_evaluate.py --query_path super_hexagon.jsonl --model_id /mnt/raid11/datasets/owa/checkpoints/super_hexagon/repeat_10-epoch_5-lr1e-4-seminit_arrow
    ```
8. Online-evalute your model. Note that only this process must be executed in Windows desktop.
    ```sh
    python .\scripts\05_online_evaluation.py C:\Users\MilkClouds\Downloads\repeat_10-epoch_20-lr4e-5-seminit_arrow\
    ```
from pathlib import Path

import webdataset as wds
from tqdm import tqdm

# Parameters
input_dir = Path("~/data/vpt_filtered_data").expanduser()  # Directory containing the files
output_dir = Path("~/data/vpt_filtered_data_webdataset").expanduser()  # Directory to save the webdataset shards
output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
output_pattern = f"{output_dir}/shard-%06d.tar"  # e.g., shard-000000.tar, shard-000001.tar
shard_size = 1000  # number of samples per shard
allowed_extensions = [".mcap", ".jsonl", ".mp4"]  # Add other extensions if needed

# Collect base names
basenames = sorted(set(f.stem for f in input_dir.glob("*") if f.is_file() and f.suffix in allowed_extensions))
shard_writer = wds.ShardWriter(output_pattern, maxcount=shard_size)

for idx, base in enumerate(tqdm(basenames)):
    sample = {}
    for ext in allowed_extensions:
        fpath = input_dir / f"{base}{ext}"
        if fpath.exists():
            with open(fpath, "rb") as f:
                sample[ext] = f.read()  # TODO: Might need to remove dot in extension here

    if sample:
        sample["__key__"] = base  # required for wds
        shard_writer.write(sample)

shard_writer.close()

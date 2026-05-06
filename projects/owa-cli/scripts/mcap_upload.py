import shutil
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from mcap_owa.highlevel import OWAMcapReader

api = HfApi()

with tempfile.TemporaryDirectory() as tmp_dir:
    # copy `example.mcap` and `example.mkv` from `../../docs/data`
    # to `tmp_dir` and upload them to the Hub
    shutil.copy("../../docs/data/example.mcap", tmp_dir)
    shutil.copy("../../docs/data/example.mkv", tmp_dir)

    # read over top 1024 messages and save into jsonl file
    with OWAMcapReader(Path(tmp_dir) / "example.mcap") as reader:
        messages = []
        for i, message in enumerate(reader.iter_messages()):
            if i >= 1024:
                break
            messages.append(message)
        df = pd.DataFrame(messages)
        df.to_json(Path(tmp_dir) / "example.jsonl", orient="records", lines=True)

    api.upload_large_folder(
        repo_id="open-world-agents/example-dataset",
        repo_type="dataset",
        folder_path=tmp_dir,
    )

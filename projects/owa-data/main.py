from datasets import load_from_disk
from transformers import AutoTokenizer

from owa.data.episode_tokenizer import EpisodeTokenizer
from owa.data.transforms import decode_map

# Load event dataset
event_dataset = load_from_disk("/mnt/raid12/datasets/owa/data/super-hexagon-event")

event_dataset.set_transform(decode_map)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Base")
event_tokenizer = EpisodeTokenizer()
event_tokenizer.prepare_model(tokenizer=tokenizer)


def iterator():
    episode_path = None
    for sample in event_dataset["train"]:
        if episode_path is not None and sample["episode_path"] != episode_path:
            break
        episode_path = sample["episode_path"]
        yield sample["mcap_message"]


for tokenized in event_tokenizer.tokenize(iterator()):
    print(tokenized)

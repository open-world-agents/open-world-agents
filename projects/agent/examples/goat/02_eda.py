from pathlib import Path

import orjson
import typer
from datasets import DatasetDict, load_from_disk

from dataset.utils import transform

app = typer.Typer()


@app.command()
def visualize(dataset_path: Path):
    dataset_dict: DatasetDict = load_from_disk(dataset_path.as_posix())
    print(f"Loaded dataset from {dataset_path}")
    print(f"{dataset_dict=}")

    dataset_dict.set_transform(transform=lambda x: transform(x, decode_images=True))

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    dataset_dict = dataset_dict.shuffle(seed=23)
    for idx, data in enumerate(dataset_dict["train"].take(3)):
        print(data)
        # {'images': [<PIL.Image.Image image mode=RGB size=768x480 at 0x7FC95A6780D0>, <PIL.Image.Image image mode=RGB size=768x480 at 0x7FC95A673ED0>, <PIL.Image.Image image mode=RGB size=768x480 at 0x7FC95A673210>, <PIL.Image.Image image mode=RGB size=768x480 at 0x7FC95A671DD0>, <PIL.Image.Image image mode=RGB size=768x480 at 0x7FC95A706610>], 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '\nYou are playing Super Hexagon, a fast-paced game that requires precise control and timing.\nGiven past event trajectory, predict the future sequence of event.\n<TIMESTAMP_169><image><TIMESTAMP_174><image><TIMESTAMP_176><KEYBOARD_39_release><TIMESTAMP_179><image><TIMESTAMP_184><image><TIMESTAMP_190><image>\nNow: <TIMESTAMP_196>'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '<TIMESTAMP_205><KEYBOARD_39_press><TIMESTAMP_219><KEYBOARD_39_release>'}]}]}

        # save the images
        for jdx, image in enumerate(data["images"]):
            image.save(tmp_dir / f"image_{idx}_{jdx}.png")


def transform_debug(examples, *, decode_images: bool = True):
    conversation = [orjson.loads(x) for x in examples["conversation"]]
    examples = {
        # "images": [x["images"] for x in conversation],
        "messages": [x["messages"] for x in conversation],
    }
    return examples


@app.command()
def stat(dataset_path: Path):
    dataset_dict: DatasetDict = load_from_disk(dataset_path.as_posix())
    print(f"Loaded dataset from {dataset_path}")
    print(f"{dataset_dict=}")

    dataset_dict = dataset_dict.map(
        lambda x: transform_debug(x, decode_images=False), num_proc=16, batched=True, batch_size=16
    )

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    # Calculate statistics on the dataset
    messages = dataset_dict["train"]["messages"]

    # Statistics for assistant response lengths
    assistant_lengths = []
    user_lengths = []
    num_images = []

    for msg in messages:
        # Get the last message (assistant response)
        if len(msg) >= 2 and msg[-1]["role"] == "assistant" and msg[-1]["content"]:
            assistant_text = msg[-1]["content"][0]["text"]
            assistant_lengths.append(len(assistant_text))

        # Get user message length
        if len(msg) >= 1 and msg[0]["role"] == "user" and msg[0]["content"]:
            user_text = msg[0]["content"][0]["text"]
            user_lengths.append(len(user_text))

            # Count image tokens in user messages
            img_count = user_text.count("<image>")
            num_images.append(img_count)

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {len(messages)}")

    if assistant_lengths:
        print("\nAssistant Response Statistics:")
        print(f"Min length: {min(assistant_lengths)}")
        print(f"Max length: {max(assistant_lengths)}")
        print(f"Average length: {sum(assistant_lengths) / len(assistant_lengths):.2f}")
        print(f"Empty responses: {assistant_lengths.count(0)}")

    if user_lengths:
        print("\nUser Message Statistics:")
        print(f"Min length: {min(user_lengths)}")
        print(f"Max length: {max(user_lengths)}")
        print(f"Average length: {sum(user_lengths) / len(user_lengths):.2f}")

    if num_images:
        print("\nImage Count Statistics:")
        print(f"Min images: {min(num_images)}")
        print(f"Max images: {max(num_images)}")
        print(f"Average images: {sum(num_images) / len(num_images):.2f}")

    # print assistant and user message with maximum length
    for i, msg in enumerate(messages):
        if len(msg) >= 2 and msg[-1]["role"] == "assistant" and msg[-1]["content"]:
            assistant_text = msg[-1]["content"][0]["text"]
            if len(assistant_text) == max(assistant_lengths):
                print(f"\nAssistant message with maximum length: {assistant_text!r}")
                print(f"User message: {msg[0]['content'][0]['text']!r}")
                break
    # print user message with maximum length
    for i, msg in enumerate(messages):
        if len(msg) >= 1 and msg[0]["role"] == "user" and msg[0]["content"]:
            user_text = msg[0]["content"][0]["text"]
            if len(user_text) == max(user_lengths):
                print(f"\nUser message with maximum length: {user_text!r}")
                print(f"Assistant message: {msg[-1]['content'][0]['text']!r}")
                break

    # Optional: Generate histograms if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(assistant_lengths, bins=20)
        plt.title("Assistant Response Lengths")
        plt.xlabel("Length (chars)")
        plt.ylabel("Count")

        plt.subplot(2, 2, 2)
        plt.hist(user_lengths, bins=20)
        plt.title("User Message Lengths")
        plt.xlabel("Length (chars)")
        plt.ylabel("Count")

        plt.subplot(2, 2, 3)
        plt.hist(num_images, bins=max(num_images) if num_images else 10)
        plt.title("Images per Example")
        plt.xlabel("Number of Images")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig(tmp_dir / "dataset_statistics.png")
        print(f"\nStatistics visualization saved to {tmp_dir / 'dataset_statistics.png'}")
    except ImportError:
        print("\nMatplotlib not available. Skipping histogram generation.")


if __name__ == "__main__":
    app()

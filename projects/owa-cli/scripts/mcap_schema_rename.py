import importlib
from pathlib import Path

import typer

from mcap_owa.highlevel import OWAMcapReader, OWAMcapWriter


def convert_name(name: str):
    convert_dict = {
        "owa_env_desktop": "owa.env.desktop",
        "owa_env_gst": "owa.env.gst",
    }
    names = name.split(".")
    if names[0] in convert_dict:
        return convert_dict[names[0]] + "." + ".".join(names[1:])
    raise ValueError(f"Name {name} not found in convert_dict")


def main(file_path: Path):
    """
    Renames the schema names in the MCAP file.
    Useful when the package structure has changed. e.g. owa_env_desktop -> owa.env.desktop
    """
    msgs = []

    with OWAMcapReader(file_path) as reader:
        for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
            name = schema.name
            name = convert_name(name)

            module, class_name = name.rsplit(".", 1)
            module = importlib.import_module(module)
            cls = getattr(module, class_name)

            decoded = cls(**decoded)

            msgs.append((message.log_time, channel.topic, decoded))

    with OWAMcapWriter(file_path) as writer:
        for log_time, topic, msg in msgs:
            writer.write_message(topic=topic, message=msg, log_time=log_time)


if __name__ == "__main__":
    typer.run(main)

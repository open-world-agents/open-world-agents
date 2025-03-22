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


def process_file(file_path: Path, dry_run: bool = True):
    """Process a single MCAP file, converting schemas as needed."""
    schema_conversions = {}
    msgs = []

    with OWAMcapReader(file_path) as reader:
        for schema, channel, message, decoded in reader.reader.iter_decoded_messages():
            old_name = schema.name
            try:
                new_name = convert_name(old_name)

                # Track schema conversions for reporting
                if old_name not in schema_conversions:
                    schema_conversions[old_name] = new_name

                module, class_name = new_name.rsplit(".", 1)
                module = importlib.import_module(module)
                cls = getattr(module, class_name)

                decoded = cls(**decoded)
            except ValueError as e:
                # Skip schemas that don't need conversion
                if "not found in convert_dict" not in str(e):
                    raise

            msgs.append((message.log_time, channel.topic, decoded))

    # Show schema conversions
    if schema_conversions:
        print(f"  Conversions for {file_path.name}:")
        for old, new in schema_conversions.items():
            print(f"    {old} → {new}")
    else:
        print(f"  No conversions needed for {file_path.name}")
        return False

    # Write the file if not in dry run mode
    if not dry_run:
        with OWAMcapWriter(file_path) as writer:
            for log_time, topic, msg in msgs:
                writer.write_message(topic=topic, message=msg, log_time=log_time)
        print(f"  Saved changes to: {file_path}")

    return True


def main(directory_path: Path, dry_run: bool = True):
    """
    Rename schema names in all MCAP files within the provided directory.
    With --no-dry-run, overwrites original files after confirmation.
    """
    # Verify directory exists
    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a directory")
        raise typer.Exit(1)

    # Find all MCAP files
    mcap_files = list(directory_path.rglob("*.mcap"))

    if not mcap_files:
        print(f"No MCAP files found in {directory_path}")
        raise typer.Exit(1)

    print(f"Found {len(mcap_files)} MCAP files in {directory_path}")

    if dry_run:
        print("DRY RUN MODE: Only showing what would change")
    else:
        print("WRITE MODE: Will overwrite original files")
        if not typer.confirm("Do you want to continue? This will modify the original files", default=False):
            print("Operation cancelled.")
            return

    # Process each file
    converted_count = 0
    for file_path in mcap_files:
        was_converted = process_file(file_path, dry_run)
        if was_converted:
            converted_count += 1

    # Summary
    if converted_count == 0:
        print("\nNo files needed conversion.")
    else:
        if dry_run:
            print(f"\n{converted_count} files would be modified. Run with --no-dry-run to apply changes.")
        else:
            print(f"\n{converted_count} files were successfully modified.")


if __name__ == "__main__":
    typer.run(main)

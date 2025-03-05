"""
Design references:
- rosbags: https://ternaris.gitlab.io/rosbags/index.html
- mcap: https://mcap.dev/
"""

import json
import os
import sys
import time
from pathlib import Path

from mcap.writer import Writer
from pydantic import BaseModel, ImportString

PathType = str | bytes | os.PathLike
FORMAT_VERSION = "0.1.0"


class Event(BaseModel):
    msgtype: ImportString
    msgdata: bytes

    def deserialize(self):
        return self.msgtype.deserialize(self.msgdata)


class MCAPReader:
    def __init__(self, file_path: PathType):
        self.file_path = Path(file_path)
        self.writer = Writer(self.file_path)

    def open(self):
        self.writer.start(profile="OWA")

    def close(self):
        self.writer.finish()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def add_message(self, event_type, source, data):
        self.current_batch["timestamp"].append(time.time_ns())
        self.current_batch["event_type"].append(event_type)
        self.current_batch["source"].append(source)
        self.current_batch["data"].append(json.dumps(data))

        if len(self.current_batch["timestamp"]) >= self.batch_size:
            self.flush_current_batch()

    def flush_current_batch(self):
        if not self.current_batch["timestamp"]:
            return

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(self.current_batch["timestamp"], type=pa.int64()),
                pa.array(self.current_batch["event_type"], type=pa.string()),
                pa.array(self.current_batch["source"], type=pa.string()),
                pa.array(self.current_batch["data"], type=pa.string()),
            ],
            names=["timestamp", "event_type", "source", "data"],
        )

        self.batches.append(batch)
        self.current_batch = {"timestamp": [], "event_type": [], "source": [], "data": []}

    def flush(self):
        self.flush_current_batch()

        if not self.batches:
            return

        # Create table from all batches
        table = pa.Table.from_batches(self.batches)

        # Write to disk
        writer = pa.ipc.RecordBatchFileWriter(pa.OSFile(self.file_path.with_suffix(".arrow"), "wb"), table.schema)

        for batch in self.batches:
            writer.write_batch(batch)

        writer.close()
        self.batches = []

    def close(self):
        self.flush()


# Reading back data
def read_events(file_path, filters=None):
    reader = pa.ipc.RecordBatchFileReader(pa.OSFile(file_path, "rb"))
    table = reader.read_all()

    # Convert to dictionaries
    records = []
    for i in range(len(table)):
        record = {
            "timestamp": table["timestamp"][i].as_py(),
            "event_type": table["event_type"][i].as_py(),
            "source": table["source"][i].as_py(),
            "data": json.loads(table["data"][i].as_py()),
        }

        # Apply filters if provided
        if filters:
            include = True
            for field, op, value in filters:
                if op == "=" and record[field] != value:
                    include = False
                    break
                elif op == ">" and record[field] <= value:
                    include = False
                    break
                elif op == "<" and record[field] >= value:
                    include = False
                    break

            if include:
                records.append(record)
        else:
            records.append(record)

    return records

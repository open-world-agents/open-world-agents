#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0.0",
#   "mcap>=1.0.0",
#   "easydict>=1.10",
#   "orjson>=3.8.0",
#   "typer>=0.12.0",
#   "mcap-owa-support==0.4.2",
# ]
# [tool.uv]
# exclude-newer = "2025-06-21T00:00:00Z"
# ///

import mcap_owa

print(mcap_owa.__version__)

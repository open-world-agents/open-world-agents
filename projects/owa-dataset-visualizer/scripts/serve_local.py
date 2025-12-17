#!/usr/bin/env python3
"""Simple file server for local MCAP/MKV viewing with owa-dataset-visualizer."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(directory: Path):
    """Create a handler class that serves from the specified directory with Range support."""

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def do_GET(self):
            if self.path == "/files.json":
                self.serve_file_list()
            elif "Range" in self.headers:
                self.serve_range_request()
            else:
                super().do_GET()

        def serve_range_request(self):
            """Handle Range requests for video streaming."""
            path = self.translate_path(self.path)
            try:
                with open(path, "rb") as f:
                    file_size = Path(path).stat().st_size
                    range_header = self.headers["Range"]

                    # Parse Range header (e.g., "bytes=0-1023" or "bytes=-500")
                    start, end = 0, file_size - 1
                    if range_header.startswith("bytes="):
                        ranges = range_header[6:].split("-")
                        if not ranges[0] and ranges[1]:  # Suffix range like "bytes=-500"
                            suffix_len = int(ranges[1])
                            start = max(0, file_size - suffix_len)
                            end = file_size - 1
                        else:
                            start = int(ranges[0]) if ranges[0] else 0
                            end = int(ranges[1]) if ranges[1] else file_size - 1

                    end = min(end, file_size - 1)
                    length = end - start + 1

                    if start >= file_size or length <= 0:
                        self.send_error(416, "Range Not Satisfiable")
                        return

                    self.send_response(206)
                    self.send_header("Content-Type", self.guess_type(path))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    f.seek(start)
                    remaining = length
                    buf_size = 64 * 1024
                    while remaining > 0:
                        chunk = f.read(min(buf_size, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (OSError, FileNotFoundError):
                self.send_error(404, "File not found")

        def serve_file_list(self):
            """Scan directory for mcap/video pairs and return as JSON."""
            pairs = {}
            for f in directory.rglob("*"):
                if not f.is_file():
                    continue
                path = str(f.relative_to(directory))
                if path.endswith(".mcap"):
                    base = path[:-5]
                    pairs.setdefault(base, {})["mcap"] = path
                elif path.lower().endswith((".mkv", ".mp4", ".webm")):
                    base = path.rsplit(".", 1)[0]
                    pairs.setdefault(base, {})["video"] = path

            files = [
                {"name": Path(k).name, "path": k, "mcap": v["mcap"], "mkv": v["video"]}
                for k, v in sorted(pairs.items())
                if "mcap" in v and "video" in v
            ]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(files).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Serve local MCAP/MKV files for owa-dataset-visualizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/recordings
  %(prog)s -p 9000 /path/to/recordings
  %(prog)s .  # serve current directory
        """,
    )
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing MCAP/MKV files (default: .)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")

    handler = make_handler(directory)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving {directory} at http://{args.host}:{args.port}")
    print(f"Open visualizer: http://localhost:5173/?base_url=http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

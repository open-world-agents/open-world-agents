#!/bin/bash
# Setup test data symlinks for development
#
# Usage:
#   ./scripts/setup-test-data.sh [MCAP_PATH] [MKV_PATH]
#
# Example:
#   ./scripts/setup-test-data.sh /path/to/recording.mcap /path/to/recording.mkv
#
# If no arguments provided, uses default test data path.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PUBLIC_DIR="$PROJECT_DIR/public"

# Default test data paths (adjust for your environment)
DEFAULT_MCAP="/mnt/raid12/datasets/owa/example-apex/mcaps/example.mcap"
DEFAULT_MKV="/mnt/raid12/datasets/owa/example-apex/mcaps/example.mkv"

MCAP_PATH="${1:-$DEFAULT_MCAP}"
MKV_PATH="${2:-$DEFAULT_MKV}"

# Validate files exist
if [ ! -f "$MCAP_PATH" ]; then
    echo "Error: MCAP file not found: $MCAP_PATH"
    exit 1
fi

if [ ! -f "$MKV_PATH" ]; then
    echo "Error: MKV file not found: $MKV_PATH"
    exit 1
fi

# Create public directory and symlinks
mkdir -p "$PUBLIC_DIR"
ln -sf "$MCAP_PATH" "$PUBLIC_DIR/test.mcap"
ln -sf "$MKV_PATH" "$PUBLIC_DIR/test.mkv"

echo "Test data configured:"
echo "  MCAP: $MCAP_PATH -> public/test.mcap"
echo "  MKV:  $MKV_PATH -> public/test.mkv"
echo ""
echo "Start dev server and open:"
echo "  http://localhost:5173/?mcap=/test.mcap&mkv=/test.mkv"


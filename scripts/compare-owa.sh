#!/bin/bash

# Compare open-world-agents: upstream vs proprietary
set -e

# Clear previous outputs
> $GITHUB_STEP_SUMMARY
> $GITHUB_OUTPUT

echo "## OWA Sync Report" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Git-based comparison with detailed stats
ORIGINAL_DIR=$(pwd)
cd /tmp/upstream-owa

# Get push changes (local → upstream) with detailed diff
git reset --hard HEAD >/dev/null 2>&1
rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./ >/dev/null 2>&1
git add -A >/dev/null 2>&1
PUSH_DIFF=$(git diff --cached --name-status 2>/dev/null || true)
PUSH_STATS=$(git diff --cached --numstat 2>/dev/null || true)

# Get pull changes (upstream → local) with detailed diff
git reset --hard HEAD >/dev/null 2>&1
mkdir -p /tmp/local-copy
rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" /tmp/local-copy/ >/dev/null 2>&1
rsync -av --delete --exclude='.git' /tmp/local-copy/ ./ >/dev/null 2>&1
git add -A >/dev/null 2>&1
PULL_DIFF=$(git diff --cached --name-status 2>/dev/null || true)
PULL_STATS=$(git diff --cached --numstat 2>/dev/null || true)

# Cleanup
rm -rf /tmp/local-copy
git reset --hard HEAD >/dev/null 2>&1
cd "$ORIGINAL_DIR"

# Set outputs
if [ -n "$PULL_DIFF" ]; then
    echo "pull_has_changes=true" >> $GITHUB_OUTPUT
else
    echo "pull_has_changes=false" >> $GITHUB_OUTPUT
fi

if [ -n "$PUSH_DIFF" ]; then
    echo "push_has_changes=true" >> $GITHUB_OUTPUT
else
    echo "push_has_changes=false" >> $GITHUB_OUTPUT
fi

# Create patch if needed
if [ -n "$PUSH_DIFF" ]; then
    cd /tmp/upstream-owa
    rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./ >/dev/null 2>&1
    git add -A >/dev/null 2>&1
    git diff --cached > /tmp/rsync-changes.patch 2>/dev/null || true
    git reset --hard HEAD >/dev/null 2>&1
    cd "$ORIGINAL_DIR"
fi

# Calculate totals
calculate_totals() {
    local stats="$1"
    local added=0
    local deleted=0
    local files=0

    if [ -n "$stats" ]; then
        while IFS=$'\t' read -r add del file; do
            if [[ "$add" =~ ^[0-9]+$ ]]; then
                added=$((added + add))
            fi
            if [[ "$del" =~ ^[0-9]+$ ]]; then
                deleted=$((deleted + del))
            fi
            files=$((files + 1))
        done <<< "$stats"
    fi

    echo "$files $added $deleted"
}

# Generate coverage-style report
if [ -n "$PULL_DIFF" ] || [ -n "$PUSH_DIFF" ]; then
    # Calculate stats
    if [ -n "$PULL_STATS" ]; then
        read pull_files pull_added pull_deleted <<< $(calculate_totals "$PULL_STATS")
    else
        pull_files=0 pull_added=0 pull_deleted=0
    fi

    if [ -n "$PUSH_STATS" ]; then
        read push_files push_added push_deleted <<< $(calculate_totals "$PUSH_STATS")
    else
        push_files=0 push_added=0 push_deleted=0
    fi

    # Summary badges
    if [ $pull_files -gt 0 ]; then
        echo "<img src=\"https://img.shields.io/badge/Pull%20Files-$pull_files-blue.svg\" alt=\"Pull Files\"> <img src=\"https://img.shields.io/badge/Lines-+$pull_added%20~$pull_deleted-orange.svg\" alt=\"Pull Lines\">" >> $GITHUB_STEP_SUMMARY
    fi

    if [ $push_files -gt 0 ]; then
        echo "<img src=\"https://img.shields.io/badge/Push%20Files-$push_files-purple.svg\" alt=\"Push Files\"> <img src=\"https://img.shields.io/badge/Lines-+$push_added%20~$push_deleted-orange.svg\" alt=\"Push Lines\">" >> $GITHUB_STEP_SUMMARY
    fi

    echo "<details><summary>Click to see where and how files changed</summary>" >> $GITHUB_STEP_SUMMARY
    echo "<table><thead>" >> $GITHUB_STEP_SUMMARY
    echo "<tr><th>File</th><th>Lines Added</th><th>Lines Deleted</th><th>Direction</th></tr>" >> $GITHUB_STEP_SUMMARY
    echo "</thead><tbody>" >> $GITHUB_STEP_SUMMARY

    # Process pull changes
    if [ -n "$PULL_STATS" ]; then
        echo "$PULL_STATS" | while IFS=$'\t' read -r added deleted file; do
            if [ "$added" = "-" ]; then added="0"; fi
            if [ "$deleted" = "-" ]; then deleted="0"; fi
            echo "<tr><td><code>$file</code></td><td align=\"center\"><img src=\"https://img.shields.io/badge/+$added-brightgreen.svg\" alt=\"+$added\"></td><td align=\"center\"><img src=\"https://img.shields.io/badge/-$deleted-red.svg\" alt=\"-$deleted\"></td><td align=\"center\">📥 Pull</td></tr>" >> $GITHUB_STEP_SUMMARY
        done
    fi

    # Process push changes
    if [ -n "$PUSH_STATS" ]; then
        echo "$PUSH_STATS" | while IFS=$'\t' read -r added deleted file; do
            if [ "$added" = "-" ]; then added="0"; fi
            if [ "$deleted" = "-" ]; then deleted="0"; fi
            echo "<tr><td><code>$file</code></td><td align=\"center\"><img src=\"https://img.shields.io/badge/+$added-brightgreen.svg\" alt=\"+$added\"></td><td align=\"center\"><img src=\"https://img.shields.io/badge/-$deleted-red.svg\" alt=\"-$deleted\"></td><td align=\"center\">📤 Push</td></tr>" >> $GITHUB_STEP_SUMMARY
        done
    fi

    echo "</tbody></table>" >> $GITHUB_STEP_SUMMARY
    echo "</details>" >> $GITHUB_STEP_SUMMARY
else
    echo "✅ **Everything is in sync!**" >> $GITHUB_STEP_SUMMARY
fi

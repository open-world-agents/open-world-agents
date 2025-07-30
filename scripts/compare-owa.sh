#!/bin/bash

# Compare open-world-agents: upstream vs proprietary
set -e

# Clear previous outputs
> $GITHUB_STEP_SUMMARY
> $GITHUB_OUTPUT

echo "## 🔄 OWA Sync Status" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Git-based comparison
ORIGINAL_DIR=$(pwd)
cd /tmp/upstream-owa

# Get push changes (local → upstream)
git reset --hard HEAD >/dev/null 2>&1
rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./ >/dev/null 2>&1
git add -A >/dev/null 2>&1
PUSH_DIFF=$(git diff --cached --name-status 2>/dev/null || true)

# Get pull changes (upstream → local)
git reset --hard HEAD >/dev/null 2>&1
mkdir -p /tmp/local-copy
rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" /tmp/local-copy/ >/dev/null 2>&1
rsync -av --delete --exclude='.git' /tmp/local-copy/ ./ >/dev/null 2>&1
git add -A >/dev/null 2>&1
PULL_DIFF=$(git diff --cached --name-status 2>/dev/null || true)

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

# Generate detailed summary like coverage report
generate_file_table() {
    local diff="$1"
    local direction="$2"

    if [ -z "$diff" ]; then
        return
    fi

    echo "<details><summary>📋 $direction</summary>"
    echo "<table><thead>"
    echo "<tr><th>File</th><th>Status</th></tr>"
    echo "</thead><tbody>"

    echo "$diff" | while IFS=$'\t' read -r status file; do
        case "$status" in
            A*)
                echo "<tr><td><code>$file</code></td><td><img src=\"https://img.shields.io/badge/Added-brightgreen.svg\" alt=\"Added\"></td></tr>"
                ;;
            M*)
                echo "<tr><td><code>$file</code></td><td><img src=\"https://img.shields.io/badge/Modified-orange.svg\" alt=\"Modified\"></td></tr>"
                ;;
            D*)
                echo "<tr><td><code>$file</code></td><td><img src=\"https://img.shields.io/badge/Deleted-red.svg\" alt=\"Deleted\"></td></tr>"
                ;;
            R*)
                echo "<tr><td><code>$file</code></td><td><img src=\"https://img.shields.io/badge/Renamed-blue.svg\" alt=\"Renamed\"></td></tr>"
                ;;
        esac
    done

    echo "</tbody></table>"
    echo "</details>"
    echo ""
}

# Count changes for summary
count_by_type() {
    local diff="$1"
    local type="$2"
    echo "$diff" | grep -c "^$type" 2>/dev/null || echo "0"
}

if [ -n "$PULL_DIFF" ] || [ -n "$PUSH_DIFF" ]; then
    # Summary badges
    if [ -n "$PULL_DIFF" ]; then
        pull_total=$(echo "$PULL_DIFF" | wc -l)
        pull_added=$(count_by_type "$PULL_DIFF" "A")
        pull_modified=$(count_by_type "$PULL_DIFF" "M")
        pull_deleted=$(count_by_type "$PULL_DIFF" "D")
        echo "<img src=\"https://img.shields.io/badge/Pull%20Changes-$pull_total%20files-blue.svg\" alt=\"Pull Changes\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Added-$pull_added-brightgreen.svg\" alt=\"Added\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Modified-$pull_modified-orange.svg\" alt=\"Modified\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Deleted-$pull_deleted-red.svg\" alt=\"Deleted\">" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
    fi

    if [ -n "$PUSH_DIFF" ]; then
        push_total=$(echo "$PUSH_DIFF" | wc -l)
        push_added=$(count_by_type "$PUSH_DIFF" "A")
        push_modified=$(count_by_type "$PUSH_DIFF" "M")
        push_deleted=$(count_by_type "$PUSH_DIFF" "D")
        echo "<img src=\"https://img.shields.io/badge/Push%20Changes-$push_total%20files-purple.svg\" alt=\"Push Changes\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Added-$push_added-brightgreen.svg\" alt=\"Added\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Modified-$push_modified-orange.svg\" alt=\"Modified\"> " >> $GITHUB_STEP_SUMMARY
        echo "<img src=\"https://img.shields.io/badge/Deleted-$push_deleted-red.svg\" alt=\"Deleted\">" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
    fi

    # Detailed file tables
    if [ -n "$PULL_DIFF" ]; then
        generate_file_table "$PULL_DIFF" "📥 Pull Changes (Upstream → Local)" >> $GITHUB_STEP_SUMMARY
    fi

    if [ -n "$PUSH_DIFF" ]; then
        generate_file_table "$PUSH_DIFF" "📤 Push Changes (Local → Upstream)" >> $GITHUB_STEP_SUMMARY
    fi
else
    echo "✅ **Everything is in sync!**" >> $GITHUB_STEP_SUMMARY
fi

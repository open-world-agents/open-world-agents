#!/bin/bash

# Compare open-world-agents: upstream vs proprietary with multiple sync options
set -e

echo "## 🔄 Open World Agents Sync Options" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Rsync-based comparison (excluding .git directories)
PULL_DIFF=$(rsync -avun --delete --exclude='.git' /tmp/upstream-owa/ ./open-world-agents/ | grep -E "(deleting|>f|cd)" || true)
PUSH_DIFF=$(rsync -avun --delete --exclude='.git' ./open-world-agents/ /tmp/upstream-owa/ | grep -E "(deleting|>f|cd)" || true)

# Parse file changes for detailed reporting
parse_rsync_output() {
    local output="$1"
    local direction="$2"

    if [ -z "$output" ]; then
        return
    fi

    echo "<details>"
    echo "<summary>📋 $direction File Changes ($(echo "$output" | wc -l) files)</summary>"
    echo ""
    echo "| Status | File |"
    echo "|--------|------|"

    echo "$output" | while IFS= read -r line; do
        if [[ $line =~ ^deleting ]]; then
            file=$(echo "$line" | sed 's/deleting //')
            echo "| 🗑️ Delete | \`$file\` |"
        elif [[ $line =~ ^\>f ]]; then
            file=$(echo "$line" | sed 's/>f[+.]* //')
            echo "| ✏️ Modify | \`$file\` |"
        elif [[ $line =~ ^cd ]]; then
            file=$(echo "$line" | sed 's/cd[+.]* //')
            echo "| 📁 New Dir | \`$file\` |"
        fi
    done

    echo ""
    echo "</details>"
    echo ""
}

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

# Create rsync patch if there are push changes
if [ -n "$PUSH_DIFF" ]; then
    ORIGINAL_DIR=$(pwd)
    cd /tmp/upstream-owa
    rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./
    git add -A && git diff --cached > /tmp/rsync-changes.patch
    cd "$ORIGINAL_DIR"
fi

# Generate comparison summary
echo "### 📊 Comparison Results" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

echo "**File-based (rsync) Status:**" >> $GITHUB_STEP_SUMMARY
if [ -n "$PULL_DIFF" ]; then
    echo "- 📥 Pull: $(echo "$PULL_DIFF" | wc -l) files would change" >> $GITHUB_STEP_SUMMARY
else
    echo "- 📥 Pull: Files match upstream" >> $GITHUB_STEP_SUMMARY
fi

if [ -n "$PUSH_DIFF" ]; then
    echo "- 📤 Push: $(echo "$PUSH_DIFF" | wc -l) files would change" >> $GITHUB_STEP_SUMMARY
else
    echo "- 📤 Push: No changes to contribute" >> $GITHUB_STEP_SUMMARY
fi

echo "" >> $GITHUB_STEP_SUMMARY

# Add detailed file change tables
if [ -n "$PULL_DIFF" ]; then
    parse_rsync_output "$PULL_DIFF" "📥 Pull (Upstream → Local)" >> $GITHUB_STEP_SUMMARY
fi

if [ -n "$PUSH_DIFF" ]; then
    parse_rsync_output "$PUSH_DIFF" "📤 Push (Local → Upstream)" >> $GITHUB_STEP_SUMMARY
fi

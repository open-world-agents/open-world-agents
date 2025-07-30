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

# Generate summary
count_changes() {
    local diff="$1"
    local adds=$(echo "$diff" | grep -c "^A" || echo "0")
    local mods=$(echo "$diff" | grep -c "^M" || echo "0")
    local dels=$(echo "$diff" | grep -c "^D" || echo "0")
    echo "+$adds ~$mods -$dels"
}

if [ -n "$PULL_DIFF" ] || [ -n "$PUSH_DIFF" ]; then
    echo "<details><summary>📊 File Changes</summary>" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "| Direction | Files | Changes |" >> $GITHUB_STEP_SUMMARY
    echo "|-----------|-------|---------|" >> $GITHUB_STEP_SUMMARY

    if [ -n "$PULL_DIFF" ]; then
        pull_count=$(echo "$PULL_DIFF" | wc -l)
        pull_changes=$(count_changes "$PULL_DIFF")
        echo "| 📥 Pull (Upstream → Local) | $pull_count | $pull_changes |" >> $GITHUB_STEP_SUMMARY
    fi

    if [ -n "$PUSH_DIFF" ]; then
        push_count=$(echo "$PUSH_DIFF" | wc -l)
        push_changes=$(count_changes "$PUSH_DIFF")
        echo "| 📤 Push (Local → Upstream) | $push_count | $push_changes |" >> $GITHUB_STEP_SUMMARY
    fi

    echo "" >> $GITHUB_STEP_SUMMARY
    echo "</details>" >> $GITHUB_STEP_SUMMARY
else
    echo "✅ **Everything is in sync!**" >> $GITHUB_STEP_SUMMARY
fi

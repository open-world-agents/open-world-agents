#!/bin/bash

# Compare open-world-agents: upstream vs proprietary with multiple sync options
set -e

echo "## 🔄 Open World Agents Sync Options" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Check if we have git subtree setup
cd open-world-agents
if git remote get-url owa >/dev/null 2>&1; then
    HAS_SUBTREE=true
    UPSTREAM_COMMIT=$(git rev-parse owa/main)
    LOCAL_COMMIT=$(git rev-parse HEAD)
else
    HAS_SUBTREE=false
fi
cd ..

# Rsync-based comparison (excluding .git directories)
PULL_DIFF=$(rsync -avun --delete --exclude='.git' /tmp/upstream-owa/ ./open-world-agents/ | grep -E "(deleting|>f|cd)" || true)
PUSH_DIFF=$(rsync -avun --delete --exclude='.git' ./open-world-agents/ /tmp/upstream-owa/ | grep -E "(deleting|>f|cd)" || true)

# Set outputs
if [ -n "$PULL_DIFF" ] || [ "$HAS_SUBTREE" = true -a "$LOCAL_COMMIT" != "$UPSTREAM_COMMIT" ]; then
    echo "pull_has_changes=true" >> $GITHUB_OUTPUT
else
    echo "pull_has_changes=false" >> $GITHUB_OUTPUT
fi

if [ -n "$PUSH_DIFF" ]; then
    echo "push_has_changes=true" >> $GITHUB_OUTPUT
    # Create rsync patch
    ORIGINAL_DIR=$(pwd)
    cd /tmp/upstream-owa
    rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./
    git add -A && git diff --cached > /tmp/rsync-changes.patch
    cd "$ORIGINAL_DIR"
else
    echo "push_has_changes=false" >> $GITHUB_OUTPUT
fi

# Generate comparison summary
echo "### 📊 Comparison Results" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

if [ "$HAS_SUBTREE" = true ]; then
    echo "**Git Subtree Status:**" >> $GITHUB_STEP_SUMMARY
    if [ "$LOCAL_COMMIT" != "$UPSTREAM_COMMIT" ]; then
        echo "- 🟡 Commits differ (local: \`${LOCAL_COMMIT:0:7}\`, upstream: \`${UPSTREAM_COMMIT:0:7}\`)" >> $GITHUB_STEP_SUMMARY
    else
        echo "- ✅ Commits match" >> $GITHUB_STEP_SUMMARY
    fi
    echo "" >> $GITHUB_STEP_SUMMARY
fi

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

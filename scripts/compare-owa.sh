#!/bin/bash

# Compare open-world-agents: upstream vs proprietary
set -euo pipefail

# Get target branch from environment or default to main
TARGET_BRANCH="${TARGET_BRANCH:-main}"
echo "Comparing against upstream branch: $TARGET_BRANCH"

# Set up GitHub Actions environment variables if not set (for local testing)
if [ -z "${GITHUB_STEP_SUMMARY:-}" ]; then
    GITHUB_STEP_SUMMARY="/tmp/github_step_summary_$$.md"
    echo "GITHUB_STEP_SUMMARY not set, using: $GITHUB_STEP_SUMMARY"
fi

if [ -z "${GITHUB_OUTPUT:-}" ]; then
    GITHUB_OUTPUT="/tmp/github_output_$$.txt"
    echo "GITHUB_OUTPUT not set, using: $GITHUB_OUTPUT"
fi

# Ensure the output files exist
touch "$GITHUB_STEP_SUMMARY" "$GITHUB_OUTPUT"

# Validate environment
if [ ! -d "/tmp/upstream-owa" ]; then
    echo "Error: /tmp/upstream-owa not found. Run git clone first." >&2
    exit 1
fi

if [ ! -d "open-world-agents" ]; then
    echo "Error: open-world-agents directory not found in current directory." >&2
    exit 1
fi

# Clear previous outputs
> "$GITHUB_STEP_SUMMARY"
> "$GITHUB_OUTPUT"

echo "## OWA Sync Report" >> "$GITHUB_STEP_SUMMARY"
echo "**Target Branch:** \`$TARGET_BRANCH\`" >> "$GITHUB_STEP_SUMMARY"
echo "" >> "$GITHUB_STEP_SUMMARY"

# Git-based comparison using git's native diff output
ORIGINAL_DIR=$(pwd)
cd /tmp/upstream-owa

# Verify we're in a git repository
if [ ! -d .git ]; then
    echo "Error: /tmp/upstream-owa is not a git repository" >&2
    exit 1
fi

# Get push changes (local → upstream)
git reset --hard HEAD >/dev/null 2>&1
rsync -av --delete --exclude='.git' "$ORIGINAL_DIR/open-world-agents/" ./ >/dev/null 2>&1
git add -A >/dev/null 2>&1
PUSH_DIFF=$(git diff --cached --name-status 2>/dev/null || true)
PUSH_STAT=$(git diff --cached --stat 2>/dev/null || true)
PUSH_STAT_SUMMARY=$(git diff --cached --stat --format='' 2>/dev/null | tail -n1 || true)

# Cleanup
git reset --hard HEAD >/dev/null 2>&1
git clean -fd >/dev/null 2>&1

# Get pull changes (upstream → local)
git reset --hard HEAD >/dev/null 2>&1
rsync -av --delete --exclude='.git' ./ "$ORIGINAL_DIR/open-world-agents/" >/dev/null 2>&1
cd "$ORIGINAL_DIR"
git add -A >/dev/null 2>&1
PULL_DIFF=$(git diff --cached --name-status 2>/dev/null || true)
PULL_STAT=$(git diff --cached --stat 2>/dev/null || true)
PULL_STAT_SUMMARY=$(git diff --cached --stat --format='' 2>/dev/null | tail -n1 || true)

# Cleanup
git reset --hard HEAD >/dev/null 2>&1
git clean -fd >/dev/null 2>&1

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

# Generate git-style diff summary
if [ -n "$PULL_DIFF" ] || [ -n "$PUSH_DIFF" ]; then
    # Pull changes
    if [ -n "$PULL_STAT" ]; then
        echo "### 📥 Pull Changes (Upstream → Local)" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "$PULL_STAT" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
    fi

    # Push changes
    if [ -n "$PUSH_STAT" ]; then
        echo "### 📤 Push Changes (Local → Upstream)" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "$PUSH_STAT" >> $GITHUB_STEP_SUMMARY
        echo '```' >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
    fi

    # Generate concise summary for comment (just totals, no detailed diff)
    echo "changes_summary<<EOF" >> $GITHUB_OUTPUT
    if [ -n "$PULL_STAT_SUMMARY" ]; then
        echo "📥 **Pull Available (Upstream → Local):** $PULL_STAT_SUMMARY" >> $GITHUB_OUTPUT
        echo "" >> $GITHUB_OUTPUT
    fi
    if [ -n "$PUSH_STAT_SUMMARY" ]; then
        echo "📤 **Push Available (Local → Upstream):** $PUSH_STAT_SUMMARY" >> $GITHUB_OUTPUT
        echo "" >> $GITHUB_OUTPUT
    fi
    echo "EOF" >> $GITHUB_OUTPUT

    # Generate detailed diff for collapsible section
    echo "detailed_changes<<EOF" >> $GITHUB_OUTPUT
    if [ -n "$PULL_STAT" ]; then
        echo "### 📥 Pull Changes (Upstream → Local)" >> $GITHUB_OUTPUT
        echo '```diff' >> $GITHUB_OUTPUT
        echo "$PULL_STAT" >> $GITHUB_OUTPUT
        echo '```' >> $GITHUB_OUTPUT
        echo "" >> $GITHUB_OUTPUT
    fi
    if [ -n "$PUSH_STAT" ]; then
        echo "### 📤 Push Changes (Local → Upstream)" >> $GITHUB_OUTPUT
        echo '```diff' >> $GITHUB_OUTPUT
        echo "$PUSH_STAT" >> $GITHUB_OUTPUT
        echo '```' >> $GITHUB_OUTPUT
        echo "" >> $GITHUB_OUTPUT
    fi
    echo "EOF" >> $GITHUB_OUTPUT
else
    echo "✅ **Everything is in sync!**" >> $GITHUB_STEP_SUMMARY
fi

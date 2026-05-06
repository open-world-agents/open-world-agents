#!/bin/bash
# OWA Release Script
# WARNING: use `vuv run scripts/release/main.py version "$VERSION" --tag --push` instead
# Usage: ./scripts/release/release.sh 1.0.0

set -e

if [[ -z "$1" ]]; then
    echo "❌ Usage: $0 <version>"
    exit 1
fi

VERSION="$1"
BRANCH="release/v$VERSION"
ORIGINAL_BRANCH=$(git branch --show-current)
RELEASE_BRANCH_CREATED=false
RELEASE_BRANCH_PUSHED=false
TAG_CREATED=false

if [[ "$ORIGINAL_BRANCH" != "main" ]]; then
    echo "❌ This script must be run from the main branch. Current branch: $ORIGINAL_BRANCH"
    exit 1
fi

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        echo "❌ Error occurred during release process. Cleaning up..."
    fi
    
    # Switch back to original branch if we're on release branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" == "$BRANCH" ]]; then
        echo "🧹 Switching back to $ORIGINAL_BRANCH..."
        git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
    fi
    
    # Delete local release branch if it was created
    if [[ "$RELEASE_BRANCH_CREATED" == "true" ]]; then
        echo "🧹 Deleting local release branch..."
        git branch -D "$BRANCH" 2>/dev/null || true
    fi
    
    # Delete remote release branch if it was pushed
    if [[ "$RELEASE_BRANCH_PUSHED" == "true" ]]; then
        echo "🧹 Deleting remote release branch..."
        git push origin --delete "$BRANCH" 2>/dev/null || true
    fi
    
    # Delete tag if it was created but release failed
    if [[ "$TAG_CREATED" == "true" && $exit_code -ne 0 ]]; then
        echo "🧹 Deleting failed release tag..."
        git tag -d "v$VERSION" 2>/dev/null || true
        git push origin --delete "v$VERSION" 2>/dev/null || true
    fi
    
    if [[ $exit_code -ne 0 ]]; then
        echo "❌ Release failed and cleanup completed."
        exit $exit_code
    fi
}

# Set trap for cleanup
trap cleanup EXIT ERR

echo "🚀 Releasing v$VERSION..."

# Create release branch and update versions
echo "📝 Creating release branch..."
git checkout -b "$BRANCH"
RELEASE_BRANCH_CREATED=true

echo "📝 Updating version..."
vuv run scripts/release/main.py version "$VERSION"

echo "📤 Pushing release branch..."
git push origin "$BRANCH"
RELEASE_BRANCH_PUSHED=true

# Rebase release branch onto main
echo "🔀 Rebasing to $ORIGINAL_BRANCH..."
git checkout "$ORIGINAL_BRANCH"
git rebase "$BRANCH"

echo "📤 Pushing rebased changes to $ORIGINAL_BRANCH..."
git push origin "$ORIGINAL_BRANCH"

# Create and push tag
echo "🏷️ Creating tag..."
git tag "v$VERSION"
TAG_CREATED=true

# Cleanup branches
echo "🧹 Cleaning up branches..."
git pull origin "$ORIGINAL_BRANCH"
git branch -d "$BRANCH"
RELEASE_BRANCH_CREATED=false
git push origin --delete "$BRANCH"
RELEASE_BRANCH_PUSHED=false

echo "📤 Pushing tag..."
git push origin "v$VERSION"

# Publish if token provided
if [[ -n "$PYPI_TOKEN" ]]; then
    echo "📦 Publishing to PyPI..."
    vuv run scripts/release/main.py publish
    echo "🎉 Released v$VERSION!"
else
    echo "📦 To publish: export PYPI_TOKEN=xxx && vuv run scripts/release/main.py publish"
    echo "🎉 Release v$VERSION prepared! (PyPI publishing skipped)"
fi
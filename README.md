# closed-world-agents

This repository contains the closed-world-agents project with [open-world-agents](https://github.com/open-world-agents/open-world-agents) integrated as a git subtree.

## Subtree Management

### Setup
```bash
git remote add owa https://github.com/open-world-agents/open-world-agents
```

### Update from upstream
```bash
git fetch owa
git subtree pull --prefix=open-world-agents owa main --squash
```

### Push changes back
```bash
# Use feature branches (recommended)
git subtree push --prefix=open-world-agents owa feature/your-feature-name

# For urgent fixes
git subtree push --prefix=open-world-agents owa hotfix/fix-name
```

### Push with hidden commit history (clean single commit)

> Following does not work, it's wrong information.

When you want to push changes to open-world-agents without exposing the detailed commit history from closed-world-agents:

```bash
# Create a single commit with just the subtree changes
SHA=$(git subtree split --prefix=open-world-agents HEAD)

# Push that single commit to a branch on the remote
git push owa ${SHA}:refs/heads/feature/your-feature-name

# Or for main branch (use with caution)
git push owa ${SHA}:main
```

### Troubleshooting
```bash
# Reset if subtree gets corrupted
git rm -r open-world-agents/
git commit -m "Remove subtree for reset"
git subtree add --prefix=open-world-agents owa main --squash
```

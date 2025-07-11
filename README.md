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

### Troubleshooting
```bash
# Reset if subtree gets corrupted
git rm -r open-world-agents/
git commit -m "Remove subtree for reset"
git subtree add --prefix=open-world-agents owa main --squash
```

#!/bin/bash
# ABSOLUTE SIMPLEST METHOD - Just copy and paste this entire command

# If you're reading this, YOU NEED TO:
# 1. Open terminal on YOUR computer
# 2. Navigate to where you cloned the autocut repository
# 3. Run this EXACT command (replace YOUR_USERNAME):

cd autocut-agent && \
cp ../specifications-v1.md docs/ 2>/dev/null || true && \
cp ../AI-AGENTS-GUIDE.md docs/ 2>/dev/null || true && \
git init && \
git branch -M main && \
git add . && \
git commit -m "Initial commit: AutoCut-Agent" && \
gh repo create autocut-agent --public --source=. --remote=origin --description="Intelligent task orchestration system for Python programs" --push && \
echo "✅ DONE! Repository created at: https://github.com/YOUR_USERNAME/autocut-agent"

# That's it! One command does everything.
# Make sure you have 'gh' installed: https://cli.github.com/

#!/bin/bash
# EMERGENCY SCRIPT: Create and push autocut-agent to GitHub NOW
# Run this on YOUR machine where you have GitHub access

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  EMERGENCY: Creating AutoCut-Agent GitHub Repository NOW  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get GitHub username
if [ -z "$1" ]; then
    echo -e "${YELLOW}Enter your GitHub username:${NC}"
    read GITHUB_USERNAME
else
    GITHUB_USERNAME="$1"
fi

REPO_NAME="${2:-autocut-agent}"
VISIBILITY="${3:-public}"

echo -e "${GREEN}✓${NC} GitHub Username: $GITHUB_USERNAME"
echo -e "${GREEN}✓${NC} Repository Name: $REPO_NAME"
echo -e "${GREEN}✓${NC} Visibility: $VISIBILITY"
echo ""

# Navigate to autocut-agent directory
cd "$(dirname "$0")/../autocut-agent" || {
    echo -e "${RED}✗${NC} autocut-agent directory not found!"
    echo "Looking in: $(dirname "$0")/../autocut-agent"
    exit 1
}

echo -e "${GREEN}✓${NC} Found autocut-agent directory: $(pwd)"
echo ""

# Copy specifications if they exist
if [ -f "../specifications-v1.md" ]; then
    echo -e "${BLUE}Copying specifications to docs...${NC}"
    cp ../specifications-v1.md docs/ 2>/dev/null || mkdir -p docs && cp ../specifications-v1.md docs/
    echo -e "${GREEN}✓${NC} Copied specifications-v1.md"
fi

if [ -f "../AI-AGENTS-GUIDE.md" ]; then
    cp ../AI-AGENTS-GUIDE.md docs/ 2>/dev/null || cp ../AI-AGENTS-GUIDE.md docs/
    echo -e "${GREEN}✓${NC} Copied AI-AGENTS-GUIDE.md"
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${BLUE}Initializing git repository...${NC}"
    git init
    git branch -M main
    echo -e "${GREEN}✓${NC} Git initialized"
else
    echo -e "${GREEN}✓${NC} Git already initialized"
fi

# Add all files
echo -e "${BLUE}Adding all files...${NC}"
git add .
echo -e "${GREEN}✓${NC} Files added"

# Create initial commit if needed
if ! git log -1 &>/dev/null; then
    echo -e "${BLUE}Creating initial commit...${NC}"
    git commit -m "Initial commit: AutoCut-Agent complete project

Complete project bootstrap including:
- Comprehensive specifications and AI guide
- Complete source structure (10 modules)
- Configuration files (Poetry, Docker, VSCode)
- Setup scripts for all platforms
- Tests, examples, configs
- Complete documentation
- AI assistant configurations

Ready for development!"
    echo -e "${GREEN}✓${NC} Initial commit created"
else
    echo -e "${GREEN}✓${NC} Commits already exist"
fi

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo ""
    echo -e "${RED}✗${NC} GitHub CLI (gh) not found!"
    echo ""
    echo -e "${YELLOW}MANUAL STEPS:${NC}"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Visibility: $VISIBILITY"
    echo "4. DON'T initialize with README"
    echo "5. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo "  cd $(pwd)"
    echo "  git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "  git push -u origin main"
    exit 1
fi

# Authenticate if needed
echo -e "${BLUE}Checking GitHub authentication...${NC}"
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}Not authenticated. Starting authentication...${NC}"
    gh auth login
fi
echo -e "${GREEN}✓${NC} Authenticated"

# Check if repository already exists
if gh repo view "$GITHUB_USERNAME/$REPO_NAME" &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  Repository already exists!"
    echo ""
    echo -e "${YELLOW}Do you want to push to existing repository? (y/N):${NC}"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
    
    # Add remote and push
    if git remote get-url origin &> /dev/null; then
        git remote remove origin
    fi
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    
    echo -e "${BLUE}Pushing to existing repository...${NC}"
    git push -u origin main --force
else
    # Create new repository
    echo -e "${BLUE}Creating GitHub repository...${NC}"
    VISIBILITY_FLAG="--$VISIBILITY"
    
    gh repo create "$REPO_NAME" \
        $VISIBILITY_FLAG \
        --source=. \
        --remote=origin \
        --description="Intelligent task orchestration system for Python programs with multi-trigger support, resource management, and LLM integration" \
        --push
fi

echo ""
echo -e "${GREEN}✓${NC} Repository created and pushed!"
echo ""

# Add topics
echo -e "${BLUE}Adding repository topics...${NC}"
gh repo edit --add-topic python
gh repo edit --add-topic task-orchestration
gh repo edit --add-topic automation
gh repo edit --add-topic llm
gh repo edit --add-topic fastapi
gh repo edit --add-topic gpu-scheduling
gh repo edit --add-topic workflow-automation
echo -e "${GREEN}✓${NC} Topics added"

# Update URLs in documentation
echo -e "${BLUE}Updating documentation URLs...${NC}"
find . -type f -name "*.md" -exec sed -i.bak "s|github.com/targuy/autocut-agent|github.com/$GITHUB_USERNAME/$REPO_NAME|g" {} \;
find . -type f -name "*.md.bak" -delete

if [ -f "pyproject.toml" ]; then
    sed -i.bak "s|github.com/targuy/autocut-agent|github.com/$GITHUB_USERNAME/$REPO_NAME|g" pyproject.toml
    rm -f pyproject.toml.bak
fi

if ! git diff --quiet; then
    git add .
    git commit -m "docs: update repository URLs to $GITHUB_USERNAME/$REPO_NAME"
    git push
    echo -e "${GREEN}✓${NC} URLs updated"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              REPOSITORY CREATED SUCCESSFULLY!              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓${NC} Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo -e "${GREEN}✓${NC} Clone command: git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Visit: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "  2. Enable Issues and Discussions in Settings"
echo "  3. Set up branch protection for main"
echo "  4. Start developing!"
echo ""
echo -e "${GREEN}🎉 All done!${NC}"

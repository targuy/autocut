# How to Move AutoCut-Agent to a New Repository

This guide explains how to move the `autocut-agent/` project from the current `targuy/autocut` repository to a new standalone `targuy/autocut-agent` repository.

## Option 1: Quick Method (Copy Files)

### Step 1: Create New Repository on GitHub

1. Go to GitHub and create a new repository:
   - Name: `autocut-agent`
   - Description: "Intelligent task orchestration system for Python programs"
   - Visibility: Public or Private
   - ✅ Initialize with README: **NO** (we have our own)
   - ✅ Add .gitignore: **NO** (we have our own)
   - ✅ Add license: **NO** (we have MIT license)

### Step 2: Prepare Local Directory

```bash
# Navigate to the autocut repository
cd /path/to/autocut

# Copy the autocut-agent directory to a new location
cp -r autocut-agent /tmp/autocut-agent
cd /tmp/autocut-agent
```

### Step 3: Initialize New Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete AutoCut-Agent project bootstrap

- Enhanced specifications (specifications-v1.md)
- AI agents guide (AI-AGENTS-GUIDE.md)
- Complete directory structure
- Configuration files (pyproject.toml, requirements.txt)
- Docker support (Dockerfile, docker-compose.yml)
- VSCode workspace configuration
- Setup scripts for all platforms
- GitHub Copilot, Cursor, and Aider configurations
- README, CONTRIBUTING, LICENSE
- Example programs
- Default configurations
- Basic CLI entry point"

# Add remote
git remote add origin https://github.com/targuy/autocut-agent.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Verify

```bash
# Check repository
git remote -v

# View commit history
git log --oneline

# Check files
ls -la
```

## Option 2: Preserve Git History (Subdirectory Filter)

If you want to preserve any git history from the original repository:

### Step 1: Clone Original Repository

```bash
# Clone the original repository
git clone https://github.com/targuy/autocut.git autocut-filtered
cd autocut-filtered
```

### Step 2: Filter to Subdirectory

```bash
# Filter repository to only include autocut-agent/ directory
git filter-repo --path autocut-agent/ --path-rename autocut-agent/:

# Or using git-filter-branch (older method)
git filter-branch --subdirectory-filter autocut-agent -- --all
```

### Step 3: Push to New Repository

```bash
# Add new remote
git remote add origin https://github.com/targuy/autocut-agent.git

# Push
git push -u origin main
```

## Option 3: GitHub Import (Web Interface)

### Step 1: Create New Repository

1. Go to GitHub
2. Click "New Repository"
3. Name: `autocut-agent`
4. Import from: Leave empty
5. Create repository

### Step 2: Upload Files

1. Download the `autocut-agent/` folder as ZIP from current repository
2. Extract ZIP
3. Use GitHub's web interface to upload files:
   - Click "uploading an existing file"
   - Drag and drop all files and folders
   - Commit changes

## Post-Migration Tasks

### Update Documentation

1. **Update README.md**
   - Change repository URLs
   - Update clone instructions
   - Update issue/discussion links

2. **Update package metadata**
   - Update `pyproject.toml` with correct repository URL
   - Update author information if needed

3. **Update CI/CD** (if applicable)
   - Set up GitHub Actions
   - Configure branch protection
   - Set up status checks

### Set Up Repository Settings

1. **Branch Protection**
   - Require pull request reviews
   - Require status checks
   - Require branches to be up to date

2. **GitHub Pages** (optional)
   - Enable for documentation
   - Set source to `docs/` folder or gh-pages branch

3. **Secrets** (for CI/CD)
   - Add API keys for testing
   - Add deployment credentials

4. **Labels**
   - Create labels: bug, enhancement, documentation, etc.
   - Import label templates

5. **Milestones**
   - Create milestones for development phases
   - Link issues to milestones

### Initial Repository Setup

```bash
# Clone new repository
git clone https://github.com/targuy/autocut-agent.git
cd autocut-agent

# Run setup script
bash scripts/setup.sh

# Create development branch
git checkout -b develop

# Install dependencies
poetry install

# Run tests (when available)
pytest

# Verify everything works
autocut-agent --version
```

## Recommended Repository Structure on GitHub

```
targuy/autocut-agent
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── question.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── docs.yml
│   └── copilot-instructions.md
├── docs/
│   └── [documentation files]
├── src/
│   └── [source code]
├── tests/
│   └── [test files]
├── examples/
│   └── [example files]
├── configs/
│   └── [configuration templates]
├── scripts/
│   └── [setup scripts]
├── .gitignore
├── .env.template
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── specifications-v1.md
├── AI-AGENTS-GUIDE.md
└── PROJECT-SUMMARY.md
```

## GitHub Actions Workflow Example

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install
    
    - name: Lint with Ruff
      run: poetry run ruff check .
    
    - name: Format check with Black
      run: poetry run black --check .
    
    - name: Type check with MyPy
      run: poetry run mypy src/
    
    - name: Test with pytest
      run: poetry run pytest --cov --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Update Links in Documentation

After creating the new repository, update these files:

1. **README.md**
   ```markdown
   - Repository: https://github.com/targuy/autocut-agent
   - Issues: https://github.com/targuy/autocut-agent/issues
   - Discussions: https://github.com/targuy/autocut-agent/discussions
   ```

2. **pyproject.toml**
   ```toml
   [tool.poetry]
   repository = "https://github.com/targuy/autocut-agent"
   homepage = "https://github.com/targuy/autocut-agent"
   ```

3. **PROJECT-SUMMARY.md**
   ```markdown
   **Repository:** github.com/targuy/autocut-agent
   ```

## Checklist

Before considering the migration complete:

- [ ] New repository created on GitHub
- [ ] All files copied and committed
- [ ] README updated with correct repository URLs
- [ ] Setup script tested on clean environment
- [ ] Docker build tested
- [ ] Docker Compose tested
- [ ] Dependencies installable via Poetry
- [ ] Dependencies installable via pip
- [ ] CLI entry point works
- [ ] Documentation links updated
- [ ] LICENSE file present
- [ ] .gitignore configured correctly
- [ ] GitHub repository settings configured
- [ ] Branch protection enabled
- [ ] CI/CD workflows set up (if desired)
- [ ] Repository description set
- [ ] Topics/tags added for discoverability

## Discoverability Tags

Add these topics to the GitHub repository:

- `python`
- `task-orchestration`
- `automation`
- `llm`
- `fastapi`
- `queue-management`
- `resource-management`
- `gpu-scheduling`
- `task-scheduler`
- `workflow-automation`
- `agent`
- `microservices`

## Final Notes

### Maintenance

After migration:
1. Keep the original `autocut-agent/` folder in `targuy/autocut` as is or remove it
2. If keeping both, add a note in original repo pointing to new repo
3. Set up syncing if you want to keep them in sync

### Communication

Announce the new repository:
1. Update any documentation mentioning the old location
2. Inform team members of the new repository URL
3. Update any external links or references

### Backup

Before deleting anything:
1. Make a complete backup of the `autocut-agent/` folder
2. Verify new repository has all files
3. Test that everything works in new repository

## Getting Help

If you encounter issues:

1. Check GitHub documentation on repository creation
2. Review git documentation for advanced operations
3. Test in a temporary location first
4. Keep backups of everything

## Success Criteria

The migration is successful when:
- ✅ New repository accessible at `github.com/targuy/autocut-agent`
- ✅ All files present and intact
- ✅ Setup script runs successfully
- ✅ Dependencies install correctly
- ✅ Docker build works
- ✅ Documentation accurate
- ✅ Links all updated
- ✅ Repository properly configured

---

**Note:** This guide is included in the autocut-agent project to facilitate easy extraction and setup as a standalone repository.

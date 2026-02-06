# EMERGENCY: Create AutoCut-Agent Repository NOW
# Windows PowerShell Script
# Run: powershell -ExecutionPolicy Bypass -File create-repo-now.ps1 YOUR_USERNAME

param(
    [Parameter(Mandatory=$false)]
    [string]$GitHubUsername = "",
    
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "autocut-agent",
    
    [Parameter(Mandatory=$false)]
    [string]$Visibility = "public"
)

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  EMERGENCY: Creating AutoCut-Agent GitHub Repository NOW  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get GitHub username if not provided
if ([string]::IsNullOrEmpty($GitHubUsername)) {
    $GitHubUsername = Read-Host "Enter your GitHub username"
}

Write-Host "✓ GitHub Username: $GitHubUsername" -ForegroundColor Green
Write-Host "✓ Repository Name: $RepoName" -ForegroundColor Green
Write-Host "✓ Visibility: $Visibility" -ForegroundColor Green
Write-Host ""

# Navigate to autocut-agent directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$agentPath = Join-Path $scriptPath "autocut-agent"

if (-not (Test-Path $agentPath)) {
    Write-Host "✗ autocut-agent directory not found!" -ForegroundColor Red
    Write-Host "Looking in: $agentPath"
    exit 1
}

Set-Location $agentPath
Write-Host "✓ Found autocut-agent directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Copy specifications if they exist
$specsPath = Join-Path $scriptPath "specifications-v1.md"
$aiGuidePath = Join-Path $scriptPath "AI-AGENTS-GUIDE.md"

if (Test-Path $specsPath) {
    Write-Host "Copying specifications to docs..." -ForegroundColor Blue
    if (-not (Test-Path "docs")) { New-Item -ItemType Directory -Path "docs" | Out-Null }
    Copy-Item $specsPath "docs/" -Force
    Write-Host "✓ Copied specifications-v1.md" -ForegroundColor Green
}

if (Test-Path $aiGuidePath) {
    Copy-Item $aiGuidePath "docs/" -Force
    Write-Host "✓ Copied AI-AGENTS-GUIDE.md" -ForegroundColor Green
}

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Blue
    git init
    git branch -M main
    Write-Host "✓ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git already initialized" -ForegroundColor Green
}

# Add all files
Write-Host "Adding all files..." -ForegroundColor Blue
git add .
Write-Host "✓ Files added" -ForegroundColor Green

# Create initial commit if needed
$hasCommits = git log -1 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating initial commit..." -ForegroundColor Blue
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
    Write-Host "✓ Initial commit created" -ForegroundColor Green
} else {
    Write-Host "✓ Commits already exist" -ForegroundColor Green
}

# Check if gh is installed
$ghInstalled = Get-Command gh -ErrorAction SilentlyContinue
if (-not $ghInstalled) {
    Write-Host ""
    Write-Host "✗ GitHub CLI (gh) not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "MANUAL STEPS:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://github.com/new"
    Write-Host "2. Repository name: $RepoName"
    Write-Host "3. Visibility: $Visibility"
    Write-Host "4. DON'T initialize with README"
    Write-Host "5. Click 'Create repository'"
    Write-Host ""
    Write-Host "Then run these commands:"
    Write-Host "  cd $(Get-Location)"
    Write-Host "  git remote add origin https://github.com/$GitHubUsername/$RepoName.git"
    Write-Host "  git push -u origin main"
    exit 1
}

# Check authentication
Write-Host "Checking GitHub authentication..." -ForegroundColor Blue
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not authenticated. Starting authentication..." -ForegroundColor Yellow
    gh auth login
}
Write-Host "✓ Authenticated" -ForegroundColor Green

# Check if repository already exists
$repoExists = gh repo view "$GitHubUsername/$RepoName" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠ Repository already exists!" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to push to existing repository? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Aborted" -ForegroundColor Red
        exit 1
    }
    
    # Add remote and push
    $hasOrigin = git remote get-url origin 2>&1
    if ($LASTEXITCODE -eq 0) {
        git remote remove origin
    }
    git remote add origin "https://github.com/$GitHubUsername/$RepoName.git"
    
    Write-Host "Pushing to existing repository..." -ForegroundColor Blue
    git push -u origin main --force
} else {
    # Create new repository
    Write-Host "Creating GitHub repository..." -ForegroundColor Blue
    
    $visibilityFlag = "--$Visibility"
    
    gh repo create $RepoName `
        $visibilityFlag `
        --source=. `
        --remote=origin `
        --description="Intelligent task orchestration system for Python programs with multi-trigger support, resource management, and LLM integration" `
        --push
}

Write-Host ""
Write-Host "✓ Repository created and pushed!" -ForegroundColor Green
Write-Host ""

# Add topics
Write-Host "Adding repository topics..." -ForegroundColor Blue
gh repo edit --add-topic python
gh repo edit --add-topic task-orchestration
gh repo edit --add-topic automation
gh repo edit --add-topic llm
gh repo edit --add-topic fastapi
gh repo edit --add-topic gpu-scheduling
gh repo edit --add-topic workflow-automation
Write-Host "✓ Topics added" -ForegroundColor Green

# Update URLs in documentation
Write-Host "Updating documentation URLs..." -ForegroundColor Blue
Get-ChildItem -Path . -Filter "*.md" -Recurse | ForEach-Object {
    (Get-Content $_.FullName) -replace "github.com/targuy/autocut-agent", "github.com/$GitHubUsername/$RepoName" | Set-Content $_.FullName
}

if (Test-Path "pyproject.toml") {
    (Get-Content "pyproject.toml") -replace "github.com/targuy/autocut-agent", "github.com/$GitHubUsername/$RepoName" | Set-Content "pyproject.toml"
}

$hasChanges = git diff
if ($hasChanges) {
    git add .
    git commit -m "docs: update repository URLs to $GitHubUsername/$RepoName"
    git push
    Write-Host "✓ URLs updated" -ForegroundColor Green
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              REPOSITORY CREATED SUCCESSFULLY!              ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "✓ Repository URL: https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Green
Write-Host "✓ Clone command: git clone https://github.com/$GitHubUsername/$RepoName.git" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Blue
Write-Host "  1. Visit: https://github.com/$GitHubUsername/$RepoName"
Write-Host "  2. Enable Issues and Discussions in Settings"
Write-Host "  3. Set up branch protection for main"
Write-Host "  4. Start developing!"
Write-Host ""
Write-Host "🎉 All done!" -ForegroundColor Green

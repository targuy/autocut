# AutoCut-Agent Setup Script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "AutoCut-Agent Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1 | Select-String -Pattern "\d+\.\d+" | ForEach-Object { $_.Matches.Value }
    $requiredVersion = [version]"3.10"
    $currentVersion = [version]$pythonVersion
    
    if ($currentVersion -lt $requiredVersion) {
        Write-Host "❌ Error: Python 3.10 or higher is required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Python $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python not found. Please install Python 3.10 or higher" -ForegroundColor Red
    exit 1
}

# Check if Poetry is installed
Write-Host "Checking Poetry..." -ForegroundColor Yellow
$poetryInstalled = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryInstalled) {
    Write-Host "Poetry not found. Installing Poetry..." -ForegroundColor Yellow
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    $env:Path += ";$env:APPDATA\Python\Scripts"
    Write-Host "✓ Poetry installed" -ForegroundColor Green
} else {
    Write-Host "✓ Poetry found" -ForegroundColor Green
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
if (Test-Path "pyproject.toml") {
    poetry install
    Write-Host "✓ Dependencies installed with Poetry" -ForegroundColor Green
} else {
    pip install -r requirements.txt
    pip install -e .
    Write-Host "✓ Dependencies installed with pip" -ForegroundColor Green
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @("data", "logs", "outputs", "configs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    @"
# AutoCut-Agent Environment Variables

# API Configuration
API_SECRET_KEY=change-this-in-production

# Database (optional, defaults to SQLite)
# DATABASE_URL=postgresql://user:pass@localhost:5432/autocut

# Redis (optional)
# REDIS_URL=redis://localhost:6379/0

# LLM API Keys (optional)
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here

# SMTP for Email Alerts (optional)
# SMTP_PASSWORD=your-smtp-password
"@ | Out-File -FilePath ".env" -Encoding utf8
    Write-Host "✓ .env file created (please update with your values)" -ForegroundColor Green
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit: AutoCut-Agent setup"
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git repository already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Setup Complete! 🎉" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate virtual environment: .\.venv\Scripts\Activate.ps1"
Write-Host "2. Update .env file with your configuration"
Write-Host "3. Edit configs\default.yaml for your use case"
Write-Host "4. Run the agent: autocut-agent start --config configs\default.yaml"
Write-Host ""
Write-Host "For development:" -ForegroundColor Yellow
Write-Host "  autocut-agent start --config configs\development.yaml --dev"
Write-Host ""
Write-Host "Documentation: See README.md and docs/" -ForegroundColor Yellow
Write-Host ""

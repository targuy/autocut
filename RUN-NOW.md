# EMERGENCY: Create AutoCut-Agent Repository NOW

## 🚨 IMMEDIATE ACTION - Run This Command

### On Linux/macOS:

```bash
cd /home/runner/work/autocut/autocut
bash create-repo-now.sh YOUR_GITHUB_USERNAME
```

### On Windows:

```powershell
cd C:\path\to\autocut
powershell -ExecutionPolicy Bypass -File create-repo-now.ps1 YOUR_GITHUB_USERNAME
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username (e.g., `targuy`).

## What This Does

The script will:
1. ✅ Navigate to autocut-agent directory
2. ✅ Copy specifications and AI guide to docs
3. ✅ Initialize git repository
4. ✅ Create initial commit
5. ✅ Authenticate with GitHub (if needed)
6. ✅ Create the repository on GitHub
7. ✅ Push all files
8. ✅ Set description and topics
9. ✅ Update documentation URLs

**Time: ~30-60 seconds**

## If GitHub CLI Not Installed

### Install GitHub CLI:

**macOS:**
```bash
brew install gh
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

**Windows:**
Download from: https://cli.github.com/

Then authenticate:
```bash
gh auth login
```

## Manual Method (If Script Fails)

```bash
cd autocut-agent

# Copy specs
cp ../specifications-v1.md docs/
cp ../AI-AGENTS-GUIDE.md docs/

# Initialize git
git init
git branch -M main
git add .
git commit -m "Initial commit: AutoCut-Agent"

# Create repository on GitHub
gh repo create autocut-agent --public --source=. --remote=origin --push

# Or manually:
# 1. Go to https://github.com/new
# 2. Create repo named "autocut-agent"
# 3. Then:
git remote add origin https://github.com/YOUR_USERNAME/autocut-agent.git
git push -u origin main
```

## Result

You will have a new repository at:
```
https://github.com/YOUR_USERNAME/autocut-agent
```

With all 43+ files ready to go!

## Troubleshooting

### "gh: command not found"
Install GitHub CLI (see above)

### "Not authenticated"
Run: `gh auth login`

### "Repository already exists"
The script will ask if you want to push to it anyway

### "Permission denied"
Make script executable: `chmod +x create-repo-now.sh`

---

**Status**: ✅ Ready to run RIGHT NOW
**Time needed**: 30-60 seconds
**Result**: Complete repository on GitHub

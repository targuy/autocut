#!/bin/bash
# RUN THIS ON YOUR MACBOOK to get the autocut-agent code immediately

echo "🚨 Getting autocut-agent code on your MacBook..."
echo ""

# Check if we're in a git repo
if [ -d ".git" ]; then
    echo "✓ Found git repository"
    echo "Fetching latest changes..."
    git fetch origin
    
    echo "Checking out branch with autocut-agent..."
    git checkout copilot/initialize-autocut-agent-structure
    
    if [ -d "autocut-agent" ]; then
        echo ""
        echo "✅ SUCCESS! autocut-agent is now on your MacBook!"
        echo ""
        echo "Location: $(pwd)/autocut-agent"
        echo "Files: $(find autocut-agent -type f | wc -l) files"
        echo ""
        echo "To see the code:"
        echo "  cd autocut-agent"
        echo "  ls -la"
        echo ""
    else
        echo "❌ Error: autocut-agent directory not found after checkout"
        exit 1
    fi
else
    echo "Not in a git repository. Cloning now..."
    
    # Ask for directory
    echo "Where do you want to clone the repository?"
    echo "Press Enter for current directory, or type a path:"
    read CLONE_DIR
    
    if [ -z "$CLONE_DIR" ]; then
        CLONE_DIR="."
    fi
    
    cd "$CLONE_DIR"
    
    echo "Cloning repository..."
    git clone https://github.com/targuy/autocut.git
    
    cd autocut
    
    echo "Checking out branch with autocut-agent..."
    git checkout copilot/initialize-autocut-agent-structure
    
    if [ -d "autocut-agent" ]; then
        echo ""
        echo "✅ SUCCESS! autocut-agent is now on your MacBook!"
        echo ""
        echo "Location: $(pwd)/autocut-agent"
        echo "Files: $(find autocut-agent -type f | wc -l) files"
        echo ""
        echo "To see the code:"
        echo "  cd $(pwd)/autocut-agent"
        echo "  ls -la"
        echo ""
    else
        echo "❌ Error: autocut-agent directory not found after checkout"
        exit 1
    fi
fi

echo "Next steps:"
echo "1. cd autocut-agent"
echo "2. cat README.md"
echo "3. bash ../create-repo-now.sh YOUR_USERNAME  (to create GitHub repo)"
echo ""
echo "🎉 Done!"

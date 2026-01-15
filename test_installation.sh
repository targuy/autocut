#!/bin/bash
# Test script to validate installation script logic without actually installing

set -e

echo "Testing installation scripts..."
echo "=============================="
echo

# Test 1: Check install.sh syntax
echo "Test 1: Validating install.sh bash syntax..."
if bash -n install.sh; then
    echo "✓ install.sh syntax is valid"
else
    echo "✗ install.sh has syntax errors"
    exit 1
fi
echo

# Test 2: Check that install.sh is executable
echo "Test 2: Checking install.sh permissions..."
if [ -x install.sh ]; then
    echo "✓ install.sh is executable"
else
    echo "✗ install.sh is not executable"
    exit 1
fi
echo

# Test 3: Verify pyproject.toml is valid
echo "Test 3: Validating pyproject.toml..."
if python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"; then
    echo "✓ pyproject.toml is valid TOML"
else
    echo "✗ pyproject.toml has syntax errors"
    exit 1
fi
echo

# Test 4: Check that extras are defined
echo "Test 4: Checking Poetry extras..."
python3 << 'EOF'
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    
extras = data.get('tool', {}).get('poetry', {}).get('extras', {})
expected_extras = ['cpu', 'cuda', 'ffmpeg', 'all']

print(f"Found extras: {list(extras.keys())}")
for extra in expected_extras:
    if extra in extras:
        print(f"  ✓ {extra}: {extras[extra]}")
    else:
        print(f"  ✗ Missing extra: {extra}")
        exit(1)
EOF
echo

# Test 5: Verify optional dependencies
echo "Test 5: Checking optional dependencies..."
python3 << 'EOF'
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    
deps = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
optional_deps = ['torch', 'torchvision', 'onnxruntime', 'onnxruntime-gpu', 'imageio-ffmpeg']

for dep in optional_deps:
    if dep in deps:
        dep_config = deps[dep]
        if isinstance(dep_config, dict) and dep_config.get('optional', False):
            print(f"  ✓ {dep} is optional")
        else:
            print(f"  ✗ {dep} should be optional")
            exit(1)
    else:
        print(f"  ✗ Missing optional dependency: {dep}")
        exit(1)
EOF
echo

# Test 6: Verify no conflicts in extras
echo "Test 6: Checking for package conflicts..."
python3 << 'EOF'
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    
extras = data.get('tool', {}).get('poetry', {}).get('extras', {})

# Check CPU extra doesn't have GPU packages
cpu_extra = extras.get('cpu', [])
if 'onnxruntime-gpu' in cpu_extra:
    print("  ✗ CPU extra should not include onnxruntime-gpu")
    exit(1)
else:
    print("  ✓ CPU extra does not include GPU packages")

# Check CUDA extra has GPU packages
cuda_extra = extras.get('cuda', [])
if 'onnxruntime-gpu' in cuda_extra and 'onnxruntime' not in cuda_extra:
    print("  ✓ CUDA extra includes onnxruntime-gpu and not onnxruntime")
else:
    print("  ✗ CUDA extra should include onnxruntime-gpu")
    exit(1)
EOF
echo

# Test 7: Check documentation files exist
echo "Test 7: Checking documentation files..."
for doc in README.md INSTALL.md REQUIREMENTS_ANALYSIS.md; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc exists"
    else
        echo "  ✗ $doc is missing"
        exit 1
    fi
done
echo

# Test 8: Verify run scripts are created by install.sh
echo "Test 8: Checking for run script creation logic in install.sh..."
if grep -q "cat > run.sh" install.sh; then
    echo "  ✓ install.sh creates run.sh"
else
    echo "  ✗ install.sh should create run.sh"
    exit 1
fi
echo

echo "=============================="
echo "All tests passed! ✓"
echo
echo "Note: These tests validate script syntax and configuration."
echo "To fully test installation, run install.sh in a clean environment."

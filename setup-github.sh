#!/bin/bash
# GitHub Repository Setup Script for cuda2ripple
# 
# This script helps you create and push the cuda2ripple repository to GitHub.
# Run this after extracting the cuda2ripple.zip file.

set -e

echo "=========================================="
echo "  cuda2ripple GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Creating repository..."
    
    # Create the repository
    gh repo create cuda2ripple \
        --public \
        --description "Translate CUDA code to RIPPLE for Hexagon HVX and other SIMD targets" \
        --source . \
        --push
    
    echo ""
    echo "✅ Repository created and code pushed!"
    echo "   Visit: https://github.com/$(gh api user -q .login)/cuda2ripple"
else
    echo "GitHub CLI not found. Using git directly..."
    echo ""
    echo "Please create a repository on GitHub first:"
    echo "  1. Go to https://github.com/new"
    echo "  2. Repository name: cuda2ripple"
    echo "  3. Description: Translate CUDA code to RIPPLE for Hexagon HVX and other SIMD targets"
    echo "  4. Set to Public"
    echo "  5. Do NOT initialize with README (we have one)"
    echo "  6. Click 'Create repository'"
    echo ""
    read -p "Enter your GitHub username: " GITHUB_USER
    echo ""
    
    # Initialize git and push
    git init
    git add .
    git commit -m "Initial commit: CUDA to RIPPLE translator

Features:
- Source-level translation (CUDA C → RIPPLE C)
- IR-level translation (CUDA LLVM IR → RIPPLE LLVM IR)
- CLI, Web, and VS Code interfaces
- Hexagon HVX optimization
- 20+ translation rules for CUDA patterns"
    
    git branch -M main
    git remote add origin "https://github.com/${GITHUB_USER}/cuda2ripple.git"
    git push -u origin main
    
    echo ""
    echo "✅ Code pushed!"
    echo "   Visit: https://github.com/${GITHUB_USER}/cuda2ripple"
fi

echo ""
echo "=========================================="
echo "  Next Steps:"
echo "=========================================="
echo "  • Add topics: cuda, ripple, hexagon, hvx, simd, compiler"
echo "  • Enable GitHub Pages for documentation"
echo "  • Set up GitHub Actions for CI/CD"
echo "=========================================="

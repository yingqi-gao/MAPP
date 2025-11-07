#!/bin/bash

# MAPP Environment Setup Script
# This script sets up the Python virtual environment for the MAPP project
# Requires: System R installation with required packages

set -e  # Exit on any error

# Parse command line arguments
FORCE_REINSTALL=false
for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE_REINSTALL=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f    Remove existing venv without prompting"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
    esac
done

echo "🚀 Setting up MAPP environment..."

# Check if R is installed
echo "🔍 Checking R installation..."
if ! command -v R &> /dev/null; then
    echo "❌ R not found. Please install R first:"
    echo ""
    echo "📋 Installation instructions:"
    echo "  macOS: brew install r  OR  download from https://cran.r-project.org/bin/macosx/"
    echo "  Ubuntu: sudo apt-get install r-base r-base-dev"
    echo "  Windows: Download from https://cran.r-project.org/bin/windows/base/"
    echo ""
    exit 1
fi

# Check R version (require 4.0+, recommend 4.2+)
# Clear R_HOME temporarily to avoid version parsing issues
unset R_HOME
R_VERSION=$(R --slave -e "cat(as.character(getRversion()))" 2>/dev/null)
R_MAJOR=$(echo $R_VERSION | cut -d. -f1)
R_MINOR=$(echo $R_VERSION | cut -d. -f2)

if [ "$R_MAJOR" -lt 4 ]; then
    echo "❌ R version $R_VERSION found, but version 4.0+ is required"
    echo "Please update R to version 4.0 or higher"
    exit 1
elif [ "$R_MAJOR" -eq 4 ] && [ "$R_MINOR" -lt 2 ]; then
    echo "⚠️  R $R_VERSION found. Version 4.2+ recommended for best compatibility"
    echo "✅ R $R_VERSION is compatible (minimum 4.0+)"
else
    echo "✅ R $R_VERSION found"
fi

# Set R_HOME environment variable
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check common R installation paths
    if [ -d "/Library/Frameworks/R.framework/Resources" ]; then
        export R_HOME="/Library/Frameworks/R.framework/Resources"
    elif [ -d "/usr/local/lib/R" ]; then
        export R_HOME="/usr/local/lib/R"
    else
        echo "⚠️  R installation found but R_HOME path unclear. Setting from R itself..."
        export R_HOME=$(R RHOME)
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    export R_HOME=$(R RHOME)
else
    # Windows/other
    export R_HOME=$(R RHOME)
fi

echo "📍 R_HOME set to: $R_HOME"

# Check Python installation (require 3.11 for reproducibility)
echo "🐍 Checking Python installation..."

if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    PYTHON_VERSION=$(python3.11 --version | grep -oE '[0-9]+\.[0-9]+')
    echo "✅ Python $PYTHON_VERSION found ($PYTHON_CMD)"
else
    echo "❌ Python 3.11 not found but is required for reproducibility"
    echo ""
    echo "📋 Installation instructions:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu: sudo apt-get install python3.11 python3.11-venv"
    echo "  Windows: Download Python 3.11 from https://python.org"
    echo ""
    echo "💡 Why Python 3.11? Ensures reproducible results across all environments"
    exit 1
fi

# Check and install required R packages
echo "📦 Checking R packages..."

# Check if fdapace is installed
if ! Rscript -e "if (!require('fdapace', quietly = TRUE)) quit(status = 1)" &>/dev/null; then
    echo "📊 Installing fdapace from CRAN (using binaries)..."
    Rscript -e "
    options(repos = c(CRAN = 'https://cloud.r-project.org/'))
    # Install binaries to avoid compilation issues
    install.packages('fdapace', quiet = TRUE, dependencies = TRUE, type = 'binary')
    " || {
        echo "⚠️  Binary installation failed, trying with source..."
        Rscript -e "
        options(repos = c(CRAN = 'https://cloud.r-project.org/'))
        install.packages('fdapace', quiet = TRUE, dependencies = TRUE)
        " || {
            echo "❌ Failed to install fdapace"
            echo "💡 You may need to install gfortran: brew install gcc"
            exit 1
        }
    }
else
    echo "✅ fdapace already installed"
fi

# Check if densityFPCA is installed
if ! Rscript -e "if (!require('densityFPCA', quietly = TRUE)) quit(status = 1)" &>/dev/null; then
    echo "📊 Installing densityFPCA from GitHub..."

    # Fix macOS compilation issues for densityFPCA if needed
    MOVED_HEADERS=false
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🔧 Checking for potential macOS compilation issues..."

        # Check for conflicting system headers
        conflicting_headers=()
        for f in math.h stdlib.h string.h ctype.h wchar.h wctype.h errno.h float.h stddef.h stdio.h; do
            if [ -f "/usr/local/include/$f" ]; then
                conflicting_headers+=("$f")
            fi
        done

        if [ ${#conflicting_headers[@]} -gt 0 ]; then
            echo "⚠️  Found ${#conflicting_headers[@]} potentially conflicting headers: ${conflicting_headers[*]}"
            echo "Temporarily moving headers to avoid compilation conflicts..."
            sudo mkdir -p /usr/local/include_backup_temp
            for f in "${conflicting_headers[@]}"; do
                sudo mv "/usr/local/include/$f" /usr/local/include_backup_temp/
            done
            MOVED_HEADERS=true
            echo "✅ Headers temporarily moved"
        fi
    fi

    # Install remotes if not available and densityFPCA
    Rscript -e "
    options(repos = c(CRAN = 'https://cloud.r-project.org/'))
    if (!require('remotes', quietly = TRUE)) {
        install.packages('remotes', quiet = TRUE, type = 'binary')
    }
    remotes::install_github('jiamingqiu/densityFPCA', quiet = TRUE, force = TRUE, type = 'binary')
    " || {
        echo "⚠️  Binary installation failed, trying with source..."
        Rscript -e "
        options(repos = c(CRAN = 'https://cloud.r-project.org/'))
        if (!require('remotes', quietly = TRUE)) {
            install.packages('remotes', quiet = TRUE)
        }
        remotes::install_github('jiamingqiu/densityFPCA', quiet = TRUE, force = TRUE)
        " || {
            echo "❌ Failed to install densityFPCA"
            echo "💡 You may need to install gfortran: brew install gcc"
            # Restore headers if we moved them
            if [ "$MOVED_HEADERS" = true ]; then
                echo "🔄 Restoring headers..."
                for f in /usr/local/include_backup_temp/*; do
                    if [ -f "$f" ]; then
                        sudo mv "$f" /usr/local/include/
                    fi
                done
                sudo rmdir /usr/local/include_backup_temp 2>/dev/null || true
            fi
            exit 1
        }
    }

    # Restore headers if we moved them and installation succeeded
    if [ "$MOVED_HEADERS" = true ]; then
        echo "🔄 Restoring headers after successful installation..."
        for f in /usr/local/include_backup_temp/*; do
            if [ -f "$f" ]; then
                sudo mv "$f" /usr/local/include/
            fi
        done
        sudo rmdir /usr/local/include_backup_temp 2>/dev/null || true
        echo "✅ Headers restored"
    fi
else
    echo "✅ densityFPCA already installed"
fi

# Test that both packages load correctly
echo "🧪 Testing R packages..."
Rscript -e "
library(fdapace)
library(densityFPCA)
cat('✅ All R packages loaded successfully\n')
" || {
    echo "❌ R packages failed to load properly"
    exit 1
}

# Check if virtual environment already exists
if [ -d "venv" ]; then
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "🗑️  Removing existing virtual environment (--force mode)..."
        rm -rf venv
    else
        echo "⚠️  Virtual environment 'venv' already exists!"
        echo ""
        echo "Choose an option:"
        echo "1. Remove existing environment and create new one"
        echo "2. Exit (keep existing environment)"
        echo ""
        read -p "Enter your choice (1/2): " choice

        case $choice in
            1)
                echo "🗑️  Removing existing virtual environment..."
                rm -rf venv
                ;;
            2)
                echo "👋 Exiting. Existing environment preserved."
                echo "To activate existing environment: source venv/bin/activate"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Exiting."
                exit 1
                ;;
        esac
    fi
else
    echo "✅ No existing virtual environment found"
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
$PYTHON_CMD -m venv venv

# Verify venv was created correctly
echo "🔍 Verifying virtual environment integrity..."
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment activation script not found"
    echo "💡 Try removing venv directory and running setup again"
    exit 1
fi

# Check for Python executable in venv
if [ ! -f "venv/bin/$PYTHON_CMD" ] && [ ! -L "venv/bin/$PYTHON_CMD" ]; then
    echo "❌ Python executable not found in virtual environment"
    echo "💡 Expected: venv/bin/$PYTHON_CMD"
    ls -la venv/bin/ | grep python || true
    exit 1
fi

echo "✅ Virtual environment structure verified"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify activation worked
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip first to avoid build issues
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Verify pip works
if ! pip --version &>/dev/null; then
    echo "❌ pip is not working correctly in the virtual environment"
    exit 1
fi

# Install Python packages from requirements.txt
echo "🐍 Installing Python packages from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found"
    exit 1
fi

pip install --quiet --no-warn-script-location -r requirements.txt

echo "✅ Python environment configured with pinned versions"

# Set R_HOME in virtual environment activation script
echo "🔧 Configuring R_HOME in virtual environment..."
echo "export R_HOME=\"$R_HOME\"" >> venv/bin/activate

# Note: Jupyter/VS Code will automatically detect the venv kernel
echo "📓 Jupyter kernel ready (will appear as 'venv' in VS Code/Claude Code)"

# Test Python setup
echo "🧪 Testing Python setup..."
python -c "
try:
    import numpy, pandas, matplotlib, scipy, rpy2, torch
    print('✅ Python packages imported successfully')
    print(f'   PyTorch {torch.__version__} ready for MyersonNet')
except ImportError as e:
    print(f'❌ Python import error: {e}')
    exit(1)
" || exit 1

# Test Python-R integration with cdf.R functions
echo "🔗 Testing Python-R bridge with cdf.R functions..."
python -c "
import os
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

try:
    # Enable automatic numpy-R conversion
    numpy2ri.activate()

    # Source the cdf.R file
    cdf_r_path = os.path.join(os.getcwd(), 'mapp/methods/cdf_based/estimation/cdf.R')
    if not os.path.exists(cdf_r_path):
        print('❌ cdf.R file not found at expected location')
        exit(1)

    ro.r.source(cdf_r_path)

    # Test KDE CDF function with sample data
    sample_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kde_cdf_r = ro.globalenv['kde_cdf_r']

    # Call the R function
    cdf_func = kde_cdf_r(sample_data, 0.5, 5.5, 100)

    # Test the returned CDF function
    test_value = ro.r('function(f) f(3.0)')(cdf_func)[0]

    if 0.0 <= test_value <= 1.0:
        print('✅ Python-R bridge working - cdf.R functions accessible')
        print(f'   KDE CDF test: F(3.0) = {test_value:.3f}')
    else:
        print(f'❌ CDF function returned invalid value: {test_value}')
        exit(1)

except Exception as e:
    print(f'❌ Python-R bridge failed: {e}')
    print('   This means you cannot use cdf.R functions from Python')
    exit(1)
" || exit 1

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Test R bridge: python mapp/methods/cdf_based/estimation/rbridge.py"
echo "  3. For notebooks: Select 'venv' kernel in VS Code/Claude Code"
echo ""
echo "📂 Environment location: $(pwd)/venv"
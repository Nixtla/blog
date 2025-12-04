#!/bin/bash

# Convert QMD files to Jupyter notebooks
# Usage: ./scripts/convert_qmd_to_notebook.sh <filename.qmd>

set -e  # Exit on error

# Check if filename provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/convert_qmd_to_notebook.sh <filename.qmd>"
    echo "Example: ./scripts/convert_qmd_to_notebook.sh posts/mlforecast_exogenous_variables.qmd"
    exit 1
fi

# Get the input filename
INPUT_FILE="$1"

# Resolve full path
if [[ "$INPUT_FILE" = /* ]]; then
    # Absolute path
    FULL_PATH="$INPUT_FILE"
else
    # Relative path - check if it's in posts/
    if [ -f "posts/$INPUT_FILE" ]; then
        FULL_PATH="posts/$INPUT_FILE"
    elif [ -f "$INPUT_FILE" ]; then
        FULL_PATH="$INPUT_FILE"
    else
        echo "Error: File not found: $INPUT_FILE"
        echo "Tried: posts/$INPUT_FILE and $INPUT_FILE"
        exit 1
    fi
fi

# Extract basename without extension
BASENAME=$(basename "$INPUT_FILE" .qmd)

# Define paths
TEMP_QMD="/tmp/${BASENAME}_cleaned.qmd"
NOTEBOOKS_DIR="examples/notebooks"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Converting: $BASENAME.qmd → notebook"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1: Clean the QMD file
echo "Step 1/4: Cleaning QMD file..."
python3 scripts/qmd_to_notebook.py "$FULL_PATH" "$TEMP_QMD"

# Step 2: Convert to notebook using quarto
echo "Step 2/4: Converting to Jupyter notebook..."
quarto convert "$TEMP_QMD" -o "/tmp/${BASENAME}.ipynb"

# Step 3: Create notebooks directory if it doesn't exist
echo "Step 3/4: Creating notebooks directory..."
mkdir -p "$NOTEBOOKS_DIR"

# Step 4: Move notebook to notebooks directory
echo "Step 4/4: Moving notebook to examples/notebooks/..."
mv "/tmp/${BASENAME}.ipynb" "$NOTEBOOKS_DIR/${BASENAME}.ipynb"

# Clean up temp file
rm -f "$TEMP_QMD"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Conversion complete!"
echo "  Output: $NOTEBOOKS_DIR/${BASENAME}.ipynb"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

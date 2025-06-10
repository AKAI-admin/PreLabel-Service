#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Preamble ---
# This script updates pip and installs Python packages from a requirements.txt file.
#
# IMPORTANT:
# This script assumes you are already using a compatible Python version
# (e.g., Python 3.9, 3.10, or 3.11 for recent TensorFlow versions).
# It does NOT change your Python version.
#
# Run this script from the same directory as your 'requirements.txt' file.
# ------------------

echo "--- Step 1: Upgrading pip ---"
# Use python3 -m pip to ensure you're using the pip associated with your python3 installation.
pip install --upgrade pip

echo ""
echo "--- Step 2: Installing packages from requirements.txt ---"
# Install the packages listed in the requirements file.
pip install -r requirements.txt

echo ""
echo "--- Installation complete. ---"
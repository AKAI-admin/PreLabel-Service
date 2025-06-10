#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Preamble ---
# This script sets up a project with a specific Python version using pyenv,
# creates a virtual environment, updates pip, and installs Python packages
# from a requirements.txt file.
#
# IMPORTANT:
# This script requires 'pyenv' to manage Python versions.
# Please install it if you haven't already: https://github.com/pyenv/pyenv
#
# Run this script from the same directory as your 'requirements.txt' file.
# ------------------

PYTHON_VERSION="3.10.16"

echo "--- Step 1: Setting up Python ${PYTHON_VERSION} with pyenv ---"

if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Please install it to continue."
    echo "See: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

if ! pyenv versions --bare | grep -q -x "${PYTHON_VERSION}"; then
    echo "Python ${PYTHON_VERSION} not found with pyenv. Attempting to install..."
    pyenv install ${PYTHON_VERSION}
fi

pyenv local ${PYTHON_VERSION}
echo "Local Python version set to ${PYTHON_VERSION} for this project."


echo ""
echo "--- Step 2: Creating and activating virtual environment ---"
# 'python' will now be shimmed by pyenv to the correct version.
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
source venv/bin/activate
echo "Virtual environment activated."

echo ""
echo "--- Step 3: Upgrading pip ---"
# Use python -m pip to ensure you're using the pip associated with the virtual env.
python -m pip install --upgrade pip

echo ""
echo "--- Step 4: Installing packages from requirements.txt ---"
# Install the packages listed in the requirements file.
python -m pip install -r requirements.txt

echo ""
echo "--- Installation complete. ---"
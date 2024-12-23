#!/bin/bash


PYTHON_VERSION_FILE=".python-version"
ENV_NAME="trust"


success_message() {
    echo -e "\e[32m$1\e[0m"
}


error_message() {
    echo -e "\e[31m$1\e[0m"
    exit 1
}


if [ ! -f "$PYTHON_VERSION_FILE" ]; then
    error_message "Error: $PYTHON_VERSION_FILE not found. Please create a file specifying the Python version."
fi


PYTHON_VERSION=$(cat "$PYTHON_VERSION_FILE")


if [[ -z "$PYTHON_VERSION" ]]; then
    error_message "Error: Python version not specified in $PYTHON_VERSION_FILE."
fi


success_message "Python version $PYTHON_VERSION detected."


echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
if [ $? -ne 0 ]; then
    error_message "Error: Failed to create Conda environment."
fi
success_message "Conda environment '$ENV_NAME' created successfully."


echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    error_message "Error: Failed to activate Conda environment."
fi
success_message "Conda environment '$ENV_NAME' activated successfully."


if [ -f "environment.yml" ]; then
    echo "Found environment.yml. Installing dependencies..."
    conda env update -f environment.yml --prune
    if [ $? -ne 0 ]; then
        error_message "Error: Failed to install dependencies from environment.yml."
    fi
    success_message "Dependencies installed from environment.yml."
elif [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        error_message "Error: Failed to install dependencies from requirements.txt."
    fi
    success_message "Dependencies installed from requirements.txt."
else
    success_message "No dependency file (requirements.txt or environment.yml) found. Skipping dependency installation."
fi


echo
success_message "Setup is correct. You can now use 'conda activate $ENV_NAME' to activate the environment."

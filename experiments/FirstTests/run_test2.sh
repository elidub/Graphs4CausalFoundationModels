#!/bin/bash
# Simple benchmark run script that takes config name as argument

# Extract environment
unzip venv.zip
source venv/bin/activate

# Set up Python path - the src directory is transferred directly
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run the minimal training with config - run the __main__.py file directly
cd src/training/Minimal_Training
python3 __main__.py "$1"
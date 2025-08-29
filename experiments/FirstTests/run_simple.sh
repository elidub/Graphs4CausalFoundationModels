#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
unzip -q venv.zip
source venv/bin/activate

#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Checking venv directory..."
ls -la venv/

echo "Checking activation script..."
ls -la venv/bin/activate

echo "Activating virtual environment..."
source venv/bin/activate

echo "Setting up virtual environment paths explicitly..."
export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

#echo "Checking Python and torch after activation..."
#which python3
#echo "VIRTUAL_ENV: $VIRTUAL_ENV"
#echo "PATH: $PATH"
#python3 --version
#python3 -c "import torch; print(f'Torch version: {torch.__version__}')" || echo "Torch import failed"

#echo "Checking GPU compatibility..."
#python3 -c "
#import torch
#print(f'CUDA available: {torch.cuda.is_available()}')
#if torch.cuda.is_available():
#    print(f'CUDA version: {torch.version.cuda}')
#    print(f'GPU count: {torch.cuda.device_count()}')
#    for i in range(torch.cuda.device_count()):
#        gpu_props = torch.cuda.get_device_properties(i)
#        print(f'GPU {i}: {gpu_props.name} (Compute {gpu_props.major}.{gpu_props.minor})')
#else:
#    print('No GPU available, will use CPU')
#"

# Run SimplePFN training with config
cd src/training
python3 simple_run.py --config "../../early_test.yaml"

# Run SimplePFN training with config
cd src/training
python3 simple_run.py --config "../../early_test.yaml"

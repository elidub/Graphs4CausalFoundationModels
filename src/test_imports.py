"""Test imports for the new two-tier training architecture"""
try:
    from training import Trainer
    print("✓ Trainer imported successfully")
except Exception as e:
    print(f"✗ Trainer import error: {e}")

try:
    from training import SetupTraining
    print("✓ SetupTraining imported successfully")
except Exception as e:
    print(f"✗ SetupTraining import error: {e}")

try:
    from training import quick_train
    print("✓ quick_train imported successfully")
except Exception as e:
    print(f"✗ quick_train import error: {e}")

try:
    from training.configs import get_training_config
    print("✓ configs imported successfully")
except Exception as e:
    print(f"✗ configs import error: {e}")

try:
    from training.utils import create_dataloaders
    print("✓ utils imported successfully")
except Exception as e:
    print(f"✗ utils import error: {e}")

try:
    import wandb
    print("✓ wandb is available")
except ImportError:
    print("✗ wandb not available - install with: pip install wandb")

try:
    import matplotlib
    print("✓ matplotlib is available")
except ImportError:
    print("✗ matplotlib not available - install with: pip install matplotlib")

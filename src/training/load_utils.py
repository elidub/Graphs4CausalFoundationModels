"""
Utility functions for loading saved models and checkpoints.
"""

import os
import torch
from typing import Optional, Dict, Any, Union, Tuple


def load_model(
    model_path: str,
    model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint file
        model (torch.nn.Module, optional): Model instance to load state into
        device (str, optional): Device to load model to. Defaults to "cpu".
        
    Returns:
        Tuple containing:
            torch.nn.Module: The loaded model
            Dict[str, Any]: The metadata from the checkpoint
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract checkpoint components
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict')
    metadata = checkpoint.get('metadata', {})
    
    # Load state into model if provided
    if model is not None and model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    
    return model, metadata


def find_checkpoint_in_run(
    run_dir: str,
    checkpoint_type: str = "final",
    step: Optional[int] = None
) -> Optional[str]:
    """Find a specific checkpoint in a run directory.
    
    Args:
        run_dir (str): Path to the run directory containing checkpoints
        checkpoint_type (str): Type of checkpoint to find ("final", "interrupted", or "step")
        step (int, optional): If checkpoint_type is "step", the step number to find
        
    Returns:
        str: Path to the checkpoint file, or None if not found
    """
    if not os.path.exists(run_dir) or not os.path.isdir(run_dir):
        return None
    
    if checkpoint_type == "final":
        checkpoint_path = os.path.join(run_dir, "final.pt")
        return checkpoint_path if os.path.exists(checkpoint_path) else None
    
    elif checkpoint_type == "interrupted":
        checkpoint_path = os.path.join(run_dir, "interrupted.pt")
        return checkpoint_path if os.path.exists(checkpoint_path) else None
    
    elif checkpoint_type == "step":
        if step is None:
            raise ValueError("Step number must be provided when checkpoint_type is 'step'")
        
        checkpoint_path = os.path.join(run_dir, f"step_{step}.pt")
        return checkpoint_path if os.path.exists(checkpoint_path) else None
    
    elif checkpoint_type == "latest_step":
        # Find the latest step checkpoint
        step_files = [f for f in os.listdir(run_dir) if f.startswith("step_") and f.endswith(".pt")]
        if not step_files:
            return None
        
        # Extract step numbers and find the highest
        step_nums = []
        for f in step_files:
            try:
                step_num = int(f.replace("step_", "").replace(".pt", ""))
                step_nums.append((step_num, f))
            except ValueError:
                continue
        
        if not step_nums:
            return None
        
        # Get the file with the highest step number
        _, latest_file = max(step_nums, key=lambda x: x[0])
        return os.path.join(run_dir, latest_file)
    
    elif checkpoint_type == "best_available":
        # Try to find checkpoints in this priority order:
        # 1. final.pt
        # 2. interrupted.pt
        # 3. latest step checkpoint
        
        # Try final checkpoint
        final_path = os.path.join(run_dir, "final.pt")
        if os.path.exists(final_path):
            return final_path
        
        # Try interrupted checkpoint
        interrupted_path = os.path.join(run_dir, "interrupted.pt")
        if os.path.exists(interrupted_path):
            return interrupted_path
        
        # Try to find latest step checkpoint
        latest_step = find_checkpoint_in_run(run_dir, "latest_step")
        return latest_step
    
    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")


def find_run_directory(
    checkpoints_dir: str,
    run_name: str = None,
    latest: bool = False
) -> Optional[str]:
    """Find a specific run directory or the latest run directory.
    
    Args:
        checkpoints_dir (str): Base directory containing run subdirectories
        run_name (str, optional): Name of the run to find
        latest (bool): If True and run_name is None, find the latest run
        
    Returns:
        str: Path to the run directory, or None if not found
    """
    if not os.path.exists(checkpoints_dir) or not os.path.isdir(checkpoints_dir):
        return None
    
    if run_name:
        run_dir = os.path.join(checkpoints_dir, run_name)
        return run_dir if os.path.exists(run_dir) else None
    
    elif latest:
        # Get all subdirectories
        subdirs = [d for d in os.listdir(checkpoints_dir) 
                   if os.path.isdir(os.path.join(checkpoints_dir, d))]
        
        if not subdirs:
            return None
            
        # Get directory with the most recent modification time
        latest_dir = max(subdirs, 
                          key=lambda d: os.path.getmtime(os.path.join(checkpoints_dir, d)))
        
        return os.path.join(checkpoints_dir, latest_dir)
    
    return None

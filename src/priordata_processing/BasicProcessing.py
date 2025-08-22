
import torch
import random
import numpy as np
from typing import Dict, Tuple, Any, Optional
from scipy.stats import yeojohnson


class BasicProcessing:
    """
    A high-performance data preprocessing class for causal inference datasets.
    
    This class efficiently processes datasets in dictionary format {feature_index: torch.Tensor(N_SAMPLES, 1)}
    through a comprehensive pipeline including data transformation, target selection, shuffling, 
    feature dropout, normalization, and padding.
    
    Features:
    - Fast and safe processing modes for different use cases
    - Flexible normalization (standardization or Yeo-Johnson + standardization)
    - Configurable feature dropout and data shuffling
    - Zero-padding to specified dimensions
    - Comprehensive metadata tracking
    
    Args:
        max_num_samples (int): Maximum number of samples after padding
        max_num_features (int): Maximum number of features after padding
        dropout_prob (float, optional): Probability of dropping each feature. Defaults to 0.0
        transformation_type (str, optional): Type of transformation ('standardize' or 'yeo_johnson'). 
                                           Defaults to 'standardize'
        shuffle_data (bool, optional): Whether to shuffle samples and features. Defaults to True
        target_feature (int, optional): Specific target feature index. If None, randomly selected
        random_seed (int, optional): Random seed for reproducibility. Defaults to None
        
    Example:
        >>> processor = BasicProcessing(max_num_samples=1000, max_num_features=10)
        >>> data = {0: torch.randn(100, 1), 1: torch.randn(100, 1)}
        >>> processed_data, metadata = processor.process(data, mode='fast')
    """
    
    def __init__(
        self, 
        max_num_samples: int,
        max_num_features: int,
        dropout_prob: float = 0.0,
        transformation_type: str = 'standardize',
        shuffle_data: bool = True,
        target_feature: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize the BasicProcessing instance with configuration parameters."""
        self.max_num_samples = max_num_samples
        self.max_num_features = max_num_features
        self.dropout_prob = dropout_prob
        self.transformation_type = transformation_type
        self.shuffle_data = shuffle_data
        self.target_feature = target_feature
        self.random_seed = random_seed
        
        # Don't set global seed in __init__ - do it per process call
    
    def process(
        self, 
        dataset: Dict[int, torch.Tensor], 
        mode: str = 'fast'
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process the input dataset according to the specified pipeline.
        
        Args:
            dataset (Dict[int, torch.Tensor]): Input dataset with format {feature_index: tensor(N_SAMPLES, 1)}
            mode (str): Processing mode - 'fast' for speed optimization or 'safe' for extensive validation
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
                - Processed tensor of shape (max_num_samples, max_num_features)
                - Metadata dictionary containing processing information
                
        Raises:
            ValueError: If mode is not 'fast' or 'safe', or if dataset is invalid
            RuntimeError: If processing fails due to data inconsistencies
        """
        # Set random seed at the start of each process call for reproducibility
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        
        if mode not in ['fast', 'safe']:
            raise ValueError(f"Mode must be 'fast' or 'safe', got '{mode}'")
            
        # Input validation based on mode
        if mode == 'safe':
            self._validate_input_safe(dataset)
        else:
            self._validate_input_fast(dataset)
        
        # Store original metadata
        original_num_features = len(dataset)
        original_num_samples = list(dataset.values())[0].shape[0]
        
        # Step 0: Transform data into tensor
        data_tensor, feature_indices = self._dict_to_tensor(dataset, mode)
        
        # Step 1: Randomly decide target feature
        target_idx = self._select_target_feature(feature_indices, mode)
        
        # Step 2: Shuffle the data (features and samples)
        if self.shuffle_data:
            data_tensor, feature_indices = self._shuffle_data(data_tensor, feature_indices, mode)
        
        # Step 3: Drop out some features
        if self.dropout_prob > 0:
            data_tensor, feature_indices = self._dropout_features(data_tensor, feature_indices, mode)
        
        # Step 4: Pre-process features
        data_tensor, transformation_params = self._transform_features(data_tensor, mode)
        
        # Step 5: Pad data with zeros
        padded_tensor = self._pad_data(data_tensor, mode)
        
        # Step 6: Create metadata dictionary
        metadata = {
            'feature_names': feature_indices,
            'target_feature': target_idx,
            'transformation_type': self.transformation_type,
            'transformation_params': transformation_params,
            'original_num_samples': original_num_samples,
            'original_num_features': original_num_features,
            'final_num_samples': padded_tensor.shape[0],
            'final_num_features': padded_tensor.shape[1],
            'padding_start_sample': data_tensor.shape[0],
            'padding_start_feature': data_tensor.shape[1],
            'dropout_prob': self.dropout_prob,
            'shuffle_applied': self.shuffle_data
        }
        
        return padded_tensor, metadata
    
    def _validate_input_fast(self, dataset: Dict[int, torch.Tensor]) -> None:
        """Fast input validation with minimal checks."""
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        if not isinstance(dataset, dict):
            raise ValueError("Dataset must be a dictionary")
    
    def _validate_input_safe(self, dataset: Dict[int, torch.Tensor]) -> None:
        """Comprehensive input validation with extensive checks."""
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        if not isinstance(dataset, dict):
            raise ValueError("Dataset must be a dictionary")
        
        # Check all keys are integers
        for key in dataset.keys():
            if not isinstance(key, int):
                raise ValueError(f"All keys must be integers, got {type(key)} for key {key}")
        
        # Check all values are tensors with consistent shapes
        sample_counts = []
        for feature_idx, tensor in dataset.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"All values must be torch.Tensor, got {type(tensor)} for feature {feature_idx}")
            if tensor.dim() != 2:
                raise ValueError(f"All tensors must be 2D, got {tensor.dim()}D for feature {feature_idx}")
            if tensor.shape[1] != 1:
                raise ValueError(f"All tensors must have shape (N, 1), got {tensor.shape} for feature {feature_idx}")
            sample_counts.append(tensor.shape[0])
        
        # Check consistent sample counts
        if len(set(sample_counts)) > 1:
            raise ValueError(f"All features must have same number of samples, got {set(sample_counts)}")
        
        # Check for NaN or infinite values
        for feature_idx, tensor in dataset.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"Feature {feature_idx} contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError(f"Feature {feature_idx} contains infinite values")
    
    def _dict_to_tensor(
        self, 
        dataset: Dict[int, torch.Tensor], 
        mode: str
    ) -> Tuple[torch.Tensor, list]:
        """Convert dictionary dataset to tensor format."""
        feature_indices = sorted(dataset.keys())
        tensors = [dataset[idx] for idx in feature_indices]
        
        # Fast concatenation
        data_tensor = torch.cat(tensors, dim=1)
        
        if mode == 'safe':
            # Verify concatenation
            expected_shape = (list(dataset.values())[0].shape[0], len(dataset))
            if data_tensor.shape != expected_shape:
                raise RuntimeError(f"Tensor concatenation failed: expected {expected_shape}, got {data_tensor.shape}")
        
        return data_tensor, feature_indices
    
    def _select_target_feature(self, feature_indices: list, mode: str) -> int:
        """Select target feature index."""
        if self.target_feature is not None:
            if mode == 'safe' and self.target_feature not in feature_indices:
                raise ValueError(f"Target feature {self.target_feature} not in dataset")
            return self.target_feature
        else:
            return random.choice(feature_indices)
    
    def _shuffle_data(
        self, 
        data_tensor: torch.Tensor, 
        feature_indices: list, 
        mode: str
    ) -> Tuple[torch.Tensor, list]:
        """Shuffle samples and features."""
        # Shuffle samples
        sample_perm = torch.randperm(data_tensor.shape[0])
        data_tensor = data_tensor[sample_perm]
        
        # Shuffle features
        feature_perm = torch.randperm(data_tensor.shape[1])
        data_tensor = data_tensor[:, feature_perm]
        feature_indices = [feature_indices[i] for i in feature_perm.tolist()]
        
        return data_tensor, feature_indices
    
    def _dropout_features(
        self, 
        data_tensor: torch.Tensor, 
        feature_indices: list, 
        mode: str
    ) -> Tuple[torch.Tensor, list]:
        """Apply feature dropout."""
        num_features = data_tensor.shape[1]
        keep_mask = torch.rand(num_features) > self.dropout_prob
        
        # Ensure at least one feature remains
        if not keep_mask.any():
            keep_mask[0] = True
        
        data_tensor = data_tensor[:, keep_mask]
        feature_indices = [feature_indices[i] for i in range(len(feature_indices)) if keep_mask[i]]
        
        if mode == 'safe':
            if data_tensor.shape[1] == 0:
                raise RuntimeError("Feature dropout removed all features")
        
        return data_tensor, feature_indices
    
    def _transform_features(
        self, 
        data_tensor: torch.Tensor, 
        mode: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply feature transformations."""
        transformation_params = {'type': self.transformation_type}
        
        if self.transformation_type == 'standardize':
            # Simple standardization
            means = data_tensor.mean(dim=0, keepdim=True)
            stds = data_tensor.std(dim=0, keepdim=True)
            
            # Avoid division by zero
            stds = torch.where(stds == 0, torch.ones_like(stds), stds)
            
            data_tensor = (data_tensor - means) / stds
            transformation_params.update({'means': means, 'stds': stds})
            
        elif self.transformation_type == 'yeo_johnson':
            # Yeo-Johnson transformation followed by standardization
            lambdas = []
            for i in range(data_tensor.shape[1]):
                col_data = data_tensor[:, i].numpy()
                transformed_data, lambda_param = yeojohnson(col_data)
                data_tensor[:, i] = torch.from_numpy(transformed_data).float()
                lambdas.append(lambda_param)
            
            # Standardize after transformation
            means = data_tensor.mean(dim=0, keepdim=True)
            stds = data_tensor.std(dim=0, keepdim=True)
            stds = torch.where(stds == 0, torch.ones_like(stds), stds)
            
            data_tensor = (data_tensor - means) / stds
            transformation_params.update({
                'lambdas': lambdas,
                'means': means,
                'stds': stds
            })
        else:
            if mode == 'safe':
                raise ValueError(f"Unknown transformation type: {self.transformation_type}")
        
        return data_tensor, transformation_params
    
    def _pad_data(self, data_tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Pad data tensor to specified maximum dimensions."""
        current_samples, current_features = data_tensor.shape
        
        if mode == 'safe':
            if current_samples > self.max_num_samples:
                raise ValueError(f"Current samples {current_samples} exceeds max_num_samples {self.max_num_samples}")
            if current_features > self.max_num_features:
                raise ValueError(f"Current features {current_features} exceeds max_num_features {self.max_num_features}")
        
        # Create padded tensor
        padded_tensor = torch.zeros(self.max_num_samples, self.max_num_features, dtype=data_tensor.dtype)
        padded_tensor[:current_samples, :current_features] = data_tensor
        
        return padded_tensor
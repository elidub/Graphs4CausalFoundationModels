from __future__ import annotations
from typing import Dict, Tuple, Optional, Literal, Union
import torch

# Import required classes
from priors.causal_prior.scm.SCM import SCM
from priors.causal_prior.causal_graph.CausalDAG import CausalDAG
from priors.causal_prior.causal_graph.GraphSampler import GraphSampler
from priors.causal_prior.mechanisms.SampleMLPMechanism import SampleMLPMechanism
from priors.causal_prior.mechanisms.SampleXGBoostMechanism import SampleXGBoostMechanism
from priors.causal_prior.noise_distributions.MixedDist import MixedDist
from priors.causal_prior.noise_distributions.MixedDist_RandomSTD import MixedDistRandomStd
from priors.causal_prior.noise_distributions.Sample_STD import GammaMeanStd, ParetoMeanStd

# Try to import TabICLResamplingDist
try:
    from priors.causal_prior.noise_distributions.TabICL_prior_resampling_dist import TabICLResamplingDist
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    TabICLResamplingDist = None


class SCMBuilder:
    """
    Builder class for creating Structural Causal Models (SCMs) with configurable hyperparameters.
    
    This class provides a comprehensive interface for building SCMs with various mechanism types,
    noise distributions, and graph structures. All hyperparameters are explicitly specified in
    the constructor rather than being sampled randomly.
    
    Parameters
    ----------
    # Graph Structure Parameters
    num_nodes : int
        Number of nodes in the causal graph.
    graph_edge_prob : float
        Probability of an edge existing between any two nodes (for random graph generation).
    graph_seed : int
        Random seed for graph generation.
    
    # Mechanism Type Selection
    xgboost_prob : float, default 0.1
        Probability of using XGBoost mechanism for each node (0.0 = never, 1.0 = always).
        Remaining nodes will use MLP mechanisms.
    mechanism_seed : int, default 42
        Random seed for mechanism type selection.
    
    # MLP Mechanism Hyperparameters
    mlp_nonlins : str, default "tabicl"
        Activation function type for MLP mechanisms. Options include:
        - "mixed": combines multiple sampling strategies
        - "tabicl": uses TabICL activation functions (diverse set including RBF, sine, random functions, etc.)
        - "sophisticated_sampling_1": complex sampling strategy
        - "tanh", "sin", "neg", "id", "elu": specific activation functions
    mlp_num_hidden_layers : int, default 0
        Fixed number of hidden layers for MLP mechanisms.
    mlp_hidden_dim : int, default 16
        Width of hidden layers for MLP mechanisms.
    mlp_activation_mode : Literal["pre", "post"], default "pre"
        Whether to apply activation before ("pre") or after ("post") adding noise in MLP.
    mlp_node_shape : Tuple[int, ...], default (1,)
        Output shape per sample for MLP mechanisms.
    
    # XGBoost Mechanism Hyperparameters
    xgb_num_hidden_layers : int, default 0
        Number of hidden layers for XGBoost mechanisms (typically 0).
    xgb_hidden_dim : int, default 0
        Hidden dimension for XGBoost mechanisms (typically 0).
    xgb_activation_mode : Literal["pre", "post"], default "pre"
        Whether to apply activation before ("pre") or after ("post") adding noise in XGBoost.
    xgb_node_shape : Tuple[int, ...], default (1,)
        Output shape per sample for XGBoost mechanisms.
    xgb_n_training_samples : int, default 100
        Number of training samples for XGBoost mechanism fitting.
    xgb_add_noise : bool, default False
        Whether XGBoost mechanisms should add their own noise.
    
    # Noise Distribution Parameters
    random_additive_std : bool, default True
        If True, use MixedDistRandomStd with random standard deviations.
        If False, use MixedDist with fixed standard deviations.
    exo_std_distribution : str, default "gamma"
        Type of distribution to use for exogenous noise standard deviations when random_additive_std=True.
        Options: "gamma" (Gamma distribution), "pareto" (Pareto distribution - heavy-tailed).
    endo_std_distribution : str, default "gamma"
        Type of distribution to use for endogenous noise standard deviations when random_additive_std=True.
        Options: "gamma" (Gamma distribution), "pareto" (Pareto distribution - heavy-tailed).
    tabicl_noise_proportion : float, default 0.0
        Proportion of TabICL noise in the exogenous noise mixture (range: [0.0, 1.0]).
        When > 0, the noise mixture includes TabICL resampling distribution with this proportion,
        and the remaining probability mass is split equally among Normal, Laplace, and StudentT.
        Requires TabICL samples to be generated first (see TabICL_prior_resampling_dist.py).
    
    # Fixed Standard Deviation Parameters (used when random_additive_std=False)
    exo_std : float, default 1.0
        Fixed standard deviation for exogenous noise distributions.
    endo_std : float, default 0.1
        Fixed standard deviation for endogenous noise distributions.
    
    # Random Standard Deviation Parameters (used when random_additive_std=True)
    exo_std_mean : float, default 1.0
        Mean of the distribution for exogenous noise standard deviations.
    exo_std_std : float, default 1.0
        Standard deviation of the distribution for exogenous noise standard deviations.
    endo_std_mean : float, default 0.3
        Mean of the distribution for endogenous noise standard deviations.
    endo_std_std : float, default 0.1
        Standard deviation of the distribution for endogenous noise standard deviations.
    
    # SCM Configuration
    scm_fast : bool, default True
        Whether to use fast sampling mode in the SCM.
    use_exogenous_mechanisms : bool, default True
        Whether to use mechanisms for exogenous (root) nodes.
        If True, root nodes require mechanisms with input_dim=1 that transform scalar noise.
        If False, root nodes can omit mechanisms and get values directly from exogenous noise.
    
    # Random Number Generation
    mechanism_generator_seed : int, default 42
        Base seed for mechanism generators. Each mechanism gets seed + node_index.
    
    Examples
    --------
    >>> # Create SCM with default hyperparameters (uses Gamma distribution)
    >>> builder = SCMBuilder(num_nodes=5, graph_edge_prob=0.3, graph_seed=42)
    >>> scm = builder.build()
    
    >>> # Create SCM with custom hyperparameters and fixed noise
    >>> builder = SCMBuilder(
    ...     num_nodes=10,
    ...     graph_edge_prob=0.5,
    ...     graph_seed=123,
    ...     xgboost_prob=0.3,
    ...     mlp_num_hidden_layers=2,
    ...     mlp_hidden_dim=32,
    ...     random_additive_std=False,
    ...     exo_std=0.5,
    ...     endo_std=0.05
    ... )
    >>> scm = builder.build()
    
    >>> # Create SCM with heavy-tailed (Pareto) noise distributions
    >>> builder = SCMBuilder(
    ...     num_nodes=8,
    ...     graph_edge_prob=0.4,
    ...     graph_seed=456,
    ...     random_additive_std=True,
    ...     exo_std_distribution="pareto",
    ...     endo_std_distribution="pareto",
    ...     exo_std_mean=1.5,
    ...     exo_std_std=2.0,  # High CV for Pareto
    ...     endo_std_mean=0.5,
    ...     endo_std_std=0.8   # High CV for Pareto
    ... )
    >>> scm = builder.build()
    
    Sampling from SCMs
    ------------------
    The SCM.propagate() method returns a dictionary mapping node names to tensors:
    >>> samples = scm.propagate(100)  # Returns Dict[str, torch.Tensor]
    >>> print(f"Node 'X1' samples shape: {samples['X1'].shape}")  # (100, 1)
    
    For convenience, use build_and_sample_tensor() to get a concatenated tensor:
    >>> tensor_samples = builder.build_and_sample_tensor(100)  # Returns torch.Tensor
    >>> print(f"Concatenated shape: {tensor_samples.shape}")  # (100, num_features)
    
    Important Notes
    ---------------
    Fast vs Safe Sampling:
    - Both fast and safe modes require calling scm.sample_exogenous(num_samples) 
      before calling scm.propagate(num_samples).
    - The difference is in validation: safe mode does more error checking.
    - Use the build_and_sample() convenience method to handle this automatically.
    
    Example with any mode:
    >>> builder = SCMBuilder(num_nodes=5, graph_edge_prob=0.3, graph_seed=42, scm_fast=True)
    >>> scm = builder.build()
    >>> scm.sample_exogenous(100)  # Always required
    >>> scm.sample_endogenous_noise(100)  # Also required
    >>> samples = scm.propagate(100)
    
    Example with convenience method:
    >>> samples = builder.build_and_sample(100)  # Handles everything automatically
    
    Example with safe mode:
    >>> builder = SCMBuilder(num_nodes=5, graph_edge_prob=0.3, graph_seed=42, scm_fast=False)
    >>> scm = builder.build()
    >>> scm.sample_exogenous(100)  # Still required even in safe mode
    >>> scm.sample_endogenous_noise(100)  # Still required even in safe mode
    >>> samples = scm.propagate(100)
    """
    
    def __init__(
        self,
        *,
        # Graph Structure Parameters
        num_nodes: int,
        graph_edge_prob: float,
        graph_seed: int,
        
        # Mechanism Type Selection
        xgboost_prob: float = 0.1,
        mechanism_seed: int = 42,
        
        # MLP Mechanism Hyperparameters
        mlp_nonlins: str = "tabicl",
        mlp_num_hidden_layers: int = 0,
        mlp_hidden_dim: int = 16,
        mlp_activation_mode: Literal["pre", "post"] = "pre",
        mlp_node_shape: Tuple[int, ...] = (1,),
        
        # XGBoost Mechanism Hyperparameters
        xgb_num_hidden_layers: int = 0,
        xgb_hidden_dim: int = 0,
        xgb_activation_mode: Literal["pre", "post"] = "pre",
        xgb_node_shape: Tuple[int, ...] = (1,),
        xgb_n_training_samples: int = 100,
        xgb_add_noise: bool = False,
        
        # Noise Distribution Parameters
        random_additive_std: bool = True,
        exo_std_distribution: Literal["gamma", "pareto"] = "gamma",
        endo_std_distribution: Literal["gamma", "pareto"] = "gamma",
        tabicl_noise_proportion: float = 0.0,
        
        # Fixed Standard Deviation Parameters
        exo_std: Optional[float] = None,
        endo_std: Optional[float] = None,
        
        # Random Standard Deviation Parameters
        exo_std_mean: Optional[float] = None,
        exo_std_std: Optional[float] = None,
        endo_std_mean: Optional[float] = None,
        endo_std_std: Optional[float] = None,
        
        # SCM Configuration
        scm_fast: bool = True,
        use_exogenous_mechanisms: bool = True,
        
        # Random Number Generation
        mechanism_generator_seed: int = 42,
    ) -> None:
        # Set default values based on random_additive_std mode
        if random_additive_std:
            # Random std mode defaults
            if exo_std_mean is None:
                exo_std_mean = 1.0
            if exo_std_std is None:
                exo_std_std = 1.0
            if endo_std_mean is None:
                endo_std_mean = 0.3
            if endo_std_std is None:
                endo_std_std = 0.1
        else:
            # Fixed std mode defaults
            if exo_std is None:
                exo_std = 1.0
            if endo_std is None:
                endo_std = 0.1
        
        # Validate noise parameter exclusivity
        if random_additive_std:
            if exo_std is not None or endo_std is not None:
                raise ValueError(
                    "When random_additive_std=True, you cannot specify exo_std or endo_std. "
                    "Use exo_std_mean, exo_std_std, endo_std_mean, endo_std_std instead."
                )
        else:
            if (exo_std_mean is not None or exo_std_std is not None or 
                endo_std_mean is not None or endo_std_std is not None):
                raise ValueError(
                    "When random_additive_std=False, you cannot specify exo_std_mean, exo_std_std, "
                    "endo_std_mean, or endo_std_std. Use exo_std and endo_std instead."
                )
        
        # Store all parameters
        self.num_nodes = num_nodes
        self.graph_edge_prob = graph_edge_prob
        self.graph_seed = graph_seed
        
        self.xgboost_prob = xgboost_prob
        self.mechanism_seed = mechanism_seed
        
        self.mlp_nonlins = mlp_nonlins
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_activation_mode = mlp_activation_mode
        self.mlp_node_shape = mlp_node_shape
        
        self.xgb_num_hidden_layers = xgb_num_hidden_layers
        self.xgb_hidden_dim = xgb_hidden_dim
        self.xgb_activation_mode = xgb_activation_mode
        self.xgb_node_shape = xgb_node_shape
        self.xgb_n_training_samples = xgb_n_training_samples
        self.xgb_add_noise = xgb_add_noise
        
        self.random_additive_std = random_additive_std
        self.exo_std_distribution = exo_std_distribution
        self.endo_std_distribution = endo_std_distribution
        self.tabicl_noise_proportion = tabicl_noise_proportion
        self.exo_std = exo_std
        self.endo_std = endo_std
        self.exo_std_mean = exo_std_mean
        self.exo_std_std = exo_std_std
        self.endo_std_mean = endo_std_mean
        self.endo_std_std = endo_std_std
        
        self.scm_fast = scm_fast
        self.use_exogenous_mechanisms = use_exogenous_mechanisms
        self.mechanism_generator_seed = mechanism_generator_seed
        
        # Validate all hyperparameters
        self._validate_hyperparameters()
    
    def build(self) -> SCM:
        """
        Build and return a configured SCM based on the provided hyperparameters.
        
        Returns
        -------
        SCM
            A fully configured Structural Causal Model ready for sampling.
        """
        # Step 1: Create the causal DAG
        graph_sampler = GraphSampler(seed=self.graph_seed)
        graph = graph_sampler.sample_dag(num_nodes=self.num_nodes, p=self.graph_edge_prob)
        causal_dag = CausalDAG(g=graph, check_acyclic=True)
        
        # Step 2: Create mechanisms for each node
        mechanisms = self._create_mechanisms(causal_dag)
        
        # Step 3: Create noise distributions
        exogenous_noise, endogenous_noise = self._create_noise_distributions(causal_dag)
        
        # Step 4: Build the SCM
        scm = SCM(
            dag=causal_dag,
            mechanisms=mechanisms,
            exogenous_noise=exogenous_noise,
            endogenous_noise=endogenous_noise,
            fast=self.scm_fast,
            use_exogenous_mechanisms=self.use_exogenous_mechanisms
        )
        
        return scm
    
    def build_and_sample(self, num_samples: int, **sample_kwargs) -> Dict[str, torch.Tensor]:
        """
        Build an SCM and sample data from it in one step.
        
        This convenience method handles the noise sampling automatically
        by calling both sample_exogenous() and sample_endogenous_noise()
        before sampling, so you don't have to call them manually.
        
        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        **sample_kwargs
            Additional keyword arguments passed to SCM.propagate().
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping node names to sampled tensors.
            Each tensor has shape (batch_size, *node_shape).
        """
        scm = self.build()
        
        # Pre-sample both exogenous and endogenous noise (required for both fast and safe modes)
        scm.sample_exogenous(num_samples)
        scm.sample_endogenous_noise(num_samples)
        
        return scm.propagate(num_samples, **sample_kwargs)
    
    def build_and_sample_tensor(self, num_samples: int, **sample_kwargs) -> torch.Tensor:
        """
        Build an SCM and sample data, returning a concatenated tensor.
        
        This method samples from the SCM and concatenates all node values
        into a single tensor for convenience.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        **sample_kwargs
            Additional keyword arguments passed to SCM.propagate().
            
        Returns
        -------
        torch.Tensor
            Concatenated tensor with shape (num_samples, total_features).
            Features are ordered by topological order of nodes.
        """
        scm = self.build()
        
        # Pre-sample both exogenous and endogenous noise (required for both fast and safe modes)
        scm.sample_exogenous(num_samples)
        scm.sample_endogenous_noise(num_samples)
        
        # Sample and get the dictionary
        samples_dict = scm.propagate(num_samples, **sample_kwargs)
        
        # Get nodes in topological order and concatenate
        topo_order = scm.dag.topo_order()
        tensors = [samples_dict[node].view(num_samples, -1) for node in topo_order]
        return torch.cat(tensors, dim=1)
    
    def _create_mechanisms(self, causal_dag: CausalDAG) -> Dict[str, Union[SampleMLPMechanism, SampleXGBoostMechanism]]:
        """Create mechanisms for each node in the DAG."""
        mechanisms = {}
        
        # Create generator for mechanism type sampling
        mechanism_type_generator = torch.Generator().manual_seed(self.mechanism_seed)
        
        for i, node in enumerate(causal_dag.nodes()):
            # Determine input dimension
            if self.use_exogenous_mechanisms:
                # When using exogenous mechanisms, roots need input_dim=1 for scalar noise
                input_dim = max(1, len(causal_dag.parents(node)))
            else:
                # In traditional mode, input_dim = number of parents (can be 0 for roots)
                input_dim = len(causal_dag.parents(node))
            
            # Sample whether to use XGBoost or MLP
            use_xgboost = torch.rand(1, generator=mechanism_type_generator).item() < self.xgboost_prob
            
            # Create mechanism-specific generator
            mechanism_generator = torch.Generator().manual_seed(self.mechanism_generator_seed + i)
            
            if use_xgboost:
                mechanisms[node] = SampleXGBoostMechanism(
                    input_dim=input_dim,
                    node_shape=self.xgb_node_shape,
                    num_hidden_layers=self.xgb_num_hidden_layers,
                    hidden_dim=self.xgb_hidden_dim,
                    activation_mode=self.xgb_activation_mode,
                    n_training_samples=self.xgb_n_training_samples,
                    generator=mechanism_generator,
                    name=node,
                    add_noise=self.xgb_add_noise
                )
            else:
                mechanisms[node] = SampleMLPMechanism(
                    input_dim=input_dim,
                    node_shape=self.mlp_node_shape,
                    nonlins=self.mlp_nonlins,
                    num_hidden_layers=self.mlp_num_hidden_layers,
                    hidden_dim=self.mlp_hidden_dim,
                    activation_mode=self.mlp_activation_mode,
                    generator=mechanism_generator,
                    name=node,
                )
        
        return mechanisms
    
    def _create_noise_distributions(self, causal_dag: CausalDAG) -> Tuple[Dict[str, Union[MixedDist, MixedDistRandomStd]], Dict[str, Union[MixedDist, MixedDistRandomStd]]]:
        """Create noise distributions for exogenous and endogenous variables."""
        exogenous_variables = causal_dag.exogenous_variables()
        endogenous_variables = causal_dag.endogenous_variables()
        
        if not self.random_additive_std:
            # Use fixed standard deviations
            exogenous_noise = {var: MixedDist(std=self.exo_std) for var in exogenous_variables}
            endogenous_noise = {var: MixedDist(std=self.endo_std) for var in endogenous_variables}
        else:
            # Use random standard deviations with selected distribution type
            
            # Create exogenous noise distribution
            if self.exo_std_distribution == "gamma":
                exo_std_dist = GammaMeanStd(mean=self.exo_std_mean, std=self.exo_std_std)
            elif self.exo_std_distribution == "pareto":
                exo_std_dist = ParetoMeanStd(mean=self.exo_std_mean, std=self.exo_std_std)
            else:
                raise ValueError(f"Unknown exo_std_distribution: {self.exo_std_distribution}")
            
            # Create endogenous noise distribution
            if self.endo_std_distribution == "gamma":
                endo_std_dist = GammaMeanStd(mean=self.endo_std_mean, std=self.endo_std_std)
            elif self.endo_std_distribution == "pareto":
                endo_std_dist = ParetoMeanStd(mean=self.endo_std_mean, std=self.endo_std_std)
            else:
                raise ValueError(f"Unknown endo_std_distribution: {self.endo_std_distribution}")
            
            # Determine mixture proportions and distributions based on tabicl_noise_proportion
            if self.tabicl_noise_proportion > 0.0 and TABICL_AVAILABLE:
                import torch.distributions as dist
                from pathlib import Path
                
                # Check if TabICL samples exist
                tabicl_samples_dir = Path("/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples")
                if not tabicl_samples_dir.exists() or not list(tabicl_samples_dir.glob("*.pt")):
                    raise FileNotFoundError(
                        f"TabICL samples not found in {tabicl_samples_dir}. "
                        "Run TabICL_prior_resampling_dist.py to generate samples first."
                    )
                
                # Create TabICL distribution instance
                tabicl_dist = TabICLResamplingDist(
                    samples_dir=str(tabicl_samples_dir),
                    scale=1.0,
                    device='cpu'
                )
                
                # Calculate mixture proportions
                # TabICL gets tabicl_noise_proportion, the rest is split equally among Normal, Laplace, StudentT
                tabicl_prop = self.tabicl_noise_proportion
                remaining_prop = 1.0 - tabicl_prop
                other_prop = remaining_prop / 3.0  # Split among Normal, Laplace, StudentT
                
                distributions = [dist.Normal, dist.Laplace, dist.StudentT, tabicl_dist]
                mixture_proportions = [other_prop, other_prop, other_prop, tabicl_prop]
                
                exogenous_noise = {
                    var: MixedDistRandomStd(
                        std_dist=exo_std_dist,
                        distributions=distributions,
                        mixture_proportions=mixture_proportions
                    ) for var in exogenous_variables
                }
                endogenous_noise = {
                    var: MixedDistRandomStd(
                        std_dist=endo_std_dist,
                        distributions=distributions,
                        mixture_proportions=mixture_proportions
                    ) for var in endogenous_variables
                }
            elif self.tabicl_noise_proportion > 0.0 and not TABICL_AVAILABLE:
                raise ImportError(
                    "tabicl_noise_proportion > 0 but TabICLResamplingDist is not available. "
                    "Check that TabICL_prior_resampling_dist.py is in the correct location."
                )
            else:
                # Standard mixture without TabICL (default behavior)
                exogenous_noise = {var: MixedDistRandomStd(std_dist=exo_std_dist) for var in exogenous_variables}
                endogenous_noise = {var: MixedDistRandomStd(std_dist=endo_std_dist) for var in endogenous_variables}
        
        return exogenous_noise, endogenous_noise
    
    def _validate_hyperparameters(self) -> None:
        """
        Validate all hyperparameters for correctness and consistency.
        
        Raises
        ------
        ValueError
            If any hyperparameter is invalid or inconsistent.
        """
        # Graph structure validation
        if not isinstance(self.num_nodes, int) or self.num_nodes < 1:
            raise ValueError(f"num_nodes must be a positive integer, got {self.num_nodes}")
        
        if not (0.0 <= self.graph_edge_prob <= 1.0):
            raise ValueError(f"graph_edge_prob must be in [0.0, 1.0], got {self.graph_edge_prob}")
        
        if not isinstance(self.graph_seed, int):
            raise ValueError(f"graph_seed must be an integer, got {self.graph_seed}")
        
        # Mechanism type validation
        if not (0.0 <= self.xgboost_prob <= 1.0):
            raise ValueError(f"xgboost_prob must be in [0.0, 1.0], got {self.xgboost_prob}")
        
        if not isinstance(self.mechanism_seed, int):
            raise ValueError(f"mechanism_seed must be an integer, got {self.mechanism_seed}")
        
        # MLP mechanism validation
        valid_mlp_nonlins = {
            "mixed", "tabicl", "sophisticated_sampling_1", "sophisticated_sampling_1_normalization",
            "sophisticated_sampling_1_rescaling_normalization", "tanh", "sin", "neg", "id", "elu",
            "summed", "post", "linear"
        }
        if self.mlp_nonlins not in valid_mlp_nonlins:
            raise ValueError(f"mlp_nonlins must be one of {valid_mlp_nonlins}, got '{self.mlp_nonlins}'")
        
        if not isinstance(self.mlp_num_hidden_layers, int) or self.mlp_num_hidden_layers < 0:
            raise ValueError(f"mlp_num_hidden_layers must be non-negative integer, got {self.mlp_num_hidden_layers}")
        
        if not isinstance(self.mlp_hidden_dim, int) or self.mlp_hidden_dim < 1:
            raise ValueError(f"mlp_hidden_dim must be positive integer, got {self.mlp_hidden_dim}")
        
        if self.mlp_activation_mode not in ["pre", "post"]:
            raise ValueError(f"mlp_activation_mode must be 'pre' or 'post', got '{self.mlp_activation_mode}'")
        
        if not isinstance(self.mlp_node_shape, tuple) or not all(isinstance(x, int) and x > 0 for x in self.mlp_node_shape):
            raise ValueError(f"mlp_node_shape must be tuple of positive integers, got {self.mlp_node_shape}")
        
        # XGBoost mechanism validation
        if not isinstance(self.xgb_num_hidden_layers, int) or self.xgb_num_hidden_layers < 0:
            raise ValueError(f"xgb_num_hidden_layers must be non-negative integer, got {self.xgb_num_hidden_layers}")
        
        if not isinstance(self.xgb_hidden_dim, int) or self.xgb_hidden_dim < 0:
            raise ValueError(f"xgb_hidden_dim must be non-negative integer, got {self.xgb_hidden_dim}")
        
        if self.xgb_activation_mode not in ["pre", "post"]:
            raise ValueError(f"xgb_activation_mode must be 'pre' or 'post', got '{self.xgb_activation_mode}'")
        
        if not isinstance(self.xgb_node_shape, tuple) or not all(isinstance(x, int) and x > 0 for x in self.xgb_node_shape):
            raise ValueError(f"xgb_node_shape must be tuple of positive integers, got {self.xgb_node_shape}")
        
        if not isinstance(self.xgb_n_training_samples, int) or self.xgb_n_training_samples < 1:
            raise ValueError(f"xgb_n_training_samples must be positive integer, got {self.xgb_n_training_samples}")
        
        if not isinstance(self.xgb_add_noise, bool):
            raise ValueError(f"xgb_add_noise must be boolean, got {self.xgb_add_noise}")
        
        # Noise distribution validation
        if not isinstance(self.random_additive_std, bool):
            raise ValueError(f"random_additive_std must be boolean, got {self.random_additive_std}")
        
        # Distribution type validation
        if self.exo_std_distribution not in ["gamma", "pareto"]:
            raise ValueError(f"exo_std_distribution must be 'gamma' or 'pareto', got '{self.exo_std_distribution}'")
        
        if self.endo_std_distribution not in ["gamma", "pareto"]:
            raise ValueError(f"endo_std_distribution must be 'gamma' or 'pareto', got '{self.endo_std_distribution}'")
        
        # TabICL noise proportion validation
        if not isinstance(self.tabicl_noise_proportion, (int, float)):
            raise ValueError(f"tabicl_noise_proportion must be numeric, got {type(self.tabicl_noise_proportion)}")
        
        if not (0.0 <= self.tabicl_noise_proportion <= 1.0):
            raise ValueError(f"tabicl_noise_proportion must be in [0.0, 1.0], got {self.tabicl_noise_proportion}")
        
        # Fixed std validation (when applicable)
        if not self.random_additive_std:
            if self.exo_std is None or not isinstance(self.exo_std, (int, float)) or self.exo_std <= 0:
                raise ValueError(f"exo_std must be positive number when random_additive_std=False, got {self.exo_std}")
            
            if self.endo_std is None or not isinstance(self.endo_std, (int, float)) or self.endo_std <= 0:
                raise ValueError(f"endo_std must be positive number when random_additive_std=False, got {self.endo_std}")
        
        # Random std validation (when applicable)
        if self.random_additive_std:
            if (self.exo_std_mean is None or not isinstance(self.exo_std_mean, (int, float)) or 
                self.exo_std_mean <= 0):
                raise ValueError(f"exo_std_mean must be positive number when random_additive_std=True, got {self.exo_std_mean}")
            
            if (self.exo_std_std is None or not isinstance(self.exo_std_std, (int, float)) or 
                self.exo_std_std <= 0):
                raise ValueError(f"exo_std_std must be positive number when random_additive_std=True, got {self.exo_std_std}")
            
            if (self.endo_std_mean is None or not isinstance(self.endo_std_mean, (int, float)) or 
                self.endo_std_mean <= 0):
                raise ValueError(f"endo_std_mean must be positive number when random_additive_std=True, got {self.endo_std_mean}")
            
            if (self.endo_std_std is None or not isinstance(self.endo_std_std, (int, float)) or 
                self.endo_std_std <= 0):
                raise ValueError(f"endo_std_std must be positive number when random_additive_std=True, got {self.endo_std_std}")
            
            # Additional validation for Pareto distribution constraints
            self._validate_pareto_constraints()
        
        # SCM configuration validation
        if not isinstance(self.scm_fast, bool):
            raise ValueError(f"scm_fast must be boolean, got {self.scm_fast}")
        
        if not isinstance(self.use_exogenous_mechanisms, bool):
            raise ValueError(f"use_exogenous_mechanisms must be boolean, got {self.use_exogenous_mechanisms}")
        
        # Seed validation
        if not isinstance(self.mechanism_generator_seed, int):
            raise ValueError(f"mechanism_generator_seed must be integer, got {self.mechanism_generator_seed}")
    
    def _validate_pareto_constraints(self) -> None:
        """
        Validate constraints specific to Pareto distribution.
        
        Pareto distribution requires coefficient of variation (CV) to be large enough.
        For a valid Pareto distribution: CV > sqrt(1/2) ≈ 0.707
        
        Raises
        ------
        ValueError
            If mean/std combination is incompatible with Pareto distribution.
        """
        if self.random_additive_std:
            # Check exogenous distribution
            if self.exo_std_distribution == "pareto":
                cv_exo = self.exo_std_std / self.exo_std_mean
                if cv_exo <= 0.707:
                    raise ValueError(
                        f"Pareto distribution requires coefficient of variation > 0.707. "
                        f"Exogenous CV = {cv_exo:.3f} (mean={self.exo_std_mean}, std={self.exo_std_std}). "
                        f"Consider using 'gamma' distribution or increasing std relative to mean."
                    )
            
            # Check endogenous distribution
            if self.endo_std_distribution == "pareto":
                cv_endo = self.endo_std_std / self.endo_std_mean
                if cv_endo <= 0.707:
                    raise ValueError(
                        f"Pareto distribution requires coefficient of variation > 0.707. "
                        f"Endogenous CV = {cv_endo:.3f} (mean={self.endo_std_mean}, std={self.endo_std_std}). "
                        f"Consider using 'gamma' distribution or increasing std relative to mean."
                    )
    
    def get_config_summary(self) -> str:
        """
        Get a human-readable summary of the current configuration.
        
        Returns
        -------
        str
            A formatted string describing all hyperparameters.
        """
        lines = [
            "SCMBuilder Configuration:",
            "=" * 50,
            "",
            "Graph Structure:",
            f"  num_nodes: {self.num_nodes}",
            f"  graph_edge_prob: {self.graph_edge_prob}",
            f"  graph_seed: {self.graph_seed}",
            "",
            "Mechanism Type Selection:",
            f"  xgboost_prob: {self.xgboost_prob}",
            f"  mechanism_seed: {self.mechanism_seed}",
            "",
            "MLP Mechanism Hyperparameters:",
            f"  mlp_nonlins: {self.mlp_nonlins}",
            f"  mlp_num_hidden_layers: {self.mlp_num_hidden_layers}",
            f"  mlp_hidden_dim: {self.mlp_hidden_dim}",
            f"  mlp_activation_mode: {self.mlp_activation_mode}",
            f"  mlp_node_shape: {self.mlp_node_shape}",
            "",
            "XGBoost Mechanism Hyperparameters:",
            f"  xgb_num_hidden_layers: {self.xgb_num_hidden_layers}",
            f"  xgb_hidden_dim: {self.xgb_hidden_dim}",
            f"  xgb_activation_mode: {self.xgb_activation_mode}",
            f"  xgb_node_shape: {self.xgb_node_shape}",
            f"  xgb_n_training_samples: {self.xgb_n_training_samples}",
            f"  xgb_add_noise: {self.xgb_add_noise}",
            "",
            "Noise Distribution Parameters:",
            f"  random_additive_std: {self.random_additive_std}",
        ]
        
        if not self.random_additive_std:
            lines.extend([
                f"  exo_std: {self.exo_std}",
                f"  endo_std: {self.endo_std}",
            ])
        else:
            lines.extend([
                f"  exo_std_mean: {self.exo_std_mean}",
                f"  exo_std_std: {self.exo_std_std}",
                f"  endo_std_mean: {self.endo_std_mean}",
                f"  endo_std_std: {self.endo_std_std}",
            ])
        
        lines.extend([
            "",
            "SCM Configuration:",
            f"  scm_fast: {self.scm_fast}",
            f"  use_exogenous_mechanisms: {self.use_exogenous_mechanisms}",
            f"  mechanism_generator_seed: {self.mechanism_generator_seed}",
        ])
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with default hyperparameters
    print("Example 1: Basic SCM with default hyperparameters")
    builder = SCMBuilder(
        num_nodes=5,
        graph_edge_prob=0.3,
        graph_seed=42
    )
    print(builder.get_config_summary())
    scm = builder.build()
    print(f"Created SCM with {len(scm.dag.nodes())} nodes")
    print()
    
    # Example 2: Custom hyperparameters
    print("Example 2: SCM with custom hyperparameters")
    builder = SCMBuilder(
        num_nodes=10,
        graph_edge_prob=0.5,
        graph_seed=123,
        xgboost_prob=0.3,
        mlp_num_hidden_layers=2,
        mlp_hidden_dim=32,
        random_additive_std=False,
        exo_std=0.5,
        endo_std=0.05,
        use_exogenous_mechanisms=False
    )
    scm = builder.build()
    print(f"Created SCM with {len(scm.dag.nodes())} nodes")
    
    # Test sampling (Method 1: Manual noise sampling)
    scm.sample_exogenous(num_samples=100)  # Required for all modes
    scm.sample_endogenous_noise(num_samples=100)  # Required for all modes
    samples = scm.propagate(num_samples=100)
    print(f"Sampled data: Dict with {len(samples)} nodes, first node shape: {list(samples.values())[0].shape}")
    
    # Test sampling (Method 2: Use convenience method)
    samples2 = builder.build_and_sample(num_samples=50)
    print(f"Convenience method: Dict with {len(samples2)} nodes, first node shape: {list(samples2.values())[0].shape}")
    
    # Example 3: Using safe mode (still requires pre-sampling)
    print("\nExample 3: SCM with safe mode")
    builder_safe = SCMBuilder(
        num_nodes=3,
        graph_edge_prob=0.4,
        graph_seed=456,
        scm_fast=False  # Use safe mode
    )
    scm_safe = builder_safe.build()
    scm_safe.sample_exogenous(num_samples=50)  # Still required even in safe mode
    scm_safe.sample_endogenous_noise(num_samples=50)  # Still required even in safe mode
    samples_safe = scm_safe.propagate(num_samples=50)
    print(f"Safe mode: Dict with {len(samples_safe)} nodes, first node shape: {list(samples_safe.values())[0].shape}")
    
    # Example 4: Get concatenated tensor instead of dictionary
    print("\nExample 4: Get concatenated tensor")
    tensor_samples = builder.build_and_sample_tensor(num_samples=20)
    print(f"Concatenated tensor shape: {tensor_samples.shape}")

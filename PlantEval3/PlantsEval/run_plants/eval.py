from pathlib import Path
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Use absolute path for base directory (shared filesystem)
# This ensures results go to the right place even when running on cluster nodes
BASE_DIR = Path("/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval")


class SimpleRealization:
    """Simple wrapper to mimic CID_Dataset interface."""
    def __init__(self, data_dict):
        self.X_train = data_dict['X_train']
        self.t_train = data_dict['t_train']
        self.y_train = data_dict['y_train']
        self.X_test = data_dict['X_test']
        self.t_test = data_dict['t_test']
        self.true_cid = data_dict['true_cid']
        # Include graph matrices if present
        self.adjacency_matrix = data_dict.get('adjacency_matrix', None)
        self.ancestor_matrix = data_dict.get('ancestor_matrix', None)


def load_benchmark(filepath):
    """Load benchmark, trying standard pickle first, then dill as fallback."""
    filepath = Path(filepath)
    
    print(f"========== DEBUG: load_benchmark ==========")
    print(f"Attempting to load: {filepath}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory:")
    for f in sorted(os.listdir('.')):
        if os.path.isfile(f):
            size = os.path.getsize(f) / 1024 / 1024  # MB
            print(f"  {f} ({size:.2f} MB)")
    print(f"=========================================")
    
    # First, try standard pickle (works with _std.pkl files)
    if filepath.exists():
        print(f"File exists at: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                data = pkl.load(f)
            print(f"Successfully loaded with pickle. Type: {type(data)}")
            # Check if it's the _std format (dict with 'realizations' key)
            if isinstance(data, dict) and 'realizations' in data:
                realizations = [SimpleRealization(r) for r in data['realizations']]
                return realizations
            # Check if it's a list of dicts from convert_dill_to_pickle.py
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                print(f"Found list of {len(data)} dict realizations")
                realizations = [SimpleRealization(r) for r in data]
                return realizations
            # Otherwise it might be a CIDBenchmark object
            elif hasattr(data, 'realizations'):
                return data.realizations
        except Exception as e:
            print(f"Standard pickle failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"File NOT found at: {filepath}")
    
    # Try to find _std version as fallback
    std_path = Path(str(filepath).replace('.pkl', '_std.pkl'))
    print(f"Trying _std version: {std_path}, exists={std_path.exists()}")
    if std_path.exists():
        try:
            with open(std_path, 'rb') as f:
                data = pkl.load(f)
            # data is a list of dicts, convert to SimpleRealization objects
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                realizations = [SimpleRealization(r) for r in data]
                return realizations
        except Exception as e:
            print(f"Standard pickle _std file failed: {e}")
    
    # Fall back to trying dill
    print("Falling back to dill...")
    try:
        import dill
        with open(filepath, 'rb') as f:
            benchmark = dill.load(f)
        return benchmark.realizations
    except ImportError:
        raise ImportError(f"Could not load {filepath}. Install dill or use _std.pkl version.")


def mean_integrated_squared_error(t, y_true, y_pred):
    mise = 0
    for s in np.unique(t):
        mise += mean_squared_error(y_true[t == s], y_pred=y_pred[t == s])
    return mise

def avg_mean_integrated_squared_error(t, y_true, y_pred):
    return 0

def evaluate_pipeline(exp_name, model_pipeline, model, args):
    # Results go to shared filesystem (absolute path)
    exp_dir = BASE_DIR / "results" / exp_name
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Data is in local directory (transferred by HTCondor)
    # Use relative path from current working directory (plant_data/ subdirectory)
    data_dir = Path("plant_data")
    dataset_name = args.dataset
    if not dataset_name.endswith('.pkl'):
        dataset_name = dataset_name + ".pkl"
    data_path = str(data_dir / Path(dataset_name))
    
    realizations = load_benchmark(data_path)
    
    # Check if model supports NLL computation (DOFM with BarDistribution)
    supports_nll = (
        args.model == 'dofm' and 
        hasattr(model, 'use_bar_distribution') and 
        model.use_bar_distribution and
        hasattr(model, 'log_likelihood')
    )

    for dataset_index, cid_dataset in tqdm(enumerate(realizations)):
        
        cid_pred = model_pipeline(model, cid_dataset)
        mise = mean_integrated_squared_error(cid_dataset.t_test, cid_dataset.true_cid, cid_pred)
        amise = avg_mean_integrated_squared_error(cid_dataset.t_test, cid_dataset.true_cid, cid_pred)

        print(f"MISE: {mise}")
        print(f"ATE Rel. Err: {amise}")
        
        # Compute NLL if supported
        nll = None
        if supports_nll:
            try:
                # Get the ancestor matrix from the pipeline function
                # We need to reconstruct it the same way as in the pipeline
                import numpy as np
                from run_dofm import construct_ancestor_matrix
                
                # Get settings from pipeline function
                graph_mode = getattr(model_pipeline, 'graph_mode', 'full_graph')
                
                # Reconstruct ancestor matrix for the model's expected feature size
                model_n_features = model.model.num_features  # Should be 50
                ancestor_matrix = construct_ancestor_matrix(model_n_features, graph_mode)
                
                # Optionally apply correlation filtering if it was used
                hide_by_correlation = getattr(model_pipeline, 'hide_by_correlation', False)
                if hide_by_correlation and graph_mode == "full_graph":
                    from run_plants.run_dofm import hide_edges_by_correlation
                    
                    # Need to get the preprocessed training data
                    X_train = np.array(cid_dataset.X_train, dtype=np.float32)
                    t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
                    y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
                    
                    # Apply same preprocessing as pipeline
                    eps = 1e-8
                    
                    # Target encoding if used
                    use_target_encoding = getattr(model_pipeline, 'use_target_encoding', False)
                    if use_target_encoding:
                        categorical_indices = [2]
                        for cat_idx in categorical_indices:
                            unique_cats = np.unique(X_train[:, cat_idx])
                            target_means = {}
                            global_mean = y_train.mean()
                            for cat in unique_cats:
                                mask = X_train[:, cat_idx] == cat
                                if mask.sum() > 0:
                                    target_means[cat] = y_train[mask].mean()
                                else:
                                    target_means[cat] = global_mean
                            for cat in unique_cats:
                                train_mask = X_train[:, cat_idx] == cat
                                X_train[train_mask, cat_idx] = target_means[cat]
                    
                    # Standardize
                    X_mean = X_train.mean(axis=0, keepdims=True)
                    X_std = X_train.std(axis=0, keepdims=True) + eps
                    X_train = (X_train - X_mean) / X_std
                    
                    t_mean = t_train.mean()
                    t_std = t_train.std() + eps
                    t_train = (t_train - t_mean) / t_std
                    
                    y_min = y_train.min()
                    y_max = y_train.max()
                    y_range = y_max - y_min + eps
                    y_train = 2.0 * (y_train - y_min) / y_range - 1.0
                    
                    top_k = getattr(model_pipeline, 'top_k_edges', 10)
                    n_real_features = 6
                    X_train_unpadded = X_train[:, :n_real_features]
                    ancestor_matrix = hide_edges_by_correlation(
                        ancestor_matrix, X_train_unpadded, t_train, y_train, top_k=top_k
                    )
                
                # Get test data - need to preprocess it the same way as in pipeline
                X_test = np.array(cid_dataset.X_test, dtype=np.float32)
                t_test = np.array(cid_dataset.t_test, dtype=np.float32).reshape(-1, 1)
                y_test = np.array(cid_dataset.true_cid, dtype=np.float32).reshape(-1, 1)
                
                X_train = np.array(cid_dataset.X_train, dtype=np.float32)
                t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
                y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
                
                # Apply same preprocessing as in pipeline
                eps = 1e-8
                
                # Target encoding if used
                use_target_encoding = getattr(model_pipeline, 'use_target_encoding', False)
                if use_target_encoding:
                    categorical_indices = [2]
                    for cat_idx in categorical_indices:
                        unique_cats = np.unique(X_train[:, cat_idx])
                        target_means = {}
                        global_mean = y_train.mean()
                        for cat in unique_cats:
                            mask = X_train[:, cat_idx] == cat
                            if mask.sum() > 0:
                                target_means[cat] = y_train[mask].mean()
                            else:
                                target_means[cat] = global_mean
                        for cat in unique_cats:
                            train_mask = X_train[:, cat_idx] == cat
                            X_train[train_mask, cat_idx] = target_means[cat]
                            test_mask = X_test[:, cat_idx] == cat
                            if cat in target_means:
                                X_test[test_mask, cat_idx] = target_means[cat]
                            else:
                                X_test[test_mask, cat_idx] = global_mean
                
                # Standardize features
                X_mean = X_train.mean(axis=0, keepdims=True)
                X_std = X_train.std(axis=0, keepdims=True) + eps
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std
                
                # Standardize treatment
                t_mean = t_train.mean()
                t_std = t_train.std() + eps
                t_train = (t_train - t_mean) / t_std
                t_test = (t_test - t_mean) / t_std
                
                # Scale outcome to (-1, 1) for training data
                y_min = y_train.min()
                y_max = y_train.max()
                y_range = y_max - y_min + eps
                y_train = 2.0 * (y_train - y_min) / y_range - 1.0
                # Also scale test outcome for NLL computation
                y_test = 2.0 * (y_test - y_min) / y_range - 1.0
                
                # Pad features to model size
                model_n_features = model.model.num_features
                n_real_features = X_train.shape[1]
                if n_real_features < model_n_features:
                    pad_width = model_n_features - n_real_features
                    X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                    X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                
                # Compute log-likelihood
                log_likelihoods = model.log_likelihood(
                    X_obs=X_train,
                    T_obs=t_train,
                    Y_obs=y_train.squeeze(),
                    X_intv=X_test,
                    T_intv=t_test,
                    Y_intv=y_test.squeeze(),
                    adjacency_matrix=ancestor_matrix,
                    batched=False
                )
                
                # NLL is negative of average log-likelihood
                nll = -np.mean(log_likelihoods)
                print(f"NLL: {nll:.6f}")
                
            except Exception as e:
                print(f"Warning: Could not compute NLL: {e}")
                import traceback
                traceback.print_exc()
                nll = None

        result_dict = {
            "model": args.model, 
            "dataset": args.dataset, 
            "realization": dataset_index, 
            "mise": mise,
            "cate_preds": cid_pred,
            "amise": amise,
            "nll": nll,  # Add NLL to results
            # Store the data arrays instead of the entire object to avoid pickling issues
            "X_train": cid_dataset.X_train,
            "t_train": cid_dataset.t_train,
            "y_train": cid_dataset.y_train,
            "X_test": cid_dataset.X_test,
            "t_test": cid_dataset.t_test,
            "true_cid": cid_dataset.true_cid
        }

        with open(exp_dir / f"{args.model}_{args.dataset}_{dataset_index}", "wb") as f:
            pkl.dump(result_dict, f)
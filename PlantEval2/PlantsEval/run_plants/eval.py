from pathlib import Path
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import mean_squared_error
import numpy as np

# Use absolute path for base directory (shared filesystem)
# This ensures results go to the right place even when running on cluster nodes
BASE_DIR = Path("/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval2/PlantsEval")


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
    
    # First, try standard pickle (works with _std.pkl files)
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                data = pkl.load(f)
            # Check if it's the _std format (dict with 'realizations' key)
            if isinstance(data, dict) and 'realizations' in data:
                realizations = [SimpleRealization(r) for r in data['realizations']]
                return realizations
            # Otherwise it might be a CIDBenchmark object
            elif hasattr(data, 'realizations'):
                return data.realizations
        except Exception as e:
            print(f"Standard pickle failed: {e}, trying dill...")
    
    # Try to find _std version as fallback
    std_path = Path(str(filepath).replace('.pkl', '_std.pkl'))
    if std_path.exists():
        try:
            with open(std_path, 'rb') as f:
                data = pkl.load(f)
            realizations = [SimpleRealization(r) for r in data['realizations']]
            return realizations
        except Exception:
            pass
    
    # Fall back to trying dill
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
    # Use relative path from current working directory
    data_dir = Path("plant_data")
    # Check if dataset already ends with .pkl
    dataset_name = args.dataset
    if not dataset_name.endswith('.pkl'):
        dataset_name = dataset_name + ".pkl"
    data_path = str(data_dir / Path(dataset_name))
    realizations = load_benchmark(data_path)

    for dataset_index, cid_dataset in tqdm(enumerate(realizations)):
        
        cid_pred = model_pipeline(model, cid_dataset)
        mise = mean_integrated_squared_error(cid_dataset.t_test, cid_dataset.true_cid, cid_pred)
        amise = avg_mean_integrated_squared_error(cid_dataset.t_test, cid_dataset.true_cid, cid_pred)

        print(f"MISE: {mise}")
        print(f"ATE Rel. Err: {amise}")

        result_dict = {
            "model": args.model, 
            "dataset": args.dataset, 
            "realization": dataset_index, 
            "mise": mise,
            "cate_preds": cid_pred,
            "amise": amise,
            "cid_dataset": cid_dataset
        }

        with open(exp_dir / f"{args.model}_{args.dataset}_{dataset_index}", "wb") as f:
            pkl.dump(result_dict, f)
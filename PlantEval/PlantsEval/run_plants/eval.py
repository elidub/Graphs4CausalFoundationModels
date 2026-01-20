from pathlib import Path
from tqdm import tqdm
import dill as pkl
from sklearn.metrics import mean_squared_error
import numpy as np

# Get the base directory (PlantEval/PlantsEval)
BASE_DIR = Path(__file__).parent.parent

def mean_integrated_squared_error(t, y_true, y_pred):
    mise = 0
    for s in np.unique(t):
        mise += mean_squared_error(y_true[t == s], y_pred=y_pred[t == s])
    return mise

def avg_mean_integrated_squared_error(t, y_true, y_pred):
    return 0

def evaluate_pipeline(exp_name, model_pipeline, model, args):
    exp_dir = BASE_DIR / "results" / exp_name
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    data_dir = BASE_DIR / "plant_data"
    with open(str(data_dir  / Path(args.dataset)) + ".pkl", "rb") as f:
        cid_benchmark = pkl.load(f)

    for dataset_index, cid_dataset in tqdm(enumerate(cid_benchmark.realizations)):
        
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
    
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from tqdm import tqdm
import pickle as pkl

from CausalPFN.benchmarks import IHDPDataset, ACIC2016Dataset
from CausalPFN.benchmarks import RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset

datasets = {
    "IHDP": IHDPDataset(),
    "ACIC": ACIC2016Dataset(),
    "CPS": RealCauseLalondeCPSDataset(),
    "PSID": RealCauseLalondePSIDDataset(),
}

def relative_error(y_true, y_pred):
    if y_true == 0:
        return float('inf') if y_pred != 0 else 0.0
        
    r = abs(y_true - y_pred) / abs(y_true)

    print(f"Predicted ate: {y_pred}, True ate: {y_true}, Relative Error: {r}")
    return r

def evaluate_pipeline(exp_name, model_pipeline, model, args):
    # Use local results directory relative to RealCauseEval folder
    base_dir = Path(__file__).parent.parent / "results"
    exp_dir = base_dir / exp_name
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    dataset = datasets[args.dataset]

    all_pehe = []
    all_ate_rel_err = []

    for realization in tqdm(range(dataset.n_tables)):
        cate_dataset = dataset[realization][0]

        cate_pred = model_pipeline(model, cate_dataset)
        pehe = root_mean_squared_error(
            cate_dataset.true_cate, 
            cate_pred
        )
        ate_rel_err = relative_error(
            cate_dataset.true_cate.mean(), 
            cate_pred.mean()
        )

        print(f"PEHE: {pehe}")
        print(f"ATE Rel. Err: {ate_rel_err}")

        all_pehe.append(pehe)
        all_ate_rel_err.append(ate_rel_err)

        result_dict = {
            "model": args.model, 
            "dataset": args.dataset, 
            "realization": realization, 
            "pehe": pehe, 
            "cate_preds": cate_pred,
            "ate_rel_err": ate_rel_err
        }

        with open(exp_dir / f"{args.model}_{args.dataset}_{realization}", "wb") as f:
            pkl.dump(result_dict, f)
    
    # Print summary statistics
    import numpy as np
    print(f"\n{'='*40}")
    print(f"Summary for {args.dataset}:")
    print(f"  Average PEHE:         {np.mean(all_pehe):.4f} ± {np.std(all_pehe):.4f}")
    print(f"  Average ATE Rel Err:  {np.mean(all_ate_rel_err):.4f} ± {np.std(all_ate_rel_err):.4f}")
    print(f"{'='*40}\n")
    
    return {"pehe": all_pehe, "ate_rel_err": all_ate_rel_err}
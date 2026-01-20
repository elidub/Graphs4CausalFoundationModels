import argparse
import torch
import sys
import os
import numpy as np

from run_plants.eval import evaluate_pipeline

from tabpfn import TabPFNRegressor

sys.path.append(os.path.abspath("/work/dlclarge2/robertsj-dofm"))
from generate_plant_data import CID_Benchmark

def tabpfn_pipeline(model, cid_dataset):
    t_X_train = np.concatenate([cid_dataset.t_train.reshape(-1, 1), cid_dataset.X_train], axis=1)
    model.fit(t_X_train, cid_dataset.y_train)
    t_X_test = np.concatenate([cid_dataset.t_test.reshape(-1, 1), cid_dataset.X_test], axis=1)
    cid_pred = model.predict(t_X_test)
    return cid_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")    
    parser.add_argument("--exp_name", type=str, required=True, help="Current time of experiment")    

    args = parser.parse_args()

    print(f"--- Starting Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
        
    model = TabPFNRegressor()

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=tabpfn_pipeline,
        model=model,
        args=args)
    
print(f"--- Experiment Finished ---")    

#!/bin/bash

datasets=("CID_10_ints_10_reals")
models=("tabpfn" "catboost" "dummy")
now=$(date +"%Y-%m-%d_%H-%M-%S")

echo $now

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Submitting job for Dataset: $dataset, Model: $model"
        
        sbatch --export=ALL,DATASET=$dataset,MODEL=$model,EXP_NAME=$now submit_plants.sh
    done
done
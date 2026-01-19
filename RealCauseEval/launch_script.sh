#!/bin/bash

datasets=("ACIC" "CPS" "IHDP" "PSID")
models=("causalpfn" "causalfm" "dopfn" "s_learner_tabpfn" "t_learner_tabpfn" "x_learner_tabpfn")
now=$(date +"%Y-%m-%d_%H-%M-%S")

echo $now

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Submitting job for Dataset: $dataset, Model: $model"
        
        python -m run_baselines.dofm --model $model -- dataset $dataset --exp_name $now
    done
done
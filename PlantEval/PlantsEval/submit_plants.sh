#!/bin/bash

#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=alldlc2_gpu-l40s

source /home/robertsj/miniconda3/etc/profile.d/conda.sh

if [[ "$MODEL" == "tabpfn" ]]; then
    conda activate dofm
    pip install -U tabpfn
    export HF_TOKEN="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"
    export TABPFN_DISABLE_TELEMETRY=1
    python3 -m run_plants.run_tabpfn --dataset $DATASET --model $MODEL --exp_name $EXP_NAME
elif [[ "$MODEL" == "catboost" ]]; then
    conda activate dofm
    python3 -m run_plants.run_catboost --dataset $DATASET --model $MODEL --exp_name $EXP_NAME
elif [[ "$MODEL" == "dummy" ]]; then
    conda activate dofm
    python3 -m run_plants.dummy --dataset $DATASET --model $MODEL --exp_name $EXP_NAME
elif [[ "$MODEL" == "dofm" ]]; then
    conda activate dofm
    python3 -m run_plants.run_dofm --dataset $DATASET --model $MODEL --exp_name $EXP_NAME
fi
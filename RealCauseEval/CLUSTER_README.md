# DOFM Baseline Cluster Submission

This directory contains scripts for running the DOFM baseline evaluation on HTCondor cluster.

## Files

- `run_dofm_cluster.sh`: Bash script that runs on cluster nodes
- `submit_dofm.sub`: HTCondor submission file
- `logs/`: Directory for cluster job logs

## Usage

### Submit all datasets (ACIC, IHDP, CPS, PSID)
```bash
cd /fast/arikreuter/DoPFN_v2/CausalPriorFitting/RealCauseEval
condor_submit submit_dofm.sub
```

This will queue 4 jobs, one for each dataset.

### Submit a single dataset
To submit only one dataset, you can use:
```bash
condor_submit submit_dofm.sub -append "queue 1" -append "dataset=ACIC" -append "model_name=dofm_v2" -append "exp_name=acic_dofm_evaluation"
```

### Monitor jobs
```bash
condor_q  # View your queued/running jobs
condor_q -analyze <job_id>  # Analyze why a job is waiting
```

### Check logs
Logs are saved in the `logs/` directory with format:
- `dofm_<dataset>_<ClusterId>.<ProcId>.out` - Standard output
- `dofm_<dataset>_<ClusterId>.<ProcId>.err` - Standard error
- `dofm_<dataset>_<ClusterId>.<ProcId>.log` - HTCondor log

### Check results
Results are saved in:
```
/fast/arikreuter/DoPFN_v2/CausalPriorFitting/RealCauseEval/results/<exp_name>/
```

Each result file contains:
- model: Model name
- dataset: Dataset name
- realization: Realization number
- pehe: Precision in Estimation of Heterogeneous Effect
- cate_preds: CATE predictions
- ate_rel_err: ATE relative error

## Configuration

The submission file uses:
- Model: `final_earlytest_16773250.0` (DOFM model)
- Checkpoint: `final_model_with_bardist.pt`
- Resources: 1 GPU, 64GB RAM, 8 CPUs, 32GB disk
- Virtual environment: `/fast/arikreuter/DoPFN_v2/venv.zip`

## Notes

- The script automatically unzips the virtual environment on the cluster node
- All necessary files (source code, model, CausalPFN benchmarks) are transferred automatically
- Results are transferred back when the job completes

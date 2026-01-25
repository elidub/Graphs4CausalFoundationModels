# PlantEval3 HTCondor Submission Scripts

This directory contains scripts for running DOFM experiments on the HTCondor cluster.

## Files

- `run_dofm_cluster.sh`: Bash script that runs on cluster nodes
- `submit_dofm.sub`: HTCondor submission file for best configurations
- `submit_dofm_all.sub`: HTCondor submission file for all tested configurations

## Usage

### Submit best configurations (3 jobs)
```bash
cd /fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval/run_plants
condor_submit submit_dofm.sub
```

This submits:
- All unknown + target encoding (best overall: MISE ~2.03)
- Top-3 edges + target encoding (MISE ~2.13)  
- Top-5 edges + target encoding (MISE ~2.43)

### Submit all configurations (9 jobs)
```bash
cd /fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval/run_plants
condor_submit submit_dofm_all.sub
```

This submits all tested configurations including ablations.

### Monitor jobs
```bash
# Check job status
condor_q

# Check specific job batch
condor_q -nobatch -constraint 'JobBatchName == "dofm_planteval3_all"'

# View job details
condor_q -better-analyze <job_id>

# Check logs
tail -f logs/dofm_*.out
```

### Cancel jobs
```bash
# Cancel specific job
condor_rm <job_id>

# Cancel all your jobs
condor_rm $USER

# Cancel specific batch
condor_rm -constraint 'JobBatchName == "dofm_planteval3_all"'
```

## Configurations

### Best Configurations (submit_dofm.sub)

1. **All unknown + target encoding** (BEST)
   - Graph mode: `all_unknown`
   - Target encoding: `true`
   - MISE: 2.03 ± 0.42 (23.4% worse than CatBoost)
   
2. **Top-3 edges + target encoding**
   - Graph mode: `full_graph`
   - Correlation filtering: top-3 edges
   - Target encoding: `true`
   - MISE: 2.13 ± 0.46 (29.5% worse than CatBoost)

3. **Top-5 edges + target encoding**
   - Graph mode: `full_graph`
   - Correlation filtering: top-5 edges
   - Target encoding: `true`
   - MISE: 2.43 ± 0.31 (47.4% worse than CatBoost)

### All Configurations (submit_dofm_all.sub)

Includes all ablations:
- With/without target encoding
- With/without correlation filtering
- Different numbers of edges (3, 5, 10)
- All unknown vs full graph

## Results

Results are saved to:
```
/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval/results/<exp_name>/
```

Each result file contains:
- MISE (Mean Integrated Squared Error)
- NLL (Negative Log-Likelihood) - only for DOFM
- CATE predictions
- Test data

## Resource Requirements

Each job requests:
- 1 GPU
- 32GB RAM
- 4 CPUs
- 16GB disk

Runtime: ~2-3 hours per job (10 realizations)

## Key Findings

1. **Target encoding is critical**: Encoding the categorical `Bed_Position` feature improved performance by ~10-14%

2. **Less graph info is better**: Providing no graph information (all_unknown) outperformed full graph knowledge
   - All unknown: MISE 2.03
   - Full graph: MISE 3.43
   
3. **Correlation filtering helps**: When using graph info, keeping only top-3 strongest edges works best
   - Suggests model learns better from data than explicit graph priors
   - Three-state format {-1, 0, 1} may confuse the model

4. **Still behind CatBoost**: Best DOFM (2.03) is 23% worse than CatBoost (1.65)

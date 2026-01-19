# Hi Arik

To evaluate on a single dataset, from this directory:

```
python -m run_baselines.dofm --model dofm --dataset ACIC --exp_name test
```

The results will appear in `results/test/ACIC/dofm_ACIC_x`

To evaluate on all datasets:

`bash launch_script.sh``

The results will appear in `results/<DATE>/ACIC/dofm_ACIC_x`


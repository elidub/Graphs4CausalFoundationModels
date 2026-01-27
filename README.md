Welcome to the repository for our ICML paper "Use what you know: Causal Foundation Models with Partial Graphs"

In general, experiments with corresponding configs can be run via: 

python3 run.py --config "path/to/config"

The configs for the linear-Gaussain experimetns are in: 

- /experiments/GraphConditioning/configs_50node
- experiments/GraphConditioning/configs_50node_ancestor
- experiments/GraphConditioning/configs_50node_idk

To train an evaluate a predictive model, use /experiments/Predictive/configs/predictive.yaml


The configs to train the models for the complexmech experiments are in: experiments/complexmech/configs
from priors.causal_prior.scm.InspectSCMSamples import InspectSCMSamples
from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
from priors.causal_prior.scm.SCMBuilder import SCMBuilder
from priors.causal_prior.scm.Basic_Configs import default_sampling_config 


def main():
    BATCH_SIZE = 256
    seed = 128
    config = default_sampling_config

    sampler = SCMHyperparameterSampler(config, seed=seed)
    print(sampler.get_parameter_summary())

    sampled_params = sampler.sample()
    builder = SCMBuilder(**sampled_params)
    scm = builder.build()

    # Sample both exogenous and endogenous noise before generating data
    scm.sample_exogenous(num_samples=BATCH_SIZE)
    scm.sample_endogenous_noise(num_samples=BATCH_SIZE)

    r = scm.propagate(num_samples=BATCH_SIZE)  # this is just to demonstrate how one dataset can be sampled

    inspector = InspectSCMSamples(scm=scm, batch_size=BATCH_SIZE, reduce="mean", device="cpu")
    inspector.full_report(
        n_boot=2000,
        alpha=0.05,
        do_plots=True,
        pairwise_max_nodes=10,
        bins=40,
        benchmark_repeats=1,
        benchmark_warmup=0,
    )



    
if __name__ == "__main__":
    main()
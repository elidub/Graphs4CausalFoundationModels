from priors.causal_prior.scm.InspectSCMSamples import InspectSCMSamples
from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
from priors.causal_prior.scm.SCMBuilder import SCMBuilder
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config 


def main():
    N_SAMPLES = 4
    seed = 128
    config = default_sampling_config

    sampler = SCMHyperparameterSampler(config, seed=seed)
    print(sampler.get_parameter_summary())

    sampled_params = sampler.sample()
    builder = SCMBuilder(**sampled_params)
    scm = builder.build()

    # Sample both exogenous and endogenous noise before generating data
    scm.sample_exogenous(num_samples=N_SAMPLES)
    scm.sample_endogenous_noise(num_samples=N_SAMPLES)

    r = scm.propagate(num_samples=N_SAMPLES)  # this is just to demonstrate how one dataset can be sampled

    print(r)

    
if __name__ == "__main__":
    main()
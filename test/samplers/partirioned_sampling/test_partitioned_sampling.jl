#Script contatining tests for Partitioned Sampling

#Import Libraries
using BAT
using Test

############# Preliminars: define model, prior, likelihood and posterior ###############

## Define the model, a multimode gaussian will be taken here
σ = [0.1 0 0; 0 0.1 0; 0 0 0.1];
μ = [1 1 -1; -1 1 0; -1 -1 -1; 1 -1 0];
mixture_model = MixtureModel(MvNormal[MvNormal(μ[i,:], Matrix(Hermitian(σ)) ) for i in 1:4]);

#Define a flat prior
prior = NamedTupleDist(a = [Uniform(-50,50), Uniform(-50,50), Uniform(-50,50)])

#Define the Likelihood
likelihood = let model = mixture_model
    params -> LogDVal(logpdf(model, params.a))
end

#Define the posterior
posterior = PosteriorDensity(likelihood, prior);

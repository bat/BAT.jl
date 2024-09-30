using BAT
using AdvancedHMC
using AffineMaps
using AutoDiffOperators
using ValueShapes



context = BATContext(ad = ADModule(:ForwardDiff))

posterior = BAT.example_posterior()

target, trafo = BAT.transform_and_unshape(PriorToGaussian(), posterior, context)

s = BAT.cholesky(BAT._approx_cov(target, totalndof(varshape(target)))).L
f = BAT.CustomTransform(Mul(s))


# Metropolis-Hastings MC Sampling

propcov_result = BAT.bat_sample_impl(posterior, 
                                     MCMCSampling(adaptive_transform = f), 
                                     context
)
propcov_samples = propcov_result.result

using Plots
plot(propcov_samples)


ram_result = BAT.bat_sample_impl(posterior, 
                                 MCMCSampling(adaptive_transform = f, trafo_tuning = RAMTuning()), 
                                 context
)
ram_samples = ram_result.result
plot(ram_samples)

# Advanced Hamiltonian MC Sampling

hmc_result = BAT.bat_sample_impl(posterior,
                                 MCMCSampling(adaptive_transform = f, proposal = HamiltonianMC(), trafo_tuning = StanHMCTuning()),
                                 context
)
hmc_samples = hmc_result.result
plot(hmc_samples)


######## Comparison with old BAT

using Revise
using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using BAT, DensityInterface, IntervalSets

using BAT: next_cycle!, mcmc_tuning_reinit!!, get_rng, mcmc_tune_post_cycle!!, mcmc_iterate!!


data = vcat(
    rand(Normal(-1.0, 0.5), 500),
    rand(Normal( 2.0, 0.5), 1000)
);
hist = append!(Histogram(-2:0.1:4), data);

function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
    p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
    p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
end

true_par_values = (a = [500, 1000], mu = [-1.0, 2.0], sigma = 0.5);



likelihood = let h = hist, f = fit_function
    ## Histogram counts for each bin as an array:
    observed_counts = h.weights

    ## Histogram binning:
    bin_edges = h.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    logfuncdensity(function (params)
        ## Log-likelihood for a single bin:
        function bin_log_likelihood(i)
            ## Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_widths[i] * f(params, bin_centers[i])
            ## Avoid zero expected counts for numerical stability:
            logpdf(Poisson(expected_counts + eps(expected_counts)), observed_counts[i])
        end

        ## Sum log-likelihood over bins:
        idxs = eachindex(observed_counts)
        ll_value = bin_log_likelihood(idxs[1])
        for i in idxs[2:end]
            ll_value += bin_log_likelihood(i)
        end

        return ll_value
    end)
end;

prior = distprod(
    a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Weibull(1.2, 2)
);

posterior = PosteriorMeasure(likelihood, prior);

samples = bat_sample(posterior, MCMCSampling(proposal = MetropolisHastings(proposaldist = TDist(1.0)), nsteps = 10^5, nchains = 4)).result

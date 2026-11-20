using Random, LinearAlgebra, Statistics, Distributions, StatsBase

par_names = ["a_1", "a_2", "mu_1", "mu_2", "sigma"]

true_par_values = [500, 1000, -1.0, 2.0, 0.5]

function model_distributions(parameters::AbstractVector{<:Real})
    return (
        Normal(parameters[3], parameters[5]),
        Normal(parameters[4], parameters[5])
    )
end

data = vcat(
    rand(model_distributions(true_par_values)[1], Int(true_par_values[1])),
    rand(model_distributions(true_par_values)[2], Int(true_par_values[2]))
)

typeof(data) == Vector{Float64}

hist = append!(Histogram(-2:0.1:4), data)

function fit_function(x::Real, parameters::AbstractVector{<:Real})
    dists = model_distributions(parameters)
    return parameters[1] * pdf(dists[1], x) +
           parameters[2] * pdf(dists[2], x)
end

using Plots

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data and True Statistical Model"
)
plot!(
    -4:0.01:4, x -> fit_function(x, true_par_values),
    label = "Truth"
)
savefig("tutorial-data-and-truth.pdf")

using BAT, IntervalSets

struct HistogramLikelihood{H<:Histogram} <: AbstractDensity
    histogram::H
end

BAT.nparams(likelihood::HistogramLikelihood) = 5

function BAT.unsafe_density_logval(
    likelihood::HistogramLikelihood,
    parameters::AbstractVector{<:Real},
    exec_context::ExecContext
)

    counts = likelihood.histogram.weights

    binning = likelihood.histogram.edges[1]

    log_likelihood::Float64 = 0.0
    for i in eachindex(counts)
        bin_left, bin_right = binning[i], binning[i+1]
        bin_width = bin_right - bin_left
        bin_center = (bin_right + bin_left) / 2

        observed_counts = counts[i]

        expected_counts = bin_width * fit_function(bin_center, parameters)

        log_likelihood += logpdf(Poisson(expected_counts), observed_counts)
    end

    return log_likelihood
end

BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval), likelihood::HistogramLikelihood, parameters::AbstractVector{<:Real}) =
    ExecCapabilities(0, true, 0, true)

likelihood = HistogramLikelihood(hist)

prior = ConstDensity(
    HyperRectBounds(
        [
            0.0..10.0^4, 0.0..10.0^4,
            -2.0..0.0, 1.0..3.0,
            0.3..0.7
        ],
        reflective_bounds
    ),
    normalize
)

### Bayesian Model Definition

model = BayesianModel(likelihood, prior)


### Parameter Space Exploration via MCMC

algorithm = MetropolisHastings(MvTDistProposalSpec(1.0))

rngseed = BAT.Philox4xSeed()

chainspec = MCMCSpec(algorithm, model, rngseed)

nsamples = 10^5
nchains = 4

tuner_config = ProposalCovTunerConfig(
    λ = 0.5,
    α = 0.15..0.35,
    β = 1.5,
    c = 1e-4..1e2
)

convergence_test = BGConvergence(1.1)

init_strategy = MCMCInitStrategy(
    ninit_tries_per_chain = 8..128,
    max_nsamples_pretune = 25,
    max_nsteps_pretune = 250,
    max_time_pretune = Inf
)

burnin_strategy = MCMCBurninStrategy(
    max_nsamples_per_cycle = 1000,
    max_nsteps_per_cycle = 10000,
    max_time_per_cycle = Inf,
    max_ncycles = 30
)

BAT.Logging.set_log_level!(BAT, BAT.Logging.LOG_DEBUG)

samples, sampleids, stats, chains = rand(
    chainspec,
    nsamples,
    nchains,
    tuner_config = tuner_config,
    convergence_test = convergence_test,
    init_strategy = init_strategy,
    burnin_strategy = burnin_strategy,
    max_nsteps = 10000,
    max_time = Inf,
    granularity = 1,
    ll = BAT.Logging.LOG_INFO
)

println("Truth: $true_par_values")
println("Mode: $(stats.mode)")
println("Mean: $(stats.param_stats.mean)")
println("Covariance: $(stats.param_stats.cov)")

@assert vec(mean(samples.params, FrequencyWeights(samples.weight))) ≈ stats.param_stats.mean
@assert vec(var(samples.params, FrequencyWeights(samples.weight))) ≈ diag(stats.param_stats.cov)
@assert cov(samples.params, FrequencyWeights(samples.weight)) ≈ stats.param_stats.cov

vec(cor(samples.params, FrequencyWeights(samples.weight)))

plot(
    samples, 3,
    mean = true, std_dev = true, globalmode = true, localmode = true,
    nbins = 50, xlabel = par_names[3], ylabel = "P($(par_names[3]))",
    title = "Marginalized Distribution for mu_1"
)
savefig("tutorial-single-par.pdf")

plot(
    samples, (3, 5),
    mean = true, std_dev = true, globalmode = true, localmode = true,
    nbins = 50, xlabel = par_names[3], ylabel = par_names[5],
    title = "Marginalized Distribution for mu_1 and sigma"
)
plot!(stats, (3, 5))
savefig("tutorial-param-pair.pdf")

plot(
    samples,
    mean = false, std_dev = false, globalmode = true, localmode = false,
    nbins = 50
)
savefig("tutorial-all-params.pdf")

using TypedTables

tbl = Table(samples, sampleids)

mode_log_posterior, mode_idx = findmax(tbl.log_posterior)

fit_par_values = tbl[mode_idx].params

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data, True Model and Best Fit"
)
plot!(-4:0.01:4, x -> fit_function(x, true_par_values), label = "Truth")
plot!(-4:0.01:4, x -> fit_function(x, fit_par_values), label = "Best fit")
savefig("tutorial-data-truth-bestfit.pdf")


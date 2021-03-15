using Random, LinearAlgebra, Statistics, Distributions, StatsBase

data = vcat(
    rand(Normal(-1.0, 0.5), 500),
    rand(Normal( 2.0, 0.5), 1000)
)

typeof(data) == Vector{Float64}

hist = append!(Histogram(-2:0.1:4), data)

using Plots

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data"
)
savefig("tutorial-data.pdf")

function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
    p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
    p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
end

true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data and True Statistical Model"
)
plot!(
    -4:0.01:4, x -> fit_function(true_par_values, x),
    label = "Truth"
)
savefig("tutorial-data-and-truth.pdf")

using BAT, IntervalSets

likelihood = let h = hist, f = fit_function
    params -> begin
        # Histogram counts for each bin as an array:
        counts = h.weights

        # Histogram binning, has length (length(counts) + 1):
        binning = h.edges[1]

        # sum log-likelihood value over bins:
        ll_value::Float64 = 0.0
        for i in eachindex(counts)
            # Get information about current bin:
            bin_left, bin_right = binning[i], binning[i+1]
            bin_width = bin_right - bin_left
            bin_center = (bin_right + bin_left) / 2

            observed_counts = counts[i]

            # Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_width * f(params, bin_center)

            # Add log of Poisson probability for bin:
            ll_value += logpdf(Poisson(expected_counts), observed_counts)
        end

        return LogDVal(ll_value)
    end
end

likelihood(true_par_values)

using ValueShapes

prior = NamedTupleDist(
    a = [0.0..10.0^4, 0.0..10.0^4],
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Truncated(Normal(0.4, 2), 0.3, 0.7)
)

parshapes = varshape(prior)

posterior = PosteriorDensity(likelihood, prior)

ENV["JULIA_DEBUG"] = "BAT"

nsamples = 10^4
nchains = 4

samples = bat_sample(posterior, (nsamples, nchains), MetropolisHastings()).result

println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

unshaped.(samples).v

parshapes = varshape(posterior)

par_cov = cov(unshaped.(samples))
println("Covariance: $par_cov")

par_cov[parshapes.mu, parshapes.sigma]

par_names = ["a_1", "a_2", "mu_1", "mu_2", "sigma"]
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
plot!(BAT.MCMCBasicStats(samples), (3, 5))
savefig("tutorial-param-pair.png")

plot(
    samples,
    mean = false, std_dev = false, globalmode = true, localmode = false,
    nbins = 50
)
savefig("tutorial-all-params.png")

using TypedTables

tbl = Table(samples)

samples_mode = mode(samples)

samples_mode[] isa NamedTuple

unshaped(samples_mode)

findmode_result = bat_findmode(posterior, initial_mode = samples_mode)

fit_par_values = findmode_result.result[]

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data, True Model and Best Fit"
)
plot!(-4:0.01:4, x -> fit_function(true_par_values, x), label = "Truth")
plot!(-4:0.01:4, x -> fit_function(fit_par_values, x), label = "Best fit")
savefig("tutorial-data-truth-bestfit.pdf")

algorithm = MetropolisHastings()

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.15..0.35,
    β = 1.5,
    c = 1e-4..1e2
)

convergence = BrooksGelmanConvergence(
    threshold = 1.1,
    corrected = false
)

init = MCMCInitStrategy(
    init_tries_per_chain = 8..128,
    max_nsamples_init = 25,
    max_nsteps_init = 250,
    max_time_init = Inf
)

burnin = MCMCBurninStrategy(
    max_nsamples_per_cycle = 1000,
    max_nsteps_per_cycle = 10000,
    max_time_per_cycle = Inf,
    max_ncycles = 30
)

samples = bat_sample(
    posterior, (nsamples, nchains), algorithm,
    max_nsteps = 10 * nsamples,
    max_time = Inf,
    tuning = tuning,
    init = init,
    burnin = burnin,
    convergence = convergence,
    strict = false,
    filter = true
).result


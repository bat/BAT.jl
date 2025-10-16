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
    # Histogram counts for each bin as an array:
    observed_counts = h.weights

    # Histogram binning:
    bin_edges = h.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    params -> begin
        # Log-likelihood for a single bin:
        function bin_log_likelihood(i)
            # Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_widths[i] * f(params, bin_centers[i])
            logpdf(Poisson(expected_counts), observed_counts[i])
        end

        # Sum log-likelihood over bins:
        idxs = eachindex(observed_counts)
        ll_value = bin_log_likelihood(idxs[1])
        for i in idxs[2:end]
            ll_value += bin_log_likelihood(i)
        end

        # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
        return LogDVal(ll_value)
    end
end

likelihood(true_par_values)

using ValueShapes

prior = NamedTupleDist(
    a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Weibull(1.2, 2)
)

parshapes = varshape(prior)

posterior = PosteriorDensity(likelihood, prior)

ENV["JULIA_DEBUG"] = "BAT"

samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 4)).result

SampledDensity(posterior, samples)

println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

unshaped.(samples).v

parshapes = varshape(posterior)

par_cov = cov(unshaped.(samples))
println("Covariance: $par_cov")

par_cov[parshapes.mu, parshapes.sigma]

plot(
    samples, :(mu[1]),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1]"
)
savefig("tutorial-single-par.pdf")

plot(
    samples, (:(mu[1]), :sigma),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1] and sigma"
)
plot!(BAT.MCMCBasicStats(samples), (3, 5))
savefig("tutorial-param-pair.png")

plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 50
)
savefig("tutorial-all-params.png")

using TypedTables

tbl = Table(samples)

samples_mode = mode(samples)

samples_mode[] isa NamedTuple

unshaped(samples_mode)

findmode_result = bat_findmode(posterior, MaxDensityNelderMead(init = ExplicitInit([samples_mode])))

fit_par_values = findmode_result.result[]

plot(-4:0.01:4, fit_function, samples)

plot!(
    normalize(hist, mode=:density),
    color=1, linewidth=2, fillalpha=0.0,
    st = :steps, fill=false, label = "Data",
    title = "Data, True Model and Best Fit"
)

plot!(-4:0.01:4, x -> fit_function(true_par_values, x), color=4, label = "Truth")
savefig("tutorial-data-truth-bestfit.pdf")

mcmcalgo = MetropolisHastings(
    weighting = RepetitionWeighting(),
    tuning = AdaptiveMHTuning()
)

using Random123
rng = Philox4x()

init = MCMCChainPoolInit()

burnin = MCMCMultiCycleBurnin()

convergence = BrooksGelmanConvergence()

samples = bat_sample(
    rng, posterior,
    MCMCSampling(
        mcalg = mcmcalgo,
        nchains = 4,
        nsteps = 10^5,
        init = init,
        burnin = burnin,
        convergence = convergence,
        strict = true,
        store_burnin = false,
        nonzero_weights = true,
        callback = (x...) -> nothing
    )
).result


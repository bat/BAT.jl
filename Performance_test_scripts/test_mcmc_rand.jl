#=
# Run with:
include("test_mcmc_rand.jl"); show_scatter()
=#

using Distributions
using PDMats
using StatsBase
using IntervalSets
using Base.Test
using BAT

using BAT.Logging
set_log_level!(BAT, LOG_TRACE)
@enable_logging

exec_context = ExecContext()

#algorithm = MetropolisHastings()
#algorithm = MetropolisHastings(MHAccRejProbWeights{Float64}())
#algorithm = MetropolisHastings(MHPosteriorFractionWeights{Float64}())

algorithm = GeneralizedMetropolisHastings(MvTDistProposalSpec(1.0), 10, true, 2)



tdist = MvNormal(PDMat([1.0 1.5; 1.5 4.0]))
density = MvDistDensity(tdist)
#density = MvDistDensity(MvNormal([0., 0.], [1. 0.; 0. 1.]))


#bounds = HyperRectBounds([-2, -4], [2, 4], reflective_bounds)
#bounds = HyperRectBounds([-2, -4], [2, 4], hard_bounds)
bounds = HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
#bounds = HyperRectBounds([-5, -8], [5, 8], hard_bounds)
#bounds = HyperRectBounds([-500, -800], [500, 800], reflective_bounds)


λ = 0.5
α = BAT.ClosedInterval(0.75, 0.99)
β = 1.5
c = BAT.ClosedInterval(1e-4, 1e2)

# tuner_config = ProposalCovTunerConfig(α = 0.15..0.5)
#tuner_config = AbstractMCMCTunerConfig(algorithm)
tuner_config = AbstractMCMCTunerConfig(algorithm, λ, α, β, c)
#tuner_config = NoOpTunerConfig()

convergence_test = GRConvergence()


chainspec = MCMCSpec(algorithm, density, bounds, AbstractRNGSeed())

nsamples = 10^8
max_nsteps = 2500
nchains = 4

samples_mh, sampleids_mh, stats_mh = @time @inferred rand(
    chainspec,
    nsamples,
    nchains,
    exec_context,
    tuner_config = tuner_config,
    convergence_test = convergence_test,
    max_nsteps = max_nsteps,
    max_time = Inf,
    burnin_strategy = MCMCBurninStrategy(
        max_ncycles = 30
    ),
    granularity = 2,
    ll = LOG_INFO
)

@assert(length(samples_mh) == length(sampleids_mh))

info("Generated $(count(x -> x > 0, samples_mh.weight)) samples, total weight = $(sum(samples_mh.weight)).")
# info("Samples weights are $(samples_mh.weight)")

cov_samples_mh = cov(samples_mh.params, FrequencyWeights(samples_mh.weight), 2; corrected=true)
# cov_samples_mh = cov(samples_mh.params, FrequencyWeights(ones(eltype(samples_mh.weight), size(samples_mh.weight))), 2; corrected=true)
info("Samples parameter covariance: $cov_samples_mh")

@assert samples_mh.params[:, findmax(samples_mh.log_value)[2]] == stats_mh.mode
info("Stats parameter covariance: $(stats_mh.param_stats.cov)")

samples_ref, sampleids_ref, stats_ref = @time Base.rand(
    MCMCSpec(DirectSampling(), density, bounds),
    round(Int, sum(samples_mh.weight) / nchains),
    nchains,
    exec_context,
    granularity = 1
)

info("Generated $(count(x -> x > 0, samples_ref.weight)) reference samples.")

using Plots
pyplot(size = (1024,768), foreground_color_grid = "gray80")
#gr(size = (1024,768), foreground_color_grid = "gray80")

function show_scatter()
    plot(bounds, (1, 2))
    plot!(samples_ref, (1, 2), color = :blue, label = "reference")
    plot!(samples_mh, (1, 2))
    plot!(stats_mh, (1, 2))
end

function show_hists()
    plot(
        begin
            plot(bounds, (1, 2))
            plot!(samples_mh, (1, 2), seriestype = :histogram2d, normalize = true, nbins = 200)
            plot!(stats_mh, (1, 2))
        end,
        begin
            plot(bounds, (1, 2))
            plot!(samples_ref, (1, 2), label = "reference", seriestype = :histogram2d, normalize = true, nbins = 200)
            plot!(stats_ref, (1, 2))
        end
    )
end


#=
samples_mh = rand(MCMCSpec(MetropolisHastings(), density), 500, 4);
=#

#=

stephist(samples_mh.weight, yscale = :log10, label="weight", title = "Sample weight distribution", xlabel = "weight", ylabel = "n", size = (1024, 1024), bins = -0.5:1:20.5, markersize = 1)
stephist(samples_mh.weight, yscale = :log10, label="weight", title = "Sample weight distribution", xlabel = "weight", ylabel = "n", size = (1024, 1024), bins = 0:0.1:10, markersize = 1)

plot(fit(Histogram, samples_mh.params[1, :], Weights(samples_mh.weight), closed = :left, nbins = 50), st = :step, markersize = 1)

=#

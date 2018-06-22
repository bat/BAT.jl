#=
# Run with:
include("test_mcmc_mh_detailed.jl"); show_scatter()
=#

using Distributions, PDMats, StatsBase, IntervalSets
using Base.Test
using BAT

using BAT.Logging
set_log_level!(BAT, LOG_TRACE)
@enable_logging

exec_context = ExecContext()

algorithm = MetropolisHastings(MHAccRejProbWeights{Float64}())
#algorithm = GeneralizedMetropolisHastings(MvTDistProposalSpec(), 10, true, 1)

tdist = MvNormal(PDMat([1.0 1.5; 1.5 4.0]))
density = MvDistDensity(tdist)

bounds = HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
#bounds = HyperRectBounds([-5, -8], [5, 8], hard_bounds)


chainspec = MCMCSpec(algorithm, density, bounds)

nsamples = 10^9
max_nsteps = 2500
nchains = 4
ll = LOG_TRACE

chain = chainspec(zero(Int64), exec_context)
# info(chain.state.samples.params)

smpls = DensitySampleVector(chain)
cb = MCMCAppendCallback(smpls, false)

mcmc_iterate!(
    cb,
    chain,
    exec_context,
    max_nsamples = 1000,
    max_nsteps = 1000,
    max_time = Inf,
    ll = ll
)


info("Number of samples: $(length(smpls))")
info("Sum of sample weights: $(sum(smpls.weight))")
info("Eff. nsamples by entropy: $(exp(BAT.mh_shannon_entropy(smpls.weight)))")
info("Eff. acceptance ratio: $(BAT.eff_acceptance_ratio(chain.state))")


function test_tuning()
    tuner_config = AbstractMCMCTunerConfig(algorithm)

    convergence_test = GRConvergence()
    init_strategy = MCMCInitStrategy(tuner_config)

    tuners = mcmc_init(
        chainspec,
        nchains,
        exec_context,
        tuner_config,
        convergence_test,
        init_strategy;
        ll = ll,
    )

    burnin_strategy = MCMCBurninStrategy(tuner_config)
    strict_mode = false

    mcmc_tune_burnin!(
        (),
        tuners,
        convergence_test,
        burnin_strategy,
        exec_context;
        strict_mode = strict_mode,
        ll = ll
    )

    granularity = 2

    chains = map(x -> x.chain, tuners)

    samples = DensitySampleVector.(chains)
    sampleids = MCMCSampleIDVector.(chains)
    stats = MCMCBasicStats.(chains)

    nonzero_weights = granularity <= 1
    callbacks = [
        BAT.MCMCMultiCallback(
            MCMCAppendCallback(samples[i], nonzero_weights),
            MCMCAppendCallback(sampleids[i], nonzero_weights),
            MCMCAppendCallback(stats[i], nonzero_weights)
        ) for i in eachindex(chains)
    ]

    max_time = Inf

    mcmc_iterate!(
        callbacks,
        chains,
        exec_context;
        max_nsamples = Int64(nsamples),
        max_nsteps = max_nsteps,
        max_time = max_time,
        ll = ll
    )
end

test_tuning()


using Plots
pyplot(size = (1024,768), foreground_color_grid = "gray80")

function show_scatter()
    plot(smpls[find(x -> x > 0, smpls.weight)], (1, 2))
end


#=

using Plots
pyplot(size = (1024,768), foreground_color_grid = "gray80")

stephist(smpls.weight, bins = 0:0.01:4)

# plot(smpls, (1, 2))
plot(smpls[find(x -> x > 0, smpls.weight)], (1, 2))
=#

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
using JLD

using BAT.Logging


α_vec = collect(0.1:0.1:1.0)
m_vec = collect(2:4:20)
df_vec = collect(1.0:10.0:1.0)
iter = collect(1:1:1)

#α_vec = collect(1:1:3)
#m_vec = collect(1:5:16)
#df_vec = collect(1.0:10.0:11.0)
#ter = collect(1:1:3)

data_mean = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), size(iter, 1), 2)
data_cov = zeros(Float64, size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), size(iter, 1), 2, 2)
data_tuned = Array{Bool}(size(α_vec, 1), size(m_vec, 1), size(df_vec, 1), size(iter, 1), 1)
data_tuned[:,:,:,:] = true

set_log_level!(BAT, LOG_TRACE)
@enable_logging

exec_context = ExecContext()

#algorithm = MetropolisHastings()
#algorithm = MetropolisHastings(MHAccRejProbWeights{Float64}())
#algorithm = MetropolisHastings(MHPosteriorFractionWeights{Float64}())

#.......................................................algorithm = GeneralizedMetropolisHastings(MvTDistProposalSpec(1.0), 100, true, 2)

#....................................................lol = view(b,:,1,1)


tdist = MvNormal(PDMat([1.0 1.5; 1.5 4.0]))
density = MvDistDensity(tdist)


#bounds = HyperRectBounds([-2, -4], [2, 4], reflective_bounds)
#bounds = HyperRectBounds([-2, -4], [2, 4], hard_bounds)
bounds = HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
#bounds = HyperRectBounds([-5, -8], [5, 8], hard_bounds)
#bounds = HyperRectBounds([-500, -800], [500, 800], reflective_bounds)


λ = 0.5
#...................................α = BAT.ClosedInterval(0.75, 0.99)
β = 3.0
c = BAT.ClosedInterval(1e-4, 1e2)

# tuner_config = ProposalCovTunerConfig(α = 0.15..0.5)
#tuner_config = AbstractMCMCTunerConfig(algorithm)
#......................tuner_config = AbstractMCMCTunerConfig(algorithm, λ, α, β, c)
#tuner_config = NoOpTunerConfig()

convergence_test = GRConvergence()


#.........................................................chainspec = MCMCSpec(algorithm, density, bounds, AbstractRNGSeed())

nsamples = 10^9
max_nsteps = 2500
nchains = 4

for i in indices(α_vec, 1)
    for m in indices(m_vec, 1)
        for k in indices(df_vec, 1)
            for w in iter

                println("------------------------alpha is $(α_vec[i]), m is $(m_vec[m]), df is $(df_vec[k]) and the iteration is $w")
                algorithm = GeneralizedMetropolisHastings(MvTDistProposalSpec(df_vec[k]), m_vec[m], true, 2)

                lol = view(data_tuned, i, m, k, w, :)

                α = BAT.ClosedInterval(α_vec[i] - 0.1, α_vec[i])

                tuner_config = AbstractMCMCTunerConfig(algorithm, λ, α, β, c)

                chainspec = MCMCSpec(algorithm, density, bounds, AbstractRNGSeed())

                #samples_mh, sampleids_mh, stats_mh = 0
                samples_mh, sampleids_mh, stats_mh = @time @inferred rand(
                    chainspec,
                    nsamples,
                    nchains,
                    lol,
                    exec_context,
                    tuner_config = tuner_config,
                    convergence_test = convergence_test,
                    max_nsteps = max_nsteps,
                    max_time = Inf,
                    burnin_strategy = MCMCBurninStrategy(
                        max_ncycles = 10
                    ),
                    granularity = 2,
                    ll = LOG_INFO
                )

                @assert(length(samples_mh) == length(sampleids_mh))

                info("Generated $(count(x -> x > 0, samples_mh.weight)) samples, total weight = $(sum(samples_mh.weight)).")
                # info("Samples weights are $(samples_mh.weight)")

                cov_samples_mh = cov(samples_mh.params, FrequencyWeights(samples_mh.weight), 2; corrected=true)
                mean_samples_mh = mean(Array(samples_mh.params), FrequencyWeights(samples_mh.weight), 2)

                data_mean[i, m, k, w, :] = mean_samples_mh[:]
                data_cov[i, m, k, w, :, :] = cov_samples_mh[:, :]

                info("Samples parameter mean: $mean_samples_mh")
                # cov_samples_mh = cov(samples_mh.params, FrequencyWeights(ones(eltype(samples_mh.weight), size(samples_mh.weight))), 2; corrected=true)
                info("Samples parameter covariance: $cov_samples_mh")

                @assert samples_mh.params[:, findmax(samples_mh.log_value)[2]] == stats_mh.mode
                info("Stats parameter covariance: $(stats_mh.param_stats.cov)")
                #println("!!!!!!!!!!!!!!!!!!!!!!!!!!   THE SIZE OF THE ARRAY IS $(size(samples_mh.weight, 1))")
            end
        end
    end
end

save("data_mean.jld", "data_mean", data_mean)
save("data_cov.jld", "data_cov", data_cov)
save("data_tuned.jld", "data_tuned", data_tuned)

#coco = load("data.jld")["data"]

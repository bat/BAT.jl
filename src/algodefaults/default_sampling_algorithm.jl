# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_sample), ::Val{:rng}, ::Any) = bat_rng()


bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::AnyIIDSampleable) = IIDSampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleVector) = OrderedResampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::AbstractDensity) = MCMCSampling(
    algorithm = MetropolisHastings(),
    init = MCMCChainPoolInit(),
    burnin = MCMCMultiCycleBurnin(
        max_nsamples_per_cycle = max(div(max_nsamples, 10), 100),
        max_nsteps_per_cycle = max(div(max_nsteps, 10), 100)
    ),
    convergence = BrooksGelmanConvergence(),
    strict = false
)


#=
For HamiltonianMC

#!!!!!!!!!!!!!!!! N samples steps evals

# MCMCBurninStrategy for HamiltonianMC
function MCMCBurninStrategy(algorithm::HamiltonianMC, nsamples::Integer, max_nsteps::Integer, tuner_config::MCMCTuningAlgorithm)
    max_nsamples_per_cycle = nsamples
    max_nsteps_per_cycle = max_nsteps
    MCMCBurninStrategy(
        max_nsamples_per_cycle = max_nsamples_per_cycle,
        max_nsteps_per_cycle = max_nsteps_per_cycle,
        max_ncycles = 1
    )
end
=#

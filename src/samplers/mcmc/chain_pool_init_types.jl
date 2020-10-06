# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCChainPoolInit <: MCMCInitAlgorithm

 MCMC chain pool initialization strategy.

Fields:
* `init_tries_per_chain`: Interval that specifies the minimum and maximum
  number of tries per MCMC chain to find a suitable starting position. Many
  candidate chains will be created and run for a short time. The chains with
  the best performance will be selected for tuning/burn-in and MCMC sampling
  run. Defaults to `IntervalSets.ClosedInterval(8, 128)`.
* `max_nsamples_init`: Maximum number of MCMC samples for each candidate
  chain. Defaults to 25. Definition of a sample depends on sampling algorithm.
* `max_nsteps_init`: Maximum number of MCMC steps for each candidate chain.
  Defaults to 250. Definition of a step depends on sampling algorithm.
* `max_time_init::Int`: Maximum wall-clock time to spend per candidate chain,
  in seconds. Defaults to `Inf`.
"""
@with_kw struct MCMCChainPoolInit <: MCMCInitAlgorithm
    init_tries_per_chain::ClosedInterval{Int64} = ClosedInterval(8, 128)
    max_nsamples_init::Int64 = 25
    max_nsteps_init::Int64 = 250
    max_time_init::Float64 = Inf
end

export MCMCChainPoolInit

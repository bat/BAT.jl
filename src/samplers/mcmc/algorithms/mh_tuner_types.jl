# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to AdaptiveMHTuning docstring.

"""
    AdaptiveMHTuning(...) <: MHProposalDistTuning

Adaptive MCMC tuning strategy for Metropolis-Hastings samplers.

Adapts the proposal function based on the acceptance ratio and covariance
of the previous samples.

Fields:

* `λ`: Controls the weight given to new covariance information in adapting
  the proposal distribution. Defaults to `0.5`.

* `α`: Metropolis-Hastings acceptance ratio target, tuning will try to
  adapt the proposal distribution to bring the acceptance ratio inside this
  interval. Defaults to `IntervalSets.ClosedInterval(0.15, 0.35)`

* `β`: Controls how much the spread of the proposal distribution is
  widened/narrowed depending on the current MH acceptance ratio.

* `c`: Interval for allowed scale/spread of the proposal distribution.
  Defaults to `ClosedInterval(1e-4, 1e2)`.

* `r`: Reweighting factor. Take accumulated sample statistics of previous
  tuning cycles into account with a relative weight of `r`. Set to `0` to
  completely reset sample statistics between each tuning cycle.

Constructors:

```julia
AdaptiveMHTuning(
    λ::Real,
    α::IntervalSets.ClosedInterval{<:Real},
    β::Real,
    c::IntervalSets.ClosedInterval{<:Real},
    r::Real
)
```
"""
@with_kw struct AdaptiveMHTuning <: MHProposalDistTuning
    λ::Float64 = 0.5
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)
    β::Float64 = 1.5
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)
    r::Real = 0.5
end

export AdaptiveMHTuning


(config::AdaptiveMHTuning)(chain::MHIterator) = ProposalCovTuner(config, chain)

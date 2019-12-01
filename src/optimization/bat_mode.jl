# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.AbstractModeEstimator

Abstract type for BAT optimization algorithms.

A typical application for optimization in BAT is mode estimation
(see [`bat_mode`](@ref)),
"""
abstract type AbstractModeEstimator end


"""
    BAT.default_mode_estimator(posterior)

Get BAT's default optimization algorithm for `posterior`.
"""
function default_mode_estimator end

default_mode_estimator(posterior::Distribution) = ModeAsDefined()
default_mode_estimator(posterior::DistLikeDensity) = ModeAsDefined()
default_mode_estimator(posterior::DensitySampleVector) = MaxDensitySampleSearch()
default_mode_estimator(posterior::AbstractDensity) = MaxDensityNelderMead()


"""
    bat_mode(
        posterior::BAT.AnyPosterior,
        [algorithm::BAT.AbstractModeEstimator]
    )::DensitySampleVector

Estiate the global mode of `posterior`.

Returns a NamedTuple of the shape

```julia
(mode = X::DensitySampleVector, ...)
```

Properties others than `mode` are algorithm-specific, they are also by default
not part of the stable BAT API.
"""
function bat_mode end
export bat_mode

@inline function bat_mode(posterior::AnyPosterior; kwargs...)
    algorithm = default_mode_estimator(posterior)
    bat_mode(posterior, algorithm; kwargs...)
end



"""
    ModeAsDefined <: AbstractModeEstimator

Constructors:

    ModeAsDefined()

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.
"""
struct ModeAsDefined <: AbstractModeEstimator end
export ModeAsDefined


function bat_mode(posterior::AnyPosterior, algorithm::ModeAsDefined)
    (mode = StatsBase.mode(posterior),)
end

function bat_mode(posterior::DistributionDensity, algorithm::ModeAsDefined)
    (mode = StatsBase.mode(posterior.dist),)
end



"""
    MaxDensitySampleSearch <: AbstractModeEstimator

Constructors:

    MaxDensitySampleSearch()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct MaxDensitySampleSearch <: AbstractModeEstimator end
export MaxDensitySampleSearch


function bat_mode(posterior::DensitySampleVector, algorithm::MaxDensitySampleSearch)
    v, i = _get_mode(posterior)
    (mode = v, mode_idx = i)
end



"""
    MaxDensityNelderMead <: AbstractModeEstimator

Constructors:

    MaxDensityNelderMead()

Estimate the mode of the posterior using Nelder-Mead optimization (via
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)).
"""
struct MaxDensityNelderMead <: AbstractModeEstimator end
export MaxDensityNelderMead

function bat_mode(posterior::AnyPosterior, algorithm::MaxDensityNelderMead; initial_mode = missing)
    x = _get_initial_mode(posterior, initial_mode)
    conv_posterior = convert(AbstractDensity, posterior)
    r = Optim.maximize(p -> density_logval(conv_posterior, p), x, Optim.NelderMead())
    (mode = Optim.minimizer(r.res), info = r)
end


_get_initial_mode(posterior::AnyPosterior, ::Missing) =
    _get_initial_mode(posterior, rand(getprior(posterior)))

_get_initial_mode(posterior::AnyPosterior, samples::DensitySampleVector) =
    _get_initial_mode(posterior, bat_mode(samples).mode)

_get_initial_mode(posterior::AnyPosterior, x::AbstractArray{<:Real}) = Array(x)
_get_initial_mode(posterior::AnyPosterior, x::Array{<:Real}) = x

_get_initial_mode(posterior::AnyPosterior, x) = x



"""
    MaxDensityLBFGS <: AbstractModeEstimator

Constructors:

    MaxDensityLBFGS()

Estimate the mode of the posterior using LBFGS optimization (via
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)). The gradient
of the posterior is computer by forward-mode auto-differentiation (via
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)).
"""
struct MaxDensityLBFGS <: AbstractModeEstimator end
export MaxDensityLBFGS


function bat_mode(posterior::AnyPosterior, algorithm::MaxDensityLBFGS; initial_mode = missing)
    x = _get_initial_mode(posterior, initial_mode)
    conv_posterior = convert(AbstractDensity, posterior)
    r = Optim.maximize(p -> density_logval(conv_posterior, p), x, Optim.LBFGS(); autodiff = :forward)
    (mode = Optim.minimizer(r.res), info = r)
end

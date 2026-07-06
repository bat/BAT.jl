# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type BAT.AbstractModeEstimator

Abstract type for BAT optimization algorithms.

A typical application for optimization in BAT is mode estimation
(see [`bat_findmode`](@ref)),
"""
abstract type AbstractModeEstimator end


"""
    bat_findmode(
        target::BAT.MeasureLike,
        [algorithm::BAT.AbstractModeEstimator],
        [context::BATContext]
    )

Estimate the global mode of `target`.

Returns a NamedTuple of the shape

```julia
(result = v, evaluated::EvaluatedMeasure, ...)
```

with `v` the estimated mode variate.


Result properties not listed here are algorithm-specific and are not part
of the stable public API.

# Implementation

`bat_findmode` uses [`evalmeasure`](@ref) internally. Do not specialize
`bat_findmode`.
"""
function bat_findmode end
export bat_findmode


function bat_findmode(target::MeasureLike, algorithm, context::BATContext)
    orig_context = deepcopy(context)

    em = evalmeasure(target, algorithm, context)
    r = (;result = mode(em), evaluated = em, _evalresult_nt(em)...)

    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_findmode(target::MeasureLike) = bat_findmode(target, get_batcontext())

bat_findmode(target::MeasureLike, algorithm) = bat_findmode(target, algorithm, get_batcontext())

function bat_findmode(target::MeasureLike, context::BATContext)
    algorithm = bat_default_withdebug(bat_findmode, Val(:algorithm), target);
    bat_findmode(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_findmode), ::Val{:algorithm}, x::AbstractModeEstimator)
    "Using mode estimator $x"
end


"""
    bat_bgml(
        likelihood, prior,
        [algorithm::BAT.MaxDensityAlgorithm],
        [context::BATContext]
    )

Estimate the maximum (log-)likelihood parameter point using Bayesian-Guided
Maximum Likelihood (BGML).

BGML runs the optimization `algorithm` in the transformed space where
the transformation is derived from `prior` and `algorithm`. Typically
it is the space in which `prior` becomes a standard multivariate normal
(or other standard) distribution. The optimization target is purely
`logdensityof(likelihood)`. The given `prior` only informs the choice of
parameter space. As a likelihood is invariant under reparameterization, the
result is not biased by the choice of `prior`, provided that `prior` does
not vanish in valid parameter regions of non-negligible likelihood and that
the optimizer finds the global maximum of the likelihood within the search
space. The numerical result may still depend on `prior` through the
parameterization of the search space and the starting values.

Returns a NamedTuple of the shape

```julia
(result = v, ...)
```

!!! note

    Do not add methods to `bat_bgml`, add methods to
    `bat_bgml_impl` instead.
"""
function bat_bgml end
export bat_bgml

function bat_bgml_impl end

function bat_bgml(likelihood, prior, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_bgml_impl(likelihood, prior, algorithm, context)
    # The result is a likelihood maximizer, not a mode of the posterior
    # measure, so it must not be registered as one (no Val(:mode) here):
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_bgml(likelihood, prior) = bat_bgml(likelihood, prior, get_batcontext())

bat_bgml(likelihood, prior, algorithm) = bat_bgml(likelihood, prior, algorithm, get_batcontext())

function bat_bgml(likelihood, prior, context::BATContext)
    algorithm = bat_default_withdebug(bat_bgml, Val(:algorithm), likelihood, prior);
    bat_bgml(likelihood, prior, algorithm, context)
end


function argchoice_msg(::typeof(bat_bgml), ::Val{:algorithm}, x::AbstractModeEstimator)
    "Using mode estimator $x for Bayesian-Guided Maximum Likelihood (BGML) estimation"
end


"""
    bat_marginalmode(
        target::DensitySampleVector,
        algorithm::AbstractModeEstimator,
        [context::BATContext]
    )

*Experimental feature, not part of stable public API.*

Estimates a marginal mode of `target` by finding the maximum of marginalized posterior for each dimension.

Returns a NamedTuple of the shape

```julia
(result = v, ...)
```

!!! note

    Do not add add methods to `bat_marginalmode`, add methods to
    `bat_marginalmode_impl` instead.
"""
function bat_marginalmode end
export bat_marginalmode

function bat_marginalmode_impl end


function bat_marginalmode(measure::MeasureLike, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_marginalmode_impl(measure, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_marginalmode(measure::MeasureLike) = bat_marginalmode(measure, get_batcontext())

bat_marginalmode(measure::MeasureLike, algorithm) = bat_marginalmode(measure, algorithm, get_batcontext())

function bat_marginalmode(measure::MeasureLike, context::BATContext)
    algorithm = bat_default_withdebug(bat_marginalmode, Val(:algorithm), measure);
    bat_marginalmode(measure, algorithm, context)
end

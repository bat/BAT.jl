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
        target::BAT.AnySampleable,
        [algorithm::BAT.AbstractModeEstimator],
        [context::BATContext]
    )::DensitySampleVector

Estimate the global mode of `target`.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_findmode`, add methods to
    `bat_findmode_impl` instead.
"""
function bat_findmode end
export bat_findmode

function bat_findmode_impl end


function bat_findmode(target::AnySampleable, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_findmode_impl(target, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_findmode(target::AnySampleable) = bat_findmode(target, default_context())

bat_findmode(target::AnySampleable, algorithm) = bat_findmode(target, algorithm, default_context())

function bat_findmode(target::AnySampleable, context::BATContext)
    algorithm = bat_default_withdebug(bat_findmode, Val(:algorithm), target, context);
    bat_findmode(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_findmode), ::Val{:algorithm}, x::AbstractModeEstimator)
    "Using mode estimator $x"
end



"""
    bat_marginalmode(
        samples::DensitySampleVector;
        nbins::Union{Integer, Symbol} = 200
    )::DensitySampleVector

*Experimental feature, not part of stable public API.*

Estimates a marginal mode of `samples` by finding the maximum of marginalized posterior for each dimension.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

`nbins` specifies the number of bins that are used for marginalization. The default value is `nbins=200`. The optimal number of bins can be estimated using  the following keywords:

* `:sqrt`  — Square-root choice

* `:sturges` — Sturges' formula

* `:rice` — Rice Rule

* `:scott` — Scott's normal reference rule

* `:fd` —  Freedman–Diaconis rule

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_marginalmode`, add methods to
    `bat_marginalmode_impl` instead.
"""
function bat_marginalmode end
export bat_marginalmode

function bat_marginalmode_impl end


function bat_marginalmode(samples::DensitySampleVector)
    r = bat_marginalmode_impl(samples::DensitySampleVector)
    result_with_args(r, NamedTuple())
end

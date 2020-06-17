# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BAT.AbstractModeEstimator

Abstract type for BAT optimization algorithms.

A typical application for optimization in BAT is mode estimation
(see [`bat_findmode`](@ref)),
"""
abstract type AbstractModeEstimator end


"""
    bat_findmode(
        posterior::BAT.AnyPosterior,
        [algorithm::BAT.AbstractModeEstimator];
        initial_mode::Union{Missing,DensitySampleVector,Any} = missing
    )::DensitySampleVector

Estiate the global mode of `posterior`.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

Properties others than `mode` are algorithm-specific, they are also by default
not part of the stable BAT API.
"""
function bat_findmode end
export bat_findmode


function bat_findmode(posterior::AnyPosterior; kwargs...)
    algorithm = bat_default_withdebug(bat_findmode, Val(:algorithm), posterior)
    r = bat_findmode(posterior, algorithm; kwargs...)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_findmode), ::Val{:algorithm}, x::AbstractModeEstimator)
    "Using mode estimator $x"
end



"""
    bat_marginalmode(
        samples::DensitySampleVector;
        nbins::Union{Integer, Symbol} = 200
    )::DensitySampleVector

Estimates a local mode of `samples` by finding the maximum of marginalized posterior for each dimension.

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

"""
function bat_marginalmode end
export bat_marginalmode

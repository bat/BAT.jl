# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AutocorLenAlgorithm

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type AutocorLenAlgorithm end
export AutocorLenAlgorithm



"""
    bat_integrated_autocorr_len(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        algorithm::AutocorLenAlgorithm = GeyerAutocorLen()
    )

Estimate the integrated autocorrelation length of variate series `v`,
separately for each degree of freedom.

Returns a NamedTuple of the shape

```julia
(result = integrated_autocorr_len, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_integrated_autocorr_len`, add methods to
    `bat_integrated_autocorr_len_impl` instead.
"""
function bat_integrated_autocorr_len end
export bat_integrated_autocorr_len

function bat_integrated_autocorr_len_impl end


function bat_integrated_autocorr_len(
    v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
    algorithm = bat_default_withdebug(bat_integrated_autocorr_len, Val(:algorithm), v)
)
    r = bat_integrated_autocorr_len_impl(v, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_integrated_autocorr_len), ::Val{:algorithm}, x::AutocorLenAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

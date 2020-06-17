# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AutocorLenAlgorithm

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
of the stable BAT API.
"""
function bat_integrated_autocorr_len end
export bat_integrated_autocorr_len


function bat_integrated_autocorr_len(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}})
    algorithm = bat_default_withdebug(bat_integrated_autocorr_len, Val(:algorithm), v)
    r = bat_integrated_autocorr_len(v, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_integrated_autocorr_len), ::Val{:algorithm}, x::AutocorLenAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end



@doc doc"""
    bat_eff_sample_size(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        [algorithm::AutocorLenAlgorithm]
    )

    bat_eff_sample_size(
        smpls::DensitySampleVector,
        [algorithm::AutocorLenAlgorithm];
        use_weights=true
    )

Estimate effective sample size estimation for variate series `v`, resp.
density samples `smpls`, separately for each degree of freedom.

* `use_weights`: Take sample weights into account, using Kish's approximation

Returns a NamedTuple of the shape

```julia
(result = X::AbstractVector{<:Real}, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.
"""
function bat_eff_sample_size end
export bat_eff_sample_size

function bat_eff_sample_size(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}})
    algorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), v)
    r = bat_eff_sample_size(v, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function bat_eff_sample_size(smpls::DensitySampleVector; use_weights = true)
    algorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), smpls)
    r = bat_eff_sample_size(smpls, algorithm; use_weights = use_weights)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_eff_sample_size), ::Val{:algorithm}, x::AutocorLenAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

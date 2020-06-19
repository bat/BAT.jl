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

!!! note

    Do not add add methods to `bat_integrated_autocorr_len`, add methods to
    `bat_integrated_autocorr_len_impl` instead (same arguments).
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



@doc doc"""
    bat_eff_sample_size(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        [algorithm::AutocorLenAlgorithm]
    )

    bat_eff_sample_size(
        smpls::DensitySampleVector,
        [algorithm::AutocorLenAlgorithm]
    )

Estimate effective sample size estimation for variate series `v`, resp.
density samples `smpls`, separately for each degree of freedom.

Returns a NamedTuple of the shape

```julia
(result = X::AbstractVector{<:Real}, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.

!!! note

    Do not add add methods to `bat_eff_sample_size`, add methods to
    `bat_eff_sample_size_impl` instead (same arguments).
"""
function bat_eff_sample_size end
export bat_eff_sample_size

function bat_eff_sample_size_impl end


function bat_eff_sample_size(
    v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
    algorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), v)
)
    r = bat_eff_sample_size_impl(v, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function bat_eff_sample_size(
    smpls::DensitySampleVector,
    algorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), smpls);
    kwargs...
)
    r = bat_eff_sample_size_impl(smpls, algorithm; kwargs...)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_eff_sample_size), ::Val{:algorithm}, x::AutocorLenAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

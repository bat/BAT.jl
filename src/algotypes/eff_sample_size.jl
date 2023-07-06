# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type EffSampleSizeAlgorithm

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type EffSampleSizeAlgorithm end
export EffSampleSizeAlgorithm



"""
    bat_eff_sample_size(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        [algorithm::EffSampleSizeAlgorithm]
    )

    bat_eff_sample_size(
        smpls::DensitySampleVector,
        [algorithm::EffSampleSizeAlgorithm]
    )

Estimate effective sample size estimation for variate series `v`, resp.
density samples `smpls`, separately for each degree of freedom.

Returns a NamedTuple of the shape

```julia
(result = eff_sample_size, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_eff_sample_size`, add methods to
    `bat_eff_sample_size_impl` instead.
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
    algorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), smpls)
)
    r = bat_eff_sample_size_impl(smpls, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_eff_sample_size), ::Val{:algorithm}, x::EffSampleSizeAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

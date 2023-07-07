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
        [algorithm::EffSampleSizeAlgorithm],
        [context::BATContext]
    )

    bat_eff_sample_size(
        smpls::DensitySampleVector,
        [algorithm::EffSampleSizeAlgorithm],
        [context::BATContext]
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


const _ESSTarget = Union{
    AbstractVector{<:Real},
    AbstractVectorOfSimilarVectors{<:Real},
    DensitySampleVector
}

function bat_eff_sample_size(target::_ESSTarget, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_eff_sample_size_impl(target, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_eff_sample_size(target::_ESSTarget) = bat_eff_sample_size(target, get_batcontext())

bat_eff_sample_size(target::_ESSTarget, algorithm) = bat_eff_sample_size(target, algorithm, get_batcontext())

function bat_eff_sample_size(target::_ESSTarget, context::BATContext)
    algorithm = bat_default_withdebug(context, bat_eff_sample_size, Val(:algorithm), target)
    bat_eff_sample_size(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_eff_sample_size), ::Val{:algorithm}, x::EffSampleSizeAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

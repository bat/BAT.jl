# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type BAT.AbstractSamplingAlgorithm

Abstract type for BAT sampling algorithms. See [`bat_sample`](@ref).
"""
abstract type AbstractSamplingAlgorithm end


"""
    bat_sample(
        target::BAT.AnySampleable,
        [algorithm::BAT.AbstractSamplingAlgorithm],
        [context::BATContext]
    )::DensitySampleVector

Draw samples from `target` using `algorithm`.

Depending on sampling algorithm, the samples may be independent or correlated
(e.g. when using MCMC).

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_sample`, add methods to
    `bat_sample_impl` instead.
"""
function bat_sample end
export bat_sample

function bat_sample_impl end


function bat_sample(target::AnySampleable, algorithm::AbstractSamplingAlgorithm, context::BATContext; kwargs...)
    orig_context = deepcopy(context)
    r = bat_sample_impl(target, algorithm, context; kwargs...)
    result_with_args(r, (algorithm = algorithm, context = orig_context), kwargs)
end

function bat_sample(target::AnySampleable, context::BATContext; kwargs...)
    algorithm = bat_default_withinfo(bat_sample, Val(:algorithm), target)
    bat_sample(target, algorithm, context; kwargs...)
end

function bat_sample(target::AnySampleable; kwargs...)
    algorithm = bat_default_withinfo(bat_sample, Val(:algorithm), target)
    bat_sample(target, algorithm, default_context(); kwargs...)
end

function bat_sample(target::AnySampleable, algorithm::AbstractSamplingAlgorithm; kwargs...)
    bat_sample(target, algorithm, default_context(); kwargs...)
end


function argchoice_msg(::typeof(bat_sample), ::Val{:algorithm}, x::AbstractSamplingAlgorithm)
    "Using sampling algorithm $x"
end



"""
    abstract type AbstractSampleGenerator

*BAT-internal, not part of stable public API.*

Abstract super type for sample generators.
"""
abstract type AbstractSampleGenerator end
export AbstractSampleGenerator


function bat_report!(md::Markdown.MD, generator::AbstractSampleGenerator)
    alg = getalgorithm(generator)
    if !(isnothing(alg) || ismissing(alg))
        markdown_append!(md, """
        ### Sample generation:

        * Algorithm: $(nameof(typeof(alg)))
        """)
    end
end

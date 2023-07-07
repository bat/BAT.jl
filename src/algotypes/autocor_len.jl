# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AutocorLenAlgorithm

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type AutocorLenAlgorithm end
export AutocorLenAlgorithm



"""
    bat_integrated_autocorr_len(
        v::_ACLenTarget,
        algorithm::AutocorLenAlgorithm = GeyerAutocorLen(),
        [context::BATContext]
    )

*Experimental feature, not yet part of stable public API.*

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


const _ACLenTarget = Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}}

function bat_integrated_autocorr_len(v::_ACLenTarget, algorithm::AutocorLenAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_integrated_autocorr_len_impl(v, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_integrated_autocorr_len(v::_ACLenTarget) = bat_integrated_autocorr_len(v, get_batcontext())

function bat_integrated_autocorr_len(v::_ACLenTarget, algorithm::AutocorLenAlgorithm)
    bat_integrated_autocorr_len(v, algorithm, get_batcontext())
end

function bat_integrated_autocorr_len(v::_ACLenTarget, context::BATContext)
    algorithm = bat_default_withdebug(context, bat_integrated_autocorr_len, Val(:algorithm), v)
    bat_integrated_autocorr_len(v, algorithm, context)
end


function argchoice_msg(::typeof(bat_integrated_autocorr_len), ::Val{:algorithm}, x::AutocorLenAlgorithm)
    "Using integrated autocorrelation length estimator $x"
end

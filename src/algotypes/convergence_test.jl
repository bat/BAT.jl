# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type ConvergenceTest

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type ConvergenceTest end



"""
    bat_convergence(
        algoutput,
        [algorithm::ConvergenceTest],
        [context::BATContext]
    )

Check if an algorithm has converged, based on it's output `algoutput`

Returns a NamedTuple of the shape

```julia
(result, ...)
```

`result` indicates whether `algoutput` and must either be a `Bool` or
support `convert(Bool, result)`. It should typically contains measures
of algorithm convergence, like a convergence value and it's threshold, etc.

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_convergence`, add methods to
    `bat_convergence_impl` instead.
"""
function bat_convergence end
export bat_convergence

function bat_convergence_impl end


function bat_convergence(result::Any, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_convergence_impl(result, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_convergence(result::Any) = bat_convergence(result, get_batcontext())

bat_convergence(result::Any, algorithm) = bat_convergence(result, algorithm, get_batcontext())

function bat_convergence(result::Any, context::BATContext)
    algorithm = bat_default_withdebug(bat_convergence, Val(:algorithm), result)
    bat_convergence(result, algorithm, context)
end


function argchoice_msg(::typeof(bat_convergence), ::Val{:algorithm}, algorithm::ConvergenceTest)
    "Using convergence algorithm $algorithm"
end



"""
struct AssumeConvergence <: ConvergenceTest

No-op convergence algorithm for [`bat_convergence`](@ref), will always declare convergence.

Constructors:

* ```$(FUNCTIONNAME)(converged::Bool = true)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct AssumeConvergence <: ConvergenceTest
    converged::Bool = true
end
export AssumeConvergence

function bat_convergence_impl(algoutput::Any, algorithm::AssumeConvergence)
    (result = algorithm.converged,)
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct PigeonsSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a posterior with local parallel tempering from Pigeons.jl. The transformed prior
provides the reference distribution and exact reference draws.

Return the final round's `2^n_rounds` samples from the target temperature in scan order.
The other temperatures aid exploration and evidence estimation. They are not independent
posterior chains. The empirical ESS uses target-trace autocorrelations. Round-trip
diagnostics are reported separately.

This functionality requires Pigeons.jl to be loaded.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PigeonsSampling{TR<:TransformIntent,E} <: AbstractSamplingAlgorithm
    "Transform the posterior and prior into an unconstrained vector space."
    pretransform::TR = (pkgext(Val(:Pigeons)); NormalBased())

    "Number of parallel-tempering rounds, at least 1."
    n_rounds::Int = 10

    "Number of temperatures, including the reference and target, at least 2."
    n_chains::Int = 10

    "Pigeons explorer, or `nothing` for its default explorer."
    explorer::E = nothing

    "Allow Pigeons to explore chains on multiple threads."
    multithreaded::Bool = false

    "Show the Pigeons sampling report."
    show_report::Bool = false
end
export PigeonsSampling

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct PigeonsSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a posterior with local parallel tempering from Pigeons.jl. The transformed prior
provides the reference distribution and exact reference draws.

This functionality requires Pigeons.jl to be loaded.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PigeonsSampling{TR<:TransformIntent,E} <: AbstractSamplingAlgorithm
    "Transform the posterior and prior into an unconstrained vector space."
    pretransform::TR = (pkgext(Val(:Pigeons)); NormalBased())

    "Number of parallel-tempering rounds."
    n_rounds::Int = 10

    "Number of tempered chains."
    n_chains::Int = 10

    "Pigeons explorer, or `nothing` for its default explorer."
    explorer::E = nothing

    "Allow Pigeons to explore chains on multiple threads."
    multithreaded::Bool = false

    "Show the Pigeons sampling report."
    show_report::Bool = false
end
export PigeonsSampling

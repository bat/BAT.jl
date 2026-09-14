# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct PigeonsSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a posterior with Pigeons.jl parallel tempering. The transformed prior is
the initial reference. Return the final round's `2^n_rounds` samples from one
target-temperature trace.

This functionality requires Pigeons.jl to be loaded.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PigeonsSampling{TR<:TransformIntent,IA<:InitvalAlgorithm,E,V} <: AbstractSamplingAlgorithm
    "Transform the posterior and prior into an unconstrained vector space."
    pretransform::TR = (pkgext(Val(:Pigeons)); NormalBased())

    "Algorithm for initial replica states in the original parameter space."
    init::IA = InitFromTarget()

    "Number of parallel-tempering rounds."
    n_rounds::Int = 10

    "Number of temperatures, or fixed-leg temperatures in stabilized mode."
    n_chains::Int = 10

    "Normalized Pigeons variational reference family, or `nothing` for a fixed reference."
    variational::V = nothing

    "Temperatures on an additional variational leg for stabilized PT. Zero disables it."
    n_chains_variational::Int = 0

    "Pigeons explorer, or `nothing` for its default explorer."
    explorer::E = nothing

    "Allow Pigeons to explore chains on multiple threads."
    multithreaded::Bool = false

    "Show the Pigeons sampling report."
    show_report::Bool = false
end
export PigeonsSampling

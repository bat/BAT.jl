# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct SliceMCMCSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a transformed target with SliceSampling.jl. The default sampler applies
stepping-out slice updates in random coordinate order.

This functionality requires SliceSampling.jl to be loaded.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SliceMCMCSampling{TR<:TransformIntent,IA<:InitvalAlgorithm,S} <:
                AbstractSamplingAlgorithm
    "Transform the target into an unconstrained vector space."
    pretransform::TR = (pkgext(Val(:SliceSampling)); NormalBased())

    "Initial-value algorithm."
    init::IA = InitFromTarget()

    "SliceSampling.jl sampler."
    sampler::S = ext_default(pkgext(Val(:SliceSampling)), Val(:SAMPLER))

    "Number of retained samples."
    nsamples::Int = 10^4

    "Number of initial samples to discard."
    n_burnin::Int = 10^3
end
export SliceMCMCSampling

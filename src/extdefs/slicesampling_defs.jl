# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct SliceMCMCSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a transformed target with SliceSampling.jl.
The default sampler visits coordinates in random order.
Each update draws a threshold uniformly below the current density.
It expands a coordinate interval and draws proposals uniformly within it.
It shrinks the interval after rejections until a proposal's density exceeds the threshold.

See [R. M. Neal, "Slice sampling"
(2003)](https://doi.org/10.1214/aos/1056562461).

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

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct EllipticalSliceMCMCSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a posterior with EllipticalSliceSampling.jl. BAT transforms the prior to a
standard Gaussian and evaluates the likelihood separately, as required by the method.

This functionality requires EllipticalSliceSampling.jl to be loaded.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct EllipticalSliceMCMCSampling{IA<:InitvalAlgorithm} <:
                AbstractSamplingAlgorithm
    "Initial-value algorithm."
    init::IA = (pkgext(Val(:EllipticalSliceSampling)); InitFromTarget())

    "Number of retained samples."
    nsamples::Int = 10^4

    "Number of initial samples to discard."
    n_burnin::Int = 10^3
end
export EllipticalSliceMCMCSampling

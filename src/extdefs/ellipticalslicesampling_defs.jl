# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct EllipticalSliceMCMCSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample a posterior with EllipticalSliceSampling.jl using a direct Gaussian prior transform.
The sampler draws a Gaussian vector to define an ellipse through the current state.
Each update draws a threshold uniformly below the current likelihood.
It draws angle proposals uniformly from a bracket spanning the ellipse.
It shrinks the bracket after rejections until a proposal's likelihood exceeds the threshold.

See [I. Murray, R. Adams and D. MacKay, "Elliptical slice sampling"
(2010)](https://proceedings.mlr.press/v9/murray10a.html).

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

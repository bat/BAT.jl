# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct WHAMISampling <: AbstractUltraNestAlgorithmReactiv

*Experimental feature, not part of stable public API.*

Whack-A-Mole Adaptive Metric Importance Sampling.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

* `pretransform::AbstractTransformTarget`: Pre-transformation to apply to the target measure before sampling.

* `nsamples::Int`: Number is independent samples to draw. WHAMIS will generate symmetical samples, so it will generate
  `2*nsamples`` samples in total, but only `nsamples` independent samples.

!!! note

    This functionality is only available when the package [MVGI](https://github.com/bat/MVGI.jl) is loaded (e.g. via
    `import MVGI`).
"""
@with_kw struct WHAMISampling{
    TR<:AbstractTransformTarget, IA<:InitvalAlgorithm, MA <: AbstractModeEstimator,
    CFG, SD<:WHAMISSchedule
} <: AbstractSamplingAlgorithm
    pretransform::TR = (pkgext(Val(:WHAMIS)); PriorToNormal())
    mapalg::MA = OptimAlg(optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG)))
    ninit::Int = 8
    nsamples::Int = 10^4
    mineff::Float64 = 0.01
    maxcycles::Int = 100
end
export WHAMISampling

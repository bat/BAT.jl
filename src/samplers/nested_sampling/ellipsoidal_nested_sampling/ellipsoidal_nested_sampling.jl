# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using .NestedSamplers               # used in this file, because this file ist only load if it is required

include("ens_bounds.jl")
include("ens_proposals.jl")

"""
    struct EllipsoidalNestedSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Uses the julia package
[NestedSamplers.jl](https://github.com/TuringLang/NestedSamplers.jl) to use nested sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)


!!! note

    This functionality is only available when the
    [NestedSamplers.jl](https://github.com/TuringLang/NestedSamplers.jl) package 
    is loaded (e.g. via
    `import`).
"""
@with_kw struct EllipsoidalNestedSampling{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()

    "Number of live-points."
    num_live_points::Int = 1000

    "Volume around the live-points."
    bound::ENSBound = ENSEllipsoidBound()

    "Algorithm used to choose new live-points."
    proposal::ENSProposal = ENSAutoProposal()
    
    "Scale factor for the volume."
    enlarge::Float64 = 1.25
    
    # "Not sure about how this works yet."
    # update_interval::Float64 =
    
    "Number of iterations before the first bound will be fit."
    min_ncall::Int64 = 2*num_live_points
    
    "Efficiency before fitting the first bound."
    min_eff::Float64 = 0.1

    # "The following four are the possible convergence criteria to end the algorithm."
    dlogz::Float64 = 0.01
    max_iters = Inf
    max_ncalls = 10^7
    maxlogl = Inf
end
export EllipsoidalNestedSampling


function bat_sample_impl(rng::AbstractRNG, target::AnyMeasureOrDensity, algorithm::EllipsoidalNestedSampling)
    
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    density, trafo = transform_and_unshape(algorithm.trafo, density_notrafo)                 # BAT prior transformation
    vs = varshape(density)
    dims = totalndof(vs)

    model = NestedModel(logdensityof(density), identity);                                   # identity, because ahead the BAT prior transformation is used instead
    bounding = ENSBounding(algorithm.bound)
    prop = ENSprop(algorithm.proposal)
    sampler = Nested(dims, algorithm.num_live_points; 
                        bounds=bounding, proposal=prop,
                        enlarge=algorithm.enlarge, min_ncall=algorithm.min_ncall, min_eff=algorithm.min_eff
    ) 

    samples_w, state = sample(model, sampler;                                               # returns samples with weights as one vector and the actual state
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Array
    )

    weights = samples_w[:, end]                                                             # the last elements of the vectors are the weights
    nsamples = size(samples_w,1)
    samples = [samples_w[i, 1:end-1] for i in 1:nsamples]                                   # the other ones (between 1 and end-1) are the samples
    logvals = map(logdensityof(density), samples)                                           # posterior values of the samples
    samples_trafo = vs.(BAT.DensitySampleVector(samples, logvals, weight = weights))
    samples_notrafo = inverse(trafo).(samples_trafo)                                            # Here the samples are retransformed
    
    logintegral = Measurements.measurement(state.logz, state.logzerr)
    ess = bat_eff_sample_size(samples_notrafo, KishESS()).result

    return (
        result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, 
        logintegral = logintegral, ess = ess,
        info = state
    )
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATNestedSamplersExt

@static if isdefined(Base, :get_extension)
    using NestedSamplers
else
    using ..NestedSamplers
end

using BAT
using HeterogeneousComputing

BAT.pkgext(::Val{:NestedSamplers}) = BAT.PackageExtension{:NestedSamplers}()

using BAT: MeasureLike, BATMeasure
using BAT: ENSBound, ENSNoBounds, ENSEllipsoidBound, ENSMultiEllipsoidBound
using BAT: ENSProposal, ENSUniformly, ENSAutoProposal, ENSRandomWalk, ENSSlice 

using Statistics, StatsBase
using DensityInterface, InverseFunctions, ValueShapes
import Measurements

using Random


function ENSBounding(bound::ENSNoBounds)
    return Bounds.NoBounds
end

function ENSBounding(bound::ENSEllipsoidBound)
    return Bounds.Ellipsoid
end

function ENSBounding(bound::ENSMultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function ENSBounding(bound::ENSBound) # If nothing ist choosen
    return Bounds.Ellipsoid           # the bound is Ellipsoid
end


function ENSprop(prop::ENSUniformly)
    return Proposals.Uniform()
end

function ENSprop(prop::ENSAutoProposal)
    return :auto     # :auto declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice
end

function ENSprop(prop::ENSRandomWalk)
    return Proposals.RWalk(;ratio=prop.ratio, walks=prop.walks, scale=prop.scale)
end

function ENSprop(prop::ENSSlice)
    return Proposals.Slice(;slices=prop.slices, scale=prop.scale)
end

function ENSprop(prop::ENSProposal) # if nothing is choosen
    return :auto
end



function BAT.bat_sample_impl(m::BATMeasure, algorithm::EllipsoidalNestedSampling, context::BATContext)
    # ToDo: Forward RNG from context!
    rng = get_rng(context)

    transformed_m, trafo = BAT.transform_and_unshape(algorithm.trafo, m, context)                 # BAT prior transformation
    dims = totalndof(varshape(transformed_m))

    if !BAT.has_uhc_support(transformed_m)
        throw(ArgumentError("$algorithm doesn't measures that are not limited to the unit hypercube"))
    end

    model = NestedModel(logdensityof(transformed_m), identity);                                   # identity, because ahead the BAT prior transformation is used instead
    bounding = ENSBounding(algorithm.bound)
    prop = ENSprop(algorithm.proposal)
    sampler = Nested(
        dims, algorithm.num_live_points; 
        bounds=bounding, proposal=prop,
        enlarge=algorithm.enlarge, min_ncall=algorithm.min_ncall, min_eff=algorithm.min_eff
    ) 

    samples_w, state = sample(rng, model, sampler;                                               # returns samples with weights as one vector and the actual state
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Array
    )

    weights = samples_w[:, end]                                                             # the last elements of the vectors are the weights
    nsamples = size(samples_w,1)
    samples = [samples_w[i, 1:end-1] for i in 1:nsamples]                                   # the other ones (between 1 and end-1) are the samples
    logvals = map(logdensityof(transformed_m), samples)                                           # posterior values of the samples
    transformed_smpls = BAT.DensitySampleVector(samples, logvals, weight = weights)
    smpls = inverse(trafo).(transformed_smpls)                                            # Here the samples are retransformed
    
    logintegral = Measurements.measurement(state.logz, state.logzerr)
    ess = bat_eff_sample_size(smpls, KishESS(), context).result

    return (
        result = smpls, result_trafo = transformed_smpls, trafo = trafo, 
        logintegral = logintegral, ess = ess,
        info = state
    )
end


end # module BATNestedSamplersExt

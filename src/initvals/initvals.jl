# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _reshape_v(target::AnyDensityLike, v::Any)
    shape = varshape(convert(AbstractDensity, target))
    reshape_variate(shape, v)
end

function _reshape_vs(target::AnyDensityLike, vs::AbstractVector)
    shape = varshape(convert(AbstractDensity, target))
    reshape_variates(shape, vs)
end


_rand_v(rng::AbstractRNG, src::Distribution) = rand(rng, src)
_rand_v(rng::AbstractRNG, src::Distribution{<:ValueShapes.StructVariate}) = rand(rng, src, ())

_reshape_rand_output(x::Any) = x
_reshape_rand_output(x::AbstractMatrix) = nestedview(x)
_rand_v(rng::AbstractRNG, src::Distribution, n::Integer) = _reshape_rand_output(rand(rng, src, n))

_rand_v(rng::AbstractRNG, src::DistLikeDensity) = rand(rng, sampler(src))
_rand_v(rng::AbstractRNG, src::DistLikeDensity, n::Integer) = _reshape_rand_output(rand(rng, sampler(src), n))

function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable)
    _rand_v(rng, convert(DistLikeDensity, src))
end  
    
function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable, n::Integer)
    _rand_v(rng, convert(DistLikeDensity, src), n)
end  

_rand_v(rng::AbstractRNG, src::DensitySampleVector) =
    first(_rand_v(rng, src, 1))

function _rand_v(rng::AbstractRNG, src::DensitySampleVector, n::Integer)
    orig_idxs = eachindex(src)
    weights = FrequencyWeights(float(src.weight))
    resampled_idxs = sample(orig_idxs, weights, n, replace=false, ordered=false)
    samples = src[resampled_idxs]
end


function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::Any)
    _reshape_v(target, _rand_v(rng, src))
end

function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::Any, n::Integer)
    _reshape_vs(target, _rand_v(rng, src, n))
end



@doc doc"""
    InitFromTarget <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct i.i.d.
sampling whatever component from th

* If the target is supports direct i.i.d. sampling, e.g. because it is a
  distribution, initial values are sampled directly from the target.

* If the target is a posterior density, initial values are sampled from the
  prior (or the prior's prior if the prior is a posterior itself, etc.).

* If the target is a sampled density, initial values are (re-)sampled from
  the available samples.
"""
struct InitFromTarget <: InitvalAlgorithm end
export InitFromTarget


function get_initsrc_from_target end

get_initsrc_from_target(target::AnyIIDSampleable) = target
get_initsrc_from_target(target::TruncatedDensity{<:DistributionDensity}) = sampler(target)

get_initsrc_from_target(target::AbstractPosteriorDensity) = get_initsrc_from_target(getprior(target))


function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::InitFromTarget)
    (result = _rand_v_for_target(rng, target, get_initsrc_from_target(target)),)
end

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::InitFromTarget)
    (result = _rand_v_for_target(rng, target, get_initsrc_from_target(target), n),)
end



@doc doc"""
    InitFromSamples <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct sampling
from a given i.i.d. sampleable source.
"""
struct InitFromSamples{SV<:DensitySampleVector} <: InitvalAlgorithm
    samples::SV
end
export InitFromSamples


function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::InitFromSamples)
    (result = _rand_v(rng, target, algorithm.samples),)
end

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::InitFromSamples)
    (result = _rand_v(rng, target, algorithm.samples, n),)
end



@doc doc"""
    InitFromIID <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by random
resampling from a given set of samples.
"""
struct InitFromIID{D<:AnyIIDSampleable} <: InitvalAlgorithm
    src::D
end
export InitFromIID


function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::InitFromIID)
    (result = _rand_v(rng, target, algorithm.src),)
end

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::InitFromIID)
    (result = _rand_v(rng, target, algorithm.src, n),)
end



@doc doc"""
    ExplicitInit <: InitvalAlgorithm

Uses initial values from a given vector of one or more values/variates. The
values are used in the order they appear in the vector, not randomly.
"""
struct ExplicitInit{V<:AbstractVector} <: InitvalAlgorithm
    xs::V
end
export ExplicitInit


function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::ExplicitInit)
    (result = first(algorithm.xs),)
end

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::ExplicitInit)
    xs = algorithm.xs
    idxs = eachindex(xs)
    (result = xs[idxs[1:n]],)
end

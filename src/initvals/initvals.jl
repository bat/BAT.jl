# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_rand_v(rng::AbstractRNG, src::Distribution) = rand(rng, src)
_rand_v(rng::AbstractRNG, src::NamedTupleDist) = rand(rng, src, ())
_rand_v(rng::AbstractRNG, src::ReshapedDist) = rand(rng, src, ())

_rand_v(rng::AbstractRNG, src::Distribution, n::Integer) = rand(rng, src, n)


_rand_v(rng::AbstractRNG, src::DensitySampleVector) =
    first(_rand_v(rng, src, 1))

function _rand_v(rng::AbstractRNG, src::DensitySampleVector, n::Integer)
    orig_idxs = eachindex(src)
    weights = FrequencyWeights(float(src.weight))
    resampled_idxs = sample(orig_idxs, weights, n, replace=false, ordered=false)
    samples = src[resampled_idxs]
end


function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable)
    _rand_v(rng, convert(Distribution, convert(DistLikeDensity, target)))
end  
    
function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable, n::Integer)
    _rand_v(rng, convert(Distribution, convert(DistLikeDensity, target)), n)
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


_get_initsrc_from_target(target::Distribution) = target

_get_initsrc_from_target(target::DistributionDensity) = convert(Distribution, target)

_get_initsrc_from_target(target::AbstractPosteriorDensity) = _get_initsrc_from_target(getprior(target))

_get_initsrc_from_target(target::SampledDensity) = target.samples


function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::ExplicitInit)
    (result = _rand_v(rng, _get_initsrc_from_target(target)),)
end

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::InitvalAlgorithm)
    (result = _rand_v(rng, _get_initsrc_from_target(target), n),)
end




@doc doc"""
    InitFromIIDSampleable <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct sampling
from a given i.i.d. sampleable source.
"""
struct InitFromIIDSampleable{D<:AnyIIDSampleable} <: InitvalAlgorithm
    dist::D
end
export InitFromIIDSampleable


function bat_initval_impl(rng::AbstractRNG, target::, algorithm::ExplicitInit)
    (result = _rand_v(rng, algorithm.dist),)
end

function bat_initval_impl(rng::AbstractRNG, target::SampledDensity, n::Integer, algorithm::InitvalAlgorithm)
    (result = _rand_v(rng, algorithm.dist, n),)
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

function bat_initval_impl(rng::AbstractRNG, target::AnyDensityLike, n::Integer, algorithm::InitvalAlgorithm)
    xs = algorithm.xs
    idxs = eachindex(xs)
    (result = xs[idxs[1:n]],)
end

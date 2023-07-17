# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_reshape_rand_n_output(x::Any) = x
_reshape_rand_n_output(x::AbstractMatrix) = nestedview(x)

_rand_v(rng::AbstractRNG, src::Distribution) = varshape(src)(rand(rng, bat_sampler(unshaped(src))))
_rand_v(rng::AbstractRNG, src::DistLikeMeasure) = varshape(src)(convert_numtype(default_var_numtype(src), rand(rng, bat_sampler(unshaped(src)))))
_rand_v(rng::AbstractRNG, src::Distribution, n::Integer) = _reshape_rand_n_output(rand(rng, bat_sampler(src), n))
_rand_v(rng::AbstractRNG, src::DistLikeMeasure, n::Integer) = _reshape_rand_n_output(convert_numtype(default_var_numtype(src), rand(rng, bat_sampler(src), n)))

function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable)
    _rand_v(rng, convert(DistLikeMeasure, src))
end  
    
function _rand_v(rng::AbstractRNG, src::AnyIIDSampleable, n::Integer)
    _rand_v(rng, convert(DistLikeMeasure, src), n)
end  

_rand_v(rng::AbstractRNG, src::DensitySampleVector) =
    first(_rand_v(rng, src, 1))

function _rand_v(rng::AbstractRNG, src::DensitySampleVector, n::Integer)
    orig_idxs = eachindex(src)
    weights = FrequencyWeights(float(src.weight))
    resampled_idxs = sample(rng, orig_idxs, weights, n, replace=false, ordered=false)
    samples = src[resampled_idxs]
end


function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::Any)
    vs_target = varshape(convert(AbstractMeasureOrDensity, target))
    vs_src = varshape(convert(AbstractMeasureOrDensity, src))
    x = _rand_v(rng, src)
    reshape_variate(vs_target, vs_src, x)
end

function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::Any, n::Integer)
    vs_target = varshape(convert(AbstractMeasureOrDensity, target))
    vs_src = varshape(convert(AbstractMeasureOrDensity, src))
    xs = _rand_v(rng, src, n)
    reshape_variates(vs_target, vs_src, xs)
end


function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::DensitySampleVector)
    first(_rand_v_for_target(rng, target, src, 1))
end

function _rand_v_for_target(rng::AbstractRNG, target::AnySampleable, src::DensitySampleVector, n::Integer)
    _rand_v(rng, src, n).v
end



"""
    struct InitFromTarget <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct i.i.d.
sampling a suitable component of that target density (e.g. it's prior)
that supports it.

* If the target is supports direct i.i.d. sampling, e.g. because it is a
  distribution, initial values are sampled directly from the target.

* If the target is a posterior density, initial values are sampled from the
  prior (or the prior's prior if the prior is a posterior itself, etc.).

* If the target is a sampled density, initial values are (re-)sampled from
  the available samples.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct InitFromTarget <: InitvalAlgorithm end
export InitFromTarget


function get_initsrc_from_target end

get_initsrc_from_target(target::AnyIIDSampleable) = target
get_initsrc_from_target(target::Renormalized{<:DistMeasure}) = bat_sampler(target)

get_initsrc_from_target(target::AbstractPosteriorMeasure) = get_initsrc_from_target(getprior(target))


function bat_initval_impl(target::AnyMeasureOrDensity, algorithm::InitFromTarget, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, get_initsrc_from_target(target)),)
end

function bat_initval_impl(target::AnyMeasureOrDensity, n::Integer, algorithm::InitFromTarget, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, get_initsrc_from_target(target), n),)
end


function bat_initval_impl(target::ReshapedDensity, algorithm::InitFromTarget, context::BATContext)
    v_orig = bat_initval_impl(parent(target), algorithm, context).result
    v = varshape(target)(unshaped(v_orig))
    (result = v,)
end

function bat_initval_impl(target::ReshapedDensity, n::Integer, algorithm::InitFromTarget, context::BATContext)
    v_orig = bat_initval_impl(parent(target), n, algorithm, context).result
    v = varshape(target).(unshaped.(v_orig))
    (result = v,)
end


function bat_initval_impl(target::Transformed, algorithm::InitFromTarget, context::BATContext)
    v_orig = bat_initval_impl(target.orig, algorithm, context).result
    v = target.trafo(v_orig)
    (result = v,)
end

function bat_initval_impl(target::Transformed, n::Integer, algorithm::InitFromTarget, context::BATContext)
    vs_orig = bat_initval_impl(target.orig, n, algorithm, context).result
    vs = BAT.broadcast_trafo(target.trafo, vs_orig)
    (result = vs,)
end


"""
    struct InitFromSamples <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct sampling
from a given i.i.d. sampleable source.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct InitFromSamples{SV<:DensitySampleVector} <: InitvalAlgorithm
    samples::SV
end
export InitFromSamples


function bat_initval_impl(target::AnyMeasureOrDensity, algorithm::InitFromSamples, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, algorithm.samples),)
end

function bat_initval_impl(target::AnyMeasureOrDensity, n::Integer, algorithm::InitFromSamples, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, algorithm.samples, n),)
end



"""
    struct InitFromIID <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by random
resampling from a given set of samples.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct InitFromIID{D<:AnyIIDSampleable} <: InitvalAlgorithm
    src::D
end
export InitFromIID


function bat_initval_impl(target::AnyMeasureOrDensity, algorithm::InitFromIID, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, algorithm.src),)
end

function bat_initval_impl(target::AnyMeasureOrDensity, n::Integer, algorithm::InitFromIID, context::BATContext)
    rng = get_rng(context)
    (result = _rand_v_for_target(rng, target, algorithm.src, n),)
end



"""
    struct ExplicitInit <: InitvalAlgorithm

Uses initial values from a given vector of one or more values/variates. The
values are used in the order they appear in the vector, not randomly.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
struct ExplicitInit{V<:AbstractVector} <: InitvalAlgorithm
    xs::V
end
export ExplicitInit


function bat_initval_impl(target::AnyMeasureOrDensity, algorithm::ExplicitInit, context::BATContext)
    rng = get_rng(context)
    (result = first(algorithm.xs),)
end

function bat_initval_impl(target::AnyMeasureOrDensity, n::Integer, algorithm::ExplicitInit, context::BATContext)
    rng = get_rng(context)
    xs = algorithm.xs
    idxs = eachindex(xs)
    (result = xs[idxs[1:n]],)
end


function apply_trafo_to_init(trafo::Function, initalg::ExplicitInit)
    xs_tr = broadcast_trafo(trafo, initalg.xs)
    ExplicitInit(xs_tr)
end

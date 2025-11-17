# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_maycopy_val(x) = x
_maycopy_val(A::AbstractArray) = copy(A)
_maycopy_val(nt::NamedTuple) = map(_maycopy_val, nt)


function _rand_v_for_target(target::BATMeasure, src::AbstractMeasure, n::Integer, context::BATContext)
    conv_src = batmeasure(src)
    xs = bat_sample_impl(conv_src, IIDSampling(nsamples = n), context).result.v
    vs_target = varshape(batmeasure(target))
    vs_src = varshape(conv_src)
    reshape_variates(vs_target, vs_src, xs)
end


function _rand_v_for_target(::BATMeasure, src::DensitySampleMeasure, n::Integer, context::BATContext)
    rand(get_gencontext(context), src^n)
end

function _rand_v_for_target(::BATMeasure, src::DensitySampleVector, n::Integer, context::BATContext)
    bat_sample_impl(src, RandResampling(nsamples = n), context).result.v
end


"""
    struct InitFromTarget <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by direct i.i.d.
sampling a suitable component of that target density (e.g. it's prior)
that supports it.

* If the target supports direct i.i.d. sampling, e.g. because it is a
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

get_initsrc_from_target(target::AbstractMeasure) = target
get_initsrc_from_target(target::WeightedMeasure) = get_initsrc_from_target(basemeasure(target))

get_initsrc_from_target(target::AbstractPosteriorMeasure) = get_initsrc_from_target(getprior(target))


function bat_initval_impl(target::MeasureLike, algorithm::InitFromTarget, context::BATContext)
    (result = _maycopy_val(first(_rand_v_for_target(target, get_initsrc_from_target(target), 1, context))),)
end

function bat_initval_impl(target::MeasureLike, n::Integer, algorithm::InitFromTarget, context::BATContext)
    (result = _rand_v_for_target(target, get_initsrc_from_target(target), n, context),)
end


function bat_initval_impl(target::BATPushFwdMeasure, algorithm::InitFromTarget, context::BATContext)
    v_orig = bat_initval_impl(target.orig, algorithm, context).result
    v = gettransform(target)(v_orig)
    (result = v,)
end

function bat_initval_impl(target::BATPushFwdMeasure, n::Integer, algorithm::InitFromTarget, context::BATContext)
    vs_orig = bat_initval_impl(target.orig, n, algorithm, context).result
    vs = BAT.transform_samples(gettransform(target), vs_orig)
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


function bat_initval_impl(target::MeasureLike, algorithm::InitFromSamples, context::BATContext)
    (result = _maycopy_val(first(_rand_v_for_target(target, algorithm.samples, 1, context))),)
end

function bat_initval_impl(target::MeasureLike, n::Integer, algorithm::InitFromSamples, context::BATContext)
    (result = _rand_v_for_target(target, algorithm.samples, n, context),)
end



"""
    struct InitFromIID <: InitvalAlgorithm

Generates initial values for sampling, optimization, etc. by random
resampling from a given set of samples.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct InitFromIID{D<:AbstractMeasure} <: InitvalAlgorithm
    src::D
end
export InitFromIID


function bat_initval_impl(target::MeasureLike, algorithm::InitFromIID, context::BATContext)
    (result = _maycopy_val(first(_rand_v_for_target(target, algorithm.src, 1, context))),)
end

function bat_initval_impl(target::MeasureLike, n::Integer, algorithm::InitFromIID, context::BATContext)
    (result = _rand_v_for_target(target, algorithm.src, n, context),)
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


function bat_initval_impl(target::MeasureLike, algorithm::ExplicitInit, context::BATContext)
    rng = get_rng(context)
    (result = _maycopy_val(first(algorithm.xs)),)
end

function bat_initval_impl(target::MeasureLike, n::Integer, algorithm::ExplicitInit, context::BATContext)
    rng = get_rng(context)
    xs = algorithm.xs
    idxs = eachindex(xs)
    (result = _maycopy_val(xs[idxs[1:n]]),)
end


function apply_trafo_to_init(f_transform::Function, initalg::ExplicitInit)
    xs_tr = transform_samples(f_transform, initalg.xs)
    ExplicitInit(xs_tr)
end

function apply_trafo_to_init(f_transform::Function, initalg::InitFromSamples)
    # ToDo: Pass context to apply_trafo_to_init
    tmp_context = BATContext()
    transformed_smpls = bat_transform_impl(f_transform, initalg.samples, SampleTransformation(), tmp_context).result
    InitFromSamples(transformed_smpls)
end

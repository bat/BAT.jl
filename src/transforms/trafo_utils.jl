# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.transform_samples(f, X::AbstractVector)
    BAT.transform_samples(f, samples::DensitySampleVector)

*BAT-internal, not part of stable public API.*

Apply the variate transformation `f` to the batch of variates `X`, resp. to
the variates of `samples`, with a result in the layout of BAT's sample
storage. The log-density values of samples are corrected by the
log-abs-det Jacobian of `f`, they become `NaN` if `f` doesn't provide one.

Transports transform the whole batch at once, log-abs-det Jacobians
included. Value shapes and `Base.Fix2(unshaped, vs)` convert between flat
and structured batches. Other functions, also those with a log-abs-det
Jacobian, are applied variate by variate on the host.
"""
function transform_samples end

transform_samples(f, X::AbstractVector) = _transform_batch(f, _canonical_batch(X))
transform_samples(f, samples::DensitySampleVector) = _transform_samples(f, samples)


_transform_batch(::typeof(identity), X::AbstractVector) = X
_transform_batch(f::ComposedFunction, X::AbstractVector) = _transform_batch(f.outer, _transform_batch(f.inner, X))
_transform_batch(f::FunctionChain, X::AbstractVector) = foldl((Y, g) -> _transform_batch(g, Y), fchainfs(f), init = X)
_transform_batch(f::TransportFunction, X::AbstractVector) = _transport_batch(f, X)
_transform_batch(vs::AbstractValueShape, X::AbstractVector) = _batch_with_shape(X, vs)

_transform_batch(f::Base.Fix2{typeof(unshaped)}, X::AbstractVector) = nestedview(flatbatch(_own_batch(X, f.x)))

_transform_batch(f, X::AbstractVector) = _canonical_batch(map(f, X))


_transform_samples(::typeof(identity), samples::DensitySampleVector) = samples
_transform_samples(f::ComposedFunction, samples::DensitySampleVector) = _transform_samples(f.outer, _transform_samples(f.inner, samples))
_transform_samples(f::FunctionChain, samples::DensitySampleVector) = foldl((Y, g) -> _transform_samples(g, Y), fchainfs(f), init = samples)

function _transform_samples(f::TransportFunction, samples::DensitySampleVector)
    Y, logd_Y = _transport_batch(f, samples.v, samples.logd)
    return _with_new_variates(samples, Y, logd_Y)
end

# Reshaping has a zero log-abs-det Jacobian:
_transform_samples(f::Union{AbstractValueShape,Base.Fix2{typeof(unshaped)}}, samples::DensitySampleVector) =
    _with_new_variates(samples, _transform_batch(f, samples.v), copy(samples.logd))

_transform_samples(f, samples::DensitySampleVector) = _transform_samples_impl(f, samples, _trafo_ladj_available(f, samples.v))

function _transform_samples_impl(f, samples::DensitySampleVector, ::Val{true})
    X = samples.v
    ys_ladjs = [with_logabsdet_jacobian(f, X[i]) for i in eachindex(samples.logd)]
    Y = _canonical_batch(map(first, ys_ladjs))
    return _with_new_variates(samples, Y, samples.logd .- map(getsecond, ys_ladjs))
end

function _transform_samples_impl(f, samples::DensitySampleVector, ::Val{false})
    Y = _transform_batch(f, samples.v)
    return _with_new_variates(samples, Y, Fill(NaN, length(eachindex(samples.logd))))
end

_with_new_variates(samples::DensitySampleVector, Y::AbstractVector, logd::AbstractVector{<:Real}) =
    DensitySampleVector((Y, logd, deepcopy(samples.weight), deepcopy(samples.info), deepcopy(samples.aux)))


"""
    BAT._transport_batch(f::TransportFunction, X::AbstractVector)
    BAT._transport_batch(f::TransportFunction, X::AbstractVector, logd::AbstractVector{<:Real})

*BAT-internal, not part of stable public API.*

Transport the batch of variates `X` via `f`, with a result in the layout
of BAT's sample storage. With `logd`, returns the transported batch and
the log-density values corrected by the log-abs-det Jacobians of `f`.
"""
function _transport_batch end

_transport_batch(f::TransportFunction, X::AbstractVector) = _own_batch(transports(f, X), f.ν)

function _transport_batch(f::TransportFunction, X::AbstractVector, logd::AbstractVector{<:Real})
    length(eachindex(logd)) == length(eachindex(X)) || throw(DimensionMismatch("Number of variates and log-density values differ"))
    Y, ladjs = transports_with_ladj(f, X)
    return _own_batch(Y, f.ν), logd .- ladjs
end


function _trafo_ladj_available(f, xs::AbstractVector)
    inferred_withladj = Core.Compiler.return_type(with_logabsdet_jacobian, Tuple{typeof(f), eltype(xs)})
    return _trafo_ladj_available_impl(f, xs, inferred_withladj)
end

_trafo_ladj_available_impl(::Any, ::AbstractVector, ::Type{<:Tuple{<:Any,<:Any}}) = Val(true)
_trafo_ladj_available_impl(::Any, ::AbstractVector, ::Type{<:NoLogAbsDetJacobian}) = Val(false)
# Inference-opaque case, probe at runtime (with_logabsdet_jacobian itself,
# not the bare transform, which never returns a tuple):
_trafo_ladj_available_impl(f, xs::AbstractVector, ::Any) = Val(with_logabsdet_jacobian(f, first(xs)) isa Tuple{<:Any,<:Any})


function _transform_dsv!!(f, dsv_y::DensitySampleVector, dsv_x::DensitySampleVector)
    xs = dsv_x.v
    ys_ladjs = with_logabsdet_jacobian.(f, xs)

    dsv_y.v .= first.(ys_ladjs)
    dsv_y.logd .= dsv_x.logd .- getsecond.(ys_ladjs)
    dsv_y.weight .= dsv_x.weight
    dsv_y.info .= dsv_x.info
    dsv_y.aux .= dsv_x.aux

    return dsv_y
end


# Transporting a sample transports its log-density value along:

function (f::TransportFunction)(s::DensitySample)
    v, ladj = with_logabsdet_jacobian(f, s.v)
    DensitySample(v, s.logd - ladj, s.weight, s.info, s.aux)
end

Base.Broadcast.broadcasted(f::TransportFunction, smpls::DensitySampleVector) = transform_samples(f, smpls)
Base.Broadcast.broadcasted(f::TransportFunction, xs::ShapedAsNTArray) = transform_samples(f, xs)

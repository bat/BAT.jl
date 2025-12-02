# This file is a part of BAT.jl, licensed under the MIT License (MIT).


transform_samples(::typeof(identity), inputs::AbstractVector) = inputs

function transform_samples(
    f::Any,
    xs::AbstractVector
)
    unshaped_f = _unshaped_trafo(f)
    x_shape, y_shape = _trafo_input_output_shape(f, xs)
    _transform_xs_impl(f, unshaped_f, xs, x_shape, y_shape)
end


transform_samples(::typeof(identity), inputs::DensitySampleVector) = inputs

function transform_samples(
    f::Any,
    xsv::DensitySampleVector
)
    unshaped_f = _unshaped_trafo(f)
    x_shape, y_shape = _trafo_input_output_shape(f, xsv.v)
    ladjavail = _trafo_ladj_available(f, xsv.v)
    _transform_xsv_impl(f, unshaped_f, xsv, x_shape, y_shape, ladjavail)
end


function _transform_xs_impl(
    f::Any,
    ::Missing,
    xs::AbstractVector,
    ::Union{AbstractValueShape,Missing},
    ::Missing
)
    return f.(xs)
end

function _transform_xs_impl(
    f::Any,
    ::Missing,
    xs::AbstractVector,
    ::AbstractValueShape,
    y_shape::AbstractValueShape
)
    ys_unshaped = _trafo_create_unshaped_ys(f, xs, y_shape)

    @assert axes(ys_unshaped) == axes(xs)
    @assert ys_unshaped isa ArrayOfSimilarArrays

    @threads for i in eachindex(ys_unshaped, xs)
        y = f(xs[i])
        ys_unshaped[i] = unshaped(y, y_shape)
    end

    ys = y_shape.(ys_unshaped)
    return ys
end

function _transform_xs_impl(
    ::Any,
    f_unshaped::Any,
    xs::AbstractVector,
    x_shape::AbstractValueShape,
    y_shape::AbstractValueShape
)
    xs_unshaped = _trafo_unshape_inputs(xs, x_shape)

    ys_unshaped = _trafo_create_unshaped_ys(f_unshaped, xs_unshaped, y_shape)

    @assert axes(ys_unshaped) == axes(xs)
    @assert ys_unshaped isa ArrayOfSimilarArrays

    @threads for i in eachindex(ys_unshaped, xs_unshaped)
        y = f_unshaped(xs_unshaped[i])
        ys_unshaped[i] = y
    end

    ys = y_shape.(ys_unshaped)
    return ys
end


function _transform_xs_impl_withlogd(
    f::Any,
    ::Missing,
    xs::AbstractVector,
    logd_xs::AbstractVector,
    ::Union{AbstractValueShape,Missing},
    ::Missing
)
    ys_ladjs = Base.Fix1(with_logabsdet_jacobian, f).(xs)
    ys = map(first, ys_ladjs)
    ladjs = map(x -> x[2], ys_ladjs)
    logd_ys = logd_xs - ladjs
    return ys, logd_ys
end

function _transform_xs_impl_withlogd(
    f::Any,
    ::Missing,
    xs::AbstractVector,
    logd_xs::AbstractVector,
    ::AbstractValueShape,
    y_shape::AbstractValueShape
)
    ys_unshaped = _trafo_create_unshaped_ys(f, xs, y_shape)
    logd_ys = zero(logd_xs)

    @assert axes(ys_unshaped) == axes(xs)
    @assert ys_unshaped isa ArrayOfSimilarArrays

    @threads for i in eachindex(ys_unshaped, xs)
        y, ladj = with_logabsdet_jacobian(f, xs[i])
        ys_unshaped[i] = unshaped(y, y_shape)
        logd_ys[i] = logd_xs[i] - ladj
    end

    ys = y_shape.(ys_unshaped)
    return ys, logd_ys
end

function _transform_xs_impl_withlogd(
    ::Any,
    f_unshaped::Any,
    xs::AbstractVector,
    logd_xs::AbstractVector,
    x_shape::AbstractValueShape,
    y_shape::AbstractValueShape
)
    xs_unshaped = _trafo_unshape_inputs(xs, x_shape)

    ys_unshaped = _trafo_create_unshaped_ys(f_unshaped, xs_unshaped, y_shape)
    logd_ys = zero(logd_xs)

    @assert axes(ys_unshaped) == axes(xs)
    @assert ys_unshaped isa ArrayOfSimilarArrays

    @threads for i in eachindex(ys_unshaped, xs_unshaped)
        y, ladj = with_logabsdet_jacobian(f_unshaped, xs_unshaped[i])
        ys_unshaped[i] = y
        logd_ys[i] = logd_xs[i] - ladj
    end

    ys = y_shape.(ys_unshaped)
    return ys, logd_ys
end



function _transform_xsv_impl(
    f::Any, f_unshaped::Any,
    xsv::DensitySampleVector,
    x_shape::Union{AbstractValueShape,Missing},
    y_shape::Union{AbstractValueShape,Missing},
    ::Val{false}
)
    ys = _transform_xs_impl(f, f_unshaped, xsv.v, x_shape, y_shape)

    DensitySampleVector((
        ys,
        Fill(NaN, length(eachindex(xsv.logd))),
        deepcopy(xsv.weight),
        deepcopy(xsv.info),
        deepcopy(xsv.aux),
    ))
end


function _transform_xsv_impl(
    f::Any, f_unshaped::Any,
    xsv::DensitySampleVector,
    x_shape::Union{AbstractValueShape,Missing},
    y_shape::Union{AbstractValueShape,Missing},
    ::Val{true}
)
    ys, logd_ys = _transform_xs_impl_withlogd(f, f_unshaped, xsv.v, xsv.logd, x_shape, y_shape)

    DensitySampleVector((
        ys,
        logd_ys,
        deepcopy(xsv.weight),
        deepcopy(xsv.info),
        deepcopy(xsv.aux),
    ))
end



_unshaped_trafo(::Any) = missing


const _ESCompatible = Union{
    AbstractArray{<:Real},
    AbstractArray{<:AbstractArray{<:Real}},
    ShapedAsNTArray
}

const _VSCompatible = Union{
    Real,
    AbstractArray{<:Real},
    NamedTuple{names, <:Tuple{Vararg{Union{Real,AbstractArray{<:Real}}}}} where names
}

_get_point_shape(xs::_ESCompatible) = elshape(xs)

_get_point_shape(xs::AbstractVector) = _get_point_shape_impl(xs, eltype(xs))
_get_point_shape_impl(xs::AbstractVector, ::Type{<:_VSCompatible}) = valshape(first(xs))
_get_point_shape_impl(xs::AbstractVector, ::Type) = missing

_maybe_valshape(::Any) = missing
_maybe_valshape(x::_VSCompatible) = valshape(x)


_trafo_unshape_inputs(xs::AbstractVector, x_shape::AbstractValueShape) = inverse(x_shape).(xs)
_trafo_unshape_inputs(xs::ArrayOfSimilarVectors{<:Real}, ::AbstractValueShape) = unshaped.(xs)
_trafo_unshape_inputs(xs::ShapedAsNTArray, ::AbstractValueShape) = unshaped.(xs)
_trafo_unshape_inputs(xs::DensitySampleVector, ::AbstractValueShape) = unshaped.(xs)


function _trafo_input_output_shape(f, xs::AbstractVector)
    x_shape = _get_point_shape(xs)
    maybe_y_shape = _trafo_resultshape(f, x_shape)
    y_shape = _trafo_output_shape_impl(f, xs, maybe_y_shape)
    return x_shape, y_shape
end

_trafo_output_shape_impl(::Any, ::AbstractVector, y_shape::AbstractValueShape) = y_shape

function _trafo_output_shape_impl(f::Any, xs::AbstractVector, ::Missing)
    dummy_x = first(xs)
    dummy_y = f(dummy_x)
    return _maybe_valshape(dummy_y)
end

_trafo_resultshape(f, x_shape::AbstractValueShape) = resultshape(f, x_shape)
_trafo_resultshape(::Any, ::Missing) = missing


function _trafo_ladj_available(f, xs::AbstractVector)
    inferred_withladj = Core.Compiler.return_type(with_logabsdet_jacobian, Tuple{typeof(f), eltype(xs)})
    return _trafo_ladj_available_impl(f, xs, inferred_withladj)
end

_trafo_ladj_available_impl(::Any, ::AbstractVector, ::Type{<:Tuple{<:Any,<:Any}}) = Val(true)
_trafo_ladj_available_impl(::Any, ::AbstractVector, ::Type{<:NoLogAbsDetJacobian}) = Val(false)
_trafo_ladj_available_impl(f, xs::AbstractVector, ::Any) = Val(f(first(xs)) isa Tuple{<:Any,<:Any})


function _trafo_output_numtype(f, xs::AbstractVector)
    realnumtype(_trafo_output_type(f, xs))
end


function _trafo_output_type(f, xs::AbstractVector)
    R = Core.Compiler.return_type(f, Tuple{eltype(xs)})
    return _trafo_output_type_impl(f, xs, R, R)
end

_trafo_output_type_impl(::Any, ::AbstractVector, ::DataType, ::Type{R}) where R = R

function _trafo_output_type_impl(f::Any, xs::AbstractVector, ::Type, ::Type)
    return typeof(f(first(xs)))
end


function _trafo_create_unshaped_ys(f, xs, y_shape::AbstractValueShape)
    R = _trafo_output_numtype(f, xs)
    cpunit = get_compute_unit(xs)
    m = totalndof(y_shape)
    n = length(eachindex(xs))
    return nestedview(allocate_array(cpunit, R, (m, n)))
end


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

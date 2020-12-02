# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    NamedTupleTransform <: VariateTransform

*BAT-internal, not part of stable public API.*
"""
struct NamedTupleTransform{
    names,
    ST <: VariateSpace,
    SF <: VariateSpace,
    N,
    TT <: NTuple{N,VariateTransform{<:VariateForm}},
    AT <: NTuple{N,ValueShapes.ValueAccessor},
} <: VariateTransform{ValueShapes.NamedTupleVariate{names},ST,SF}
    _internal_transforms::NamedTuple{names,TT}
    _internal_shape::NamedTupleShape{names,AT}
end 


_nt_acc_nt_type(::NamedTupleShape{names,AT}) where {names,AT} = AT

function NamedTupleTransform(transforms::NamedTuple{names,<:NTuple{N,VariateTransform}}) where {names,N}
    trg = product_varspace(map(target_space, values(transforms))...)
    src = product_varspace(map(source_space, values(transforms))...)
    shape = NamedTupleShape(map(varshape, transforms))

    ST = typeof(trg)
    SF = typeof(src)
    TT = typeof(transforms)
    AT = _nt_acc_nt_type(shape)

    NamedTupleTransform{names,ST,SF,T,TT,AT}(transforms, shape)
end


@inline NamedTupleTransform(;named_dists...) = NamedTupleTransform(values(named_dists))



@inline _transforms(d::NamedTupleTransform) = getfield(d, :_internal_transforms)
@inline _shape(d::NamedTupleTransform) = getfield(d, :_internal_shape)


@inline Base.keys(d::NamedTupleTransform) = keys(_transforms(d))

@inline Base.values(d::NamedTupleTransform) = values(_transforms(d))

@inline function Base.getproperty(d::NamedTupleTransform, s::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy:
    if s == :_internal_transforms
        getfield(d, :_internal_transforms)
    elseif s == :_internal_shape
        getfield(d, :_internal_shape)
    else
        getproperty(_transforms(d), s)
    end
end

@inline function Base.propertynames(d::NamedTupleTransform, private::Bool = false)
    names = propertynames(_transforms(d))
    if private
        (names..., :_internal_transforms, :_internal_shape)
    else
        names
    end
end


@inline Base.map(f, dist::NamedTupleTransform) = map(f, _transforms(dist))


Base.merge(a::NamedTuple, dist::NamedTupleTransform{names}) where {names} = merge(a, _transforms(dist))
Base.merge(a::NamedTupleTransform) = a
Base.merge(a::NamedTupleTransform, b::NamedTupleTransform, cs::NamedTupleTransform...) = merge(NamedTupleTransform(;a..., b...), cs...)


ValueShapes.varshape(trafo::NamedTupleTransform) = _shape(trafo)

target_space(trafo::NamedTupleTransform) = trafo.target_space
source_space(trafo::NamedTupleTransform) = MixedSpace


# trafo_fwd(trafo::VariateTransformChain, v::Any) = ...

# trafo_inv(trafo::VariateTransformChain, v::Any) = ...

# ToDo: Finish implementation


# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Leaves of a value shape tree in flat order, as path => accessor pairs. Paths
# are Symbols for top-level fields and dotted Exprs like :(x.b.c) for nested
# fields, offsets are absolute within the flat vector:
_leaf_accessors(vs::NamedTupleShape) = _push_leaves!(Pair{Union{Symbol,Expr},ValueAccessor}[], vs, nothing, 0)

function _push_leaves!(leaves, vs::NamedTupleShape, prefix, offset::Int)
    for k in keys(vs)
        acc = getproperty(vs, k)
        _push_leaves!(leaves, acc.shape, _subpath(prefix, k), offset + acc.offset)
    end
    return leaves
end

function _push_leaves!(leaves, vs::AbstractValueShape, path, offset::Int)
    push!(leaves, path => ValueAccessor(vs, offset))
    return leaves
end

_subpath(::Nothing, k::Symbol) = k
_subpath(prefix, k::Symbol) = Expr(:., prefix, QuoteNode(k))


# Accessor, with absolute offset, of the field selected by a path like :a or :(x.b.c):
function _resolve_path(vs::NamedTupleShape, path::Symbol)
    path in keys(vs) || throw(ArgumentError("Value shape has no field $path"))
    return getproperty(vs, path)
end

function _resolve_path(vs::NamedTupleShape, path::Expr)
    Meta.isexpr(path, :., 2) && path.args[2] isa QuoteNode || throw(ArgumentError("Invalid parameter path $path"))
    parent = _resolve_path(vs, path.args[1])
    parent.shape isa NamedTupleShape || throw(ArgumentError("Parameter $(path.args[1]) is not a named tuple, cannot select $path"))
    child = _resolve_path(parent.shape, path.args[2].value)
    return ValueAccessor(child.shape, parent.offset + child.offset)
end


_index_arg(i::Integer) = i
_index_arg(s::Symbol) = s === :(:) ? Colon() : throw(ArgumentError("Invalid index $s in parameter selection"))
_index_arg(ex::Expr) = @capture(ex, s_:e_) ? (s:e) : throw(ArgumentError("Invalid index $ex in parameter selection"))

# Flat indices and value shape of the parameters selected by a field path like
# :a or :(x.b.c), optionally with an element selection like :(a[2]), :(a[2:4])
# or :(a[1, :]). Constants select no indices but keep their value:
function _select_params(vs::NamedTupleShape, sel::Union{Symbol,Expr})
    if @capture(sel, path_[args__])
        acc = _resolve_path(vs, path)
        acc.shape isa NamedTupleShape && throw(ArgumentError("$path is a named tuple, select its fields by name instead of $sel"))
        idxargs = map(_index_arg, args)
        if acc.shape isa ConstValueShape
            return Int[], ConstValueShape(acc.shape.value[idxargs...])
        else
            idxs = (1:totalndof(vs))[acc][idxargs...]
            return idxs isa Integer ? (Int[idxs], ScalarShape{Real}()) : (collect(vec(idxs)), ArrayShape{Real}(size(idxs)...))
        end
    else
        acc = _resolve_path(vs, sel)
        return collect(acc.offset .+ (1:acc.len)), acc.shape
    end
end


# Flat index of the parameter selected by `name`, or the flat indices of a
# multivariate parameter:
function asindex(vs::NamedTupleShape, name::Union{Symbol,Expr})
    idxs, _ = _select_params(vs, name)
    isempty(idxs) && throw(ArgumentError("$name selects no free parameter in $vs"))
    return length(idxs) == 1 ? only(idxs) : idxs
end

asindex(ntd::NamedTupleDist, name::Union{Expr,Symbol}) = asindex(varshape(ntd), name)


# Name of the free parameter with the given flat index:
getstring(vs::NamedTupleShape, idx::Integer) = all_active_names(vs)[idx]
getstring(prior::NamedTupleDist, idx::Integer) = getstring(varshape(prior), idx)


# Names of the free parameters in flat order, elements of multivariate
# parameters as "a[i]":
function all_active_names(vs::NamedTupleShape)
    mapreduce(vcat, _leaf_accessors(vs); init = String[]) do (path, acc)
        n = acc.len
        n > 1 ? ["$path[$i]" for i in 1:n] : fill("$path", n)
    end
end

all_active_names(vs::ScalarShape) = ["v"]
all_active_names(vs::ArrayShape) = ["v_$i" for i in 1:totalndof(vs)]


# Names and values of the constant parameters:
_const_parameters(::AbstractValueShape) = Pair{String,Any}[]

function _const_parameters(vs::NamedTupleShape)
    leaves = filter(leaf -> last(leaf).shape isa ConstValueShape, _leaf_accessors(vs))
    return Pair{String,Any}[string(path) => acc.shape.value for (path, acc) in leaves]
end


# Whether a value shape contains a constant anywhere, nested shapes included:
_has_const_shape(::AbstractValueShape) = false
_has_const_shape(::ConstValueShape) = true
_has_const_shape(vs::NamedTupleShape) = any(acc -> _has_const_shape(acc.shape), values(vs))

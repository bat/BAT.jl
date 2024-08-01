# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# bring vsel to a normalized shape, i.e.: 
#
# AbstractVector{<:Integer} -> AbstractVector{<:Integer}
# AbstractVector{AbstractUnitRange} -> AbstractVector{AbstractUnitRange}
# 
# Expr -> Vector{Union{Symbol,Expr})
# 
# Symbol -> Vector{Union{Symbol,Expr})

function _normalize_vsel_unshaped(vsel::Union{AbstractVector{I}, AbstractVector{UnitRange{I}}}) where I <: Integer
    return vsel
end

function _normalize_vsel_unshaped(vsel::Integer)
    return Int[vsel]
end

function _normalize_vsel_unshaped(vsel::AbstractVector)
    @argcheck all(broadcast(x -> x isa Union{T, AbstractVector{T}, AbstractVector{UnitRange{T}}} where T <: Integer, vsel)) throw(ArgumentError("Samples or distribution are unshaped, please use Integer, Vector of Integers, or UnitRange for indexing."))
    return Union{typeof.(vsel)...}[vsel...]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Union{UR, AbstractVector{UR}} where UR <: AbstractRange)
    @argcheck (vsel isa AbstractVector{UR} where UR <: AbstractUnitRange ? length(vsel) == 1 : true) throw(ArgumentError("Invalid selection of indexes vsel = $vsel, please use a single UnitRange or a Vector of a single UnitRange."))
    rng = vsel isa AbstractVector{UR} where UR <: AbstractUnitRange ? vsel[1] : vsel
    return UnitRange[rng]
end

# function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Integer)
#     n_vsel = _all_exprs(vs)[vsel]
#     return typeof(n_vsel)[n_vsel]
# end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Union{I, AbstractVector{I}}) where I <: Integer
    return vsel isa AbstractVector ? Integer[vsel...] : Integer[vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Expr)
    if isexpr(vsel, :vect) || isexpr(vsel, :call)
        if @capture(vsel, [s_:e_]) || @capture(vsel, (s_:e_))
            return _normalize_vsel_shaped(vs, s:e)

        elseif @capture(vsel, [dims__]) && all(broadcast(x -> x isa Integer, dims))
            return _normalize_vsel_shaped(vs, [dims...])
        else
            vsel_norm = [_normalize_vsel_shaped(vs, el isa QuoteNode ? el.value : el) for el in vsel.args]
            vcat(vsel_norm...) # for user convenience, does :([a[1,2], b, c]) -> [:(a[1,2]), :b, :c]
            return Union{typeof.(n_vsel)...}[n_vsel...]
        end
    end
    return Expr[vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Symbol)   
    return Symbol[vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::AbstractVector)
    vsel_norm = [_normalize_vsel_shaped(vs, el isa QuoteNode ? el.value : el) for el in vsel]
    n_vsel = vcat(vsel_norm...)
    return Union{typeof.(n_vsel)...}[n_vsel...]
end


# Retrieve flat indexes of the selected dims of samples
# If shaped, build new shape

function marg_idxs_unshaped(samples::DensitySampleVector, 
                            vsel::AbstractVector            
)
    flat_v = ValueShapes.flatview(samples.v)
    return (firstindex(flat_v) - 1) .+ vsel
end

function marg_idxs_shaped(samples::DensitySampleVector,
                          vsel::Union{AbstractVector{E}, AbstractVector{S}, AbstractVector{Union{S, E}}} where E <: Expr where S <: Symbol
)
    shapes = AbstractValueShape[]
    sel_idxs = Int[]

    vs = varshape(samples)
    flat_v = ValueShapes.flatview(unshaped.(samples).v)

    flat_idxs = axes(flat_v, 1)
    shaped_idxs = vs(flat_idxs)

    for el in vsel 
        # :a
        if el isa Symbol
            shape_acc = getproperty(vs, el)
            tmp_shape = shape_acc.shape
            if !(shape isa ConstValueShape)
                idx_acc = getproperty(shaped_idxs, el)
                append!(sel_idxs, idx_acc isa Integer ? Int64[idx_acc] : vec(idx_acc))
            end
        # :(a[1]) or :(a[1:4]) or :(a[1,2,3]) or :(a[1,2,:])
        elseif @capture(el, a_[args__])
            shape_acc = getproperty(vs, a)
            isa_const = shape_acc.shape isa ConstValueShape
            if !isa_const
                idx_acc = getproperty(shaped_idxs, a)
            end

            if @capture(el, a_[s_:e_])
                if isa_const
                    tmp_shape = ConstValueShape(shape_acc.shape.value[s:e])
                else
                    tmp_shape = ArrayShape{Real}(length(s:e))
                    append!(sel_idxs, vec(idx_acc[s:e]))
                end
            elseif @capture(el, a_[dim_])
                if isa_const
                    tmp_shape = ConstValueShape(shape_acc.shape.value[dim])
                else
                    tmp_shape = ScalarShape{Real}()
                    append!(sel_idxs, idx_acc[dim])
                end
            else
                slice_dims = Int[]

                for (i,arg) in enumerate(args) 
                    if @capture(arg, (s_:e_))
                        args[i] = s:e
                        append!(slice_dims, length(s:e))
                    else
                        append!(slice_dims, 0)
                    end
                end
                sliced_dims = slice_dims .> 0

                whole_dims = args .== :(:)
                args[whole_dims] .= Colon()

                if isa_const
                    tmp_shape = ConstValueShape(shape_acc.shape.value[args...])
                else
                    new_dims = shape_acc.shape.dims .* whole_dims .+ slice_dims
                    new_axes = new_dims[whole_dims .| sliced_dims]
            
                    tmp_shape = length(new_axes) > 0 ? ArrayShape{Real}(new_axes...) : ScalarShape{Real}()

                    idxs = idx_acc[args...]
                    append!(sel_idxs, ndims(idxs) > 0 ? vec(idxs) : idxs)
                end
            end
        end
        push!(shapes, tmp_shape)
    end
    vsel = encode_name.(vsel)
    new_shape = NamedTupleShape(NamedTuple(vsel .=> shapes))

    return sel_idxs, new_shape
end 

function marg_idxs_shaped(samples::DensitySampleVector,
                          vsel::AbstractVector{UR} where UR <: AbstractUnitRange
)   
    rng = vsel[1]
    vs = varshape(samples)
    all_accs = values(vs)
    all_shapes = getproperty.(all_accs, :shape)
    all_keys = keys(vs)
    consts = [broadcast(x -> x isa ConstValueShape, all_shapes)...]
    non_const_accs = all_accs[.!consts]
    non_const_shapes = all_shapes[.!consts]
    non_const_keys = all_keys[.!consts]
    non_const_vs = NamedTupleShape(NamedTuple(non_const_keys .=> non_const_shapes))
    @argcheck sum(length.(non_const_accs)) >= length(rng) throw(ArgumentError("The selected range of flat indexes is larger than the number of free parameters. Please only select free parameters by flat index or use Symbols or Expressions."))

    strs = _all_exprs_as_strings(non_const_vs)[rng]
    flat_v = ValueShapes.flatview(unshaped.(samples).v)

    sel_idxs = (firstindex(flat_v) - 1) .+ rng
    new_shapes = fill(ScalarShape{Real}(), length(rng))

    encoded_keys = encode_name.(strs)
    new_shape = NamedTupleShape(NamedTuple(encoded_keys .=> new_shapes))

    return sel_idxs, new_shape
end 

function marg_idxs_shaped(samples::DensitySampleVector,
    vsel::AbstractVector{I} where I <: Integer
)   
    vs = varshape(samples)
    all_accs = values(vs)
    all_shapes = getproperty.(all_accs, :shape)
    all_keys = keys(vs)
    consts = [broadcast(x -> x isa ConstValueShape, all_shapes)...]
    non_const_accs = all_accs[.!consts]
    non_const_shapes = all_shapes[.!consts]
    non_const_keys = all_keys[.!consts]
    non_const_vs = NamedTupleShape(NamedTuple(non_const_keys .=> non_const_shapes))
    @argcheck all(broadcast(x -> x <= sum(length.(non_const_accs)), vsel)) throw(ArgumentError("One or more selected flat indexes is out of bounds of the free parameters. Please only select free parameters by flat index or use Symbols or Expressions."))
    all_strs = _all_exprs_as_strings(non_const_vs)
    strs = String[all_strs[idx] for idx in vsel]
    flat_v = ValueShapes.flatview(unshaped.(samples).v)
    sel_idxs = (firstindex(flat_v) - 1) .+ vsel
    new_shapes = fill(ScalarShape{Real}(), length(vsel))
    encoded_keys = encode_name.(strs)
    new_shape = NamedTupleShape(NamedTuple(encoded_keys .=> new_shapes))

    return sel_idxs, new_shape
end 

# Encode dim names to preserve indexing information 
#
# unicode encodings to preserve indexing information:
# :a⌞2⌟     = : a \llcorner 2 \lrcorner 
# :a⌞2ː4⌟   = : a \llcorner 2 \lmrk 4 \lrcorner 
# :a⌞1ˌ3ˌ5⌟ = : a \llcorner 1 \verti 3 \verti 5 \lrcorner

function encode_name(name::Union{Expr, String})
    # Nested replace to make Julia v1.6 happy: # ToDo: simplify, now that we require Julia v1.10
    code = replace(replace(replace(replace(replace(
        string(name)," " => ""), "[" => "⌞"), ":" => "ː"), "," => "ˌ"), "]" => "⌟"
    )
    return Symbol(code)
end 

function encode_name(name::Symbol)
    return name
end


#=
"""
    bat_marginalize(
        samples::DensitySampleVector, 
        vsel::Union{E, S, I, U, AbstractVector{Union{E, S, I, U}}} where E<:Expr where S<:Symbol where I<:Integer where U<:UnitRange
    )

Create a new `DensitySampleVector` containing the selected 
dimensions `vsel` of the input `samples`.

Returns a NamedTuple `(result = marg_samples::DensitySampleVector)`. If the input is shaped, the new samples
will have the shape of the selected dimensions.

`samples` may be shaped or unshaped, vsel may be an `Int`, `UnitRange`, 
`Symbol`, `Expression`, or a Vector of these types.

To select certain elements or whole slices of a multidimensional parameter `a` of 
`samples` do:

`vsel` = :(a[1]) or `vsel` = :(a[1,2]) or `vsel` = :(a[1,:,:])

with the brackets containing the cartesian index of the selected values. 

To preserve the indexing information of the selection, `vsel` is encoded using 
unicode characters. The new names in the marginalized samples are then `symbol`s
of these codes:

:(a[2])     => :a⌞2⌟    
:(a[2:4])   => :a⌞2ː4⌟   
:(a[1,2,:]) => :a⌞1ˌ2ˌː⌟ 

If `samples` is shaped, and `vsel` is an `Int` `i` or a vector of `Int`s `[i,j,k]`, 
the i-th, j-th, and k-th parameters of `samples` are selected automatically.

If `samples` is unshaped, `vsel` may only be an ``Int` `i` or a vector 
of `Int`s `[i,j,k]`. Then the i-th, j-th, and k-th columns of `samples` are selected.

For convenience, `vsel` can be of the shape 

:([a[1,2], a[5:8], c]) instead of [:(a[1,2]), :(a[5:8]), :c]

The logd of the new samples is set to NaN, the info and aux fields of `samples` are copied.

Example:

```julia 
marg_shaped_samples = bat_marginalize(orig_shaped_samples, :(a[1,2,:]))

varshape(marg_samples) == NamedTupleShape{(:a⌞1ˌ2ˌː⌟,), Tuple{ValueAccessor{ArrayShape{Real, 1}}}}((a⌞1ˌ2ˌː⌟ = ValueAccessor{ArrayShape{Real, 1}}(ArrayShape{Real, 1}((3,)), 0, 3),), 3)
```

```julia
marg_unshaped_samples = bat_marginalize(orig_unshaped_smaples, [1,2,3])
'''
"""
=#

# ToDo: Turn this into a proper public API:
function bat_marginalize(samples::DensitySampleVector, 
                         vsel
)
    shaped = isshaped(samples)
    vs = varshape(samples)
    vsel = vsel isa Tuple ? Union{typeof.(vsel)...}[vsel...] : vsel

    if shaped
       vsel = _normalize_vsel_shaped(vs, vsel)
       flat_idxs, new_shape = marg_idxs_shaped(samples, vsel)
    else
       vsel = _normalize_vsel_unshaped(vsel)
       flat_idxs = marg_idxs_unshaped(samples, vsel)
    end

    @argcheck !isempty(flat_idxs) throw(ArgumentError("The selected parameters for the input data to be marginalized to yielded only constant Values, please select at least one free parameter."))
    info = samples.info
    aux = samples.aux
    logd = fill(NaN, size(samples.v, 1))

    # if only_consts
    #     new_accs = values(new_shape)
    #     n_params = sum(length.(new_accs))
    #     flat_orig_data = zeros(n_params, size(samples.v, 1))
    #     marg_data = ndims(flat_orig_data) > 1 ? nestedview(flat_orig_data) : flat_orig_data
    # else

    flat_orig_data = ValueShapes.flatview(unshaped.(samples).v)
    marg_data = flat_orig_data[flat_idxs, :]
    marg_data = ndims(marg_data) > 1 ? nestedview(marg_data) : marg_data
   
    marg_samples = shaped ? new_shape.(DensitySampleVector(marg_data, logd, info = info, aux = aux)) : DensitySampleVector(marg_data, logd, info = info, aux = aux)
    return (result = marg_samples,)
end

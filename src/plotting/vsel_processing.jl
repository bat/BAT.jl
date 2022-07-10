# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# bring vsel to a normalized shape, i.e.: 
#
# AbstractVector{<:Integer} -> AbstractVector{<:Integer}
# AbstractVector{AbstractUnitRange} -> AbstractVector{AbstractUnitRange}
# 
# Expr -> Vector{Union{Symbol,Expr})
# 
# Symbol -> Vector{Union{Symbol,Expr})

function _normalize_vsel_unshaped(vsel::Union{AbstractVector{T}, AbstractVector{UnitRange{T}}}) where T <: Integer
    return vsel
end

function _normalize_vsel_unshaped(vsel::Integer) where T <: Integer
    return Int[vsel]
end

function _normalize_vsel_unshaped(vsel::AbstractVector)
    @argcheck all(broadcast(x -> x isa Union{T, AbstractVector{T}, AbstractVector{UnitRange{T}}} where T <: Integer, vsel)) throw(ArgumentError("Samples or distribution are unshaped, please use Integer, Vector of Integers, or UnitRange for indexing."))
    return vsel
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::AbstractVector{UnitRange}) 
    @argcheck length(vsel) == 1 throw(ArgumentError("Can't use multidimensional cartesian index for shaped samples or distribution, please use keys instead"))
    return [key for key in keys(vs)[vsel[1]]]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Integer)
    return [keys(vs)[vsel]]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::AbstractVector{Integer})
    return [keys(vs)[i] for i in vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Expr)
    if isexpr(vsel, :vect) || isexpr(vsel, :call)
        if @capture(vsel, [s_:e_]) || @capture(vsel, (s_:e_))
            return [keys(vs)[i] for i in s:e]

        elseif @capture(vsel, [dims__]) && all(broadcast(x -> x isa Integer, dims))
            return [keys(vs)[i] for i in dims]
        else
            vsel_norm = [_normalize_vsel_shaped(vs, el isa QuoteNode ? el.value : el) for el in vsel.args]
            return vcat(vsel_norm...) # for user convenience, does :([a[1,2], b, c]) -> [:(a[1,2]), :b, :c]
        end
    end
    return [vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Symbol)   
    return [vsel]
end

function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::AbstractVector)
    vsel_norm = [_normalize_vsel_shaped(vs, el isa QuoteNode ? el.value : el) for el in vsel]
    return vcat(vsel_norm...)
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
                         vsel::AbstractVector
)
    shapes = []
    sel_idxs = Int[]

    vs = varshape(samples)
    flat_idxs = axes(ValueShapes.flatview(unshaped.(samples).v), 1)
    shaped_idxs = vs(flat_idxs)

    for el in vsel 

        # :a
        if el isa Symbol
            shape_acc = getproperty(vs, el)
            idx_acc = getproperty(shaped_idxs, el)

            shape = shape_acc.shape
            append!(sel_idxs, idx_acc isa Integer ? Int64[idx_acc] : vec(idx_acc))

        # :(a[1]) or :(a[1:4]) or :(a[1,2,3]) or :(a[1,2,:])
        elseif @capture(el, a_[args__])
            shape_acc = getproperty(vs, a)
            idx_acc = getproperty(shaped_idxs, a)

            if @capture(el, a_[s_:e_])
                shape = ArrayShape{Real}(length(s:e))
                append!(sel_idxs, vec(idx_acc[s:e]))
            elseif @capture(el, a_[dim_])
                shape = ScalarShape{Real}()
                append!(sel_idxs, idx_acc[dim])
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

                new_dims = shape_acc.shape.dims .* whole_dims .+ slice_dims

                new_axes = new_dims[whole_dims .| sliced_dims]
        
                shape = length(new_axes) > 0 ? ArrayShape{Real}(new_axes...) : ScalarShape{Real}()

                idxs = idx_acc[args...]
                append!(sel_idxs, ndims(idxs) > 0 ? vec(idxs) : idxs)

            end
        end
        el = encode_name(el)
        append!(shapes, (el = shape,))
    end
    vsel = encode_name.(vsel)
    new_shape = NamedTupleShape(NamedTuple(vsel .=> shapes))

    return sel_idxs, new_shape
end 


# Encode dim names to preserve indexing information 
#
# unicode encodings to preserve indexing information:
# :a⌞2⌟     = : a \llcorner 2 \lrcorner 
# :a⌞2ː4⌟   = : a \llcorner 2 \lmrk 4 \lrcorner 
# :a⌞1ˌ3ˌ5⌟ = : a \llcorner 1 \verti 3 \verti 5 \lrcorner

function encode_name(name::Expr)
    code = replace(string(name)," " => "", "[" => "⌞", ":" => "ː", "," => "ˌ", "]" => "⌟")     
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
function bat_marginalize(samples::DensitySampleVector, 
                         vsel
)
    shaped = isshaped(samples)
    vs = varshape(samples)
    vsel = vsel isa Tuple ? [vsel...] : vsel

    if shaped
       vsel = _normalize_vsel_shaped(vs, vsel)
    else
       vsel = _normalize_vsel_unshaped(vsel)
    end

    if shaped
        flat_idxs, new_shape = marg_idxs_shaped(samples, vsel)
    else
        flat_idxs = marg_idxs_unshaped(samples, vsel)
    end

    flat_orig_data = ValueShapes.flatview(unshaped.(samples).v)
    marg_data = flat_orig_data[flat_idxs, :]
    marg_data = ndims(marg_data) > 1 ? nestedview(marg_data) : marg_data
    logd = fill(NaN, length(ndims(marg_data) > 1 ? axes(marg_data, 2) : marg_data))
    info = samples.info
    aux = samples.aux

    marg_samples = shaped ? new_shape.(DensitySampleVector(marg_data, logd, info = info, aux = aux)) : DensitySampleVector(marg_data, logd, info = info, aux = aux)
    return (result = marg_samples,)
end

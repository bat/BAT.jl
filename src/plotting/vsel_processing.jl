# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Parameter selections (`vsel`) select parameters by name (`:a`), path
# (`:(x.b.c)`) or element (`:(a[2])`, `:(a[1, :])`), or select free
# parameters by their position (an integer or a range of integers) in the
# standard-variate layout of the samples. Tuples and vectors of these, and
# expressions like `:([a, b[2]])` or `:(1:3)`, select several parameters.

# The selections in `vsel` as a vector of integers, symbols and expressions:
_normalize_vsel(vsel::Union{Integer,Symbol}) = Any[vsel]
_normalize_vsel(vsel::AbstractRange{<:Integer}) = Any[vsel...]
_normalize_vsel(vsel::QuoteNode) = _normalize_vsel(vsel.value)
_normalize_vsel(vsel::Union{Tuple,AbstractVector}) = reduce(vcat, map(_normalize_vsel, vsel); init = Any[])

function _normalize_vsel(vsel::Expr)
    if isexpr(vsel, :vect)
        return _normalize_vsel(vsel.args)
    elseif @capture(vsel, s_:e_) && s isa Integer && e isa Integer
        return _normalize_vsel(s:e)
    elseif isexpr(vsel, :call)
        throw(ArgumentError("Invalid parameter selection $vsel, expected a name, an indexed name, a range or a vector of these."))
    else
        return Any[vsel]
    end
end

_normalize_vsel(vsel) = throw(ArgumentError("Invalid parameter selection $vsel"))


# Positions of the free parameters selected by `vsel` in the standard-variate
# layout of `vs`, and the variate shape of the selection with names encoded
# via `encode_name`. Constants select no positions but keep their value:
function _resolve_vsel(vs::AbstractValueShape, vsel)
    names = all_active_names(vs)
    selections = map(sel -> _resolve_sel(vs, names, sel), _normalize_vsel(vsel))
    idxs = reduce(vcat, map(first, selections); init = Int[])
    sel_shape = NamedTupleShape(NamedTuple{(map(sel -> sel[2], selections)...,)}(map(last, selections)))
    return idxs, sel_shape
end

function _resolve_sel(::AbstractValueShape, names::AbstractVector{<:AbstractString}, sel::Integer)
    checkbounds(Bool, names, sel) || throw(ArgumentError("Parameter position $sel is out of bounds, there are $(length(names)) free parameters"))
    return Int[sel], encode_name(names[sel]), ScalarShape{Real}()
end

function _resolve_sel(vs::NamedTupleShape, ::AbstractVector{<:AbstractString}, sel::Union{Symbol,Expr})
    idxs, sel_shape = _select_params(vs, sel)
    return idxs, encode_name(sel), sel_shape
end

_resolve_sel(vs::AbstractValueShape, ::AbstractVector{<:AbstractString}, sel::Union{Symbol,Expr}) =
    throw(ArgumentError("Variates of shape $vs have no named parameters, select $sel by position instead"))


# Values of the free parameters at positions `idxs` of all samples, one row
# per parameter:
function _flat_params(samples::DensitySampleVector, idxs)
    X = flatbatch(samples.v)
    return X[firstindex(X, 1) - 1 .+ idxs, :]
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

The logd of the new samples is set to NaN, the weight, info and aux fields of `samples` are copied.

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
function bat_marginalize(samples::DensitySampleVector, vsel)
    vs = varshape(samples)
    flat_idxs, sel_shape = _resolve_vsel(vs, vsel)
    @argcheck !isempty(flat_idxs) throw(ArgumentError("The selected parameters for the input data to be marginalized to yielded only constant Values, please select at least one free parameter."))

    marg_data = _flat_params(samples, flat_idxs)
    v = vs isa NamedTupleShape ? unflatbatch(sel_shape, marg_data) : nestedview(marg_data)
    logd = fill!(similar(samples.logd), NaN)
    marg_samples = DensitySampleVector((v, logd, copy(samples.weight), copy(samples.info), copy(samples.aux)))
    return (result = marg_samples,)
end




function isshaped(samples::Union{DensitySampleVector, NamedTupleDist, ReshapedDist})
    
    isa(varshape(samples), NamedTupleShape) ? (return true) : (return false)
end



function _normalize_vsel_unshaped(vsel::Union{T, AbstractVector{T}, AbstractVector{UnitRange{T}}}) where T <: Integer

    return vsel isa Integer ? Int[vsel] : vsel
end

function _normalize_vsel_unshaped(vsel::AbstractVector)

    @argcheck all(broadcast(x -> x isa Union{T, AbstractVector{T}, AbstractVector{UnitRange{T}}} where T <: Integer, vsel)) throw(ArgumentError("Samples or distribution are unshaped, please use Integer, Vector of Integers, or UnitRange for indexing."))

    return vsel
end


function _normalize_vsel_shaped(vs::AbstractValueShape, vsel::Union{T, AbstractVector{T}, AbstractVector{UnitRange{T}}}) where T <: Integer

    if vsel isa AbstractVector{UnitRange{T}}    

        @argcheck length(vsel) == 1 throw(ArgumentError("Can't use multidimensional cartesian index for shaped samples or distribution, please use keys instead"))

        return [key for key in keys(vs)[vsel[1]]]

    elseif vsel isa Integer

        return [keys(vs)[vsel]]
    else
        
        return [keys(vs)[i] for i in vsel]
    end
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
            append!(sel_idxs, vec(idx_acc))

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

                new_axes = new_dims[whole_dims .|| sliced_dims]
        
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



function encode_name(name::Expr)

    name = string(name)
    code = ""

    for char in name
        if char == '['
            code *= '⌞'
        elseif char == ':'
            code *= 'ː'
        elseif char == ','
            code *= 'ˌ'
        elseif char == ']'
            code *= '⌟'
        elseif char == ' '
            code = code
        else
            code *= char
        end
    end
            
    return Symbol(code)
end 

function encode_name(name::Symbol)

    return name
end



function bat_marginalize(samples::DensitySampleVector, 
                         vsel::Union{E, S, I, U, AbstractVector{Union{E, S, I, U}}} where E<:Expr where S<:Symbol where I<:Integer where U<:UnitRange # maybe just use Any
)

    shaped = isshaped(samples)
    vs = varshape(samples)

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

    return marg_samples
end

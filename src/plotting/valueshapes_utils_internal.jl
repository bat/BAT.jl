function isshaped(samples::BAT.DensitySampleVector)
    isa(varshape(samples), NamedTupleShape) ? (return true) : (return false)
end

function asindex(samples::BAT.DensitySampleVector, name::Union{Expr, Symbol})
    if isshaped(samples)
        return asindex(varshape(samples), name)
    else
        throw(ArgumentError("Samples are unshaped. Key :$name cannot be matched. Use index instead."))
    end
end


function asindex(
    x::Union{DensitySampleVector, NamedTupleDist, MarginalDist},
    key::Integer
)
    return key
end

#for MarginalDist
function asindex(marg::MarginalDist, name::Union{Expr, Symbol})
    idx = asindex(marg.origvalshape, name)
    if idx in marg.dims
        return idx
    else
        throw(ArgumentError("Key :$name not in MarginalDist"))
    end
end


function getstring(samples::BAT.DensitySampleVector, idx::Integer)
    if isshaped(samples)
        vs = varshape(samples)
        names = all_active_names(vs)
        return names[idx]
    else
        throw(ArgumentError("Samples are unshaped. Key :$name cannot be matched. Use index instead."))
    end
end

function getstring(marg::MarginalDist, idx::Integer)
    if idx in marg.dims
        vs = marg.origvalshape
        names = all_active_names(vs)
        return names[idx]
    else
        throw(ArgumentError("Index $idx not in MarginalDist"))
    end
end

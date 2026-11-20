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

asindex(marg::MarginalDist, name::Union{Expr, Symbol}) = asindex(varshape(marg), name)


# Name of the free parameter at position `idx`:
function getstring(x::Union{DensitySampleVector, MarginalDist}, idx::Integer)
    names = all_active_names(varshape(x))
    checkbounds(Bool, names, idx) || throw(ArgumentError("Parameter position $idx is out of bounds, there are $(length(names)) free parameters"))
    return names[idx]
end

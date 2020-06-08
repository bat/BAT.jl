# Returns index corresponding to the name in NamedTupleDist.
# If the name corresponds to a multivariate distribution,
# an array containig all indices of this distribution is returned.
function asindex(vs::NamedTupleShape, name::Symbol)
    name = string(name)
    if name in (rn = repeatednames(vs))
        indxs = findall(x-> x == string(name), rn)
        length(indxs) > 1 ? (return indxs) : (return indxs[1])
    else
        throw(ArgumentError("Symbol :$name found not in $vs."))
    end
end


# Returns index corresponding to the name in NamedTupleDist.
function asindex(vs::NamedTupleShape, name::Expr)
    name = string(name)
    if name in (an = allnames(vs))
        return findfirst(x-> x == name, an)[1]
    else
        throw(ArgumentError("Symbol :$name found not in $vs."))
    end
end

# for samples
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

# for prior
function asindex(ntd::NamedTupleDist, name::Union{Expr, Symbol})
    return asindex(varshape(ntd), name)
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


# Return the name corresponding to the index as Symbol (for univariate)
# or Expr for (multivariate) distributions
function getname(vs::NamedTupleShape, idx::Integer)
    names = allnames(vs)
    return Meta.parse(names[idx])
end

# Return the name corresponding to the index as String
function getstring(vs::NamedTupleShape, idx::Integer)
    names = allnames(vs)
    return names[idx]
end

function getstring(prior::NamedTupleDist, idx::Integer)
    vs = varshape(prior)
    names = allnames(vs)
    return names[idx]
end


function getstring(samples::BAT.DensitySampleVector, idx::Integer)
    if isshaped(samples)
        vs = varshape(samples)
        names = allnames(vs)
        return names[idx]
    else
        throw(ArgumentError("Samples are unshaped. Key :$name cannot be matched. Use index instead."))
    end
end

function getstring(marg::MarginalDist, idx::Integer)
    println(marg.dims)
    if idx in marg.dims
        vs = marg.origvalshape
        names = allnames(vs)
        return names[idx]
    else
        throw(ArgumentError("Index $idx not in MarginalDist"))
    end
end

# Return array of strings with the names of all indices.
# For a multivariate distribution, names for each dimension are created by appending "[i]" to the name.
function allnames(vs::NamedTupleShape)
    subindxs = [collect(1:getproperty(vs, k).len) for k in keys(vs)]
    names = [length(subindxs[i]) > 1 ? subname.(string(keys(vs)[i]), subindxs[i]) : [string(keys(vs)[i])] for i in 1:length(vs)]

    return reduce(vcat, names)
end


# Return array of strings with the names of all indices.
# For a multivariate distribution, the name is repeated for all its dimensions.
function repeatednames(vs::NamedTupleShape)
    names = [[string(k) for i in 1:getproperty(vs, k).len] for k in keys(vs)]
    return reduce(vcat, names)
end

# Generate sub-name for one dimension of a multivariate distribution
function subname(name::String, indx::Integer)
    return name*"["*string(indx)*"]"
end

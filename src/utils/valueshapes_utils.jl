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
    if name in (an = all_active_names(vs))
        return findfirst(x-> x == name, an)[1]
    else
        throw(ArgumentError("Symbol :$name found not in $vs."))
    end
end


# for prior
function asindex(ntd::NamedTupleDist, name::Union{Expr, Symbol})
    return asindex(varshape(ntd), name)
end

# Return the name corresponding to the index as Symbol (for univariate)
# or Expr for (multivariate) distributions
function getname(vs::NamedTupleShape, idx::Integer)
    names = all_active_names(vs)(vs)
    return Meta.parse(names[idx])
end

# Return the name corresponding to the index as String
function getstring(vs::NamedTupleShape, idx::Integer)
    names = all_active_names(vs)
    return names[idx]
end

function getstring(prior::NamedTupleDist, idx::Integer)
    vs = varshape(prior)
    names = all_active_names(vs)
    return names[idx]
end


# Return array of strings with the names of all indices.
# For a multivariate distribution, names for each dimension are created by appending "[i]" to the name.
function allnames(vs::NamedTupleShape)
    subindxs = [collect(1:getproperty(vs, k).len) for k in keys(vs)]
    names = [length(subindxs[i]) > 1 ? subname.(string(keys(vs)[i]), subindxs[i]) : [string(keys(vs)[i])] for i in 1:length(vs)]

    return reduce(vcat, names)
end

function all_active_names(vs::NamedTupleShape)
    subindxs = [collect(1:getproperty(vs, k).len) for k in keys(vs)]
    names = [length(subindxs[i]) > 1 ? subname.(string(keys(vs)[i]), subindxs[i]) : [string(keys(vs)[i])] for i in 1:length(vs) if getproperty(vs, keys(vs)[i]).len>0]

    return reduce(vcat, names)
end

function all_active_names(vs::ScalarShape)
    return "v"
end

function all_active_names(vs::ArrayShape)
    ndims = totalndof(vs)
    return ["v_$i" for i in 1:ndims]
end

# Return array of strings with the names of all indices.
# For a multivariate distribution, the name is repeated for all its dimensions.
function repeatednames(vs::NamedTupleShape)
    names = [[string(k) for i in 1:getproperty(vs, k).len] for k in keys(vs) if getproperty(vs, k).len>0]
    return reduce(vcat, names)
end

# Generate sub-name for one dimension of a multivariate distribution
function subname(name::String, indx::Integer)
    return name*"["*string(indx)*"]"
end


get_fixed_names(vs::AbstractValueShape) = Vector{String}()

function get_fixed_names(vs::NamedTupleShape)
    active_names = all_active_names(vs)
    all_names = allnames(vs)
    fixed_names = [n for n in all_names if !in(n, active_names)]
end


function _trim(a::Array{T})::Tuple{Array{Int64},Array{Int64}} where {T<:AbstractFloat}

	#Functions first calculates 68% central percentile then returns list of deleted
	#indicies (i.e. outside the 68%) and accepted indicies (i.e. inside 68% percentile).

    if length(a) < 3
        return (Vector{Int64}(), collect(range(1,stop=length(a))))
    end
    # 1-sigma range of a unit normal distribution
    lower = quantile(a, 0.15865)
    upper = quantile(a, 0.84135)
    indices = findall(x-> x >= lower && x <= upper, a)
    return (setdiff(range(1,stop=length(a)), indices), indices)
end

function trim(x::Array)
    deleteat!(x, _trim(x)[1])
    x
end

function trim(
    res::IntermediateResults{T},
    dotrimming::Bool)::Array{Int64, 1} where {T<:AbstractFloat}

    deleteids, remainingids = _trim(res.integrals)

    if dotrimming && length(remainingids) > 1
		@debug "Trimming integration results: $(length(deleteids)) entries out of $(length(res.integrals)) deleted"
        deleteat!(res.integrals, deleteids)
        deleteat!(res.volumeID, deleteids)
        res.Y = res.Y[:, remainingids]
	elseif dotrimming && length(remainingids) < 2
		@warn "Trimming is NOT successful. No entries will be deleted"
    end

    deleteids
end





function split_dataset(dataset::DataSet{T, I})::Tuple{DataSet{T, I}, DataSet{T, I}} where {T<:Real, I<:Integer}

    ds1 = DataSet(dataset.data[:, 1:2:end], dataset.logprob[1:2:end], dataset.weights[1:2:end], dataset.nsubsets, dataset.subsetsize)
    ds2 = DataSet(dataset.data[:, 2:2:end-1], dataset.logprob[2:2:end-1], dataset.weights[2:2:end-1], dataset.nsubsets, dataset.subsetsize)

    return ds1, ds2
end

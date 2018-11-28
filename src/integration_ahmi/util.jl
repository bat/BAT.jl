
function _trim(x::Array{T})::Tuple{Array{Int64},Array{Int64}} where {T<:AbstractFloat}
    ids = sortperm(x)
    distances = [x[ids[i+1]] - x[ids[i]] for i=1:length(x)-1]

    if length(x) < 3
        return Array{Int64, 1}(0), collect(1:length(x))
    end

    density = zeros(length(x))
    for i in eachindex(density)
        if i == 1
            density[i] = 0.25 / distances[1]
        elseif i == length(density)
            density[i] = 0.25 / distances[end]
        else
            density[i] = 1.0 / (distances[i-1] + distances[i])
        end
    end
    #normalize density
    density /= sum(density)

    #remove tails -> trim to 1σ
    cutoff = (1 - 0.6827) * 0.5

    prob = 0.0

    start = 1
    stop = length(x)

    for i in eachindex(x)
        prob += density[i]
        if prob >= cutoff
            start = max(1, i - 1)
            break
        end
    end

    j = length(x)
    prob = 0.0
    for i in eachindex(x)
        prob += density[j]
        if prob >= cutoff
            stop = min(length(x), j + 1)
            break
        end
        j -= 1
    end

    sort!(ids[[1:start..., stop:length(x)...]]), sort!(ids[[start+1:stop-1...]])
end


function trim(x::Array)
    deleteat!(x, _trim(x)[1])
    x
end

function trim(
    res::IntermediateResults{T},
    dotrimming::Bool)::Array{Int64, 1} where {T<:AbstractFloat}

    deleteids, remainingids = _trim(res.integrals)
    @log_msg LOG_DEBUG "Trimming integration results: $(length(deleteids)) entries out of $(length(res.integrals)) deleted"

    if dotrimming
        deleteat!(res.integrals, deleteids)
        deleteat!(res.volumeID, deleteids)
        res.Y = res.Y[:, remainingids]
    end

    deleteids
end





function split_dataset(dataset::DataSet{T, I})::Tuple{DataSet{T, I}, DataSet{T, I}} where {T<:Real, I<:Integer}
    N = dataset.N
    n = floor(Int64, N / 2)

    ds1 = DataSet(dataset.data[:, 1:n], dataset.logprob[1:n], dataset.weights[1:n], dataset.nsubsets, dataset.subsetsize)
    ds2 = DataSet(dataset.data[:, n+1:N], dataset.logprob[n+1:N], dataset.weights[n+1:N], dataset.nsubsets, dataset.subsetsize)

    return ds1, ds2
end

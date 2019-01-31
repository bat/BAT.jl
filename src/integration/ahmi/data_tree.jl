
function create_search_tree(
    dataset::DataSet{T, I},
    progressbar::Progress;
    mincuts::I = 8,
    maxleafsize::I = 200)::SpacePartitioningTree{T, I} where {T<:AbstractFloat, I<:Integer}

    sugg_cuts = (dataset.N / maxleafsize) ^ (1 / dataset.P)
    cuts = ceil(I, max(mincuts, sugg_cuts))

    recdepth = ceil(I, log(dataset.N / maxleafsize) / log(cuts))

    #define dimension list
    dimensionlist = if cuts > mincuts
        [i for i = 1:dataset.P]
    else
        [i for i = 1:recdepth]
    end

    leafsize = ceil(I, dataset.N / cuts^recdepth)
    @log_msg LOG_DEBUG "cuts $cuts\tleafsize $leafsize\tRec. Depth $recdepth"
    cutlist = zeros(T, 0)

    if recdepth > 0
        createleafs(dataset, progressbar, dimensionlist, cutlist, cuts, leafsize, 1)
    end

    dataset.partitioningtree = SpacePartitioningTree(cuts, leafsize, dimensionlist, length(dimensionlist), cutlist)
end

function createleafs(
    dataset::DataSet{T, I},
    progressbar::Progress,
    dimensionlist::Vector{I},
    cutlist::Vector{T},
    cuts::I,
    leafsize::I,
    StartID::I = 0) where {T<:AbstractFloat, I<:Integer}

    remainingRec::I = length(dimensionlist)

    thisLeaf::I = leafsize * cuts^(remainingRec - 1)
    bigLeaf::I = leafsize * cuts^remainingRec

    startInt::I = StartID
    stopInt::I = StartID + bigLeaf - 1
    if stopInt > dataset.N
        stopInt = dataset.N
    end

    sortID = sortperm(dataset.data[dimensionlist[1], startInt:stopInt])

    dataset.data[:, startInt:stopInt] = dataset.data[:, sortID.+startInt.-1]
    dataset.logprob[startInt:stopInt] = dataset.logprob[sortID.+startInt.-1]
    dataset.weights[startInt:stopInt] = dataset.weights[sortID.+startInt.-1]
    dataset.ids[startInt:stopInt] =     dataset.ids[sortID.+startInt.-1]
    dataset.sortids[startInt:stopInt] = dataset.sortids[sortID.+startInt.-1]

    @assert remainingRec >= 1
    start::I = 0
    stop::I = StartID - 1

    for i = 1:cuts
        if stop == dataset.N
            continue
        end

        start = stop + 1
        stop += thisLeaf
        if stop > dataset.N
            stop = dataset.N
        end

        push!(cutlist, dataset.data[dimensionlist[1], start])

        if remainingRec > 1
            createleafs(dataset, progressbar, dimensionlist[2:end], cutlist, cuts, leafsize, start)
        else
            next!(progressbar)
        end
    end
end

function search(
    dataset::DataSet{T, I},
    searchvol::HyperRectVolume{T},
    searchpoints::Bool = false)::SearchResult where {T<:AbstractFloat, I<:Integer}

    res = SearchResult(T, I)
    #searchpoints = false
    search!(res, dataset, searchvol, searchpoints)

    res
end

function search!(
    result::SearchResult{T, I},
    dataset::DataSet{T, I},
    searchvol::HyperRectVolume{T},
    searchpoints::Bool = false) where {T<:AbstractFloat, I<:Integer}

    result.points = 0
    resize!(result.pointIDs, 0)
    result.maxLogProb = -Inf
    result.minLogProb = Inf
    result.maxWeightProb = -Inf
    result.minWeightProb = Inf

    currentRecursion::I = 0
    currentDimension::I = 0
    treePos = zeros(I, 0)

    maxRecursion::I = dataset.partitioningtree.recursiondepth
    maxI::I = length(dataset.partitioningtree.cutlist)

    i::I = 1
    if maxI == 0
        #only on leaf
        for n::I = 1:dataset.N
            inVol = true
            for p::I = 1:dataset.P
                if dataset.data[p, n] < searchvol.lo[p] || dataset.data[p, n] > searchvol.hi[p]
                    inVol = false
                    break
                end
            end

            if inVol
                result.points += 1

                if searchpoints
                    push!(result.pointIDs, n)
                end
                prob = dataset.logprob[n]
                w = log(dataset.weights[n]) + prob
                result.maxLogProb = max(result.maxLogProb, prob)
                result.minLogProb = min(result.minLogProb, prob)
                result.maxWeightProb = max(result.maxWeightProb, w)
                result.minWeightProb = min(result.minWeightProb, w)
            end
        end
        return

    end
    while i <= maxI
        if currentRecursion < maxRecursion
            currentRecursion += 1
        end

        if length(treePos) < currentRecursion
            push!(treePos, 1)
        else
            treePos[currentRecursion] += 1
        end
        while treePos[currentRecursion] > dataset.partitioningtree.cuts
            deleteat!(treePos, currentRecursion)
            currentRecursion -= 1
            treePos[currentRecursion] += 1
        end
        currentDimension = dataset.partitioningtree.dimensionlist[currentRecursion]


        diff::I = 1
        for r::I = 1:(maxRecursion - currentRecursion)
            diff += dataset.partitioningtree.cuts^r
        end
        low::T = dataset.partitioningtree.cutlist[i]
        high::T = i + diff > maxI ? dataset.partitioningtree.cutlist[end] : dataset.partitioningtree.cutlist[i+diff]
        if treePos[currentDimension] == dataset.partitioningtree.cuts
            high = Inf
        end


        if searchvol.lo[currentDimension] > high || searchvol.hi[currentDimension] < low
            #skip this interval
            i += diff
            if currentDimension < maxRecursion
                currentRecursion -= 1
            end
            continue
        end

        #if on deepest recursion check for points
        if currentRecursion == maxRecursion
            startID, stopID = getDataPositions(dataset, treePos)

            searchInterval!(result, dataset, searchvol, startID, stopID, searchpoints)
        end
        i += 1
    end

end

@inline function getDataPositions(
    dataset::DataSet{T, I},
    TreePos::Vector{I}) where {T<:AbstractFloat, I<:Integer}

    maxRecursion = dataset.partitioningtree.recursiondepth
    startID = 1
    recCntr = maxRecursion
    for t in TreePos
        recCntr -= 1
        startID += dataset.partitioningtree.leafsize * dataset.partitioningtree.cuts^recCntr * (t-1)
    end
    stopID = startID + dataset.partitioningtree.leafsize - 1
    if stopID > dataset.N
        stopID = dataset.N
    end

    return startID, stopID
end


function searchInterval!(
    result::SearchResult{T, I},
    dataset::DataSet{T, I},
    searchvol::HyperRectVolume{T},
    start::I,
    stop::I,
    searchpoints::Bool) where {T<:AbstractFloat, I<:Integer}

    dimsort = dataset.partitioningtree.dimensionlist[dataset.partitioningtree.recursiondepth]

    for i = start:stop
        if dataset.data[dimsort, i] > searchvol.hi[dimsort]
            break
        end
        inVol = true
        for p = 1:dataset.P
            if dataset.data[p, i] < searchvol.lo[p] || dataset.data[p, i] > searchvol.hi[p]
                inVol = false
                break
            end
        end

        if inVol
            result.points += 1

            if searchpoints
                push!(result.pointIDs, i)
            end
            prob = dataset.logprob[i]
            w = log(dataset.weights[i]) + prob
            result.maxLogProb = max(result.maxLogProb, prob)
            result.minLogProb = min(result.minLogProb, prob)
            result.maxWeightProb = max(result.maxWeightProb, w)
            result.minWeightProb = min(result.minWeightProb, w)
        end
    end
end

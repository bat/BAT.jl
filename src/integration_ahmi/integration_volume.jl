
"""
    IntegrationVolume(dataset::DataSet{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)::IntegrationVolume{T, I}

creates an integration region by calculating the point cloud an the volume of the spatial volume.
"""
function IntegrationVolume(
    dataset::DataSet{T, I},
    spvol::HyperRectVolume{T},
    searchpts::Bool = true
)::IntegrationVolume{T, I} where {T<:AbstractFloat, I<:Integer}

    cloud = PointCloud(dataset, spvol, searchpts)
    vol = prod(spvol.hi - spvol.lo)

    IntegrationVolume(cloud, spvol, vol)
end


"""
    modify_integrationvolume!(intvol::IntegrationVolume{T, I}, dataset::DataSet{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)

updates an integration volume with new boundaries. Recalculates the pointcloud and volume.
"""
function modify_integrationvolume!(
    intvol::IntegrationVolume{T, I},
    dataset::DataSet{T, I},
    spvol::HyperRectVolume{T},
    searchpts::Bool = true
) where {T<:AbstractFloat, I<:Integer}

    if ndims(intvol.spatialvolume) != ndims(spvol)
        intvol.spatialvolume = deepcopy(spvol)
    else
        copy!(intvol.spatialvolume, spvol)
    end

    PointCloud!(intvol.pointcloud, dataset, spvol, searchpts)

    intvol.volume = prod(spvol.hi - spvol.lo)

    nothing
end

function shrink_integrationvol!(
    volume::IntegrationVolume{T, I},
    dataset::DataSet{T, I},
    newrect::HyperRectVolume{T}) where {T<:AbstractFloat, I<:Integer}

    i = volume.pointcloud.points
    for j = 1:i
        inV = true
        for p = 1:dataset.P
            if dataset.data[p, volume.pointcloud.pointIDs[i]] < newrect.lo[p] || dataset.data[p, volume.pointcloud.pointIDs[i]] > newrect.hi[p]
                inV = false
                break
            end
        end
        if !inV
            deleteat!(volume.pointcloud.pointIDs, i)
        end
        i -= 1
    end
    copy!(volume.spatialvolume, newrect)
    volume.pointcloud.points = length(volume.pointcloud.pointIDs)
    volume.volume = prod(newrect.hi .- newrect.lo)

    nothing
end

function update!(
    volume::IntegrationVolume{T, I},
    dataset::DataSet{T, I}) where {T<:AbstractFloat, I<:Integer}

    res = search(dataset, volume.spatialvolume, true)

    _update!(volume, res, false, true)

    #volume.volume doesn't change
    volume.pointcloud.probfactor = exp(volume.pointcloud.maxLogProb - volume.pointcloud.minLogProb)
    volume.pointcloud.probweightfactor = exp(volume.pointcloud.maxWeightProb - volume.pointcloud.minWeightProb)

    @log_msg LOG_DEBUG "Hyperrectangle updated. Points:\t$(volume.pointcloud.points)\tVolume:\t$(volume.volume)\tProb. Factor:\t$(volume.pointcloud.probfactor)"
    nothing
end

function _update!(
    volume::IntegrationVolume{T, I},
    searchres::SearchResult{T, I},
    addpts::Bool, #if true adds the points to the volume, if false assumes to replace the points in the volume
    searchpts::Bool) where {T<:AbstractFloat, I<:Integer}

    volume.pointcloud.points = searchres.points + (addpts ? volume.pointcloud.points : 0)

    volume.pointcloud.maxLogProb = volume.pointcloud.maxLogProb > searchres.maxLogProb && addpts ? volume.pointcloud.maxLogProb : searchres.maxLogProb
    volume.pointcloud.minLogProb = volume.pointcloud.minLogProb < searchres.minLogProb && addpts ? volume.pointcloud.minLogProb : searchres.minLogProb

    volume.pointcloud.maxWeightProb = volume.pointcloud.maxWeightProb > searchres.maxWeightProb && addpts ? volume.pointcloud.maxWeightProb : searchres.maxWeightProb
    volume.pointcloud.minWeightProb = volume.pointcloud.minWeightProb < searchres.minWeightProb && addpts ? volume.pointcloud.minWeightProb : searchres.minWeightProb

    if searchpts && searchres.points > 0
        start = addpts ? length(volume.pointcloud.pointIDs) + 1 : 1
        resize!(volume.pointcloud.pointIDs, volume.pointcloud.points)
        copyto!(volume.pointcloud.pointIDs, start, searchres.pointIDs, 1)
    end

    nothing
end

function resize_integrationvol(
    original::IntegrationVolume{T, I},
    dataset::DataSet{T, I},
    changed_dim::I,
    newrect::HyperRectVolume{T},
    searchpts::Bool = false)::IntegrationVolume{T, I} where {T<:AbstractFloat, I<:Integer}

    result = deepcopy(original)

    resize_integrationvol!(result, original, dataset, datatre, changed_dim, newrect, searchpts)
end

function resize_integrationvol!(
    result::IntegrationVolume{T, I},
    original::IntegrationVolume{T, I},
    dataset::DataSet{T, I},
    changed_dim::I,
    newrect::HyperRectVolume{T},
    searchpts::Bool,
    searchVol::HyperRectVolume{T}) where {T<:AbstractFloat, I<:Integer}

    copy!(searchVol, newrect)
    increase = true

    #increase
    if original.spatialvolume.lo[changed_dim] > newrect.lo[changed_dim]
        searchVol.hi[changed_dim] = original.spatialvolume.lo[changed_dim]
        searchVol.lo[changed_dim] = newrect.lo[changed_dim]
    elseif original.spatialvolume.hi[changed_dim] < newrect.hi[changed_dim]
        searchVol.lo[changed_dim] = original.spatialvolume.hi[changed_dim]
        searchVol.hi[changed_dim] = newrect.hi[changed_dim]
    else
        increase = false
        if original.spatialvolume.lo[changed_dim] < newrect.lo[changed_dim]
            searchVol.lo[changed_dim] = original.spatialvolume.lo[changed_dim]
            searchVol.hi[changed_dim] = newrect.lo[changed_dim]
        elseif original.spatialvolume.hi[changed_dim] > newrect.hi[changed_dim]
            searchVol.hi[changed_dim] = original.spatialvolume.hi[changed_dim]
            searchVol.lo[changed_dim] = newrect.hi[changed_dim]
        else
            #check if pts inside vol are corrected
            @log_msg LOG_ERROR "resize_integrationvol(): Volume $original didn't change.", searchVol.lo[changed_dim], "\n", searchVol.hi[changed_dim], "\n", original.spatialvolume.lo[changed_dim], "\n", original.spatialvolume.hi[changed_dim]
        end
    end


    result.pointcloud.points = original.pointcloud.points
    if searchpts
        if result.pointcloud.pointIDs != original.pointcloud.pointIDs
            result.pointcloud.pointIDs = deepcopy(original.pointcloud.pointIDs)
        end
    end

    if increase
        res = search(dataset, searchVol, searchpts)

        _update!(result, res, true, searchpts)
    else
        res = search(dataset, searchVol, searchpts)
        result.pointcloud.points = original.pointcloud.points - res.points
        if searchpts
            newids = search(dataset, newrect, searchpts).pointIDs
            resize!(result.pointcloud.pointIDs, result.pointcloud.points)
            copyto!(result.pointcloud.pointIDs, newids)
        end
    end

    result.volume = prod(newrect.hi - newrect.lo)
    result.pointcloud.probfactor = exp(result.pointcloud.maxLogProb - result.pointcloud.minLogProb)
    result.pointcloud.probweightfactor = exp(result.pointcloud.maxWeightProb - result.pointcloud.minWeightProb)
    copy!(result.spatialvolume, newrect)

    nothing
end

function Base.copy!(
    target::IntegrationVolume{T, I},
    src::IntegrationVolume{T, I}) where {T<:AbstractFloat, I<:Integer}

    target.volume = src.volume

    copy!(target.spatialvolume, src.spatialvolume)
    target.pointcloud.points = src.pointcloud.points

    copy!(target.pointcloud, src.pointcloud)

    nothing
end

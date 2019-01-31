
"""
    PointCloud{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, hyperrect::HyperRectVolume{T}, searchpts::Bool = false)::PointCloud

creates a point cloud by searching the data tree for points which are inside the hyper-rectangle
The parameter searchpts determines if an array of the point IDs is created as well
"""
function PointCloud(
    dataset::DataSet{T, I},
    hyperrect::HyperRectVolume{T},
    searchpts::Bool)::PointCloud{T, I} where {T<:AbstractFloat, I<:Integer}

    result = PointCloud(T, I)

    PointCloud!(result, dataset, hyperrect, searchpts)

    result
end


function PointCloud!(
    cloud::PointCloud{T, I},
    dataset::DataSet{T, I},
    hyperrect::HyperRectVolume{T},
    searchpts::Bool) where {T<:AbstractFloat, I<:Integer}

    search!(cloud.searchres, dataset, hyperrect, searchpts)

    cloud.points = cloud.searchres.points

    resize!(cloud.pointIDs, cloud.points)
    copyto!(cloud.pointIDs, cloud.searchres.pointIDs)

    cloud.maxWeightProb = cloud.searchres.maxWeightProb
    cloud.minWeightProb = cloud.searchres.minWeightProb
    cloud.maxLogProb = cloud.searchres.maxLogProb
    cloud.minLogProb = cloud.searchres.minLogProb

    cloud.probfactor = exp(cloud.maxLogProb - cloud.minLogProb)
    cloud.probweightfactor = exp(cloud.maxWeightProb - cloud.minWeightProb)

    nothing
end

function create_pointweights(
    dataset::DataSet{T, I},
    volumes::Vector{IntegrationVolume{T, I, V}},
    ids::Array{I})::Vector{T} where {T<:AbstractFloat, I<:Integer, V<:SpatialVolume}

    pweights = zeros(T, dataset.N)

    for i in ids
        for p in eachindex(volumes[i].pointcloud.pointIDs)
            pweights[volumes[i].pointcloud.pointIDs[p]] += 1
        end
    end

    pweights
end

function Base.copy!(
    target::PointCloud{T, I},
    src::PointCloud{T, I}) where {T<:AbstractFloat, I<:Integer}

    target.points = src.points

    resize!(target.pointIDs, length(src.pointIDs))
    copyto!(target.pointIDs, src.pointIDs)

    target.maxLogProb = src.maxLogProb
    target.minLogProb = src.minLogProb

    target.maxWeightProb = src.maxWeightProb
    target.minWeightProb = src.minWeightProb

    target.probfactor = src.probfactor
    target.probweightfactor = src.probweightfactor

    nothing
end

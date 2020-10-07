

mutable struct SpacePartitioningTree{
    T<:AbstractFloat,
    I<:Integer}

    cuts::I
    leafsize::I

    dimensionlist::Vector{I}
    recursiondepth::I
    cutlist::Vector{T}
end
SpacePartitioningTree(T::DataType, I::DataType) = SpacePartitioningTree{T, I}(zero(I), zero(I), zeros(I, 0), zero(I), zeros(I, 0))
isinitialized(x::SpacePartitioningTree) = !(iszero(x.cuts) && iszero(x.leafsize) && isempty(x.dimensionlist) && iszero(x.recursiondepth) && isempty(x.cutlist))


"""
    DataSet{T<:AbstractFloat, I<:Integer}

*AHMI-internal, not part of stable public API.*

Holds the MCMC output. For construction use constructor: function DataSet{T<:Real}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})
# Variables
- 'data' : An P x N array with N data points with P parameters.
- 'logprob' : The logarithmic probability for each samples stored in an array
- 'weights' : How often each sample occurred. Set to an array of ones if working directly on MCMC output
- 'ids' : Array which is used to assign each sample to a batch, required for the cov. weighed uncertainty estimation
- .sortids : an array of indices which stores the original ordering of the samples (the space partitioning tree reorders the samples), required to calculate an effective sample size.
- 'N' : number of samples
- 'P' : number of parameters
- 'nsubsets' : the number of batches
- 'iswhitened' : a boolean value which indicates whether the data set is iswhitened
- 'isnew' : a boolean value which indicates whether the data set was swapped out with a different one (it is possible to redo the integration with a different sample set using previously generated hyper-rectangles)
- 'partitioningtree' : The space partitioning tree, used to efficiently identify samples in a point cloud
- 'startingIDs' : The Hyper-Rectangle Seed Samples are stored in this array
- 'tolerance' : A threshold required for the hyper-rectangle creation process.
"""
mutable struct DataSet{T<:AbstractFloat, I<:Integer}
    data::Array{T, 2}
    logprob::Array{T, 1}
    weights::Array{T, 1}
    ids::Array{I, 1}    #used to divide the dataset into sub-sets
    sortids::Array{I, 1}#used to calculate the ess on the unsorted dataset
    N::I
    P::I
    nsubsets::I    #number of sub-sets
    subsetsize::T
    iswhitened::Bool
    isnew::Bool
    partitioningtree::SpacePartitioningTree
    startingIDs::Array{I, 1}
    tolerance::T
end


function DataSet(
    data:: Array{T, 2},
    logprob::Array{T, 1},
    weights::Array{I, 1},
    nsubsets::Integer = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, Int} where {T<:AbstractFloat, I<:Integer}

    DataSet(data, logprob, convert(Array{T, 1}, weights), nsubsets, subsetsize)
end

function DataSet(
    data:: Array{T, 2},
    logprob::Array{T, 1},
    weights::Array{T, 1},
    nsubsets::Integer = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, Int} where {T<:AbstractFloat}

    P, N = size(data)

    if iszero(nsubsets)
        nsubsets = 10
    end

    maxbatchsize = sum(weights) / 10 / nsubsets
    if iszero(subsetsize)
        subsetsize = 100.0
    end
    subsetsize = minimum(typeof(maxbatchsize), (maxbatchsize, subsetsize))

    ids = zeros(Int, N)
    cnt = 1

    batch_currentsize = 0.0

    for i = 1:N
        ids[i] = cnt
        batch_currentsize += weights[i]

        if batch_currentsize >= subsetsize
            cnt += 1
            if cnt > nsubsets
                cnt = 1
            end
            batch_currentsize = 0.0
        end
    end
    DataSet(data, logprob, weights, ids, [i for i=1:N], N, P, nsubsets, subsetsize, false, true, SpacePartitioningTree(T, Int), zeros(Int, 0), T(0))
end

Base.show(io::IO, data::DataSet) = print(io, "DataSet: $(data.N) samples, $(data.P) parameters")
Base.eltype(data::DataSet{T, I}) where {T<:AbstractFloat, I<:Integer} = (T, I)

"""
    HMISettings

*AHMI-internal, not part of stable public API.*

holds the settings for the hm_integrate function. There are several default constructors available:
HMIFastSettings()
HMIStandardSettings()
HMIPrecisionSettings()

#Variables
- 'whitening_method::Symbol' : which whitening method to use
- 'max_startingIDs::Integer' : influences how many starting ids are allowed to be generated
- 'max_startingIDs_fraction::AbstractFloat' : how many points are considered as possible starting points as a fraction of total points available
- 'rect_increase::AbstractFloat' : describes the procentual rectangle volume increase/decrease during hyperrectangle creation. Low values can increase the precision if enough points are available but can cause systematically wrong results if not enough points are available.
- 'use_all_rects::Bool' : All rectangles are used for the integration process no matter how big their overlap is. If enabled the rectangles are weighted by their overlap.
- 'useMultiThreading' : activate multithreading support.
- 'warning_minstartingids' : the required minimum amount of starting samples
- 'dotrimming' : determines whether the integral estimates are trimmed (1σ trim) before combining them into a final result (more robust)
- 'uncertainty_estimators' : A dictionary of different uncertainty estimator functions. Currently three functions are available: hm_combineresults_legacy! (outdated, overestimates uncertainty significantly in higher dimensions), hm_combineresults_covweighted! (very fast) and hm_combineresults_analyticestimation! (recommended)
end
"""
mutable struct HMISettings
    whitening_function!::Function
    max_startingIDs::Integer
    max_startingIDs_fraction::AbstractFloat
    rect_increase::AbstractFloat
    useMultiThreading::Bool
    warning_minstartingids::Integer
    dotrimming::Bool
    uncertainty_estimators::Dict{String, Function}
end
HMIFastSettings() =      return HMISettings(cholesky_whitening!, 100,   0.1, 0.1, true, 16, true, Dict("cov. weighted result" => hm_combineresults_covweighted!))
HMIStandardSettings() =  return HMISettings(cholesky_whitening!, 1000,  0.5, 0.1, true, 16, true, Dict("cov. weighted result" => hm_combineresults_covweighted!, "analytic result" => hm_combineresults_analyticestimation!))
HMIPrecisionSettings() = return HMISettings(cholesky_partial_whitening!, 10000, 2.5, 0.1, true, 16, true, Dict("cov. weighted result" => hm_combineresults_covweighted!))
# HMIPrecisionSettings() = return HMISettings(cholesky_whitening!, 10000, 2.5, 0.1, true, 16, true, Dict("cov. weighted result" => hm_combineresults_covweighted!, "analytic result" => hm_combineresults_analyticestimation!))

"""
    WhiteningResult{T<:AbstractFloat}

*AHMI-internal, not part of stable public API.*

Stores the information obtained during the Whitening Process
# Variables
- 'determinant' : The determinant of the whitening matrix
- 'targetprobfactor' : The suggested target probability factor
- 'whiteningmatrix' : The whitening matrix
- 'meanvalue' : the mean vector of the input data
"""
struct WhiteningResult{T<:AbstractFloat}
    determinant::T
    targetprobfactor::T
    whiteningmatrix::Matrix{T}
    meanvalue::Vector{T}
end
WhiteningResult(T::DataType) = WhiteningResult(zero(T), zero(T), zeros(T, 0, 0), zeros(T, 0))
Base.show(io::IO, wres::WhiteningResult) = print(io, "Whitening Result: Determinant: $(wres.determinant), Target Prob. Factor: $(wres.targetprobfactor)")
isinitialized(x::WhiteningResult) = !(iszero(x.determinant) && iszero(x.targetprobfactor) && isempty(x.whiteningmatrix) && isempty(x.meanvalue))

"""
    SearchResult{T<:AbstractFloat, I<:Integer}

*AHMI-internal, not part of stable public API.*

Stores the results of the space partitioning tree's search function

# Variables
- 'pointIDs' : the IDs of samples found, might be empty because it is optional
- 'points' : The number of points found.
- 'maxLogProb' : the maximum log. probability of the points found.
- 'minLogProb' : the minimum log. probability of the points found.
- 'maxWeightProb' : the weighted minimum log. probability found.
- 'minWeightProb' : the weighted maximum log. probfactor found.
"""
mutable struct SearchResult{T<:AbstractFloat, I<:Integer}
    pointIDs::Vector{I}
    points::I
    maxLogProb::T
    minLogProb::T
    maxWeightProb::T
    minWeightProb::T
end

function SearchResult(T::DataType, I::DataType)
    @assert T<:AbstractFloat
    @assert I<:Integer
    return SearchResult{T, I}(zeros(I, 0), I(0), T(0), T(0), T(0), T(0))
end
Base.show(io::IO, sres::SearchResult) = print(io, "Search Result: Points: $(sres.points), Max. Log. Prob.: $(sres.maxLogProb), Min. Log. Prob.: $(sres.minLogProb)")


"""
    PointCloud{T<:AbstractFloat, I<:Integer}

*AHMI-internal, not part of stable public API.*

Stores the information of the points of an e.g. HyperRectVolume
# Variables
- 'maxLogProb' : The maximum log. probability of one of the points inside the hyper-rectangle
- 'minLogProb' : The minimum log. probability of one of the points inside the hyper-rectangle
- 'maxWeightProb' : the weighted max. log. probability
- 'minWeightProb' : the weighted min. log. probability
- 'probfactor' : The probability factor of the hyper-rectangle
- 'probweightfactor' : The weighted probability factor
- 'points' : The number of points inside the hyper-rectangle
- 'pointIDs' : the IDs of the points inside the hyper-rectangle, might be empty because it is optional and costs performance
- 'searchres' : used to boost performance
"""
mutable struct PointCloud{T<:AbstractFloat, I<:Integer}
    maxLogProb::T
    minLogProb::T
    maxWeightProb::T
    minWeightProb::T
    probfactor::T
    probweightfactor::T
    points::I
    pointIDs::Vector{I}
    searchres::SearchResult{T, I}
end
PointCloud(T::DataType, I::DataType) = PointCloud(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(I), zeros(I, 0), SearchResult(T, I))
Base.show(io::IO, cloud::PointCloud) = print(io, "Point Cloud with $(cloud.points) points, probability factor: $(cloud.probfactor)")


"""
    IntegrationVolume{T<:AbstractFloat, I<:Integer}

*AHMI-internal, not part of stable public API.*

# Variables
- 'pointcloud' : holds the point cloud of the integration volume
- 'spatialvolume' : the boundaries of the integration volume
- 'volume' : the volume

Hold the point cloud and the spatial volume for integration.
"""
mutable struct IntegrationVolume{T<:AbstractFloat, I<:Integer, V<:SpatialVolume}
    pointcloud::PointCloud{T, I}
    spatialvolume::V
    volume::T
end
Base.show(io::IO, vol::IntegrationVolume) = print(io, "Hyperrectangle: $(vol.pointcloud.points) points, $(vol.volume) Volume")



mutable struct IntermediateResults{T<:AbstractFloat}
    integrals::Array{T, 1}
    volumeID::Array{Int, 1}
    Y::Array{T, 2}
end
IntermediateResults(T::DataType, n::Integer) = IntermediateResults(zeros(T, n), [Int(i) for i=1:n], zeros(T, 0, 0))
Base.length(x::IntermediateResults) = length(x.integrals)

mutable struct HMIEstimate{T<:AbstractFloat}
    estimate::T
    uncertainty::T
    weights::Array{T, 1}
end
HMIEstimate(T::DataType) = HMIEstimate(zero(T), zero(T), zeros(T, 0))
function HMIEstimate(a::HMIEstimate{T}, b::HMIEstimate{T})::HMIEstimate{T} where {T<:AbstractFloat}
    val = mean([a.estimate, b.estimate], AnalyticWeights([1 / a.uncertainty^2, 1 / b.uncertainty^2]))
    unc = 1 / sqrt(1 / a.uncertainty^2 + 1 / b.uncertainty^2)
    HMIEstimate(val, unc, [a.weights..., b.weights...])
end
Base.show(io::IO, ires::HMIEstimate) = print(io, "$(round(ires.estimate, sigdigits=6))  +-  $(round(ires.uncertainty, sigdigits=6))")


mutable struct HMIResult{T<:AbstractFloat}
    result1::HMIEstimate{T}
    result2::HMIEstimate{T}
    final::HMIEstimate{T}
    dat1::Dict{String, Any}
    dat2::Dict{String, Any}
end
HMIResult(T::DataType) = HMIResult(HMIEstimate(T), HMIEstimate(T), HMIEstimate(T), Dict{String, Any}(), Dict{String, Any}())

"""
    HMIData{T<:AbstractFloat, I<:Integer}

*AHMI-internal, not part of stable public API.*

Includes all the informations of the integration process, including a list of hyper-rectangles, the results of the whitening transformation,
the starting ids, and the average number of points and volume of the created hyper-rectangles.

# Variables
- 'dataset1' : Data Set 1
- 'dataset2' : Data Set 2
- 'whiteningresult' : contains the whitening matrix and its determinant, required to scale the final integral estimate
- 'volumelist1' : An array of integration volumes created using dataset1, but filled with samples from dataset2
- 'volumelist2' : An array of integration volumes created using dataset2, but filled with samples from dataset1
- 'cubelist1' : An array of small hyper-cubes created around seeding samples of dataset 1
- 'cubelist2' : An array of small hyper-cubes created around seeding samples of dataset 2
- 'iterations1' : The number of volume adapting iterations for the creating volumelist1
- 'iterations2' : The number of volume adapting iterations for the creating volumelist2
- 'rejectedrects1' : An array of ids, indicating which hyper-rectangles of volumelist1 were rejected due to trimming
- 'rejectedrects2' : An array of ids, indicating which hyper-rectangles of volumelist2 were rejected due to trimming
- 'integralestimates' : A dictionary containing the final integral estimates with uncertainty estimation using different uncertainty estimators. Also includes all intermediate results required for the integral estimate combination
"""
mutable struct HMIData{T<:AbstractFloat, I<:Integer, V<:SpatialVolume}
    dataset1::DataSet{T, I}
    dataset2::DataSet{T, I}
    whiteningresult::WhiteningResult{T}
    volumelist1::Vector{IntegrationVolume{T, I, V}}
    volumelist2::Vector{IntegrationVolume{T, I, V}}
    cubelist1::Vector{V}
    cubelist2::Vector{V}
    iterations1::Vector{I}
    iterations2::Vector{I}
    rejectedrects1::Vector{I}
    rejectedrects2::Vector{I}
    integrals1::IntermediateResults{T}
    integrals2::IntermediateResults{T}
    integralestimates::Dict{String, HMIResult}
end

function HMIData(
    dataset1::DataSet{T, I},
    dataset2::DataSet{T, I},
    dataType::DataType = HyperRectVolume{T})::HMIData where {T<:AbstractFloat, I<:Integer}

    HMIData(
        dataset1,
        dataset2,
        WhiteningResult(T),
        Vector{IntegrationVolume{T, I, dataType}}(undef, 0),
        Vector{IntegrationVolume{T, I, dataType}}(undef, 0),
        Vector{dataType}(undef, 0),
        Vector{dataType}(undef, 0),
        zeros(I, 0),
        zeros(I, 0),
        zeros(I, 0),
        zeros(I, 0),
        IntermediateResults(T, 0),
        IntermediateResults(T, 0),
        Dict{String, HMIResult}()
    )
end

function HMIData(dataset::DataSet{T, I})::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}
    HMIData(split_dataset(dataset)...)
end


@deprecate HMIData(bat_samples::Tuple{DensitySampleVector, MCMCBasicStats, AbstractVector{<:MCMCIterator}}) HMIData(bat_samples[1])

function HMIData(samples::DensitySampleVector)
    logprob = samples.logd
    weights = samples.weight
    samples = convert(Array{eltype(logprob), 2}, flatview(samples.v))
    ds = DataSet(samples, logprob, weights)
    HMIData(ds)
end



function Base.show(io::IO, ires::HMIData)
    output = "Parameters: $(ires.dataset1.P)\tTotal Samples: $(ires.dataset1.N + ires.dataset2.N)"
    output *= "\nData Set 1: $(length(ires.volumelist1)) Volumes\nData Set 2: $(length(ires.volumelist2)) Volumes"

    for pair in ires.integralestimates
        output *= "\n\nIntegral Estimate ($(pair[1])):\n\t $(pair[2].final)"
    end

    println(io, output)
end

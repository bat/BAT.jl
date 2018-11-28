
"""
    create_hypercube{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume

creates a hypercube shaped spatial volume
"""
function create_hypercube(
    origin::Vector{T},
    edgelength::T)::HyperRectVolume{T} where {T<:AbstractFloat}

    dim = length(origin)
    lo = zeros(T, dim)
    hi = zeros(T, dim)

    _setcubeboundaries!(lo, hi, origin, edgelength)

    HyperRectVolume(lo, hi)
end

"""
    create_hypercube!{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume

resizes a hypercube shaped spatial volume
"""
function modify_hypercube!(
    rect::HyperRectVolume{T},
    neworigin::Vector{T},
    newedgelength::T) where {T<:AbstractFloat}

    _setcubeboundaries!(rect.lo, rect.hi, neworigin, newedgelength)

    nothing
end

@inline function _setcubeboundaries!(
    lo::Vector{T},
    hi::Vector{T},
    origin::Vector{T},
    edgelength::T) where {T<:AbstractFloat}

    for i = 1:length(lo)
        lo[i] = origin[i] - edgelength * 0.5
        hi[i] = origin[i] + edgelength * 0.5
    end

    nothing
end

"""
    find_hypercube_centers(dataset::DataSet{T, I}, whiteningresult::WhiteningResult, settings::HMISettings)::Vector{I}

finds possible starting points for the hyperrectangle creation
"""
function find_hypercube_centers(
    dataset::DataSet{T, I},
    whiteningresult::WhiteningResult{T},
    settings::HMISettings)::Bool where {T<:AbstractFloat, I<:Integer}

    sortLogProb = sortperm(dataset.logprob, rev = true)

    NMax = min(settings.max_startingIDs, round(I, sqrt(dataset.N * settings.max_startingIDs_fraction)))
    NConsidered = round(I, sqrt(dataset.N) * settings.max_startingIDs_fraction)
    @log_msg LOG_DEBUG "Considered seed samples $NConsidered"

    discardedsamples = falses(dataset.N)

    testlength = find_density_test_cube_edgelength(dataset.data[:, sortLogProb[1]], dataset, round(I, sqrt(dataset.N)))
    @log_msg LOG_DEBUG "Edge length of global mode cube: $testlength"

    maxprob = dataset.logprob[sortLogProb[1]]
    startingsamples = zeros(I, NMax)
    cntr = 0
    @showprogress for n::I in sortLogProb[1:NConsidered]
        if discardedsamples[n]
            continue
        end
        if cntr == NMax || dataset.logprob[n] < maxprob - log(whiteningresult.targetprobfactor)
            break
        end

        cntr += 1
        startingsamples[cntr] = n

        cubevol = create_hypercube(dataset.data[:, n], testlength)
        cube = IntegrationVolume(dataset, cubevol, true)

        discardedsamples[cube.pointcloud.pointIDs] .= true
    end
    resize!(startingsamples, cntr)

    success = true
    if cntr < settings.warning_minstartingids
        stop = floor(I, length(sortLogProb) * 0.2)
        step = floor(I, stop / settings.warning_minstartingids)
        if step == 0 step = 1 end
        startingsamples = sortLogProb[1:step:stop]
        success = false
        @log_msg LOG_WARNING "Returned minimum number of starting points: $(settings.warning_minstartingids)"
    end


    @log_msg LOG_DEBUG "Selected Starting Samples: $cntr out of $(dataset.N) points"
    dataset.startingIDs = startingsamples

    success
end

function find_density_test_cube_edgelength(
    mode::Vector{T},
    dataset::DataSet{T, I},
    points::I = 100)::T where {T<:AbstractFloat, I<:Integer}

    find_density_test_cube(mode, dataset, points)[1]
end

function find_density_test_cube(
    mode::Vector{T},
    dataset::DataSet{T, I},
    points::I)::Tuple{T, IntegrationVolume{T, I}} where {T<:AbstractFloat, I<:Integer}

    P = dataset.P

    l::T = 1.0
    tol::T = 1.0
    mult::T = 2.0^(1.0 / P)
    last_change = 0

    rect = create_hypercube(mode, l)
    intvol = IntegrationVolume(dataset, rect, false)
    pt = intvol.pointcloud.points

    iterations = 0
    while pt < points / tol || pt > points * tol
        iterations += 1
        tol += 0.001 * iterations
        if pt > points
            l /= mult
            mult = last_change == -1 ? mult^2.0 : mult^0.5
            last_change = -1
        else
            l *= mult
            mult = last_change == 1 ? mult^2.0 : mult^0.5
            last_change = 1
        end

        modify_hypercube!(rect, mode, l)
        modify_integrationvolume!(intvol, dataset, rect, false)
        pt = intvol.pointcloud.points
    end

    #@log_msg LOG_TRACE "Tolerance Test Cube: Iterations $iterations\tPoints: $(intvol.pointcloud.points)\ttarget Points: $points"

    l, intvol
end

function create_initialhypercube(
    mode::Array{T, 1},
    dataset::DataSet{T, I},
    targetprobfactor::T) where {T<:AbstractFloat, I<:Integer}

    edgelength::T = 1.0


    cube = create_hypercube(mode, edgelength)
    vol = IntegrationVolume(dataset, cube, true)

    while vol.pointcloud.points > 0.01 * dataset.N
        edgelength *= 0.5^(1/dataset.P)
        modify_hypercube!(cube, mode, edgelength)
        modify_integrationvolume!(vol, dataset, cube, true)
    end

    tol = 1.0
    step = 0.7
    direction = 0
    PtsIncrease = 0.0

    it = 0
    while vol.pointcloud.probfactor < targetprobfactor / tol || vol.pointcloud.probfactor > targetprobfactor
        tol += 0.01 * it
        it += 1

        if vol.pointcloud.probfactor > targetprobfactor
            #decrease side length
            edgelength *= step

            step = adjuststepsize!(step,direction == -1)
            direction = -1
        else
            #increase side length
            edgelength /= step

            step = adjuststepsize!(step, direction == 1)
            direction = 1
        end
        PtsIncrease = vol.pointcloud.points
        modify_hypercube!(cube, mode, edgelength)
        modify_integrationvolume!(vol, dataset, cube, true)

        PtsIncrease = vol.pointcloud.points / PtsIncrease

        if vol.pointcloud.points > 0.01 * dataset.N && vol.pointcloud.probfactor < targetprobfactor
            break
        end
    end

    return cube, vol
end

@enum Edge LowerEdge UpperEdge
@enum Adaption IncreaseVolume DecreaseVolume NoChange Init
function modify_edge!(
    dataset::DataSet{T, I},
    change_mod::T,
    adaption::Adaption,
    edge::Edge,
    spatialvolume::HyperRectVolume{T},
    searchvol::HyperRectVolume{T},
    vol::IntegrationVolume{T, I, HyperRectVolume{T}},
    newvol::IntegrationVolume{T, I, HyperRectVolume{T}},
    dim::I) where {T<:AbstractFloat, I<:Integer}

    margin = spatialvolume.hi[dim] - spatialvolume.lo[dim]
    @assert margin != 0

    buffer = (edge == UpperEdge) ? spatialvolume.hi[dim] : spatialvolume.lo[dim]
    if edge == UpperEdge
        if adaption == IncreaseVolume
            spatialvolume.hi[dim] += margin * change_mod
        elseif adaption == DecreaseVolume
            spatialvolume.hi[dim] -= margin * change_mod
        else
            @log_msg LOG_ERROR "No edge modification possible: Volume change not specified"
        end
    else
        if adaption == IncreaseVolume
            spatialvolume.lo[dim] -= margin * change_mod
        elseif adaption == DecreaseVolume
            spatialvolume.lo[dim] += margin * change_mod
        else
            @log_msg LOG_ERROR "No edge modification possible: Volume change not specified"
        end
    end
    prevpts = vol.pointcloud.points
    resize_integrationvol!(newvol, vol, dataset, dim, spatialvolume, false, searchvol)
    pts_change = newvol.pointcloud.points / prevpts
    return pts_change, buffer, margin
end
function accept_edge_modification!(
    vol::IntegrationVolume{T, I, HyperRectVolume{T}},
    newvol::IntegrationVolume{T, I, HyperRectVolume{T}}) where {T<:AbstractFloat, I<:Integer}

    copy!(vol, newvol)
end
function reject_edge_modification!(
    vol::IntegrationVolume{T, I, HyperRectVolume{T}},
    newvol::IntegrationVolume{T, I, HyperRectVolume{T}},
    spatialvolume::HyperRectVolume{T},
    buffer::T,
    edge::Edge,
    dim::I) where {T<:AbstractFloat, I<:Integer}

    newvol.pointcloud.points = vol.pointcloud.points
    if edge == UpperEdge
        spatialvolume.hi[dim] = buffer
    elseif edge == LowerEdge
        spatialvolume.lo[dim] = buffer
    else
        @log_msg LOG_ERROR "No Edge modification to reject"
    end
end

function adapt_dimension!(
    dataset::DataSet{T, I},
    vol::IntegrationVolume{T, I, HyperRectVolume{T}},
    newvol::IntegrationVolume{T, I, HyperRectVolume{T}},
    spatialvolume::HyperRectVolume{T},
    searchvol::HyperRectVolume{T},
    edge::Edge,
    dim::I,
    increase::T,
    decrease::T,
    targetprobfactor::T,
    max_iterations_per_dimension::I) where {T<:AbstractFloat, I<:Integer}

    ptsTolInc::T = dataset.tolerance
    ptsTolDec::T = dataset.tolerance * 1.1

    last_change = Init
    current_iteration = 0
    local change = false

    while last_change != NoChange &&
        vol.pointcloud.probfactor > 1.0 &&
        current_iteration < max_iterations_per_dimension

        current_iteration += 1
        pts_change, buffer, margin = modify_edge!(dataset, increase, IncreaseVolume, edge, spatialvolume, searchvol, vol, newvol, dim)

        if newvol.pointcloud.probfactor < targetprobfactor && pts_change > (1.0 + increase / ptsTolInc) && last_change != DecreaseVolume
            accept_edge_modification!(vol, newvol)
            change = true
            last_change = IncreaseVolume
        else
            reject_edge_modification!(vol, newvol, spatialvolume, buffer, edge, dim)
            pts_change, buffer, margin = modify_edge!(dataset, decrease, DecreaseVolume, edge, spatialvolume, searchvol, vol, newvol, dim)

            if pts_change > (1.0 - decrease / ptsTolDec) && last_change != IncreaseVolume
                accept_edge_modification!(vol, newvol)
                change = true
                last_change = DecreaseVolume
            else
                reject_edge_modification!(vol, newvol, spatialvolume, buffer, edge, dim)
                last_change = NoChange
            end
        end
    end

    change, current_iteration
end

function adapt_hyperrectangle!(
    dataset::DataSet{T, I},
    vol::IntegrationVolume{T, I, HyperRectVolume{T}},
    newvol::IntegrationVolume{T, I, HyperRectVolume{T}},
    spatialvolume::HyperRectVolume{T},     #spatial volume
    searchvol::HyperRectVolume{T}, #adaptively proposed changes to the spatial volume
    increase::T,
    decrease::T,
    dim::I,
    targetprobfactor::T,
    max_iterations_per_dimension::I) where {T<:AbstractFloat, I<:Integer}

    change1, iterations1 = adapt_dimension!(dataset, vol, newvol, spatialvolume, searchvol,
        LowerEdge, dim, increase, decrease, targetprobfactor, max_iterations_per_dimension)

    change2, iterations2 = adapt_dimension!(dataset, vol, newvol, spatialvolume, searchvol,
        UpperEdge, dim, increase, decrease, targetprobfactor, max_iterations_per_dimension)


    change1 && change2, iterations1 + iterations2
end
"""
This function creates a hyper-rectangle around each starting sample.
It starts by building a hyper-cube  and subsequently adapts each face individually,
thus turning the hyper-cube into a hyper-rectangle.
The faces are adjusted in a way to match the shape of the distribution as best as possible.
"""
function create_hyperrectangle(
    id::I,
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    mode = dataset.data[:, id]
    initialcube, vol = create_initialhypercube(mode, dataset, targetprobfactor)

    local change = true
    volbuffer::IntegrationVolume{T, I} = deepcopy(vol)


    spatialvolume::HyperRectVolume{T} = deepcopy(vol.spatialvolume)
    searchvol::HyperRectVolume{T} = deepcopy(spatialvolume)

    increase_default = settings.rect_increase
    increase = increase_default
    decrease = 1.0 - 1.0 / (1.0 + increase)

    min_points = 5
    max_iterations_per_dimension = 20
    total_iterations = 0

    while change && vol.pointcloud.probfactor > 1.0

        if vol.pointcloud.points * increase < min_points
            increase *= ceil(I, min_points / (vol.pointcloud.points * increase))
            decrease = 1.0 - 1.0 / (1.0 + increase)
        elseif increase > increase_default && vol.pointcloud.points * increase_default > 2 * min_points
            increase = increase_default
            decrease = 1.0 - 1.0 / (1.0 + increase)
        end

        for p::I = 1:dataset.P
            change, iterations = adapt_hyperrectangle!(dataset, vol, volbuffer, spatialvolume, searchvol,
                increase, decrease, p, targetprobfactor, max_iterations_per_dimension)
            total_iterations += iterations
        end
    end

    res = search(dataset, vol.spatialvolume, true)
    resize!(vol.pointcloud.pointIDs, res.points)
    copyto!(vol.pointcloud.pointIDs, res.pointIDs)
    vol.pointcloud.points = res.points
    vol.pointcloud.maxLogProb = res.maxLogProb
    vol.pointcloud.minLogProb = res.minLogProb
    vol.pointcloud.probfactor = exp(vol.pointcloud.maxLogProb - vol.pointcloud.minLogProb)
    vol.pointcloud.probweightfactor = exp(vol.pointcloud.maxWeightProb - vol.pointcloud.minWeightProb)

    vol, initialcube, total_iterations
end

@inline function adjuststepsize!(Step, Increase::Bool)
    if Increase
        return Step * 0.5
    else
        return Step * 2.0
    end
end

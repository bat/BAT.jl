function hm_create_integrationvolumes!(
    result::HMIData{T, I, HyperRectVolume{T}},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if isempty(result.volumelist1) || isempty(result.volumelist2)
        nvols = (isempty(result.volumelist1) ? length(result.dataset1.startingIDs) : 0) +
                (isempty(result.volumelist2) ? length(result.dataset2.startingIDs) : 0)

        @info "Create $nvols Hyperrectangles using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        isempty(result.volumelist1) && hm_create_integrationvolumes_dataset!(
            result.dataset1, result.volumelist1, result.cubelist1, result.iterations1, result.whiteningresult.targetprobfactor,
                progressbar, settings)
        isempty(result.volumelist2) && hm_create_integrationvolumes_dataset!(
            result.dataset2, result.volumelist2, result.cubelist2, result.iterations2, result.whiteningresult.targetprobfactor,
                progressbar, settings)

        finish!(progressbar)
    end


    nvols = (result.dataset1.isnew ? length(result.volumelist1) : 0) +
            (result.dataset2.isnew ? length(result.volumelist2) : 0)

    if nvols > 0
        @info "Updating $nvols Hyperrectangles using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        result.dataset2.isnew && hm_update_integrationvolumes_dataset!(result.dataset2, result.volumelist1, progressbar)
        result.dataset1.isnew && hm_update_integrationvolumes_dataset!(result.dataset1, result.volumelist2, progressbar)

        finish!(progressbar)
    end
end

function hm_create_integrationvolumes_dataset!(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, HyperRectVolume{T}}, 1},
    cubes::Array{HyperRectVolume{T}, 1},
    iterations_per_volume::Array{I, 1},
    targetprobfactor::T,
    progressbar::Progress,
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}


    maxPoints::I = 0
    totalpoints::I = 0

    thread_volumes = Vector{IntegrationVolume{T, I, HyperRectVolume{T}}}(undef, length(dataset.startingIDs))
    thread_cubes = Vector{HyperRectVolume{T}}(undef, length(dataset.startingIDs))
    thread_iterations = Vector{I}(undef, length(dataset.startingIDs))

    atomic_centerID = Atomic{I}(1)

    @mt BAT.hyperrectangle_creationproccess!(dataset, targetprobfactor, settings,
        thread_volumes, thread_cubes, thread_iterations, atomic_centerID, progressbar)

    for i in eachindex(thread_volumes)
        if isassigned(thread_volumes, i) == false
        elseif thread_volumes[i].pointcloud.probfactor == 1.0 || thread_volumes[i].pointcloud.points < dataset.P * 4
        else
            push!(volumes, thread_volumes[i])
            push!(cubes, thread_cubes[i])
            push!(iterations_per_volume, thread_iterations[i])
            maxPoints = max(maxPoints, thread_volumes[i].pointcloud.points)
            totalpoints += thread_volumes[i].pointcloud.points
        end
    end

    dataset.isnew = true
end

function hm_update_integrationvolumes_dataset!(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, HyperRectVolume{T}}, 1},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

    maxPoints = zero(T)

    @mt for i in workpart(eachindex(volumes), mt_nthreads(), threadid())
        update!(volumes[i], dataset)

        maxPoints = max(maxPoints, volumes[i].pointcloud.points)

        @critical next!(progressbar)
    end

    #remove rectangles with less than 1% points of the largest rectangle (in terms of points)
    j = length(volumes)
    for i = 1:length(volumes)
        if volumes[j].pointcloud.points < maxPoints * 0.01 || volumes[j].pointcloud.points < dataset.P * 4
            deleteat!(volumes, j)
        end
        j -= 1
    end
    dataset.isnew = false
end


"""
    hyperrectangle_creationproccess!(...)

*AHMI-internal, not part of stable public API.*

This function assigns each thread its own hyper-rectangle to build, if in multithreading-mode.
"""
function hyperrectangle_creationproccess!(
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings,
    integrationvolumes::Vector{IntegrationVolume{T, I, HyperRectVolume{T}}},
    cubevolumes::Vector{HyperRectVolume{T}},
    total_iterations::Vector{I},
    atomic_centerID::Atomic{I},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

    while true
        #get new starting sample
        idc = atomic_add!(atomic_centerID, 1)
        if idc > length(dataset.startingIDs)
            break
        end
        id = dataset.startingIDs[idc]

        #update progress bar
        @critical next!(progressbar)

        #create hyper-rectangle
        integrationvolumes[idc], cubevolumes[idc], total_iterations[idc] = create_hyperrectangle(id, dataset, targetprobfactor, settings)

        @debug "Hyperrectangle created. Points:\t$(integrationvolumes[idc].pointcloud.points)\tVolume:\t$(integrationvolumes[idc].volume)\tProb. Factor:\t$(integrationvolumes[idc].pointcloud.probfactor)"
    end
end

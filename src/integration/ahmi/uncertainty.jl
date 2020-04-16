
function calculate_overlap(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, V}, 1},
    integralestimates::IntermediateResults)::Array{T, 2} where{T<:AbstractFloat, I<:Integer, V<:SpatialVolume}

    M = length(integralestimates)

    overlap = zeros(T, M, M)

    sortedsets = SortedSet.([volumes[integralestimates.volumeID[i]].pointcloud.pointIDs for i = 1:M])

    @mt for i in workpart(1:M, mt_nthreads(), threadid())
        for j = 1:M
            intersectpts = intersect(sortedsets[i], sortedsets[j])
            unionpts = union(sortedsets[i], sortedsets[j])

            overlap[i, j] = sum(dataset.weights[collect(intersectpts)]) / sum(dataset.weights[collect(unionpts)])
        end
    end
    overlap
end


function pdf_gauss(Z::Float64, μ::Float64,  σ_sq::Float64)::Float64
    return 1.0 / sqrt(2 * pi * σ_sq) * exp(-0.5 * (Z - μ)^2 / σ_sq)
end


function binomial_p_estimate_wald(n_total, n_success, nsigmas = 1)
    p_hat = n_success / n_total
    @assert p_hat >= 0 && p_hat <= 1
    @assert n_total > 0 && n_success > 0
    p_err = nsigmas * sqrt(p_hat * (1 - p_hat) / n_total)
end

function binomial_p_estimate_wilson(n_total, n_success, nsigmas = 1)
    n = n_total
    n_S = n_success
    n_F = n_total - n_S
    z = nsigmas
    z2 = z^2

    p_val = (n_S + z2/2) / (n + z2)

    p_err = z / (n + z2) * sqrt(n_S * n_F / n + z2 / 4)
end

function calculateuncertainty(dataset::DataSet{T, I}, volume::IntegrationVolume{T, I}, determinant::T, integral::T) where {T<:AbstractFloat, I<:Integer}

    #resorting the samples to undo the reordering of the space partitioning tree
    reorderscheme = sortperm(dataset.sortids[volume.pointcloud.pointIDs], rev = true)
    original_ordered_sample_ids = volume.pointcloud.pointIDs[reorderscheme]

    vol_samples = dataset.data[:, original_ordered_sample_ids]
    vol_weights = dataset.weights[original_ordered_sample_ids]

    x_min = 1.0 / dataset.N
    x_max = 1.0

    vol_weight = sum(vol_weights)
    total_weight = sum(dataset.weights)
    r = vol_weight / total_weight
    y_r = (x::Float64) -> Float64(x_min + (x_max - x_min) * x)


    ess = BAT.effective_sample_size(vol_samples, vol_weights)
    ess_total = BAT.effective_sample_size(dataset.data[:, dataset.sortids], dataset.weights[dataset.sortids])

    f = 1 ./ exp.(dataset.logprob[volume.pointcloud.pointIDs])

    μ_Z = mean(f)
    @assert ess>0

    σ_μZ_sq = var(f) / ess

    f_max = maximum(f)
    f_min = minimum(f)

    y = (x::Float64) -> Float64(f_min + (f_max - f_min) * x)
    
    integrand1(x) = pdf_gauss(y(x), μ_Z, σ_μZ_sq) / y(x)^2 * (f_max - f_min)
    integrand2(x) = pdf_gauss(y(x), μ_Z, σ_μZ_sq) / y(x) * (f_max - f_min)

    integral1 = QuadGK.quadgk(integrand1,0,1, rtol=0.001)[1]
    integral2 = QuadGK.quadgk(integrand2,0,1, rtol = 0.001)[1]
    
    uncertainty_Z = (integral1 - integral2^2) * volume.volume / r / determinant
    #set uncertainty to 0 if it is negative (numerical error)
    if uncertainty_Z < 0
        uncertainty_Z = 0
    else
        uncertainty_Z = sqrt(uncertainty_Z)
    end

    scaling_factor = ess / vol_weight
    scaling_factor_total = ess_total / total_weight

    uncertainty_r = binomial_p_estimate_wald(scaling_factor_total * total_weight, scaling_factor * vol_weight)
    uncertainty_r *= integral / r #error propagation

    uncertainty_tot = sqrt(uncertainty_Z^2 + uncertainty_r^2)

    return uncertainty_r, uncertainty_Z, uncertainty_tot, f_min, f_max, μ_Z, σ_μZ_sq, integral1, integral2, ess
end

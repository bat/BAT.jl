
function no_whitening(
    dataset::DataSet{T, I})::WhiteningResult{T} where {T<:AbstractFloat, I<:Integer}

    datamean = zeros(T, dataset.P)

    transform_data!(dataset, Array{T, 2}(LinearAlgebra.I, dataset.P, dataset.P), datamean)
end


function cholesky_whitening!(
    dataset::DataSet{T, I})::WhiteningResult{T} where {T<:AbstractFloat, I<:Integer}

    datamean = zeros(T, dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    for n in 1:dataset.N
        buffer = view(dataset.data, :, n) - datamean
        setindex!(dataset.data, buffer, :, n)
    end

    #cov of a Transpose object yields the wrong (0.0) result in 1D, if all weights are equal (e.g. direct sampling)!
    covmatrix = cov(convert(typeof(dataset.data), transpose(dataset.data)), FrequencyWeights(dataset.weights), corrected=true)
    symcovmatrix = Symmetric(covmatrix)
    covmatrix_inv = inv(symcovmatrix)
    w = cholesky(covmatrix_inv).U
    wres = convert(Matrix{T}, w)

    transform_data!(dataset, wres, datamean)
end

function cholesky_partial_whitening!(
    dataset::DataSet{T, I})::WhiteningResult{T} where {T<:AbstractFloat, I<:Integer}

    datamean = zeros(T, dataset.P)
	avoid_whitening_dims = find_steep_edges(dataset)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    for n in 1:dataset.N
        buffer = view(dataset.data, :, n) - datamean
        setindex!(dataset.data, buffer, :, n)
    end

    covmatrix = cov(convert(typeof(dataset.data), transpose(dataset.data)), FrequencyWeights(dataset.weights), corrected=true)

	for param in avoid_whitening_dims # replace non-diagonal elements with zeros to preserve rectangular bounds.
		covmatrix[1:param-1, param] .= 0.0
		covmatrix[param+1:end, param] .= 0.0
		covmatrix[param, param+1:end] .= 0.0
		covmatrix[param, 1:param-1] .= 0.0
	end

    symcovmatrix = Symmetric(covmatrix)
    covmatrix_inv = inv(symcovmatrix)
    w = cholesky(covmatrix_inv).U
    wres = convert(Matrix{T}, w)

    transform_data!(dataset, wres, datamean)
end

function find_steep_edges(dataset::DataSet{T, I})::Array{Int64,1} where {T<:AbstractFloat, I<:Integer}

    N_bins::Int64 = 100
    treshold::Float64 = 0.00
    avoid_whitening_dims::Array{Int64,1} = []

    for param in 1:dataset.P
        hist = fit(Histogram, dataset.data[param, :], FrequencyWeights(dataset.weights), nbins = N_bins, closed = :left)
        treshold = 0.01 * maximum(hist.weights)
		if ((hist.weights[1] < treshold) && (hist.weights[end] < treshold)) == false
            push!(avoid_whitening_dims, param)
        end
    end

    return avoid_whitening_dims
end



function statistical_whitening!(
    dataset::DataSet{T, I})::WhiteningResult{T} where {T<:AbstractFloat, I<:Integer}

    datamean = zeros(T, dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    dataset.data .-= datamean

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)

    E = eigfact(covmatrix).vectors
    w_d = transpose(E)
    D = Diagonal(inv(E) * covmatrix * E)
    w = inv(full(sqrt.(D))) * w_d
    wres = convert(Matrix{T}, w)

    transform_data!(dataset, wres, datamean)
end

function transform_data!(
    dataset::DataSet{T, I},
    W::Matrix{T},
    datamean::Vector{T},
    substractmean::Bool = false)::WhiteningResult{T} where {T<:AbstractFloat, I<:Integer}

    local determinant::T

    if substractmean
        dataset.data .-= datamean
    end

    if diag(W) == ones(T, dataset.P)
        determinant = 1.0
    else
        dataset.data = W * dataset.data

        determinant = abs(det(W))
    end

    maxP::T = maximum(dataset.logprob)
    suggTargetProb::T = 500 #exp(maxP - partialsort(dataset.logprob, floor(Int64, dataset.N * 0.2)))

    dataset.iswhitened = true

    @debug "Determinant:\t" * string(determinant)
    @debug "Suggested Target Probability Factor:\t" * string(suggTargetProb)

    WhiteningResult(determinant, suggTargetProb, W, datamean)
end

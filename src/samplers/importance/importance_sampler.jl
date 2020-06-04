abstract type ImportanceSampler <: AbstractSamplingAlgorithm end

struct SobolSampler <: ImportanceSampler end


function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::ImportanceSampler,
    bounds::Vector{Vector{Float64}}; # TODO default: use bounds from prior (if available)
)

    n_samples = n[1]
    n_chains = n[2]
    sample_arr = Vector{Array{Array{Float64, 1},1}}(undef, n_chains)
    stats_arr =  Vector{Array{NamedTuple, 1}}(undef, n_chains)

    Threads.@threads for i in 1:n_chains

        sample_arr[i] = get_samples(algorithm, bounds, n_samples)

        #stats_arr[i] = stats
    end

    samples = vcat(sample_arr...)
    # stats = vcat(stats_arr...)

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end


function get_samples(algorithm::SobolSampler, bounds::Vector{Vector{Float64}}, n_samples::Int)
    dim = length(bounds)
    mins = [bounds[i][1] for i in 1:dim]
    maxs = [bounds[i][2] for i in 1:dim]
    sobol = SobolSeq(mins, maxs)
    p = hcat([[Sobol.next!(sobol)] for i in 1:n_samples])
    return p
end

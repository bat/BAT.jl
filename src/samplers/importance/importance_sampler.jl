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
        sample_arr[i] = get_samples(algorithm, bounds)
        #stats_arr[i] = stats
    end

    samples = vcat(sample_arr...)
    stats = vcat(stats_arr...)

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end


function get_samples(algorithm::SobolSampler, bounds::Vector{Vector{Float64}})
    #...
    # return here an Array of Array with samples
end

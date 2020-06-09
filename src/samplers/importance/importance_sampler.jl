export SobolSampler, GridSampler, PriorSampler

abstract type ImportanceSampler <: AbstractSamplingAlgorithm end

struct SobolSampler <: ImportanceSampler end
struct GridSampler <: ImportanceSampler end
struct PriorSampler <: ImportanceSampler end


function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::ImportanceSampler;
    bounds::Vector{<:Tuple{Real, Real}} = get_prior_bounds(posterior)
)
    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    samples = get_samples(algorithm, bounds, n_samples, posterior)
    stats = [(stat = nothing, ) for i in n_samples] # TODO

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end


function get_samples(algorithm::SobolSampler, bounds::Vector{<:Tuple{Real, Real}}, n_samples::Int, posterior::AnyPosterior)
    dim = length(bounds)
    mins = [bounds[i][1] for i in 1:dim]
    maxs = [bounds[i][2] for i in 1:dim]
    sobol = SobolSeq(mins, maxs)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)
    return p
end


function get_samples(algorith::GridSampler, bounds::Vector{<:Tuple{Real, Real}}, n_samples::Int, posterior::AnyPosterior)
    dim = length(bounds)
    ppa = n_samples^(1/dim)
    ranges = [range(bounds[i][1], bounds[i][2], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


function get_samples(algorithm::PriorSampler, bounds::Vector{<:Tuple{Real, Real}}, n_samples::Int, posterior::AnyPosterior)
    p = rand(getprior(posterior).dist, n_samples)
    return collect(eachcol(p))
end

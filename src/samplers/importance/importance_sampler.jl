export SobolSampler, GridSampler, PriorSampler

abstract type ImportanceSampler <: AbstractSamplingAlgorithm end

struct SobolSampler <: ImportanceSampler end
struct GridSampler <: ImportanceSampler end
struct PriorSampler <: ImportanceSampler end


function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::ImportanceSampler;
    bounds::Any = finite_param_bounds(getprior(posterior).dist, hard_bounds)
)
    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]
    bounds = convert2HyperRectBounds(bounds)

    samples = get_samples(algorithm, bounds, n_samples, posterior)
    stats = [(stat = nothing, ) for i in n_samples] # TODO
    logvals = density_logval.(Ref(posterior), samples)
    weights = exp.(logvals)

    bat_samples = DensitySampleVector(samples, varshape(posterior), logval = logvals, weight = weights)
    return (result = bat_samples, chains = stats)
end


function get_samples(algorithm::SobolSampler, bounds::HyperRectBounds, n_samples::Int, posterior::AnyPosterior)
    sobol = SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)
    return p
end


function get_samples(algorith::GridSampler, bounds::HyperRectBounds, n_samples::Int, posterior::AnyPosterior)
    dim = length(bounds.bt)
    ppa = n_samples^(1/dim)
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


function get_samples(algorithm::PriorSampler, bounds::HyperRectBounds, n_samples::Int, posterior::AnyPosterior)
    p = rand(getprior(posterior).dist, n_samples)
    return collect(eachcol(p))
end

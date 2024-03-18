"""
    struct SobolSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SobolSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    nsamples::Int = 10^5
end
export SobolSampler



"""
    struct GridSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from equidistantly distributed points in each dimension.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct GridSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    ppa::Int = 100
end
export GridSampler


function bat_sample_impl(
    m::BATMeasure,
    algorithm::Union{SobolSampler, GridSampler},
    context::BATContext
)
    transformed_measure, trafo = transform_and_unshape(algorithm.trafo, m, context)

    if !has_uhc_support(transformed_measure)
        throw(ArgumentError("$algorithm doesn't measures that are not limited to the unit hypercube"))
    end

    samples = _gen_samples(transformed_measure, algorithm, context)

    logvals = map(logdensityof(transformed_measure), samples)
    weights = exp.(logvals)
    # ToDo: Renormalize weights

    est_integral = mean(weights)
    # ToDo: Add integral error estimate
    @show samples
    transformed_smpls = DensitySampleVector(samples, logvals, weight = weights)
    smpls = inverse(trafo).(transformed_smpls)

    return (result = smpls, result_trafo = transformed_smpls, trafo = trafo, integral = est_integral)
end


function _gen_samples(m::BATMeasure, algorithm::SobolSampler, context::BATContext)
    T = get_precision(context)
    n = getdof(m)
    # ToDo: Use BAT context for precision, etc:
    x = Vector{T}(undef, n)
    X = VectorOfSimilarVectors(Matrix{T}(undef, n, algorithm.nsamples))
    sobol = Sobol.SobolSeq(getdof(m))
    for i in 1:algorithm.nsamples
        Sobol.next!(sobol, x)
        X[i] .= x
    end
    return X
end


function _gen_samples(m::BATMeasure, algorithm::GridSampler, context::BATContext)
    dim = _rv_dof(m)
    ppa = algorithm.ppa
    # ToDo: Use BAT context for precision, etc:
    ranges = [range(0.0, 1.0, length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


"""
    struct PriorImportanceSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Importance sampler using IID samples from the prior.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PriorImportanceSampler <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export PriorImportanceSampler

function bat_sample_impl(
    posterior::AbstractPosteriorMeasure,
    algorithm::PriorImportanceSampler,
    context::BATContext
)
    shape = varshape(posterior)

    prior = convert_for(bat_sample, getprior(posterior))
    prior_samples = bat_sample_impl(prior, IIDSampling(nsamples = algorithm.nsamples), context).result
    unshaped_prior_samples = unshaped.(prior_samples)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = map(logdensityof(unshaped(posterior)), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    est_integral = mean(weight)
    # ToDo: Add integral error estimate

    posterior_samples = shape.(DensitySampleVector(v, posterior_logd, weight = weight))

    return (result = posterior_samples, prior_samples = prior_samples, integral = est_integral)
end

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
    pretransform::TR = PriorToUniform()
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
    pretransform::TR = PriorToUniform()
    ppa::Int = 100
end
export GridSampler


function evalmeasure_impl(
    m::BATMeasure,
    algorithm::Union{SobolSampler, GridSampler},
    context::BATContext
)
    transformed_m, f_pretransform = transform_and_unshape(algorithm.pretransform, m, context)
    transformed_m_uneval = unevaluated(transformed_m)
    n_dof = some_dof(transformed_m_uneval)

    if !has_uhc_support(transformed_m_uneval)
        throw(ArgumentError("$algorithm doesn't measures that are not limited to the unit hypercube"))
    end

    samples = _gen_samples(transformed_m_uneval, algorithm, context)

    # TODO: Parallelize
    logvals = map(logdensityof(transformed_m_uneval), samples)
    weights = exp.(logvals)
    # ToDo: Renormalize weights

    est_integral = mean(weights)
    # ToDo: Add integral error estimate
    # @show samples #disable for testing

    transformed_smpls = DensitySampleVector(v = samples, samples = logvals, weight = weights)
    smpls = inverse(f_pretransform).(transformed_smpls)

    ess = bat_eff_sample_size_impl(smpls, KishESS(), context).result

    dsm = DensitySampleMeasure(smpls, dof = n_dof, ess = ess)

    evalresult = (result_trafo = transformed_smpls, f_pretransform = f_pretransform)

    return EvalMeasureImplReturn(;
        empirical = dsm,
        dof = n_dof,
        mass = est_integral,
        evalresult = evalresult
    )
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

function evalmeasure_impl(
    measure::BATMeasure,
    algorithm::PriorImportanceSampler,
    context::BATContext
)
    m = unevaluated(measure)
    shape = varshape(m)

    prior = convert_for(bat_sample, getprior(m))
    prior_samples = samplesof(evalmeasure(prior, IIDSampling(nsamples = algorithm.nsamples), context))
    unshaped_prior_samples = unshaped.(prior_samples)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = map(logdensityof(unshaped(m)), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    est_integral = mean(weight)
    # ToDo: Add integral error estimate

    smpls = shape.(DensitySampleVector(v = v, logd = posterior_logd, weight = weight))

    ess = bat_eff_sample_size_impl(smpls, KishESS(), context).result

    # ToDo: DOF
    dsm = DensitySampleMeasure(smpls, ess = ess)

    evalresult = (;prior_samples = prior_samples)

    return EvalMeasureImplReturn(;
        empirical = dsm,
        dof = n_dof,
        mass = est_integral,
        evalresult = evalresult
    )
end

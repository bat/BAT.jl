# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BridgeSampling <: IntegrationAlgorithm

*Experimental feature, not part of stable public API.*

BridgeSampling integration algorithm.

See [X.-L. Meng and W. H. Wong, "Simulating ratios of normalizing
constants via a simple identity: a theoretical exploration"
(1996)](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n43/j6n43.htm).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct BridgeSampling{TR<:TransformIntent,ESS<:EffSampleSizeAlgorithm} <: IntegrationAlgorithm
    pretransform::TR = NormalBased()    
    essalg::ESS = EffSampleSizeFromAC()
    strict::Bool = true
    # ToDo: add argument for proposal density generator
end
export BridgeSampling

function evalmeasure_impl(em::EvaluatedMeasure, algorithm::BridgeSampling, context::BATContext)
    @argcheck !isnothing(empiricalof(em))
    if unevaluated(em) isa DensitySampleMeasure
        throw(ArgumentError("BridgeSampling requires a target with an evaluable density, a purely sample-based measure is not sufficient"))
    end
    transformed_m, _ = transform_and_unshape(algorithm.pretransform, em, context)
    renomalized_m, logweight = auto_renormalize(transformed_m)
    renomalized_m_uneval, renormalized_smpled = unevaluated(renomalized_m), empiricalof(renomalized_m)

    renormalized_smpls = samplesof(renormalized_smpled)
    (value, error) = bridge_sampling_integral(renomalized_m_uneval, renormalized_smpls, algorithm.strict, algorithm.essalg, context)
    rescaled_value, rescaled_error = exp(BigFloat(log(value) - logweight)), exp(BigFloat(log(error) - logweight))
    mass = Measurements.measurement(rescaled_value, rescaled_error)

    return EvaluatedMeasure(em;
        mass = mass,
        evalinfo = MeasureEvalInfo(algorithm, (;logweight = logweight))
    )
end


#TODO: Use EvaluatedMeasure to get proposal
function bridge_sampling_integral(
    target_density::BATMeasure, 
    target_samples::DensitySampleVector, 
    proposal_density::BATMeasure, 
    proposal_samples::DensitySampleVector, 
    strict::Bool,
    ess_alg::EffSampleSizeAlgorithm,
    context::BATContext
    )

    # Total weights normalize the weighted means below; the sample counts
    # entering the bridge mixture fractions and the error estimate are
    # effective (Kish) counts instead: sample weights carry no provenance,
    # so raw weight sums are not meaningful observation counts (they
    # change under rescaling that leaves the represented measure
    # unchanged), while Kish's effective count is scale-invariant and
    # reduces to the actual count for unit weights. Both are computed from
    # canonical relative weights, in which neither the sums nor their
    # squares can overflow: squaring the sum of raw integer repetition
    # weights wraps around where `Int` is 32 bits wide.
    u1 = _canonical_rel_weights(target_samples.weight)
    u2 = _canonical_rel_weights(proposal_samples.weight)
    W1_total = sum(u1)
    W2_total = sum(u2)
    N1 = W1_total^2 / sum(abs2, u1)
    N2 = W2_total^2 / sum(abs2, u2)

    #####################
    # Evaluate integral #
    #####################
    #calculate elements for iterative determination of marginal likelhood
    l1 = [exp(target_samples.logd[i]-logdensityof(proposal_density,x)) for (i,x) in enumerate(target_samples.v)]
    l2 = [exp(logdensityof(target_density,x)-proposal_samples.logd[i]) for (i,x) in enumerate(proposal_samples.v)]
    s1 = N1/(N2+N1)
    s2 = N2/(N1+N2)

    #calculate marginal likelhood iteratively
    prev_int = 0
    counter = 0
    current_int = 0.1
    while abs(current_int-prev_int)/current_int > 10^(-15)
        prev_int = current_int
        numerator = sum(@. u2 * l2 / (s1 * l2 + s2 * prev_int)) / W2_total
        denominator = sum(@. u1 / (s1 * l1 + s2 * prev_int)) / W1_total

        current_int = numerator/denominator
        if !isfinite(current_int)
            msg = "The bridge sampling iteration became non-finite"
            if strict
                throw(ErrorException(msg))
            else
                @warn(msg)
                # Fall back to the last finite iterate instead of
                # propagating a non-finite mass estimate:
                current_int = prev_int
                break
            end
        end
        if counter == 500
            msg = "The iterative scheme is not converging!!"
            if strict
                throw(ErrorException(msg))
            else
                @warn(msg)
                break
            end
        end
        counter=counter+1
    end

    #################
    #Evaluate error #
    #################
    # RMSE estimate following Q. F. Gronau et al., "A tutorial on bridge
    # sampling" (2017), https://doi.org/10.1016/j.jmp.2017.09.005
    #pre calculate objects for error estimate
    # ToDo: Make this type-stable:
    f1 = [exp(logdensityof(target_density,x))/current_int/(s1*exp(logdensityof(target_density,x))/current_int+s2*exp(proposal_samples.logd[i])) for (i,x) in enumerate(proposal_samples.v)]
    f2 = [[exp(logdensityof(proposal_density,x))/(s1*exp(target_samples.logd[i])/current_int+s2*exp(logdensityof(proposal_density,x)))] for (i,x) in enumerate(target_samples.v)]
    # The info fields carry the sampling-process provenance, so the ESS
    # of the derived quantities can be computed properly:
    f2_density_vector = DensitySampleVector(v = f2, logd = target_samples.logd, weight = target_samples.weight, info = target_samples.info)

    # Probability-weight semantics: sample weights carry no provenance, and
    # the frequency-weight Bessel correction degenerates for non-integer
    # weights. Canonical relative weights again, because the bias
    # correction of a probability weighting multiplies the weight sum by
    # the number of samples, which overflows for integer weights where
    # `Int` is 32 bits wide:
    mean1, var1 = StatsBase.mean_and_var(f1, ProbabilityWeights(u2), corrected = true)
    mean2, var2 = mean(f2_density_vector)[1],cov(f2_density_vector)[1]

    N1_eff = bat_eff_sample_size_impl(f2_density_vector,ess_alg,context).result[1] 
    # calculate  Root mean squared error
    r_MSE = sqrt(var1/(mean1^2*N2)+(var2/mean2^2)/N1_eff)*current_int 

    value, error = current_int, r_MSE
    return (Float64(value)::Float64, Float64(error)::Float64) # Force type stability, see above.
end


# ToDo: Rework to operate on an EvaluatedMeasure directly:
function bridge_sampling_integral(
    target_measure::BATMeasure,
    target_samples::DensitySampleVector,
    strict::Bool,
    ess_alg::EffSampleSizeAlgorithm,
    context::BATContext
    )

    num_samples = size(target_samples.weight)[1]
    n_first = floor(Int,num_samples/2)
    first_batch = target_samples[1:n_first]
    second_batch = target_samples[n_first+1:end]
    
    #####################
    # proposal function #
    #####################
    
    #Determine proposal function
    post_mean = vec(mean(first_batch))
    post_cov = Array(cov(first_batch)) #TODO: other covariance approximations
    post_cov_pd = PDMat(cholesky(Positive, post_cov))

    proposal_measure = batmeasure(MvNormal(post_mean,post_cov_pd))
    held_out_ess = bat_eff_sample_size_impl(second_batch, KishESS(), context).result
    n_proposal = clamp(round(Int, held_out_ess), 1, length(second_batch))
    proposal_samples = samplesof(evalmeasure(
        proposal_measure, IIDSampling(nsamples = n_proposal), context,
    ))
    proposal_measure = batmeasure(proposal_measure)

    bridge_sampling_integral(target_measure,second_batch,proposal_measure,proposal_samples,strict,ess_alg,context)
end

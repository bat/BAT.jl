# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BridgeSampling <: IntegrationAlgorithm

*Experimental feature, not part of stable public API.*

BridgeSampling integration algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct BridgeSampling{TR<:AbstractTransformTarget,ESS<:EffSampleSizeAlgorithm} <: IntegrationAlgorithm
    pretransform::TR = PriorToNormal()    
    essalg::ESS = EffSampleSizeFromAC()
    strict::Bool = true
    # ToDo: add argument for proposal density generator
end
export BridgeSampling

function evalmeasure_impl(measure::BATMeasure, algorithm::BridgeSampling, context::BATContext)
    @argcheck measure isa EvaluatedMeasure
    @argcheck !isnothing(empiricalof(measure))
    transformed_m, _ = transform_and_unshape(algorithm.pretransform, measure, context)
    renomalized_m, logweight = auto_renormalize(transformed_m)
    renomalized_m_uneval, renormalized_smpled = unevaluated(renomalized_m), empiricalof(renomalized_m)

    renormalized_smpls = samplesof(renormalized_smpled)
    (value, error) = bridge_sampling_integral(renomalized_m_uneval, renormalized_smpls, algorithm.strict, algorithm.essalg, context)
    rescaled_value, rescaled_error = exp(BigFloat(log(value) - logweight)), exp(BigFloat(log(error) - logweight))
    mass = Measurements.measurement(rescaled_value, rescaled_error)

    return EvalMeasureImplReturn(;
        mass = mass,
        evalresult = (;logweight = logweight)
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

    N1 = Int(sum(target_samples.weight))
    N2 = Int(sum(proposal_samples.weight))

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
        numerator = 0
        for (i, w) in enumerate(proposal_samples.weight)
            numerator += w*(l2[i]/(s1*l2[i]+s2*prev_int))
        end
        numerator = numerator/N2

        denominator = 0
        for (i, w) in enumerate(target_samples.weight)
            denominator += w/(s1*l1[i]+s2*prev_int)
        end
        denominator = denominator/N1

        current_int = numerator/denominator
        if counter == 500
            msg = "The iterative scheme is not converging!!"
            if strict
                throw(ErrorException(msg))
            else
                @warn(msg)
            end
        end
        counter=counter+1
    end

    #################
    #Evaluate error #
    #################
    #pre calculate objects for error estimate
    # ToDo: Make this type-stable:
    f1 = [exp(logdensityof(target_density,x))/current_int/(s1*exp(logdensityof(target_density,x))/current_int+s2*exp(proposal_samples.logd[i])) for (i,x) in enumerate(proposal_samples.v)]
    f2 = [[exp(logdensityof(proposal_density,x))/(s1*exp(target_samples.logd[i])/current_int+s2*exp(logdensityof(proposal_density,x)))] for (i,x) in enumerate(target_samples.v)]
    f2_density_vector = DensitySampleVector(v = f2, logd = target_samples.logd, weight=target_samples.weight)

    mean1, var1 = StatsBase.mean_and_var(f1, FrequencyWeights(proposal_samples.weight), corrected = true)
    mean2, var2 = mean(f2_density_vector)[1],cov(f2_density_vector)[1]

    N1_eff = bat_eff_sample_size_impl(f2_density_vector,ess_alg,context).result[1] 
    # calculate  Root mean squared error
    r_MSE = sqrt(var1/(mean1^2*N2)+(var2/mean2^2)/N1_eff)*current_int 

    value, error = current_int, r_MSE
    return (Float64(value)::Float64, Float64(error)::Float64) # Force type stability, see above.
end


#!!!!!! Use EvaluatedMeasure
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
    proposal_samples = samplesof(evalmeasure(proposal_measure, IIDSampling(nsamples=Int(sum(second_batch.weight))), context))
    proposal_measure = batmeasure(proposal_measure)

    bridge_sampling_integral(target_measure,second_batch,proposal_measure,proposal_samples,strict,ess_alg,context)
end

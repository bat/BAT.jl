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


function _bridge_weights(weights::AbstractVector{<:Real})
    AnalyticWeights(weights ./ sum(weights))
end


function _bridge_weight_ess(weights::AbstractVector{<:Real})
    inv(sum(abs2, _bridge_weights(weights)))
end


function _bridge_weighted_mean_and_var(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    weight_ess = _bridge_weight_ess(weights)
    StatsBase.mean_and_var(values, _bridge_weights(weights); corrected = weight_ess > 1)
end


function _bridge_target_ess(
    samples::DensitySampleVector,
    ess_alg::EffSampleSizeAlgorithm,
    context::BATContext,
)
    weight_ess = _bridge_weight_ess(samples.weight)
    autocorr_ess = bat_eff_sample_size_impl(samples, ess_alg, context).result[1]
    isfinite(autocorr_ess) && autocorr_ess > 0 ? min(weight_ess, autocorr_ess) : weight_ess
end


function bat_integrate_impl(m::BATMeasure, algorithm::BridgeSampling, context::BATContext)
    @argcheck m isa EvaluatedMeasure
    @argcheck !ismissing(maybe_samplesof(m))
    transformed_m, _ = transform_and_unshape(algorithm.pretransform, m, context)
    renomalized_m, logweight = auto_renormalize(transformed_m)
    renomalized_m_uneval, renormalized_smpls = unevaluated(renomalized_m), maybe_samplesof(renomalized_m)
    @assert !ismissing(renormalized_smpls)

    (value, error) = bridge_sampling_integral(renomalized_m_uneval, renormalized_smpls, algorithm.strict, algorithm.essalg, context)
    rescaled_value, rescaled_error = exp(BigFloat(log(value) - logweight)), exp(BigFloat(log(error) - logweight))
    result = Measurements.measurement(rescaled_value, rescaled_error)
    return (result = result, logweight = logweight)
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

    N1 = _bridge_weight_ess(target_samples.weight)
    N2 = _bridge_weight_ess(proposal_samples.weight)

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
        numerator = mean(l2 ./ (s1 .* l2 .+ s2 .* prev_int), _bridge_weights(proposal_samples.weight))
        denominator = mean(1 ./ (s1 .* l1 .+ s2 .* prev_int), _bridge_weights(target_samples.weight))

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
    f2_density_vector = DensitySampleVector(f2,target_samples.logd,weight=target_samples.weight)

    mean1, var1 = _bridge_weighted_mean_and_var(f1, proposal_samples.weight)
    mean2, var2 = _bridge_weighted_mean_and_var(only.(f2), target_samples.weight)

    N1_eff = _bridge_target_ess(f2_density_vector, ess_alg, context)
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
    post_cov = Array(_cov(
        first_batch.v,
        _bridge_weights(first_batch.weight);
        corrected = _bridge_weight_ess(first_batch.weight) > 1,
    )) #TODO: other covariance approximations
    post_cov_pd = PDMat(cholesky(Positive, post_cov))

    proposal_measure = batmeasure(MvNormal(post_mean,post_cov_pd))
    n_proposal = round(Int, _bridge_weight_ess(second_batch.weight))
    proposal_samples = bat_sample_impl(proposal_measure, IIDSampling(nsamples=n_proposal), context).result
    proposal_measure = batmeasure(proposal_measure)

    bridge_sampling_integral(target_measure,second_batch,proposal_measure,proposal_samples,strict,ess_alg,context)
end

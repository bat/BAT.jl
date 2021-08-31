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
@with_kw struct BridgeSampling{TR<:AbstractDensityTransformTarget,ESS<:EffSampleSizeAlgorithm} <: IntegrationAlgorithm
    trafo::TR = PriorToGaussian()    
    essalg::ESS = EffSampleSizeFromAC()
    strict::Bool = true
    # ToDo: add argument for proposal density generator
end
export BridgeSampling


function bat_integrate_impl(target::SampledDensity, algorithm::BridgeSampling)
    transformed_target, trafo = bat_transform(algorithm.trafo, target)
    density = unshaped(transformed_target.density)
    samples = unshaped.(transformed_target.samples)
    
    integral = bridge_sampling_integral(density, samples,algorithm.strict, algorithm.essalg )
    (result = integral,)
end


function bridge_sampling_integral(
    target_density::AbstractDensity, 
    target_samples::DensitySampleVector, 
    proposal_density::AbstractDensity, 
    proposal_samples::DensitySampleVector, 
    strict::Bool,
    ess_alg::EffSampleSizeAlgorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), target_samples)
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
    f1 = [exp(logdensityof(target_density,x))/current_int/(s1*exp(logdensityof(target_density,x))/current_int+s2*exp(proposal_samples.logd[i])) for (i,x) in enumerate(proposal_samples.v)]
    f2 = [[exp(logdensityof(proposal_density,x))/(s1*exp(target_samples.logd[i])/current_int+s2*exp(logdensityof(proposal_density,x)))] for (i,x) in enumerate(target_samples.v)]
    f2_density_vector = DensitySampleVector(f2,target_samples.logd,weight=target_samples.weight)

    mean1, var1 = StatsBase.mean_and_var(f1, FrequencyWeights(proposal_samples.weight))
    mean2, var2 = mean(f2_density_vector)[1],cov(f2_density_vector)[1]

    N1_eff = bat_eff_sample_size(f2_density_vector,ess_alg).result[1] 
    # calculate  Root mean squared error
    r_MSE = sqrt((var1/(mean1^2*N2)+(var2/mean2^2)/N1_eff)*current_int) 

    
    integral = Measurements.measurement(current_int, r_MSE)

    return integral
end



function bridge_sampling_integral(
    target_density::AbstractDensity,
    target_samples::DensitySampleVector,
    strict::Bool,
    ess_alg::EffSampleSizeAlgorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), target_samples)
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

    proposal_density = MvNormal(post_mean,post_cov_pd)
    proposal_samples = bat_sample(proposal_density,IIDSampling(nsamples=Int(sum(second_batch.weight)))).result
    proposal_density = convert(DistLikeDensity, proposal_density)

    bridge_sampling_integral(target_density,second_batch,proposal_density,proposal_samples,strict,ess_alg)
end

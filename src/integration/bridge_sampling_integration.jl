# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function bridge_sampling_integral end
export bridge_sampling_integral


function bridge_sampling_integral(
    target_density::AbstractDensity, 
    target_samples::DensitySampleVector, 
    proposal_density::AnyIIDSampleable, 
    proposal_samples::DensitySampleVector, 
    ess_alg::EffSampleSizeAlgorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), target_samples)
    )
    
    proposal_density = unshaped(convert(DistLikeDensity, proposal_density))

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
    end


    #################
    #Evaluate error #
    #################
    #pre calculate objects of error estimate
    f1 = [exp(logdensityof(target_density,x))*current_int/(s1*exp(logdensityof(target_density,x))*current_int+s2*exp(proposal_samples.logd[i])) for (i,x) in enumerate(proposal_samples.v)]
    f2 = [[exp(logdensityof(proposal_density,x))/(s1*exp(target_samples.logd[i])*current_int+s2*exp(logdensityof(proposal_density,x)))] for (i,x) in enumerate(target_samples.v)]
    mean1 = mean(f1)  #TODO: weighted variance
    var1 = var(f1)
    mean2 = sum([f2[i][1]*w for (i, w) in enumerate(target_samples.weight)])/N1 
    var2 = sum([w*(f2[i][1]-mean2)^2 for (i, w) in enumerate(target_samples.weight)])/(N1-1)
    f2_density_vector = DensitySampleVector(f2,target_samples.logd,weight=target_samples.weight)
    N1_eff = bat_eff_sample_size(f2_density_vector,ess_alg).result[1] 

    # calculate relative Root mean squared error
    r_MSE = sqrt(var1/(mean1^2*N2)+ *(var2/mean2^2)/N1_eff)

    integral = Measurements.measurement(current_int, r_MSE)

    return integral
end


function bridge_sampling_integral(
    target_density::AbstractDensity,
    target_samples::DensitySampleVector, 
    ess_alg::EffSampleSizeAlgorithm = bat_default_withdebug(bat_eff_sample_size, Val(:algorithm), target_samples)
    )

    num_samples = size(target_samples.weight)[1]
    n_first = floor(Int,num_samples/2)
    first_batch = target_samples[1:n_first]
    second_batch = target_samples[n_first+1:end]
    
    #####################
    # proposal function #
    #####################
    #expand first batch
    len = Int(sum(first_batch.weight))
    first_batch_flat = collect(flatview(unshaped.(first_batch.v)))
    first_batch_expanded = zeros((size(first_batch_flat)[1], len))
    counter =1
    for (i, w) in enumerate(first_batch.weight)
        for _=1:w
            first_batch_expanded[:,counter] = first_batch_flat[:,i]
            counter += 1
        end
    end    
    
    #Determine proposal function
    post_mean = vec(StatsBase.mean(first_batch_expanded, dims=2))
    post_cov = Matrix(StatsBase.cov(first_batch_expanded, dims=2)) #TODO: other covariance approximations
    

    proposal_density =  MvNormal(post_mean,post_cov)
    proposal_samples = bat_sample(proposal_density,IIDSampling(nsamples=Int(sum(second_batch.weight)))).result

    bridge_sampling_integral(target_density,second_batch,proposal_density,proposal_samples,ess_alg)
end

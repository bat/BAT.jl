############################################################################################################
# Here are transformation functions to convert datatypes
############################################################################################################

# If the prior for the parameter is a constant value transform it to an Normaldistribution without standard deviation
function const_to_normaldistribution(c::ValueShapes.ConstValueDist)
    return Distributions.Normal(c.value, 0)
end

# Otherwise return the Distribution
function const_to_normaldistribution(c)
    return c
end

# This function transforms a batprior to an array for NestedSamplers
function batPrior_to_array(posterior::AnyDensityLike)
    p = collect(values(BAT.getprior(posterior).dist._internal_distributions))
    return const_to_normaldistribution.(p)
end

# For plotting this function creates bat-standardversions of samples
function chain_to_batsamples(chain::Chains, posterior::AnyDensityLike)
    shape = BAT.varshape(BAT.getprior(posterior))
    weights = chain.value.data[:, end]                                                      # The last elements of the vectors are the weights
    logvals = zeros(length(weights)) ############### @TODO: implement the correct logvals, they dont seem to be in the chain

    nsamples = size(chain.value.data,1)
    samples = [chain.value.data[i, 1:end-1] for i in 1:nsamples]                            # The other ones between 1 and end-1 are the samples
    
    return shape.(BAT.DensitySampleVector(samples, logvals, weight = weights))
end


function batPosterior_to_nestedModel(posterior::AnyDensityLike; num_live_points::Int64, bound::TNS_Bound, proposal::TNS_Proposal, enlarge::Float64, min_ncall::Int64, min_eff::Float64)
    
    # The likelihoodfunction has to be a fuction of only x for NestedSamplers 
    function nestedSamplers_Likelihood(x)
        ks = keys(BAT.varshape(posterior))
        nx = (;zip(ks, x)...)
        return BAT.eval_logval_unchecked(BAT.getlikelihood(posterior),nx)
    end

    priors = batPrior_to_array(posterior)                                                    # NestedSamplers expects an array as prior

    model = NestedModel(nestedSamplers_Likelihood, priors);

    bounding = TNS_Bounding(bound)
    prop = TNS_prop(proposal)
    sampler = Nested(BAT.getprior(posterior).dist._internal_shape._flatdof, num_live_points; 
                    bounding, prop, enlarge, 
                    #update_interval, # ?
                    min_ncall, min_eff 
    ) 

    return model, sampler
end
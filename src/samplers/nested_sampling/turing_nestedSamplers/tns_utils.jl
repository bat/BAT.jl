############################################################################################################
# Here are transformation functions to convert datatypes
############################################################################################################
function const2normal(c::ValueShapes.ConstValueDist)
    return Distributions.Normal(c.value, 0)
end

function const2normal(c)
    return c
end

function prior2array(posterior)
    p = collect(values(BAT.getprior(posterior).dist._internal_distributions))
    return const2normal.(p)
end

function chain2batsamples(chain::Chains, shape)
    weights = chain.value.data[:, end]                                                      # The last elements of the vectors are the weights
    logvals = zeros(length(weights))

    samples = [chain.value.data[i, 1:end-1] for i in 1:length(chain.value.data[:, 1])]      # The other ones are samples
    return shape.(BAT.DensitySampleVector(samples, logvals, weight = weights))
end


function batPosterior2nestedModel(posterior; num_live_points, bound::TNS_Bound, proposal::TNS_Proposal, enlarge, min_ncall, min_eff)
    
    # The likelihoodfunction has to be a fuction of only x for NestedSamplers 
    function nestedLikelihood(x)
        ks = keys(BAT.varshape(posterior))
        nx = (;zip(ks, x)...)
        return BAT.eval_logval_unchecked(BAT.getlikelihood(posterior),nx)
    end

    priors = prior2array(posterior)                                                         # NestedSamplers expects an array as prior

    model = NestedModel(nestedLikelihood, priors);

    bounding = TNS_Bounding(bound)
    prop = TNS_prop(proposal)
    sampler = Nested(BAT.getprior(posterior).dist._internal_shape._flatdof, num_live_points; 
                    bounding, prop, enlarge, 
                    #update_interval, # ?
                    min_ncall, min_eff 
    ) 

    return model, sampler
end
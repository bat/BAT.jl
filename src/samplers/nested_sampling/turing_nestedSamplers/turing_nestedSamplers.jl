# They are used in this file, because this file ist only load if it is required
using .NestedSamplers
using .MCMCChains: Chains

include("tns_bounds.jl")
include("tns_proposals.jl")
include("tns_utils.jl")

@with_kw struct TuringNestedSamplersTR{TR<:AbstractDensityTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()

    num_live_points::Int = 1000                         # the number of live-points
    bound::TNS_Bound = TNS_MultiEllipsoidBound()        # volume around the live-points
    proposal::TNS_Proposal = TNS_AutoProposal()         # algorithm to choose new live-points
    enlarge::Float64 = 1.25                             # Scale-factor for the volume
    # update_interval::Float64 =                        # Not sure about how this works yet
    min_ncall::Int64 = 2*num_live_points                # number of iterations before the first bound will be fit
    min_eff::Float64 = 0.10                             # efficiency before fitting the first bound

    # The following four are the possible convergence criteria to end the algorithm
    dlogz::Float64 = 0.1
    max_iters = Inf
    max_ncalls = Inf
    maxlogl = Inf
end
export TuringNestedSamplersTR

@with_kw struct TuringNestedSamplers <: AbstractSamplingAlgorithm
    num_live_points::Int = 1000                         # the number of live-points
    bound::TNS_Bound = TNS_MultiEllipsoidBound()        # volume around the live-points
    proposal::TNS_Proposal = TNS_AutoProposal()         # algorithm to choose new live-points
    enlarge::Float64 = 1.25                             # Scale-factor for the volume
    # update_interval::Float64 =                        # Not sure about how this works yet
    min_ncall::Int64 = 2*num_live_points                # number of iterations before the first bound will be fit
    min_eff::Float64 = 0.10                             # efficiency before fitting the first bound

    # The following four are the possible convergence criteria to end the algorithm
    dlogz::Float64 = 0.1
    max_iters = Inf
    max_ncalls = Inf
    maxlogl = Inf
end
export TuringNestedSamplers


############################################################################################################
# Here is the bat_sample implementation for the NestedSamplers package
############################################################################################################
function bat_sample_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::TuringNestedSamplers)
    
    posterior = convert(AbstractDensity, target)
    model, sampler = batPosterior_to_nestedModel(posterior; 
        algorithm.num_live_points, algorithm.bound, algorithm.proposal,
        algorithm.enlarge, algorithm.min_ncall, algorithm.min_eff
    )
    
    chain, state = sample(model, sampler; 
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Chains
    )

    res = chain_to_batsamples(chain, posterior);
    return ( ########### Here is more to do i think
        result = res,
    )

end

function bat_sample_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::TuringNestedSamplersTR)
    
    density_notrafo = convert(AbstractDensity, target)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    vs = varshape(shaped_density)
    density = unshaped(shaped_density)

    # The likelihoodfunction has to be a fuction of only x for NestedSamplers 
    function nestedSamplers_Likelihood(x)
        #ks = keys(BAT.varshape(density_notrafo))
        ks = keys(BAT.varshape(density.likelihood.orig))
        nx = (;zip(ks, x)...)
        return BAT.eval_logval_unchecked(density.likelihood.orig,nx)
    end

    prior_trafo(x) = x                                                                            # identity, because we Use the BAT.PriorTrafo instead
    model = NestedModel(nestedSamplers_Likelihood, prior_trafo);

    bounding = TNS_Bounding(algorithm.bound)
    prop = TNS_prop(algorithm.proposal)
    sampler = Nested(totalndof(vs), algorithm.num_live_points; 
                        bounding, prop,
                        algorithm.enlarge, algorithm.min_ncall, algorithm.min_eff
    ) 

    chain, state = sample(model, sampler; 
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Chains
    )

    weights = chain.value.data[:, end]                                                            # the last elements of the vectors are the weights
    nsamples = size(chain.value.data,1)
    samples = [chain.value.data[i, 1:end-1] for i in 1:nsamples]                                  # the other ones between 1 and end-1 are the samples
    logvals = map(logdensityof(density), samples)                                                 # posterior values of the samples
    samples_trafo = vs.(BAT.DensitySampleVector(samples, logvals, weight = weights))
    samples_notrafo = inv(trafo).(samples_trafo)

    z = samples_notrafo.v.__internal_data.data
    samples_notrafo = [Vector([z[1,i],z[2,i],z[3,i]]) for i in 1:nsamples]                        # Shaping the from of the samples to the same like samples_trafo

    shape = BAT.varshape(BAT.getprior(density_notrafo))
    res = shape.(BAT.DensitySampleVector(samples_notrafo, logvals, weight = weights))
    return ( ########### Here is more to do i think
        result = res, result_trafo = samples_trafo, trafo = trafo,
    )

end

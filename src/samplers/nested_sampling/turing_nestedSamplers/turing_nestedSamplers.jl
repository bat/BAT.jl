using NestedSamplers
using MCMCChains: Chains

include("tns_bounds.jl")
include("tns_proposals.jl")
include("tns_utils.jl")


@with_kw struct TuringNestedSamplers <: AbstractSamplingAlgorithm
    
    num_live_points::Int = 1000
    
    bound::TNS_Bound = MultiEllipsoidBound()
    
    proposal::TNS_Proposal = AutoProposal() 

    enlarge::Float64 = 1.25

    # Not sure about how this works yet
    # update_interval =
    
    min_ncall::Int64 = 2*num_live_points
    
    min_eff::Float64 = 0.10

    # The following four are the possible convergence criteria
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
    model, sampler = batPosterior2nestedModel(posterior; 
        algorithm.num_live_points, algorithm.bound, algorithm.proposal,
        algorithm.enlarge, algorithm.min_ncall, algorithm.min_eff
    )
    
    chain, state = sample(model, sampler; 
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Chains
    )
    
    res = chain2batsamples(chain, BAT.varshape(BAT.getprior(posterior)));
    return ( ########### Here is more to do i think
        result = res,
    )

end

# They are used in this file, because this file ist only load if it is required
using NestedSamplers
using MCMCChains: Chains

include("tns_bounds.jl")
include("tns_proposals.jl")
include("tns_utils.jl")


@with_kw struct TuringNestedSamplers <: AbstractSamplingAlgorithm
    
    num_live_points::Int = 1000                         # the number of live-points
    
    bound::TNS_Bound = MultiEllipsoidBound()            # volume around the live-points
    
    proposal::TNS_Proposal = AutoProposal()             # algorithm to choose new live-points

    enlarge::Float64 = 1.25                             # Scale-factor for the volume

    # Not sure about how this works yet
    # update_interval::Float64 =
    
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
    model, sampler = batPosterior_2_nestedModel(posterior; 
        algorithm.num_live_points, algorithm.bound, algorithm.proposal,
        algorithm.enlarge, algorithm.min_ncall, algorithm.min_eff
    )
    
    chain, state = sample(model, sampler; 
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Chains
    )
    
    res = chain_2_batsamples(chain, BAT.varshape(BAT.getprior(posterior)));
    return ( ########### Here is more to do i think
        result = res,
    )

end

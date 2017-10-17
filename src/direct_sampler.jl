# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DirectSampling <: MCMCAlgorithm{AcceptRejectState} end
export DirectSampling


# ToDo: Specialized version of rand_initial_params for DirectSampling:
#
#     rand_initial_params!(rng::AbstractRNG, algorithm::DirectSampling, target::TargetSubject, x::StridedVecOrMat{<:Real}) = ...


AbstractMCMCTunerConfig(algorithm::DirectSampling) = NoOpTunerConfig()



function AcceptRejectState(
    algorithm::DirectSampling,
    target::AbstractTargetSubject, #{<:MvDistTargetDensity},
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    AcceptRejectState(
        GenericProposalDist(target.tdensity.d),
        current_sample
    )
end


function mcmc_propose_accept_reject!(
    chain::MCMCIterator{<:DirectSampling},
    exec_context::ExecContext
)
    state = chain.state
    rng = chain.rng
    target = chain.target
    pdist = state.pdist

    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    current_params = current_sample.params
    proposed_params = proposed_sample.params

    current_log_value = current_sample.log_value
    T = typeof(current_log_value)

    #info(pdist)
    #@assert false

    # Propose new parameters:
    rand!(rng, pdist.s, proposed_params)

    accepted = if proposed_params in target.bounds
        proposed_log_value = T(target_logval(target.tdensity, proposed_params, exec_context))
        proposed_sample.log_value = proposed_log_value
        current_sample.weight = 1
        true
    else
        # Reject:
        proposed_sample.log_value = T(-Inf)
        false
    end
end

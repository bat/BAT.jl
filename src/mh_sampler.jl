# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MetropolisHastings{
    Q<:ProposalDistSpec
} <: MCMCAlgorithm{AcceptRejectState}
    q::Q
end

export MetropolisHastings

MetropolisHastings() = MetropolisHastings(MvTDistProposalSpec())


mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::UnboundedParams) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)


function AcceptRejectState(
    algorithm::MetropolisHastings,
    target::AbstractTargetSubject,
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    AcceptRejectState(
        algorithm.q(P, nparams(current_sample)),
        current_sample
    )
end


#=
function mcmc_propose_accept_reject!(
    proposed_params::Vector{<:Real}
    algorithm::MetropolisHastings,
    current_params::Vector{<:Real},
    current_log_value::T
) with {T<:Real}
    # Propose new parameters:
    proposal_rand!(rng, pdist, proposed_params, current_params)
    apply_bounds!(proposed_params, bounds, false)

    if proposed_params in bounds
        # Evaluate target density at new parameters:
        proposed_log_value = T(target_logval(tdensity, proposed_params, exec_context))

        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(pdist)
            T(0)
        else
            log_tp_fwd = proposal_logpdf(pdist, proposed_params, current_params)
            log_tp_rev = proposal_logpdf(pdist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        current_log_value = current_sample.log_value
        proposed_sample.log_value = proposed_log_value

        # Metropolis-Hastings accept/reject:
        rand(rng) < exp(proposed_log_value - current_log_value - log_tpr)
    else
        # Reject:
        false, T(-Inf)
    end
end
=#


function mcmc_iterate!(
    callback,
    chain::MCMCChain{<:MetropolisHastings},
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int = 1000,
    max_time::Float64 = Inf,
    ll::LogLevel = LOG_NONE
)
    algorithm = chain.algorithm
    cbfunc = mcmc_callback(callback)

    start_time = time()
    nsteps = 0
    nsamples = 0

    if !mcmc_compatible(algorithm, chain.state.pdist, chain.target.bounds)
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    T = typeof(chain.state.current_sample.log_value)

    while nsamples < max_nsamples && nsteps < max_nsteps && (time() - start_time) < max_time
        target = chain.target
        state = chain.state
        rng = chain.rng

        tdensity = target.tdensity
        bounds = target.bounds

        pdist = state.pdist

        current_sample = state.current_sample
        proposed_sample = state.proposed_sample

        if state.proposal_accepted
            reset_rng_counters!(rng, MCMCSampleID(chain))
            copy!(current_sample, proposed_sample)
            state.current_nreject = 0
            state.proposal_accepted = false
        end



        current_params = current_sample.params
        proposed_params = proposed_sample.params

        current_log_value = current_sample.log_value

        # Propose new parameters:
        proposal_rand!(rng, pdist, proposed_params, current_params)
        apply_bounds!(proposed_params, bounds, false)



        accepted = if proposed_params in bounds
            # Evaluate target density at new parameters:
            proposed_log_value = T(target_logval(tdensity, proposed_params, exec_context))

            # log of ratio of forward/reverse transition probability
            log_tpr = if issymmetric(pdist)
                T(0)
            else
                log_tp_fwd = proposal_logpdf(pdist, proposed_params, current_params)
                log_tp_rev = proposal_logpdf(pdist, current_params, proposed_params)
                T(log_tp_fwd - log_tp_rev)
            end

            current_log_value = current_sample.log_value
            proposed_sample.log_value = proposed_log_value

            # Metropolis-Hastings accept/reject:
            rand(rng) < exp(proposed_log_value - current_log_value - log_tpr)
        else
            # Reject:
            proposed_sample.log_value = T(-Inf)
            false
        end



        nsteps += 1
        state.nsteps += 1

        if accepted
            current_sample.weight = state.current_nreject + 1
            state.proposal_accepted = true
            nsamples += 1
            state.nsamples += 1
        else
            state.current_nreject += 1
        end

        if accepted
            cbfunc(1, chain)
        else
            cbfunc(2, chain)
        end
    end
    chain
end

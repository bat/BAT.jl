# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MHWeightingScheme{T<:Real} end
export MHWeightingScheme

struct MHMultiplicityWeights{T<:Real} <: MHWeightingScheme{T} end
export MHMultiplicityWeights
MHMultiplicityWeights() = MHMultiplicityWeights{Int}()

struct MHAccRejProbWeights{T<:AbstractFloat} <: MHWeightingScheme{T} end
export MHAccRejProbWeights
MHAccRejProbWeights() = MHAccRejProbWeights{Float64}()



struct MetropolisHastings{
    Q<:ProposalDistSpec,
    W<:Real,
    WS<:MHWeightingScheme{W}
} <: MCMCAlgorithm
    proposalspec::Q
    weighting_scheme::WS

    MetropolisHastings(proposalspec::Q,weighting_scheme::WS) where {
        Q<:ProposalDistSpec, W<:Real, WS<:MHWeightingScheme{W}} =
        new{Q,W,WS}(proposalspec, weighting_scheme)
end

export MetropolisHastings


MetropolisHastings(proposalspec::ProposalDistSpec = MvTDistProposalSpec()) =
    MetropolisHastings(proposalspec, MHMultiplicityWeights())

MetropolisHastings(weighting_scheme::MHWeightingScheme) =
    MetropolisHastings(MvTDistProposalSpec(), weighting_scheme)


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::NoParamBounds) = true

mcmc_compatible(::MetropolisHastings, proposaldist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(proposaldist) || all(x -> x == hard_bounds, bounds.bt)


_sample_weight_type(::Type{MetropolisHastings{Q,W,WS}}) where {Q,W,WS} = W



mutable struct MHIterator{
    SP<:MCMCSpec,
    R<:AbstractRNG,
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector,
    IV<:MCMCSampleIDVector
} <: MCMCIterator
    spec::SP
    rng::R
    info::MCMCIteratorInfo
    proposaldist::Q
    samples::SV
    sampleids::IV
    nsamples::Int64
    stepno::Int64
end


function MHIterator(
    spec::MCMCSpec,
    info::MCMCIteratorInfo,
    initial_params::AbstractVector{P},
) where {P<:Real}
    stepno::Int64 = 0
    rng = spec.rngseed()
    reset_rng_counters!(rng, info.id, info.cycle, stepno)

    postr = spec.posterior
    npar = nparams(postr)
    alg = spec.algorithm

    params_vec = Vector{P}(undef, npar)
    if isempty(initial_params)
        initial_params!(params_vec, rng, postr, alg)
    else
        params_vec .= initial_params
    end
    !(params_vec in param_bounds(postr)) && throw(ArgumentError("Parameter(s) out of bounds"))

    proposaldist = alg.proposalspec(P, npar)

    # ToDo: Make numeric type configurable:

    (log_prior_value, log_posterior_value) = eval_prior_posterior_logval_strict!(postr, params_vec)

    T = typeof(log_posterior_value)
    W = _sample_weight_type(typeof(alg))

    current_sample = DensitySample(params_vec, log_posterior_value, convert(T, log_prior_value), one(W))

    samples = push!(DensitySampleVector{P,T,W}(npar), current_sample)
    sampleids = push!(MCMCSampleIDVector(), MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE))
    @assert axes(samples, 1) == axes(sampleids, 1)

    nsamples::Int64 = 0

    chain = MHIterator(
        spec,
        rng,
        info,
        proposaldist,
        samples,
        sampleids,
        nsamples,
        stepno
    )

    chain
end


function reset_rng_counters!(chain::MHIterator)
    reset_rng_counters!(chain.rng, chain.info.id, chain.info.cycle, chain.stepno)
end


function (spec::MCMCSpec{<:MetropolisHastings})(
    chainid::Integer,
)
    P = float(eltype(param_bounds(spec.posterior)))

    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)

    MHIterator(spec, info, Vector{P}())
end


@inline _current_sample_idx(chain::MHIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::MHIterator) = lastindex(chain.samples)

function _available_samples_idxs(chain::MHIterator)
    sampletype = chain.sampleids.sampletype
    @uviews sampletype begin
        from = firstindex(chain.samples)

        to = if samples_available(chain)
            lastidx = lastindex(chain.samples)
            @assert sampletype[from] == ACCEPTED_SAMPLE
            @assert sampletype[lastidx] == CURRENT_SAMPLE
            lastidx - 1
        else
            from - 1
        end

        r = from:to
        @assert all(x -> x > INVALID_SAMPLE, view(sampletype, r))
        r
    end
end


mcmc_spec(chain::MHIterator) = chain.spec

getrng(chain::MHIterator) = chain.rng

mcmc_info(chain::MHIterator) = chain.info

nsteps(chain::MHIterator) = chain.stepno

nsamples(chain::MHIterator) = chain.nsamples

sample_type(chain::MHIterator) = eltype(chain.samples)


function samples_available(chain::MHIterator)
    i = _current_sample_idx(chain::MHIterator)
    chain.sampleids.sampletype[i] == ACCEPTED_SAMPLE
end


function get_samples!(appendable, chain::MHIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        idxs = _available_samples_idxs(chain)
        samples = chain.samples

        @uviews samples begin
            # if nonzero_weights
                for i in idxs
                    if !nonzero_weights || samples.weight[i] > 0
                        push!(appendable, samples[i])
                    end
                end
            # else
            #     append!(appendable, view(samples, idxs))
            # end
        end
    end
    appendable
end


function get_sample_ids!(appendable, chain::MHIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        idxs = _available_samples_idxs(chain)
        samples = chain.samples
        sampleids = chain.sampleids

        @uviews samples sampleids begin
            # if nonzero_weights
                for i in idxs
                    if !nonzero_weights || samples.weight[i] > 0
                        push!(appendable, sampleids[i])
                    end
                end
            # else
            #     append!(appendable, view(sampleids, idxs))
            # end
        end
    end
    appendable
end


function next_cycle!(chain::MHIterator)
    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.nsamples = 0
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)
    resize!(chain.sampleids, 1)

    i = _current_sample_idx(chain)
    @assert chain.sampleids[i].sampletype == CURRENT_SAMPLE

    chain.samples.weight[i] = 1
    chain.sampleids[i] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE)

    chain
end


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MHIterator
)
    alg = algorithm(chain)

    if !mcmc_compatible(alg, chain.proposaldist, param_bounds(getposterior(chain)))
        error("Implementation of algorithm $alg does not support current parameter bounds with current proposal distribution")
    end

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = getrng(chain)
    pstr = getposterior(chain)

    proposaldist = chain.proposaldist
    samples = chain.samples
    sampleids = chain.sampleids

    # Grow samples and sampleids vector by one:
    resize!(samples, size(samples, 1) + 1)
    push!(sampleids, MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, PROPOSED_SAMPLE))
    @assert axes(samples, 1) == axes(sampleids, 1)

    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    @assert current != proposed

    accepted = @uviews samples sampleids begin
        current_params = samples.params[current]
        proposed_params = samples.params[proposed]

        # Propose new parameters:
        samples.weight[proposed] = 0
        proposal_rand!(rng, proposaldist, proposed_params, current_params)

        current_log_posterior = samples.log_posterior[current]
        T = typeof(current_log_posterior)

        # Evaluate prior and likelihood with proposed parameters:
        proposed_log_prior, proposed_log_posterior =
            eval_prior_posterior_logval!(T, pstr, proposed_params)

        samples.log_posterior[proposed] = proposed_log_posterior
        samples.log_prior[proposed] = proposed_log_prior

        p_accept = if proposed_log_posterior > -Inf
            # log of ratio of forward/reverse transition probability
            log_tpr = if issymmetric(proposaldist)
                T(0)
            else
                log_tp_fwd = distribution_logpdf(proposaldist, proposed_params, current_params)
                log_tp_rev = distribution_logpdf(proposaldist, current_params, proposed_params)
                T(log_tp_fwd - log_tp_rev)
            end

            p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
            T(clamp(p_accept_unclamped, 0, 1))
        else
            zero(T)
        end

        @assert p_accept >= 0
        accepted = rand(chain.rng, float(typeof(p_accept))) < p_accept

        if accepted
            sampleids.sampletype[current] = ACCEPTED_SAMPLE
            sampleids.sampletype[proposed] = CURRENT_SAMPLE
            chain.nsamples += 1
        else
            sampleids.sampletype[proposed] = REJECTED_SAMPLE
        end

        delta_w_current, w_proposed = _mh_weights(alg, p_accept, accepted)
        samples.weight[current] += delta_w_current
        samples.weight[proposed] = w_proposed

        callback(1, chain)

        if accepted
            current_params .= proposed_params
            samples.log_posterior[current] = samples.log_posterior[proposed]
            samples.log_prior[current] = samples.log_prior[proposed]
            samples.weight[current] = samples.weight[proposed]
            sampleids[current] = sampleids[proposed]
        end

        accepted
    end # @uviews

    if accepted
        resize!(samples, 1)
        resize!(sampleids, 1)
    end

    chain
end


function _mh_weights(
    algorithm::MetropolisHastings{Q,W,<:MHMultiplicityWeights},
    p_accept::Real,
    accepted::Bool
) where {Q,W}
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end


function _mh_weights(
    algorithm::MetropolisHastings{Q,W,<:MHAccRejProbWeights},
    p_accept::Real,
    accepted::Bool
) where {Q,W}
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end


eff_acceptance_ratio(chain::MHIterator) = nsamples(chain) / nsteps(chain)

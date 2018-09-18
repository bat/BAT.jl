# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct MTMState{
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector,
    IV<:MCMCSampleIDVector
} <: AbstractMCMCState
    pdist::Q
    samples::SV  # First element is the current sample in the chain
    help_samples_1::SV
    help_samples_2::SV  # First element is the current sample in the chain
    sampleids::IV
    eff_acceptratio_sum::Double{Float64}
    nsamples::Int64
    nsteps::Int64
end

function MTMState(
    pdist::AbstractProposalDist,
    current_sample::DensitySample{P,T,W},
    nproposals::Int
) where {P,T,W}
    npar = nparams(current_sample)

    params = ElasticArray{T}(npar, 2)
    helpers_1 = ElasticArray{T}(npar, nproposals)
    helpers_2 = ElasticArray{T}(npar, nproposals)
    fill!(params, zero(T))
    fill!(helpers, zero(T))

    helpers_2[:, 1] = current_sample.params
    params[:, 1] = current_sample.params

    log_value_params = Vector{T}(2)
    log_value_helpers_1 = Vector{T}(nproposals)
    log_value_helpers_1 = Vector{T}(nproposals)
    fill!(log_value_params, NaN)
    fill!(log_value_helpers_1, NaN)
    fill!(log_value_helpers_2, NaN)
    log_value_helpers_2[1] = current_sample.log_value
    log_value_params[1] = current_sample.log_value

    weight_params = Vector{W}(2)
    weight_helpers_1 = Vector{W}(nproposals)
    weight_helpers_2 = Vector{W}(nproposals)
    fill!(weight_params, zero(W))
    fill!(weight_helpers_1, zero(W))
    fill!(weight_helpers_2, zero(W))
    weight_params[1] = current_sample.weight

    samples = DensitySampleVector(params, log_value_params, weight_params)
    help_samples_1 = DensitySampleVector(helpers_1, log_value_helpers_1, weight_helpers_1)
    help_samples_2 = DensitySampleVector(helpers_2, log_value_helpers_2, weight_helpers_2)
    accepted = fill(false, )

    sampleids = append!(MCMCSampleIDVector(), fill(MCMCSampleID(-1, -1, -1, -1), 2))

    eff_acceptratio_sum = zero(Double{Float64})
    nsamples = 0
    nsteps = 0

    MTMState(
        pdist,
        samples,
        help_samples_1,
        help_samples_1,
        sampleids,
        eff_acceptratio_sum,
        nsamples,
        nsteps
    )
end


nparams(state::MTMState) = nparams(state.pdist)

nsteps(state::MTMState) = state.nsteps

nsamples(state::MTMState) = state.nsamples

eff_acceptance_ratio(state::MTMState) = Float64(state.eff_acceptratio_sum / state.nsteps)


function next_cycle!(state::MTMState)
    state.samples.weight[1] = one(eltype(state.samples.weight))
    state.eff_acceptratio_sum = zero(state.eff_acceptratio_sum)
    state.nsamples = zero(state.nsamples)
    state.nsteps = zero(state.nsteps)
    state
end


density_sample_type(state::MTMState) = eltype(state.samples)


function MCMCBasicStats(state::MTMState)
    L = promote_type(eltype(state.samples.log_value), Float64)
    P = promote_type(eltype(state.samples.params), Float64)
    m = nparams(state)
    MCMCBasicStats{L, P}(m)
end


function nsamples_available(state::MTMState; nonzero_weights::Bool = false)
    # ignore nonzero_weights for now
    # ToDo: Handle this properly
    length(state.samples)
end


function Base.append!(xs::DensitySampleVector, state::MTMState)
    if nsamples_available(state) > 0
        new_samples = view(state.samples, (firstindex(state.samples) + 1):lastindex(state.samples))  # Memory allocation!
        append!(xs, new_samples)
    end
    xs
end


function nsamples_available(chain::MCMCIterator{<:MCMCAlgorithm{MTMState}}, nonzero_weights::Bool = false)
    weight = mh_view_proposed(chain.state.samples.weight)  # Memory allocation!
    if nonzero_weights
        count(w -> w > 0, weight)
    else
        length(linearindices(weight))
    end
end


function get_samples!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{MTMState}}, nonzero_weights::Bool)::typeof(appendable)
    samples = mh_view_proposed(chain.state.samples)  # Memory allocation!
    if nonzero_weights
        idxs = find(w -> w > 0, samples.weight)  # Memory allocation!
        filtered_samples = view(samples, idxs)  # Memory allocation!
        append!(appendable, filtered_samples)
    else
        append!(appendable, samples)
    end
    appendable
end


function get_sample_ids!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{MTMState}}, nonzero_weights::Bool)::typeof(appendable)
    weight = mh_view_proposed(chain.state.samples.weight)  # Memory allocation!
    sampleids = mh_view_proposed(chain.state.sampleids)  # Memory allocation!
    if nonzero_weights
        idxs = find(w -> w > 0, weight)  # Memory allocation!
        filtered_sampleids = view(sampleids, idxs)  # Memory allocation!
        append!(appendable, filtered_sampleids)
    else
        append!(appendable, sampleids)
    end
    appendable
end



struct MultiTryMethod{
    Q<:ProposalDistSpec,
} <: MCMCAlgorithm{MTMState}
    q::Q
    nproposals::Int
    optimize_pt::Bool
    #eff_acceptratio_method::Int
end

export MultiTryMethod

MultiTryMethod(q::ProposalDistSpec = MvTDistProposalSpec(), nproposals::Int = 10, optimize_pt = Bool) =
    MultiTryMethod(q, nproposals, optimize_pt, 1)


mcmc_compatible(::MultiTryMethod, ::AbstractProposalDist, ::NoParamBounds) = true

mcmc_compatible(::MultiTryMethod, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)

sample_weight_type(::Type{<:MultiTryMethod}) = Float64  # ToDo: Allow for other floating point types



function MCMCIterator(
    algorithm::MultiTryMethod,
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = Vector{P}(),
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    target = likelihood * prior
    pdist = algorithm.q(P, nparams(target))

    cycle = zero(Int)
    reset_rng_counters!(rng, id, cycle, 0)

    params_vec = Vector{P}(nparams(target))
    if isempty(initial_params)
        rand_initial_params!(rng, algorithm, prior, params_vec)
    else
        params_vec .= initial_params
    end

    !(params_vec in param_bounds(target)) && throw(ArgumentError("Initial parameter(s) out of bounds"))

    m = length(params_vec)

    log_value = density_logval(target, params_vec, exec_context)
    L = typeof(log_value)
    W = sample_weight_type(typeof(algorithm))

    current_sample = DensitySample(
        params_vec,
        log_value,
        one(W)
    )

    state = MTMState(
        pdist,
        current_sample,
        algorithm.nproposals
    )

    chain = MCMCIterator(
        algorithm,
        target,
        state,
        rng,
        id,
        cycle,
        false,
        false
    )

    chain
end


exec_capabilities(mcmc_step!, callback::AbstractMCMCCallback, chain::MCMCIterator{<:MCMCAlgorithm{MTMState}}) =
    exec_capabilities(density_logval!, mh_view_proposed(chain.state.samples.log_value), chain.target, mh_view_proposed(chain.state.samples.params))


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MCMCAlgorithm{MTMState}},
    exec_context::ExecContext,
    ll::LogLevel
)
    state = chain.state
    algorithm = chain.algorithm

    if !mcmc_compatible(algorithm, chain.state.pdist, param_bounds(chain.target))
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    state.nsteps += 1
    reset_rng_counters!(chain)

    rng = chain.rng
    target = chain.target
    pdist = state.pdist

    samples = state.samples
    helpers_1 = state.help_samples_1
    helpers_2 = state.help_samples_2

    sampleids = state.sampleids

    params = samples.params
    help_params_1 = helpers_1.params
    help_params_2 = helpers_2.params

    weights = samples.weight
    help_weights_1 = helpers_1.weight
    help_weights_2 = helpers_2.weight

    logdensity = samples.log_value
    help_logdensity_1 = helpers_1.log_value
    help_logdensity_2 = helpers_2.log_value

    T = eltype(logdensity)

    current_sample_idx = mh_idx_current(samples)
    nproposed = length(mh_idxs_proposed(samples))

    @assert firstindex(samples) == firstindex(params) == firstindex(weights) == firstindex(logdensity)

    fill!(sampleids, MCMCSampleID(chain.id, chain.cycle, nsteps(state), 1))
    sampleids[current_sample_idx] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 0)

    # Propose new parameter values
    is_inbounds_1 = BitVector(size(help_params_1, 2))  # Memory allocation!
    is_inbounds_2 = BitVector(size(help_params_2, 2))  # Memory allocation!

    mtm_multi_propose!(rng, pdist, target, view(params, :, 1), help_params_1, is_inbounds)
    inbounds_idxs_1 = find(is_inbounds_1)  # Memory allocation!
    #@assert inbounds_idxs[mh_idx_current(inbounds_idxs)] == current_sample_idx
    #inbounds_idxs_proposed = inbounds_idxs[mh_idxs_proposed(inbounds_idxs)]  # Memory allocation!

    # Evaluate log(target) for all proposed parameter vectors
    help_params_proposed_inbounds_1 = help_params_1[:, inbounds_idxs_1]
    help_logdensity_proposed_inbounds_1 = help_logdensity_1[inbounds_idxs_1]
    density_logval!(help_logdensity_proposed_inbounds_1, target, help_params_proposed_inbounds_1, exec_context)
    logdensity[2] .= -Inf  # Memory allocation?
    #logdensity[inbounds_idxs_proposed] = logdensity_proposed_inbounds

    help_weights_proposed_inbounds_1 = help_weights_1[inbounds_idxs_1]

    _mtm_weight_1!(help_weights_proposed_inbounds_1, pdist, view(params, :, 1), help_params_proposed_inbounds_1, help_logdensity_proposed_inbounds_1)

    _select_candidate!(params, help_params_proposed_inbounds_1, logdensity, help_logdensity_proposed_inbounds_1, help_weights_proposed_inbounds_1)

    mtm_multi_propose!(rng, pdist, target, view(params, :, 2), help_params_2, is_inbounds)
    inbounds_idxs_2 = find(is_inbounds_2)

    help_params_proposed_inbounds_2 = help_params_2[:, inbounds_idxs_2]
    help_logdensity_proposed_inbounds_2 = help_logdensity_2[inbounds_idxs_2]
    density_logval!(help_logdensity_proposed_inbounds_2, target, help_params_proposed_inbounds_2, exec_context)
    help_weights_proposed_inbounds_2 = help_weights_1[inbounds_idxs_2]

    _mtm_weight_1!(help_weights_proposed_inbounds_2, pdist, view(params, :, 2), help_params_proposed_inbounds_2, help_logdensity_proposed_inbounds_2)

    p_accept = _MTM_accept_reject!(params, weights, help_weights_proposed_inbounds_1, help_weights_proposed_inbounds_2)


    if (weights[2] != 0.)
        state.nsamples += 1
        _swap!(samples, current_sample_idx, samples, current_sample_idx + 1)
        sampleids[current_sample_idx] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 0)
        sampleids[current_sample_idx + 1] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 2)
    end

    eff_acceptratio_method = algorithm.eff_acceptratio_method
    state.eff_acceptratio_sum += p_accept

    callback(1, chain)

    chain
end




function mtm_multi_propose!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, source::AbstractVector{<:Real}, params::AbstractMatrix{<:Real}, is_inbounds::BitVector) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(is_inbounds, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of inbound bools"))
    #current_params = view(params, :, 1)  # memory allocation
    #proposed_params = view(params, :, 2:size(params, 2))  # memory allocation

    # Propose new parameters:
    proposal_rand!(rng, pdist, params, source)
    #proposal_rand!(rng, pdist, params, current_params + rand(rng, pdist.s))
    apply_bounds!(params, param_bounds(target), false)

    is_inbounds .= false
    #is_inbounds[1] = true
    n_proposals = size(params, 2)
    n_proposals_inbounds = 0
    proposal_idxs_inbounds_tmp = Vector{Int}(n_proposals)  # Memory allocation
    fill!(proposal_idxs_inbounds_tmp, 0)
    @inbounds for j in indices(proposed_params, 2)
        if proposed_params[:, j] in param_bounds(target)
            is_inbounds[j+1] = true
        end
    end
    nothing
end



function _mtm_weight_1!(mtm_W::AbstractVector{<:AbstractFloat}, pdist::GenericProposalDist, source::AbstractVector{<:Real}, params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(params, 2) != indices(mtm_W, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of mtm_W"))

    # ToDo: Optimize for symmetric proposals?
    p_d = similar(logdensity, size(params, 2))
    distribution_logpdf!(p_d, pdist, params, source)  # Memory allocation due to view

    mtm_W .+= logdensity .+ p_d
    #mtmt_W[1] = 0.0
    mtm_W .-= maximum(mtm_W)
    mtm_W .= exp.(mtm_W)
    normalize!(mtm_W, 1)
    @assert sum(mtm_W) ≈ 1
    @assert mtmt_W[1] ≈ 0

    mtm_W
end


function _select_candidate!(params::AbstractMatrix{<:Real}, help_params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}, help_logdensity::Vector{<:Real}, help_weight::Vector{<:Real})
    indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(params, 2) != 2 && throw(ArgumentError("Number of parameter is not 2"))
    indices(help_params, 2) != indices(help_logdensity, 1) && throw(ArgumentError("Number of help_parameter sets doesn't match number of help_log(density) values"))
    indices(help_params, 2) != indices(help_weight, 1) && throw(ArgumentError("Number of help_parameter sets doesn't match number of help_weight values"))


    cum_help_weight = similar(help_weight)
    @assert firstindex(cum_help_weight) == firstindex(help_weight)
    cumsum!(cum_help_weight, help_weight)

    threshold = rand(rng, eltype(cum_help_weight))
    accepted_sample_idx = searchsortedfirst(cum_help_weight, threshold)

    params[:, 2] = help_params[:, accepted_sample_idx]
    logdensity[2] = help_logdensity[accepted_sample_idx]

end


function _MTM_accept_reject!(params::AbstractMatrix{<:Real}, weights::Vector{<:Real}, help_weights_1::Vector{<:Real}, help_weights_2::Vector{<:Real})
    indices(params, 2) != indices(weights, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of weight values"))
    indices(params, 2) != 2 && throw(ArgumentError("Number of parameter is not 2"))

    weights[2] = 0.
    firts_jump = sum(help_weights_1)
    second_jump = sum(help_weights_2)

    p_accept = if fisrt_jump / second_jump > -Inf && fisrt_jump / second_jump < Inf
        clamp(first_jump / second_jump, 0., 1.)
    else
        0.
    end

    threshold = rand(rng, eltype(cum_help_weight))

    if p_accept >= threshold
        weights[2] = 1.
    end

    p_accept

end









    function _mtm_weight_2!(mtm_W::AbstractVector{<:AbstractFloat}, pdist::GenericProposalDist, params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}) # TODO include checks for input, optimize and write test
        indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
        indices(params, 2) != indices(mtm_W, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of mtm_W"))


        mtm_W .+= logdensity
        #mtmt_W[1] = 0.0
        mtm_W .-= maximum(mtm_W)
        mtm_W .= exp.(mtm_W)
        mtmt_W[1] = 0.0
        normalize!(mtm_W, 1)
        @assert sum(mtm_W) ≈ 1
        @assert mtmt_W[1] ≈ 0

        mtm_W
    end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct MTMState{
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector,
    IV<:MCMCSampleIDVector
} <: AbstractMCMCState
    pdist::Q
    samples::SV  # First element is the current sample in the chain
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

    params = ElasticArray{T}(npar, nproposals + 1)
    fill!(params, zero(T))
    params[:, 1] = current_sample.params

    log_value = Vector{T}(nproposals + 1)
    fill!(log_value, NaN)
    log_value[1] = current_sample.log_value

    weight = Vector{W}(nproposals + 1)
    fill!(weight, zero(W))
    weight[1] = current_sample.weight

    samples = DensitySampleVector(params, log_value, weight)
    accepted = fill(false, )

    sampleids = append!(MCMCSampleIDVector(), fill(MCMCSampleID(-1, -1, -1, -1), nproposals + 1))

    eff_acceptratio_sum = zero(Double{Float64})
    nsamples = 0
    nsteps = 0

    MTMState(
        pdist,
        samples,
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
    eff_acceptratio_method::Int
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
    sampleids = state.sampleids
    params = samples.params
    weights = samples.weight
    logdensity = samples.log_value
    T = eltype(logdensity)

    current_sample_idx = mh_idx_current(samples)
    nproposed = length(mh_idxs_proposed(samples))

    @assert firstindex(samples) == firstindex(params) == firstindex(weights) == firstindex(logdensity)

    fill!(sampleids, MCMCSampleID(chain.id, chain.cycle, nsteps(state), 1))
    sampleids[current_sample_idx] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 0)

    # Propose new parameter values
    is_inbounds = BitVector(size(params, 2))  # Memory allocation!
    mh_multi_propose!(rng, pdist, target, params, is_inbounds)
    inbounds_idxs = find(is_inbounds)  # Memory allocation!
    @assert inbounds_idxs[mh_idx_current(inbounds_idxs)] == current_sample_idx
    inbounds_idxs_proposed = inbounds_idxs[mh_idxs_proposed(inbounds_idxs)]  # Memory allocation!

    # Evaluate log(target) for all proposed parameter vectors
    params_proposed_inbounds = params[:, inbounds_idxs_proposed]
    logdensity_proposed_inbounds = logdensity[inbounds_idxs_proposed]
    density_logval!(logdensity_proposed_inbounds, target, params_proposed_inbounds, exec_context)
    logdensity[2:end] .= -Inf  # Memory allocation?
    logdensity[inbounds_idxs_proposed] = logdensity_proposed_inbounds

    params_inbounds = params[:, inbounds_idxs]  # Memory allocation!
    logdensity_inbounds = logdensity[inbounds_idxs]  # Memory allocation!

    # Calculate Tjelmeland's P_T1
    P_T_inbounds = Vector{eltype(logdensity)}(size(logdensity_inbounds, 1))  # Memory allocation!
    P_T1_inbounds = similar(P_T_inbounds, size(logdensity_inbounds, 1))  # Memory allocation!

    @assert firstindex(P_T_inbounds) == firstindex(P_T1_inbounds) == firstindex(logdensity)

    _tjl_multipropT1!(P_T1_inbounds, pdist, params_inbounds, logdensity_inbounds)

    if algorithm.optimize_pt
        # Calculate Tjelmeland's P_T2
        P_T2_inbounds = similar(P_T1_inbounds)  # Memory allocation!
        @assert firstindex(P_T2_inbounds) == firstindex(P_T1_inbounds)
        _tjl_multipropT2!(P_T2_inbounds::AbstractVector{<:AbstractFloat}, P_T1_inbounds::AbstractVector{<:AbstractFloat})
        P_T_inbounds .= P_T2_inbounds
    else
        P_T_inbounds .= P_T1_inbounds
    end

    @assert sum(P_T_inbounds) ≈ 1
    @assert all(x -> x >= 0, P_T_inbounds)

    currsmpl_weight = weights[current_sample_idx]
    fill!(weights, 0)
    weights[inbounds_idxs] = P_T_inbounds
    weights[current_sample_idx] += currsmpl_weight

    cumP_T_inbounds = similar(P_T_inbounds)  # Memory allocation
    @assert firstindex(cumP_T_inbounds) == firstindex(P_T_inbounds)
    cumsum!(cumP_T_inbounds, P_T_inbounds)

    threshold = rand(rng, eltype(cumP_T_inbounds))
    accepted_sample_idx = inbounds_idxs[searchsortedfirst(cumP_T_inbounds, threshold)]

    eff_acceptratio_method = algorithm.eff_acceptratio_method
    if eff_acceptratio_method == 1
        # Use mean of classic-MH accept probabilitis of the proposals
        from = mh_idx_current(P_T_inbounds)
        to_idxs = mh_idxs_proposed(P_T_inbounds)
        if length(to_idxs) > 0
            T = Double{eltype(P_T_inbounds)}
            sum_mh_accpet_classic = zero(T)
            for to in to_idxs
                sum_mh_accpet_classic += T(mh_classic_accept_probability(pdist, params, logdensity, from, to))
            end
            mean_mh_accpet_classic = sum_mh_accpet_classic / length(to_idxs)
            state.eff_acceptratio_sum += mean_mh_accpet_classic
        end
    elseif eff_acceptratio_method == 2
        # Use total acceptance probability of proposed samples for eff_acceptratio_sum
        total_acceptance_prob = 1 - first(P_T_inbounds)
        state.eff_acceptratio_sum += total_acceptance_prob
        #println("Total acceptance prob is : $total_acceptance_prob")
    elseif eff_acceptratio_method == 3
        # Use mean acceptance probability of proposed samples for eff_acceptratio_sum
        mean_acceptance_prob = mean(mh_view_proposed(P_T_inbounds))  # Memory allocation!
        state.eff_acceptratio_sum += mean_acceptance_prob
    elseif eff_acceptratio_method == 4
        # Use maximum acceptance probability of proposed samples for eff_acceptratio_sum
        max_acceptance_prob = maximum(mh_view_proposed(P_T_inbounds))  # Memory allocation!
        state.eff_acceptratio_sum += max_acceptance_prob
    elseif eff_acceptratio_method == 5
        # Use minimum acceptance probability of proposed samples for eff_acceptratio_sum
        min_acceptance_prob = minimum(mh_view_proposed(P_T_inbounds))  # Memory allocation!
        state.eff_acceptratio_sum += min_acceptance_prob
    elseif eff_acceptratio_method == 6
        # Calculate addition to naccept based on entropy of proposed samples
        eff_nsamples = mh_entroy_eff_nsamples(P_T_inbounds)
        state.eff_acceptratio_sum += eff_nsamples / nproposed
    elseif eff_acceptratio_method == 7
        # Use maximum of classic-MH accept probabilitis of the proposals
        from = mh_idx_current(P_T_inbounds)
        to_idxs = mh_idxs_proposed(P_T_inbounds)
        if length(to_idxs) > 0
            T = Double{eltype(P_T_inbounds)}
            max_mh_accpet_classic = zero(T)
            for to in to_idxs
                max_mh_accpet_classic = max(max_mh_accpet_classic, T(mh_classic_accept_probability(pdist, params, logdensity, from, to)))
            end
            state.eff_acceptratio_sum += max_mh_accpet_classic
        end
    elseif eff_acceptratio_method == 8
        # Use minimum of classic-MH accept probabilitis of the proposals
        from = mh_idx_current(P_T_inbounds)
        to_idxs = mh_idxs_proposed(P_T_inbounds)
        if length(to_idxs) > 0
            T = Double{eltype(P_T_inbounds)}
            min_mh_accpet_classic = one(T)
            for to in to_idxs
                min_mh_accpet_classic = min(min_mh_accpet_classic, T(mh_classic_accept_probability(pdist, params, logdensity, from, to)))
            end
            state.eff_acceptratio_sum += min_mh_accpet_classic
        end
    else
        throw(ArgumentError("Invalid eff_acceptratio_method: $eff_acceptratio_method"))
    end

    if (accepted_sample_idx != current_sample_idx)
        state.nsamples += 1
        _swap!(samples, current_sample_idx, samples, accepted_sample_idx)
        sampleids[current_sample_idx] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 0)
        sampleids[accepted_sample_idx] = MCMCSampleID(chain.id, chain.cycle, nsteps(state), 2)
    end

    callback(1, chain)

    chain
end


function mh_multi_propose!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, params::AbstractMatrix{<:Real}, is_inbounds::BitVector) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(is_inbounds, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of inbound bools"))
    current_params = view(params, :, 1)  # memory allocation
    proposed_params = view(params, :, 2:size(params, 2))  # memory allocation

    # Propose new parameters:
    #proposal_rand!(rng, pdist, proposed_params, current_params)
    proposal_rand!(rng, pdist, proposed_params, current_params + rand(rng, pdist.s))
    apply_bounds!(proposed_params, param_bounds(target), false)

    is_inbounds .= false
    is_inbounds[1] = true
    n_proposals = size(proposed_params, 2)
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


function mh_classic_accept_probability(pdist::AbstractProposalDist, params::AbstractMatrix{P}, logdensity::AbstractVector{T}, from::Integer, to::Integer) where {P<:Real,T<:AbstractFloat}
    current_log_value = logdensity[from]
    proposed_log_value = logdensity[to]

    log_tpr = if issymmetric(pdist)
        zero(T)
    else
        current_params = view(params, :, 1)  # Memory allocation!
        proposed_params = view(params, :, 1)  # Memory allocation!
        log_tp_fwd = distribution_logpdf(pdist, proposed_params, current_params)
        log_tp_rev = distribution_logpdf(pdist, current_params, proposed_params)
        T(log_tp_fwd - log_tp_rev)
    end

    p_accept = if proposed_log_value > -Inf
        clamp(T(exp(proposed_log_value - current_log_value - log_tpr)), zero(T), one(T))
    else
        zero(T)
    end

    @assert !isnan(p_accept)
    p_accept
end


function mh_entroy_eff_nsamples(P::AbstractVector{<:Real})
    sum(P) ≈ 1 || throw(ArgumentError("Sum of propabilities of proposals must be one"))
    p_reject = first(P)
    P_accept = mh_view_proposed(P)  # Memory allocation!
    sum_p_accept = 1 - p_reject
    T = eltype(P)
    if sum_p_accept > 0
        nstates_eff = exp(mh_shannon_entropy(P_accept))
        T(sum_p_accept * nstates_eff)
    else
        zero(T)
    end
end


function mh_shannon_entropy(P::AbstractVector{<:Real}, norm_factor = inv(sum(P)))
    - sum(P) do p
        p2 = float(p * norm_factor)
        p2 ≈ 0 ? zero(p2) : p2 * log(p2)
    end
end


@inline mh_idx_current(A::AbstractVector) = first(linearindices(A))

@inline function mh_idxs_proposed(A::AbstractVector)
    idxs = linearindices(A)
    (first(idxs) + 1):last(idxs)
end

@inline mh_view_proposed(A::AbstractVector) =
    view(A, mh_idxs_proposed(A))


@inline mh_idx_current(A::AbstractMatrix) = first(axes(A, 2))

@inline function mh_idxs_proposed(A::AbstractMatrix)
    idxs = axes(A, 2)
    (first(idxs) + 1):last(idxs)
end

@inline mh_view_proposed(A::AbstractMatrix) =
    view(A, :, mh_idxs_proposed(A))

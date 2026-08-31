# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct StanLikeTuning <: MCMCTransformTuning

*BAT-internal, not part of stable public API.*

Tunes MCMC space transformations from sample covariance estimates,
accumulated over Stan-like adaptation windows of doubling size (see the
[Stan HMC parameters documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#hmc-algorithm-parameters)).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct StanLikeTuning <: MCMCTransformTuning
    "width of initial fast adaptation interval"
    init_buffer::Int = 75

    "width of final fast adaptation interval"
    term_buffer::Int = 50

    "initial width of slow adaptation interval"
    window_size::Int = 25
end


mutable struct StanLikeTunerState{S<:MCMCBasicStats} <: MCMCTransformTunerState
    tuning::StanLikeTuning
    stats::S
    i::Int
    window_start::Int
    window_end::Int
    window_splits::Vector{Int}
end

function create_trafo_tuner_state(tuning::StanLikeTuning, chain_state::MCMCChainState, n_steps_hint::Integer)
    chain_state.f_transform isa MulAdd || throw(ArgumentError(
        "StanLikeTuning requires an affine adaptive space transformation (like TriangularAffineTransform)"
    ))
    StanLikeTunerState(tuning, MCMCBasicStats(chain_state), 0, 0, 0, Int[])
end


# Stan's windowed adaptation schedule, see
# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function _stan_adaptation_windows(init_buffer::Integer, term_buffer::Integer, window_size::Integer, n_adapts::Integer)
    window_start = init_buffer + 1
    window_end = n_adapts - term_buffer

    window_splits = Int[]
    next_window = init_buffer + window_size
    while next_window <= window_end
        # Extend the current window to the end of the full window if the
        # remainder would be too short for another doubling:
        if next_window + 2 * window_size > window_end
            next_window = window_end
        end
        push!(window_splits, next_window)
        window_size *= 2
        next_window += window_size
    end
    if !isempty(window_splits) && last(window_splits) == n_adapts
        pop!(window_splits)
    end

    return window_start, window_end, window_splits
end

function _init_stanlike_tuner!(tuner::StanLikeTunerState, max_nsteps::Integer)
    tuning = tuner.tuning
    tuner.i = 0
    tuner.window_start, tuner.window_end, tuner.window_splits =
        _stan_adaptation_windows(tuning.init_buffer, tuning.term_buffer, tuning.window_size, Int(max_nsteps - 1))
    return nothing
end

function mcmc_trafo_tuning_init!!(tuner::StanLikeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _init_stanlike_tuner!(tuner, max_nsteps)
end

function mcmc_trafo_tuning_reinit!!(tuner::StanLikeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _init_stanlike_tuner!(tuner, max_nsteps)
end

function mcmc_tune_trafo_post_step!!(
    f_transform::Function,
    tuner::StanLikeTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    step_info::MCMCStepInfo
)
    tuner.i += 1

    if tuner.window_start <= tuner.i <= tuner.window_end
        # Accumulate the post-step chain states: proposals count only if
        # selected by accept/reject, each state enters with unit weight.
        # The sample weights on the step's samples are not assigned yet
        # when tuning runs, and Stan-style covariance estimation weights
        # every kept state equally:
        accepted = chain_state.accepted
        idxs = step_info.walker_order
        v = ifelse.(accepted[idxs], proposed.x.v[idxs], current.x.v[idxs])
        logd = ifelse.(accepted[idxs], proposed.x.logd[idxs], current.x.logd[idxs])
        foreach(
            sample -> push!(tuner.stats, sample),
            DensitySample.(v, logd, 1, nothing, nothing),
        )
    end

    f_transform_new = f_transform
    if tuner.i in tuner.window_splits
        A = f_transform.A
        T = eltype(A)
        n_dims = size(A, 2)

        M = convert(Array, tuner.stats.param_stats.cov)
        A_new = T.(cholesky(Positive, M).L)
        reweight_relative!(tuner.stats, 0)

        f_transform_new = MulAdd(A_new, zeros(T, n_dims))
    end

    return f_transform_new, tuner, chain_state
end

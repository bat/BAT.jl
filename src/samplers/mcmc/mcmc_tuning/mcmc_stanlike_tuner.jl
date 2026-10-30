# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct StanLikeTuning <: MCMCTransformTuning

*BAT-internal, not part of stable public API.*

Tunes MCMC space transformations from sample covariance estimates,
accumulated over Stan-like adaptation windows of doubling size (see the
[Stan HMC parameters documentation](https://mc-stan.org/docs/reference-manual/mcmc.html#hmc-algorithm-parameters)).

Each explicit tuning initialization or reinitialization clears the window
statistics, retaining the learned transformation. Iterator chunks and chain
cycle changes alone do not restart adaptation; initialization output does not
seed this tuner. Each window uses the kept post-step states with unit weights.

At initialization, buffers must be nonnegative, the initial window width
positive, and the nonnegative step horizon representable as `Int`. Horizons
zero and one are inactive. The schedule retains BAT's one-step margin and
omits a split at the last adaptation step; only windows with retained splits
collect observations. After the final split, the internal counter stops.

A window with fewer than two kept states leaves the transformation unchanged.
Otherwise, finite covariance estimates use the existing positive-factorization
policy, including for singular covariance. Nonfinite covariance or a
nonfinite/singular converted factor raises an error before installing new
geometry. Such an error need not roll back accumulated statistics or the step
index; reinitialize the tuner before reuse.

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
    @argcheck tuning.init_buffer >= 0
    @argcheck tuning.term_buffer >= 0
    @argcheck tuning.window_size > 0
    chain_state.f_transform isa MulAdd || throw(ArgumentError(
        "StanLikeTuning requires an affine adaptive space transformation (like TriangularAffineTransform)"
    ))
    StanLikeTunerState(tuning, MCMCBasicStats(chain_state), 0, 0, 0, Int[])
end


function _stanlike_step_count(value::Integer, name::Symbol; positive::Bool = false)
    lower = positive ? 1 : 0
    lower <= value <= typemax(Int) || throw(ArgumentError(
        "StanLikeTuning $name must be an integer in $lower:$(typemax(Int))"
    ))
    return Int(value)
end


# Stan-like windowed adaptation, retaining BAT's split locations; see
# https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function _stan_adaptation_windows(init_buffer::Integer, term_buffer::Integer, window_size::Integer, n_adapts::Integer)
    init_buffer = _stanlike_step_count(init_buffer, :init_buffer)
    term_buffer = _stanlike_step_count(term_buffer, :term_buffer)
    window_size = _stanlike_step_count(window_size, :window_size; positive = true)
    n_adapts = _stanlike_step_count(n_adapts, :n_adapts)

    window_splits = Int[]
    term_buffer <= n_adapts || return 1, 0, window_splits
    window_end = n_adapts - term_buffer
    init_buffer < window_end || return 1, 0, window_splits
    window_size <= window_end - init_buffer || return 1, 0, window_splits

    # The checks above bound both additions by window_end.
    window_start = init_buffer + 1
    next_window = init_buffer + window_size
    while true
        # Compare remaining distance before doubling: neither this test nor
        # the terminal iteration needs an overflowing lookahead/addition.
        if window_size > (window_end - next_window) ÷ 2
            push!(window_splits, window_end)
            break
        end
        push!(window_splits, next_window)
        # Here 2 * window_size <= window_end - next_window, so both
        # operations fit. Positive widths double, bounding the loop by
        # the number of bits in Int.
        window_size *= 2
        next_window += window_size
    end
    if last(window_splits) == n_adapts
        pop!(window_splits)
    end

    # Do not collect a tail whose closing split was omitted.
    isempty(window_splits) && return 1, 0, window_splits
    return window_start, last(window_splits), window_splits
end

function _init_stanlike_tuner!(tuner::StanLikeTunerState, max_nsteps::Integer)
    tuning = tuner.tuning
    n_steps = _stanlike_step_count(max_nsteps, :max_nsteps)
    n_adapts = n_steps == 0 ? 0 : n_steps - 1
    window_start, window_end, window_splits =
        _stan_adaptation_windows(tuning.init_buffer, tuning.term_buffer, tuning.window_size, n_adapts)

    # Validate/build first, then reset this tuner's owned state in place.
    # No chain-coordinate access: MultiTrafo passes its whole chain here.
    empty!(tuner.stats)
    tuner.i = 0
    tuner.window_start, tuner.window_end, tuner.window_splits = window_start, window_end, window_splits
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
    tuner.i >= tuner.window_end && return f_transform, tuner, chain_state
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
        if nsamples(tuner.stats) < 2
            empty!(tuner.stats)
            return f_transform, tuner, chain_state
        end

        A = f_transform.A
        T = eltype(A)
        n_dims = size(A, 2)

        M = convert(Array, tuner.stats.param_stats.cov)
        all(isfinite, M) || throw(ArgumentError(
            "StanLikeTuning cannot adapt from nonfinite window covariance"
        ))
        A_new = T.(cholesky(Positive, M).L)
        all(isfinite, A_new) && all(>(zero(T)), diag(A_new)) || throw(ArgumentError(
            "StanLikeTuning covariance factor must be finite with positive diagonal in $T"
        ))

        f_transform_new = MulAdd(A_new, zeros(T, n_dims))
        empty!(tuner.stats)
    end

    return f_transform_new, tuner, chain_state
end

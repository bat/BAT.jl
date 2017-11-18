# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCTunerConfig end
export AbstractMCMCTunerConfig


abstract type AbstractMCMCTuner end
export AbstractMCMCTuner


function mcmc_init end
export mcmc_init

function mcmc_tune_burnin! end
export mcmc_tune_burnin!

function isviable end


@with_kw struct MCMCInitStrategy
    ninit_tries_per_chain::ClosedInterval{Int64} = 8..128
    max_nsamples_pretune::Int64 = 25
    max_nsteps_pretune::Int64 = 250
    max_time_pretune::Float64 = Inf
end

export MCMCInitStrategy


MCMCInitStrategy(tuner_config::AbstractMCMCTunerConfig) =
    MCMCInitStrategy()

gen_tuners(ids::Range{<:Integer},
    chainspec::MCMCSpec,
    exec_context::ExecContext,
    tuner_config::AbstractMCMCTunerConfig,
    ) = [tuner_config(chainspec(id, exec_context), init_proposal = true) for id in ids]

function mcmc_init(
    chainspec::MCMCSpec,
    nchains::Int,
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config);
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "Trying to generate $nchains viable MCMC chain(s)."

    min_nviable = minimum(init_strategy.ninit_tries_per_chain) * nchains
    max_ncandidates = maximum(init_strategy.ninit_tries_per_chain) * nchains

    ncandidates = zero(Int64)

    tuners = similar([tuner_config(chainspec(zero(Int64), exec_context), init_proposal = false)], zero(Int))
    cycle = one(Int)
    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @log_msg ll+1 "Generating $n $(cycle > 1 ? "additional " : "")MCMC chain(s)."
        new_tuners = gen_tuners(ncandidates + (one(Int64):n), chainspec, exec_context, tuner_config)
        ncandidates += n

        @log_msg ll+1 "Testing $(length(new_tuners)) MCMC chain(s)."

        chains = map(x -> x.chain, new_tuners)

        run_tuning_iterations!(
            (), new_tuners, exec_context;
            max_nsamples = max(5, div(init_strategy.max_nsamples_pretune, 5)),
            max_nsteps =  max(50, div(init_strategy.max_nsteps_pretune, 5)),
            max_time = init_strategy.max_time_pretune / 5,
            ll = ll+2
        )

        filter!(isviable, new_tuners)
        @log_msg ll+1 "Found $(length(new_tuners)) viable MCMC chain(s)."

        if !isempty(new_tuners)
            run_tuning_iterations!(
                (), new_tuners, exec_context;
                max_nsamples = init_strategy.max_nsamples_pretune,
                max_nsteps = init_strategy.max_nsteps_pretune,
                max_time = init_strategy.max_time_pretune,
                ll = ll+2
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(t.chain.state) for t in new_tuners]))

            filter!(t -> nsamples(t.chain.state) >= nsamples_thresh, new_tuners)
            @log_msg ll+1 "Found $(length(new_tuners)) MCMC chain(s) with at least $(nsamples_thresh) samples."

            append!(tuners, new_tuners)
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

    m = nchains
    tidxs = linearindices(tuners)
    n = length(tidxs)

    mode_1 = tuners[1].stats.mode
    modes = Array{eltype(mode_1)}(length(mode_1), n)
    for i in tidxs
        modes[:,i] = tuners[i].stats.mode
    end

    final_tuners = similar(tuners, zero(Int))

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m)
        clusters.converged || error("k-means clustering of MCMC chains did not converge")

        mincosts = fill(Inf, m)
        tuner_sel = fill(zero(Int), m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                tuner_sel[j] = i
            end
        end

        @assert all(j -> j in tidxs, tuner_sel)

        for i in sort(tuner_sel)
            push!(final_tuners, tuners[i])
        end
    else
        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copy!(final_tuners, tuners)
    end


    @log_msg ll "Selected $(length(final_tuners)) MCMC chain(s)."

    final_tuners
end



@with_kw struct MCMCBurninStrategy
    max_nsamples_per_cycle::Int64 = 1000
    max_nsteps_per_cycle::Int64 = 10000
    max_time_per_cycle::Float64 = Inf
    max_ncycles::Int = 30
end

export MCMCBurninStrategy


MCMCBurninStrategy(tuner_config::AbstractMCMCTunerConfig) =
    MCMCBurninStrategy()

MCMCBurninStrategy(tuner::AbstractMCMCTuner) =
    MCMCBurninStrategy()


function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:AbstractMCMCTuner},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy,
    exec_context::ExecContext;
    strict_mode::Bool = false,
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "Begin tuning of $(length(tuners)) MCMC chain(s)."

    chains = map(x -> x.chain, tuners)
    nchains = length(chains)

    user_callbacks = mcmc_callback_vector(callbacks, chains)

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_strategy.max_ncycles
        cycles += 1
        run_tuning_cycle!(
            user_callbacks, tuners, exec_context;
            max_nsamples = burnin_strategy.max_nsamples_per_cycle,
            max_nsteps = burnin_strategy.max_nsteps_per_cycle,
            max_time = burnin_strategy.max_time_per_cycle,
            ll = ll+2
        )

        stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, stats, ll = ll+2)

        ntuned = count(c -> c.tuned, chains)
        nconverged = count(c -> c.converged, chains)
        successful = (ntuned == nconverged == nchains)

        for i in eachindex(user_callbacks, tuners)
            user_callbacks[i](1, tuners[i])
        end

        @log_msg ll+1 "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    if successful
        @log_msg ll "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
        if strict_mode
            error(msg)
        else
            @log_msg LOG_WARNING msg
        end
    end

    successful
end



struct NoOpTunerConfig <: BAT.AbstractMCMCTunerConfig end
export NoOpTunerConfig

(config::NoOpTunerConfig)(chain::MCMCIterator; kwargs...) =
    NoOpTuner(chain)



struct NoOpTuner{C<:MCMCIterator} <: AbstractMCMCTuner
    chain::C
end

export NoOpTuner


MCMCInitStrategy(tuner_config::NoOpTunerConfig) =
    MCMCInitStrategy(1..1, zero(Int), zero(Int), Inf)


MCMCBurninStrategy(tuner_config::NoOpTunerConfig) =
    MCMCBurninStrategy(zero(Int), zero(Int), Inf, zero(Int))


isviable(tuner::NoOpTuner) = true


function mcmc_init(
    chainspec::MCMCSpec,
    nchains::Int,
    exec_context::ExecContext,
    tuner_config::NoOpTunerConfig,
    convergence_test::MCMCConvergenceTest,
    init_strategy::MCMCInitStrategy;
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "NoOpTuner generating $nchains MCMC chain(s)."

    [tuner_config(chainspec(id, exec_context), init_proposal = true) for id in one(Int):nchains]
end


function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy,
    exec_context::ExecContext;
    ll::LogLevel = LOG_INFO,
    kwargs...
)
    @log_msg ll "Tune/Burn-In with NoOpTuner doing nothing."
end

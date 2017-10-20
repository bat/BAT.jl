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


struct MCMCInitStrategy
    ninit_tries_per_chain::ClosedInterval{Int64}
    max_nsamples_pretune::Int64
    max_nsteps_pretune::Int64
    max_time_pretune::Float64
end

export MCMCInitStrategy

MCMCInitStrategy(
    ;
    ninit_tries_per_chain::ClosedInterval{<:Integer} = 8..128,
    max_nsamples_pretune::Integer = Int64(25),
    max_nsteps_pretune::Integer = Int64(250),
    max_time_pretune::Real = Inf
) = MCMCInitStrategy(
    ninit_tries_per_chain,
    max_nsamples_pretune,
    max_nsteps_pretune,
    max_time_pretune
)

MCMCInitStrategy(tuner_config::AbstractMCMCTunerConfig) =
    MCMCInitStrategy()



function mcmc_init(
    chainspec::MCMCSpec,
    nchains::Integer,
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config);
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "Trying to generate $nchains viable MCMC chain(s)."

    min_nviable = minimum(init_strategy.ninit_tries_per_chain) * nchains
    max_ncandidates = maximum(init_strategy.ninit_tries_per_chain) * nchains

    gen_tuners(ids::Range{<:Integer}) =
        [tuner_config(chainspec(id, exec_context), init_proposal = true) for id in ids]

    @log_msg ll+1 "Generating $(min_nviable) MCMC chain(s)."
    initial_tuners = gen_tuners(1:min_nviable)
    ncandidates = maximum(min_nviable)

    tuners = similar(initial_tuners, 0)
    cycle = 1
    @log_trace "XXXXX: $ncandidates, $max_ncandidates"
    while length(tuners) < min_nviable && (cycle==1 || ncandidates < max_ncandidates)
        if cycle == 1
            new_tuners = initial_tuners
        else
            n = min(min_nviable, max_ncandidates - ncandidates)
            @log_msg ll+1 "Generating $n additional MCMC chain(s)."
            new_tuners = gen_tuners(ncandidates + (1:n))
            ncandidates += n
        end

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

    clusters = kmeans(modes, m)
    clusters.converged || error("k-means clustering of MCMC chains did not converge")

    mincosts = fill(Inf, m)
    tuner_sel = fill(0, m)

    for i in tidxs
        j = clusters.assignments[i]
        if clusters.costs[i] < mincosts[j]
            mincosts[j] = clusters.costs[i]
            tuner_sel[j] = i
        end
    end

    @assert all(j -> j in tidxs, tuner_sel)

    final_tuners = similar(initial_tuners, 0)
    for i in sort(tuner_sel)
        push!(final_tuners, tuners[i])
    end

    @log_msg ll "Selected $(length(final_tuners)) MCMC chain(s)."

    final_tuners
end



struct MCMCBurninStrategy
    max_nsamples_per_cycle::Int64
    max_nsteps_per_cycle::Int64
    max_time_per_cycle::Float64
    max_ncycles::Int
end

export MCMCBurninStrategy

MCMCBurninStrategy(
    ;
    max_nsamples_per_cycle::Integer = Int64(1000),
    max_nsteps_per_cycle::Integer = 10000,
    max_time_per_cycle::Real = Inf,
    max_ncycles::Integer = 30
) = MCMCBurninStrategy(
    max_nsamples_per_cycle,
    max_nsteps_per_cycle,
    max_time_per_cycle,
    max_ncycles::Int
)

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

    cycles = 0
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
    MCMCInitStrategy(1..1, 0, 0, Inf)


MCMCBurninStrategy(tuner_config::NoOpTunerConfig) =
    MCMCBurninStrategy(0, 0, Inf, 0)

# MCMCBurninStrategy(tuner::NoOpTuner) =
#     MCMCBurninStrategy(0, 0, Inf, 0)


isviable(tuner::NoOpTuner) = true


function run_tuning_cycle!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    exec_context::ExecContext;
    ll::LogLevel = LOG_NONE,
    kwargs...
)
    @log_msg ll "NoOpTuner tuning cycle, leaving MCMC chain unchanged."
    nothing
end



function mcmc_tune_burnin!(
    callbacks::AbstractVector{<:Function},
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:NoOpTuner},
    convergence_test::MCMCConvergenceTest,
    exec_context::ExecContext;
    ll::LogLevel = LOG_INFO,
    kwargs...
)
    @log_msg ll "Tune/Burn-In with NoOpTuner doing nothing."
end

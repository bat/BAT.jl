# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCTunerConfig end
export AbstractMCMCTunerConfig


abstract type AbstractMCMCTuner end
export AbstractMCMCTuner


function isviable end

function tuning_init! end
export tuning_init!

function mcmc_tune_burnin! end
export mcmc_tune_burnin!

function mcmc_init end
export mcmc_init



@with_kw struct MCMCBurninStrategy
    max_nsamples_per_cycle::Int64 = 1000
    max_nsteps_per_cycle::Int64 = 10000
    max_time_per_cycle::Float64 = Inf
    max_ncycles::Int = 30
end

export MCMCBurninStrategy

MCMCBurninStrategy(chainspec::MCMCSpec, nsamples::Integer, tuner_config::AbstractMCMCTunerConfig) =
    MCMCBurninStrategy()


function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:AbstractMCMCTuner},
    chains::AbstractVector{<:MCMCIterator},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy;
    strict_mode::Bool = false
)
    @info "Begin tuning of $(length(tuners)) MCMC chain(s)."

    nchains = length(chains)

    user_callbacks = mcmc_callback_vector(callbacks, eachindex(chains))

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_strategy.max_ncycles
        cycles += 1
        run_tuning_cycle!(
            user_callbacks, tuners, chains;
            max_nsamples = burnin_strategy.max_nsamples_per_cycle,
            max_nsteps = burnin_strategy.max_nsteps_per_cycle,
            max_time = burnin_strategy.max_time_per_cycle
        )

        stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, stats)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        for i in eachindex(user_callbacks, tuners)
            user_callbacks[i](1, tuners[i])
        end

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    if successful
        @info "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
        if strict_mode
            @error msg
        else
            @warn msg
        end
    end

    successful
end



struct NoOpTunerConfig <: BAT.AbstractMCMCTunerConfig end
export NoOpTunerConfig

(config::NoOpTunerConfig)(chain::MCMCIterator; kwargs...) = NoOpTuner()


struct NoOpTuner{C<:MCMCIterator} <: AbstractMCMCTuner end

export NoOpTuner


isviable(tuner::NoOpTuner, chain::MCMCIterator) = true


function tuning_init!(tuner::NoOpTuner, chain::MCMCIterator)
    nothing
end


function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    chains::AbstractVector{<:MCMCIterator},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy;
    kwargs...
)
    @debug "Tune/Burn-In with NoOpTuner doing nothing."
end



@with_kw struct MCMCInitStrategy
    ninit_tries_per_chain::ClosedInterval{Int64} = 8..128
    max_nsamples_pretune::Int64 = 25
    max_nsteps_pretune::Int64 = 250
    max_time_pretune::Float64 = Inf
end

export MCMCInitStrategy


MCMCInitStrategy(tuner_config::AbstractMCMCTunerConfig) =
    MCMCInitStrategy()

_gen_chains(ids::AbstractRange{<:Integer},
    chainspec::MCMCSpec,
    ) = [chainspec(id) for id in ids]

function mcmc_init(
    chainspec::MCMCSpec,
    nchains::Int,
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config)
)
    @info "Trying to generate $nchains viable MCMC chain(s)."

    min_nviable::Int = minimum(init_strategy.ninit_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_strategy.ninit_tries_per_chain) * nchains

    ncandidates::Int = 0

    dummy_chain = chainspec(zero(Int64))
    dummy_tuner = tuner_config(dummy_chain)

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    cycle::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")MCMC chain(s)."

        new_chains = _gen_chains(ncandidates .+ (one(Int64):n), chainspec)
        new_tuners = tuner_config.(new_chains)
        tuning_init!.(new_tuners, new_chains)
        ncandidates += n

        @debug "Testing $(length(new_tuners)) MCMC chain(s)."

        # ToDo: Use mcmc_iterate! instead of run_tuning_iterations! ?
        run_tuning_iterations!(
            (), new_tuners, new_chains;
            max_nsamples = max(5, div(init_strategy.max_nsamples_pretune, 5)),
            max_nsteps =  max(50, div(init_strategy.max_nsteps_pretune, 5)),
            max_time = init_strategy.max_time_pretune / 5
        )

        viable_idxs = findall(isviable.(new_tuners, new_chains))
        viable_tuners = new_tuners[viable_idxs]
        viable_chains = new_chains[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC chain(s)."

        if !isempty(viable_tuners)
            # ToDo: Use mcmc_iterate! instead of run_tuning_iterations! ?
            run_tuning_iterations!(
                (), viable_tuners, viable_chains;
                max_nsamples = init_strategy.max_nsamples_pretune,
                max_nsteps = init_strategy.max_nsteps_pretune,
                max_time = init_strategy.max_time_pretune
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_tuners)) MCMC chain(s) with at least $(nsamples_thresh) samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

    m = nchains
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    mode_1 = tuners[1].stats.mode
    modes = Array{eltype(mode_1)}(undef, length(mode_1), n)
    for i in tidxs
        modes[:,i] = tuners[i].stats.mode
    end

    final_chains = similar(chains, zero(Int))
    final_tuners = similar(tuners, zero(Int))

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m)
        clusters.converged || error("k-means clustering of MCMC chains did not converge")

        mincosts = fill(Inf, m)
        chain_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                chain_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, chain_sel_idxs)

        for i in sort(chain_sel_idxs)
            push!(final_chains, chains[i])
            push!(final_tuners, tuners[i])
        end
    else
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)
    end


    @info "Selected $(length(final_tuners)) MCMC chain(s)."

    (chains = final_chains, tuners = final_tuners)
end

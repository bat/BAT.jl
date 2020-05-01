AbstractMCMCTuningStrategy(algorithm::AHMC) = NoOpTunerConfig()



mutable struct HMCIterator{
    SP<:MCMCSpec,
    R<:AbstractRNG,
    PR<:RNGPartition,
    SV<:DensitySampleVector,
    I<:AdvancedHMC.AbstractIntegrator,
    P<:AdvancedHMC.AbstractProposal,
    A<:AdvancedHMC.AbstractAdaptor
} <: MCMCIterator
    spec::SP
    rng::R
    rngpart_cycle::PR
    info::MCMCIteratorInfo
    samples::SV
    nsamples::Int64
    stepno::Int64
    h::AdvancedHMC.Hamiltonian #TODO@C rename
    t::AdvancedHMC.Transition
    integrator::I
    proposal::P
    adaptor::A
end


function HMCIterator(
    rng::AbstractRNG,
    spec::MCMCSpec,
    info::MCMCIteratorInfo,
    x_init::AbstractVector{P},
) where {P<:Real}
    stepno::Int64 = 0

    postr = spec.posterior
    npar = totalndof(postr)
    alg = spec.algorithm

    params_vec = Vector{P}(undef, npar)
    if isempty(x_init)
        mcmc_startval!(params_vec, rng, postr, alg)
    else
        params_vec .= x_init
    end
    !(params_vec in var_bounds(postr)) && throw(ArgumentError("Parameter(s) out of bounds"))


    # ToDo: Make numeric type configurable:

    log_posterior_value = apply_bounds_and_eval_posterior_logval_strict!(postr, params_vec)

    T = typeof(log_posterior_value)
    W = Float64#_sample_weight_type(typeof(alg))

    sample_info = MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,MCMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    metric = AHMCMetric(alg.metric, npar)
    logval_posterior(v) = density_logval(postr, v)

    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, alg.gradient)
    hamiltonian, t = AdvancedHMC.sample_init(rng, hamiltonian, params_vec)

    ####
    alg.integrator.step_size == 0.0 ? alg.integrator.step_size = AdvancedHMC.find_good_stepsize(hamiltonian, params_vec) : nothing
    ahmc_integrator = AHMCIntegrator(alg.integrator)

    ahmc_proposal = AHMCProposal(alg.proposal, ahmc_integrator)
    ahmc_adaptor = AHMCAdaptor(alg.adaptor, metric, ahmc_integrator)
    ####


    chain = HMCIterator(
        spec,
        rng,
        rngpart_cycle,
        info,
        samples,
        nsamples,
        stepno,
        hamiltonian,
        t,
        ahmc_integrator,
        ahmc_proposal,
        ahmc_adaptor
    )

    reset_rng_counters!(chain)

    chain
end



function run_tuning_iterations!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    chains::AbstractVector{<:HMCIterator};
    max_nsamples::Int64 = Int64(1000),
    max_nsteps::Int64 = Int64(10000),
    max_time::Float64 = Inf
)
    user_callbacks = mcmc_callback_vector(callbacks, eachindex(chains))

    combined_callbacks = broadcast(tuners, user_callbacks) do tuner, user_callback
        (level, chain) -> begin
            if level == 1
                get_samples!(tuner.stats, chain, true)
            end
            user_callback(level, chain)
        end
    end

    mcmc_iterate!(combined_callbacks, chains, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time)
    nothing
end

function run_tuning_cycle!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    chains::AbstractVector{<:HMCIterator};
    kwargs...
)
    run_tuning_iterations!(callbacks, tuners, chains; kwargs...)
    nothing
end

# just burnin no tuning
function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    chains::AbstractVector{<:MCMCIterator},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy;
    strict_mode::Bool = false
)
    @info "Begin burnin of $(length(tuners)) MCMC chain(s)."

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

        new_stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, new_stats)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (nconverged == nchains)

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



function (spec::MCMCSpec{<:AHMC})(
    rng::AbstractRNG,
    chainid::Integer
)
    #P = float(eltype(var_bounds(spec.posterior)))
    P = Float64

    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)

    HMCIterator(rng, spec, info, Vector{P}())
end


mcmc_spec(chain::HMCIterator) = chain.spec

getrng(chain::HMCIterator) = chain.rng

mcmc_info(chain::HMCIterator) = chain.info

nsteps(chain::HMCIterator) = chain.stepno

nsamples(chain::HMCIterator) = chain.nsamples

current_sample(chain::HMCIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::HMCIterator) = eltype(chain.samples)

@inline _current_sample_idx(chain::HMCIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::HMCIterator) = lastindex(chain.samples)

function reset_rng_counters!(chain::HMCIterator)
    set_rng!(chain.rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(chain.rng, 0:(typemax(Int32) - 2))
    set_rng!(chain.rng, rngpart_step, chain.stepno)
    nothing
end


function samples_available(chain::HMCIterator)
    i = _current_sample_idx(chain::HMCIterator)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function _available_samples_idxs(chain::HMCIterator)
    sampletype = chain.samples.info.sampletype
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

function get_samples!(appendable, chain::HMCIterator, nonzero_weights::Bool)::typeof(appendable)
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


function next_cycle!(chain::HMCIterator)
    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.nsamples = 0
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)

    i = _current_sample_idx(chain)
    @assert chain.samples.info[i].sampletype == CURRENT_SAMPLE

    chain.samples.weight[i] = 1
    chain.samples.info[i] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE)

    chain
end


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::HMCIterator
)
    alg = algorithm(chain)
    #
    # if !mcmc_compatible(alg, chain.proposaldist, var_bounds(getposterior(chain)))
    #     error("Implementation of algorithm $alg does not support current parameter bounds with current proposal distribution")
    # end

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = getrng(chain)
    pstr = getposterior(chain)

    samples = chain.samples

    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, PROPOSED_SAMPLE)

    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    @assert current != proposed

    accepted = @uviews samples begin
        current_params = samples.v[current]
        proposed_params = samples.v[proposed]

        # Propose new variate:
        samples.weight[proposed] = 0

        ahmc_step!(rng, alg, chain, proposed_params, current_params)

        current_log_posterior = samples.logd[current]
        T = typeof(current_log_posterior)

        # Evaluate prior and likelihood with proposed variate:
        proposed_log_posterior = apply_bounds_and_eval_posterior_logval!(T, pstr, proposed_params)

        samples.logd[proposed] = proposed_log_posterior


        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        chain.nsamples += 1


        delta_w_current, w_proposed = (0, 1) # always accepted
        samples.weight[current] += delta_w_current
        samples.weight[proposed] = w_proposed

        callback(1, chain)

        current_params .= proposed_params
        samples.logd[current] = samples.logd[proposed]
        samples.weight[current] = samples.weight[proposed]
        samples.info[current] = samples.info[proposed]

        true
    end # @uviews


    resize!(samples, 1)


    chain
end


function ahmc_step!(rng, alg, chain, proposed_params, current_params)

    τ = chain.proposal
    adaptor = chain.adaptor
    h = chain.h
    t = chain.t

    # Make a step
    chain.t = AdvancedHMC.step(rng, h, τ, t.z)
    # Adapt h and τ; what mutable is the adaptor
    tstat = AdvancedHMC.stat(chain.t)
    chain.h, chain.proposal, isadapted = AdvancedHMC.adapt!(chain.h, chain.proposal, chain.adaptor, 0, 1, chain.t.z.θ, tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    #println("intern: ", t.z.θ)
    proposed_params[:] = chain.t.z.θ
    nothing
end
# function sample(
#     rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
#     h::Hamiltonian,
#     τ::AbstractProposal,
#     θ::T,
#     n_samples::Int,
#     adaptor::AbstractAdaptor=NoAdaptation(),
#     n_adapts::Int=min(div(n_samples, 10), 1_000);
#     drop_warmup=false,
#     verbose::Bool=true,
#     progress::Bool=false,
#     (pm_next!)::Function=pm_next!
# ) where {T<:AbstractVecOrMat{<:AbstractFloat}}
#     @assert !(drop_warmup && (adaptor isa Adaptation.NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
#     # Prepare containers to store sampling results
#     n_keep = n_samples - (drop_warmup ? n_adapts : 0)
#     θs, stats = Vector{T}(undef, n_keep), Vector{NamedTuple}(undef, n_keep)
#     # Initial sampling
#     h, t = sample_init(rng, h, θ)
#     # Progress meter
#     pm = progress ? ProgressMeter.Progress(n_samples, desc="Sampling", barlen=31) : nothing
#     time = @elapsed for i = 1:n_samples
#         # Make a step
#         t = step(rng, h, τ, t.z)
#         # Adapt h and τ; what mutable is the adaptor
#         tstat = stat(t)
#         h, τ, isadapted = adapt!(h, τ, adaptor, i, n_adapts, t.z.θ, tstat.acceptance_rate)
#         tstat = merge(tstat, (is_adapt=isadapted,))
#         # Update progress meter
#         if progress
#             # Do include current iteration and mass matrix
#             pm_next!(pm, (iterations=i, tstat..., mass_matrix=h.metric))
#         # Report finish of adapation
#         elseif verbose && isadapted && i == n_adapts
#             @info "Finished $n_adapts adapation steps" adaptor τ.integrator h.metric
#         end
#         # Store sample
#         if !drop_warmup || i > n_adapts
#             j = i - drop_warmup * n_adapts
#             θs[j], stats[j] = t.z.θ, tstat
#         end
#     end
#     # Report end of sampling
#     if verbose
#         EBFMI_est = EBFMI(map(s -> s.hamiltonian_energy, stats))
#         average_acceptance_rate = mean(map(s -> s.acceptance_rate, stats))
#         if θ isa AbstractVector
#             n_chains = 1
#         else
#             n_chains = size(θ, 2)
#             EBFMI_est = "[" * join(EBFMI_est, ", ") * "]"
#             average_acceptance_rate = "[" * join(average_acceptance_rate, ", ") * "]"
#         end
#         @info "Finished $n_samples sampling steps for $n_chains chains in $time (s)" h τ EBFMI_est average_acceptance_rate
#     end
#     return θs, stats
# end

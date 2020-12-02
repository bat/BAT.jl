export HamiltonianMC

"""
    HamiltonianMC <: MCMCAlgorithm

Hamiltonian Monte Carlo sampling algorithm.

* The arguments/options of `HamiltonianMC` are still subject to change, and not yet
part of stable public API.*
"""
@with_kw struct HamiltonianMC <: MCMCAlgorithm
    metric::HMCMetric = DiagEuclideanMetric()
    gradient::Module = ForwardDiff
    integrator::HMCIntegrator = LeapfrogIntegrator()
    proposal::HMCProposal = NUTS()
    adaptor::HMCAdaptor = StanHMCAdaptor()
end


get_mcmc_tuning(algorithm::HamiltonianMC) = MCMCNoOpTuning()


# MCMCIterator subtype for HamiltonianMC
mutable struct AHMCIterator{
    AL<:HamiltonianMC,
    D<:AbstractDensity,
    R<:AbstractRNG,
    PR<:RNGPartition,
    SV<:DensitySampleVector,
    I<:AdvancedHMC.AbstractIntegrator,
    P<:AdvancedHMC.AbstractProposal,
    A<:AdvancedHMC.AbstractAdaptor,
    H<:AdvancedHMC.Hamiltonian,
} <: MCMCIterator
    algorithm::AL
    density::D
    rng::R
    rngpart_cycle::PR
    info::MCMCIteratorInfo
    samples::SV
    nsamples::Int64
    stepno::Int64
    hamiltonian::H
    transition::AdvancedHMC.Transition
    integrator::I
    proposal::P
    adaptor::A
end


function AHMCIterator(
    rng::AbstractRNG,
    algorithm::MCMCAlgorithm,
    density::AbstractDensity,
    info::MCMCIteratorInfo,
    x_init::AbstractVector{P},
) where {P<:Real}
    stepno::Int64 = 0

    npar = totalndof(density)

    params_vec = Vector{P}(undef, npar)
    params_vec .= x_init
    !(params_vec in var_bounds(density)) && throw(ArgumentError("Parameter(s) out of bounds"))

    log_posterior_value = eval_logval(density, params_vec, strict = true)

    T = typeof(log_posterior_value)
    W = Float64 # ToDo: Support other sample weight types

    sample_info = AHMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE, 0.0, 0, false, 0.0)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,AHMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    metric = AHMCMetric(algorithm.metric, npar)
    logval_posterior(v) = eval_logval(density, v)

    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, algorithm.gradient)
    hamiltonian, t = AdvancedHMC.sample_init(rng, hamiltonian, params_vec)

    algorithm.integrator.step_size == 0.0 ? algorithm.integrator.step_size = AdvancedHMC.find_good_stepsize(hamiltonian, params_vec) : nothing
    ahmc_integrator = AHMCIntegrator(algorithm.integrator)

    ahmc_proposal = AHMCProposal(algorithm.proposal, ahmc_integrator)
    ahmc_adaptor = AHMCAdaptor(algorithm.adaptor, metric, ahmc_integrator)


    chain = AHMCIterator(
        algorithm,
        density,
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


function MCMCIterator(
    rng::AbstractRNG,
    algorithm::HamiltonianMC,
    density::AbstractDensity,
    chainid::Integer,
    startpos::AbstractVector{<:Real}
)
    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)
    AHMCIterator(rng, algorithm, density, info, startpos)
end


@inline _current_sample_idx(chain::AHMCIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::AHMCIterator) = lastindex(chain.samples)


getalgorithm(chain::AHMCIterator) = chain.algorithm

getdensity(chain::AHMCIterator) = chain.density

getrng(chain::AHMCIterator) = chain.rng

mcmc_info(chain::AHMCIterator) = chain.info

nsteps(chain::AHMCIterator) = chain.stepno

nsamples(chain::AHMCIterator) = chain.nsamples

current_sample(chain::AHMCIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::AHMCIterator) = eltype(chain.samples)



function reset_rng_counters!(chain::AHMCIterator)
    set_rng!(chain.rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(chain.rng, 0:(typemax(Int32) - 2))
    set_rng!(chain.rng, rngpart_step, chain.stepno)
    nothing
end


function samples_available(chain::AHMCIterator)
    i = _current_sample_idx(chain::AHMCIterator)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function get_samples!(appendable, chain::AHMCIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        samples = chain.samples

        for i in eachindex(samples)
            st = samples.info.sampletype[i]
            if (
                (st == ACCEPTED_SAMPLE || st == REJECTED_SAMPLE) &&
                (samples.weight[i] > 0 || !nonzero_weights)
            )
                push!(appendable, samples[i])
            end
        end
    end
    appendable
end


function next_cycle!(chain::AHMCIterator)
    _cleanup_samples(chain)

    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.nsamples = 0
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)

    i = _proposed_sample_idx(chain)
    @assert chain.samples.info[i].sampletype == CURRENT_SAMPLE
    chain.samples.weight[i] = 1

    t_stat = chain.transition.stat
    
    chain.samples.info[i] = AHMCSampleID(
        chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE,
        t_stat.hamiltonian_energy, t_stat.tree_depth,
        t_stat.numerical_error, t_stat.step_size
    )

    chain
end


function _cleanup_samples(chain::AHMCIterator)
    samples = chain.samples
    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    if (current != proposed) && samples.info.sampletype[proposed] == CURRENT_SAMPLE
        # Proposal was accepted in the last step
        @assert samples.info.sampletype[current] == ACCEPTED_SAMPLE
        samples.v[current] .= samples.v[proposed]
        samples.logd[current] = samples.logd[proposed]
        samples.weight[current] = samples.weight[proposed]
        samples.info[current] = samples.info[proposed]

        resize!(samples, 1)
    end
end


function mcmc_step!(chain::AHMCIterator)
    _cleanup_samples(chain)

    samples = chain.samples
    algorithm = getalgorithm(chain)

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = getrng(chain)
    density = getdensity(chain)


    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = AHMCSampleID(
        chain.info.id, chain.info.cycle, chain.stepno, PROPOSED_SAMPLE,
        0.0, 0, false, 0.0
    )

    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    @assert current != proposed

    current_params = samples.v[current]
    proposed_params = samples.v[proposed]

    # Propose new variate:
    samples.weight[proposed] = 0

    ahmc_step!(rng, algorithm, chain, proposed_params, current_params)
    tstat = chain.transition.stat
    
    current_log_posterior = samples.logd[current]
    T = typeof(current_log_posterior)

    # Evaluate prior and likelihood with proposed variate:
    proposed_log_posterior = eval_logval(density, proposed_params, strict = false)

    samples.logd[proposed] = proposed_log_posterior

    accepted = current_params != proposed_params

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        chain.nsamples += 1

        samples.info.hamiltonian_energy[proposed] = tstat.hamiltonian_energy
        samples.info.tree_depth[proposed] = tstat.tree_depth
        samples.info.divergent[proposed] = tstat.numerical_error
        samples.info.step_size[proposed] = tstat.step_size
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = if accepted
        (0, 1)
    else
        (1, 0)
    end
    
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed

    nothing
end


function ahmc_step!(rng, algorithm, chain, proposed_params, current_params)

    chain.transition = AdvancedHMC.step(rng, chain.hamiltonian, chain.proposal, chain.transition.z)

    tstat = AdvancedHMC.stat(chain.transition)

    if typeof(algorithm.adaptor) <: StanHMCAdaptor
        i = chain.adaptor.state.i
        n_adapts = algorithm.adaptor.n_adapts
    else
        i, n_adapts = chain.info.converged ? (3, 2) : (1, 1)
    end
    
    chain.hamiltonian, chain.proposal, isadapted = AdvancedHMC.adapt!(
        chain.hamiltonian,
        chain.proposal,
        chain.adaptor,
        Int(i),
        Int(n_adapts),
        chain.transition.z.θ,
        tstat.acceptance_rate
    )

   
    if i == n_adapts
        chain.info = MCMCIteratorInfo(chain.info, tuned = isadapted)
    end

    tstat = merge(tstat, (is_adapt=isadapted,))

    
    proposed_params[:] = chain.transition.z.θ
    nothing
end

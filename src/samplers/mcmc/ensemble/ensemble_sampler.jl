"""
    EnsembleProposal

Abstract super-type for ensemble-sampler walker move proposals.
"""
abstract type EnsembleProposal end
export EnsembleProposal



"""
    StretchMove

Ensemble sampler move proposal using the
[Goodman & Weare affine-invariant stretch move](https://doi.org/10.2140/camcos.2010.5.65).

Constructor:

```julia
StretchMove(a = 2)
```

* `a`: Maximum relative move distance.
"""
@with_kw struct StretchMove <: EnsembleProposal
    a::Float64 = 2
end
export StretchMove



"""
    EnsembleMC <: MCMCAlgorithm

Affine-invariant MCMC emseble sampling algorithm.

Parallelizes the ensemble using the
[emcee (D. Foreman-Mackey et al.)](arXiv:1202.3665) approach, by dividing the
ensemble of walkers into two sets.

Constructor:

```julia
EnsembleMC()
```

By default uses the
[Goodman & Weare stretch move](https://doi.org/10.2140/camcos.2010.5.65)
to propose new walker positions.
"""
struct EnsembleMC{M<:EnsembleProposal} <: MCMCAlgorithm
    nwalkers::Int
    proposal::M
end

export EnsembleMC

# ToDo: Allow for weighted set of possible moves


EnsembleMC(nwalkers::Integer) = EnsembleMC(nwalkers, StretchMove())



# MCMCSpec for EnsembleMC
function (spec::MCMCSpec{<:EnsembleMC})(
    rng::AbstractRNG,
    chainid::Integer
)
    # ToDo: Make numeric type configurable:
    P = Float64

    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)

    V_init = VectorOfSimilarVectors{P}



    EnsembleMCIterator(rng, spec, info, V_init)

    if isempty(x_init)
        mcmc_startval!(params_vec, rng, postr, alg)
    else
        params_vec .= x_init
    end
    !(params_vec in var_bounds(postr)) && throw(ArgumentError("Parameter(s) out of bounds"))

end



mutable struct EnsembleMCIterator{
    R<:AbstractRNG,
    D<:AbstractDensity,
    AL<:EnsembleMC,
    PR<:RNGPartition,
    SV<:DensitySampleVector,
    SVE<:AbstractArray{SV}
} <: MCMCIterator
    rng::R
    density::D
    algorithm::AL
    rngpart_cycle::PR
    info::MCMCIteratorInfo
    samples::SV
    ensembles::SVE
    nsamples::Int64
    stepno::Int64
end


function EnsembleMCIterator(
    rng::AbstractRNG,
    density::AbstractDensity,
    algorithm::EnsembleMC,
    info::MCMCIteratorInfo,
    P::Type{<:Real} = Float64
)
    V_init = nestedview(AbstractArray(undef, totalndof(density), algorithm.nwalkers))

    mcmc_startval!.(v, Ref(rng), Ref(density), Ref(alg))
    all(v -> v in var_bounds(postr), V_init) || throw(ArgumentError("Parameter(s) out of bounds"))

    EnsembleMCIterator(rng, density, algorithm, info, V_init)
end



_sample_weight_type(::Type{<:EnsembleMC) = Int


function EnsembleMCIterator(
    rng::AbstractRNG,
    density::AbstractDensity,
    algorithm::EnsembleMC,
    info::MCMCIteratorInfo,
    V_init::AbstractVector,
) where {P<:Real}
    stepno::Int64 = 0

    density = spec.posterior
    npar = totalndof(postr)
    alg = spec.algorithm

    logd_vals = apply_bounds_and_eval_posterior_logval_strict!.(Ref(density), V_init)

    T = eltype(logd_vals)
    W = _sample_weight_type(typeof(algorithm))

    sample_info = MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,MCMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    chain = MHIterator(
        spec,
        rng,
        rngpart_cycle,
        info,
        proposaldist,
        samples,
        nsamples,
        stepno
    )

    reset_rng_counters!(chain)

    chain
end



mcmc_spec(chain::EnsembleMCIterator) = chain.spec

getrng(chain::EnsembleMCIterator) = chain.rng

mcmc_info(chain::EnsembleMCIterator) = chain.info

nsteps(chain::EnsembleMCIterator) = chain.stepno

nsamples(chain::EnsembleMCIterator) = chain.nsamples

current_sample(chain::EnsembleMCIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::EnsembleMCIterator) = eltype(chain.samples)

@inline _current_sample_idx(chain::EnsembleMCIterator) = firstindex(chain.samples)

@inline _proposed_sample_idx(chain::EnsembleMCIterator) = lastindex(chain.samples)


function reset_rng_counters!(chain::EnsembleMCIterator)
    set_rng!(chain.rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(chain.rng, 0:(typemax(Int32) - 2))
    set_rng!(chain.rng, rngpart_step, chain.stepno)
    nothing
end


function samples_available(chain::EnsembleMCIterator)
    i = _current_sample_idx(chain::EnsembleMCIterator)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function _available_samples_idxs(chain::EnsembleMCIterator)
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


function get_samples!(appendable, chain::EnsembleMCIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        idxs = _available_samples_idxs(chain)
        samples = chain.samples

        @uviews samples begin
            # if nonzero_weights # TODO ?
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


function next_cycle!(chain::EnsembleMCIterator)
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


AbstractMCMCTuningStrategy(algorithm::EnsembleMC) = OnlyBurninTunerConfig()


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::EnsembleMCIterator
)
    alg = algorithm(chain)

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
    chain.transition = AdvancedHMC.step(rng, chain.hamiltonian, chain.proposal, chain.transition.z)

    tstat = AdvancedHMC.stat(chain.transition)
    i, nadapt = chain.info.converged ? (3, 2) : (1, 1)

    chain.hamiltonian, chain.proposal, isadapted = AdvancedHMC.adapt!(chain.hamiltonian,
                                                                    chain.proposal,
                                                                    chain.adaptor,
                                                                    i, nadapt,
                                                                    chain.transition.z.θ,
                                                                    tstat.acceptance_rate)
    tstat = merge(tstat, (is_adapt=isadapted,))

    proposed_params[:] = chain.transition.z.θ
    nothing
end

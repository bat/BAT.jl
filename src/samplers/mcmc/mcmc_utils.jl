# TODO AC: File not included as it would overwrite BAT.jl functions


function _cov_with_fallback(m::BATMeasure)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, m)))
    n = totalndof(varshape(m))
    C = fill(T(NaN), n, n)
    try
        C[:] = cov(m)
    catch err
        if err isa MethodError
            C[:] = cov(nestedview(rand(rng, m, 10^5)))
        else
            throw(err)
        end
    end
    return C
end

_approx_cov(target::BATMeasure) = _cov_with_fallback(target)
_approx_cov(target::BAT.BATDistMeasure) = _cov_with_fallback(target)
_approx_cov(target::AbstractPosteriorMeasure) = _approx_cov(getprior(target))
#_approx_cov(target::BAT.Transformed{<:Any,<:BAT.DistributionTransform}) =
#    BAT._approx_cov(target.trafo.target_dist)
#_approx_cov(target::Renormalized) = _approx_cov(parent(target))
#_approx_cov(target::WithDiff) = _approx_cov(parent(target))

function MCMCSampleID(iter::TransformedMCMCIterator, sampletype::Int64)
    MCMCSampleID(iter.info.id, iter.info.cycle, iter.stepno, sampletype)
end

function _rebuild_density_sample(s::DensitySample, x, logd, weight=1)
    @unpack info, aux = s
    DensitySample(x, logd, weight, info, aux)
end

function reset_rng_counters!(chain::TransformedMCMCIterator)
    rng = get_rng(get_context(chain))
    set_rng!(rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, chain.stepno)
    nothing
end

function samples_available(chain::TransformedMCMCIterator)
    i = _current_sample_idx(chain)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end

function get_samples!(appendable, chain::TransformedMCMCIterator, nonzero_weights::Bool)::typeof(appendable)
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

function _cleanup_samples(chain::TransformedMCMCIterator)
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

# TODO: MD Discuss, how should this act on the transformed iterator?
function next_cycle!(chain::TransformedMCMCIterator)
    _cleanup_samples(chain)

    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    #chain.nsamples = 0 # TODO: Should this reset n_accepted ? 
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)

    i = _proposed_sample_idx(chain)
    @assert chain.samples.info[i].sampletype == CURRENT_SAMPLE
    chain.samples.weight[i] = 1

    chain.samples.info[i] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE)

    chain
end

function _mcmc_weights(
    algorithm::RepetitionWeighting,
    p_accept::Real,
    accepted::Bool
) where Q
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end

function _mcmc_weights(
    algorithm::ARPWeighting,
    p_accept::Real,
    accepted::Bool
) where Q
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end

function _update_iter_transform!(iter::TransformedMCMCIterator, f_transform::Function)
    @unpack samples, sample_z, μ = iter
    proposed_x = _proposed_sample_idx(iter)
    proposed_z = 2

    samples.v[proposed_x], ladj = with_logabsdet_jacobian(f_transform, sample_z.v[proposed_z])
    samples.logd[proposed_x] = BAT.checked_logdensityof(μ, samples.v[proposed_x])
    sample_z.logd[proposed_z] = samples.logd[proposed_x] + ladj

    iter.f_transform = f_transform
    nothing
end

# TODO MD: Relocate functions
(tuning::AdaptiveMHTuning)(chain::MHIterator) = ProposalCovTuner(tuning, chain)
(tuning::AdaptiveMHTuning)(chain::TransformedMCMCIterator) = TransformedProposalCovTuner(TransformedAdaptiveMHTuning(tuning.λ, tuning.α, tuning.β, tuning.c, tuning.r), chain) # TODO: MD: Remove, temporary wrapper


#=
# Unused?
function reset_chain(
    rng::AbstractRNG,
    chain::TransformedMCMCIterator,
)
    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))
    #TODO reset cycle count?
    chain.rngpart_cycle = rngpart_cycle
    chain.info = MCMCIteratorInfo(chain.info, cycle=0)
    chain.context = set_rng(chain.context, rng)
    # wants a next_cycle!
    # reset_rng_counters!(chain)
end
=#


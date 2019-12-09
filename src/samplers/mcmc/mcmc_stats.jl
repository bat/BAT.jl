# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCStats end
AbstractMCMCStats


Base.convert(::Type{AbstractMCMCCallback}, x::AbstractMCMCStats) = MCMCAppendCallback(x)

MCMCAppendCallback(x::AbstractMCMCStats, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_samples!, nonzero_weights)



struct MCMCNullStats <: AbstractMCMCStats end


Base.push!(stats::MCMCNullStats, sv::DensitySampleVector) = stats

Base.append!(stats::MCMCNullStats, sv::DensitySampleVector) = stats



struct MCMCBasicStats{L<:Real,P<:Real} <: AbstractMCMCStats
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{L,FrequencyWeights}
    mode::Vector{P}

    function MCMCBasicStats{L,P}(m::Integer) where {L<:Real,P<:Real}
        param_stats = BasicMvStatistics{P,FrequencyWeights}(m)
        logtf_stats = BasicUvStatistics{L,FrequencyWeights}()
        mode = fill(oob(P), m)

        new{L,P}(
            param_stats,
            logtf_stats,
            mode
        )
    end
end


function MCMCBasicStats(::Type{S}, nparams::Integer) where {
    PT<:Real, T, W, S<:DensitySample{<:AbstractVector{PT},T,W}
}
    SL = promote_type(T, Float64)
    SP = promote_type(PT, W, Float64)
    MCMCBasicStats{SL,SP}(nparams)
end

MCMCBasicStats(chain::MCMCIterator) = MCMCBasicStats(sample_type(chain), nparams(chain))

function MCMCBasicStats(sv::DensitySampleVector)
    stats = MCMCBasicStats(eltype(sv), innersize(sv.params, 1))
    append!(stats, sv)
end


function Base.empty!(stats::MCMCBasicStats)
    empty!(stats.param_stats)
    empty!(stats.logtf_stats)
    fill!(stats.mode, oob(eltype(stats.mode)))

    stats
end


function Base.push!(stats::MCMCBasicStats, s::DensitySample)
    push!(stats.param_stats, s.params, s.weight)
    if s.logdensity > stats.logtf_stats.maximum
        stats.mode .= s.params
    end
    push!(stats.logtf_stats, s.logdensity, s.weight)
    stats
end


function Base.append!(stats::MCMCBasicStats, sv::DensitySampleVector)
    for i in eachindex(sv)
        p = sv.params[i]
        w = sv.weight[i]
        l = sv.logdensity[i]
        push!(stats.param_stats, p, w)  # Memory allocation (view)!
        if sv.logdensity[i] > stats.logtf_stats.maximum
            stats.mode .= p  # Memory allocation (view)!
        end
        push!(stats.logtf_stats, l, w)
        stats
    end
    stats
end


nparams(stats::MCMCBasicStats) = stats.param_stats.m

nsamples(stats::MCMCBasicStats) = stats.param_stats.cov.sum_w

function Base.merge!(target::MCMCBasicStats, others::MCMCBasicStats...)
    for x in others
        if (x.logtf_stats.maximum > target.logtf_stats.maximum)
            target.mode .= x.mode
        end
        merge!(target.param_stats, x.param_stats)
        merge!(target.logtf_stats, x.logtf_stats)
    end
    target
end

Base.merge(a::MCMCBasicStats, bs::MCMCBasicStats...) = merge!(deepcopy(a), bs...)


function _bat_stats(mcmc_stats::MCMCBasicStats)
    (
        mode = mcmc_stats.mode,
        mean = mcmc_stats.param_stats.mean,
        cov = mcmc_stats.param_stats.cov
    )
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCStats end
AbstractMCMCStats



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
        mode = fill(P(NaN), m)

        new{L,P}(
            param_stats,
            logtf_stats,
            mode
        )
    end
end


function MCMCBasicStats(::Type{S}, ndof::Integer) where {
    PT<:Real, T, W, S<:DensitySample{<:AbstractVector{PT},T,W}
}
    SL = promote_type(T, Float64)
    SP = promote_type(PT, W, Float64)
    MCMCBasicStats{SL,SP}(ndof)
end

MCMCBasicStats(chain::MCMCIterator) = MCMCBasicStats(sample_type(chain), totalndof(getdensity(chain)))

function MCMCBasicStats(sv::DensitySampleVector{<:AbstractVector{<:Real}})
    stats = MCMCBasicStats(eltype(sv), innersize(sv.v, 1))
    append!(stats, sv)
end

MCMCBasicStats(sv::DensitySampleVector) = MCMCBasicStats(unshaped.(sv))


function Base.empty!(stats::MCMCBasicStats)
    empty!(stats.param_stats)
    empty!(stats.logtf_stats)
    fill!(stats.mode, eltype(stats.mode)(NaN))

    stats
end


function Base.push!(stats::MCMCBasicStats, s::DensitySample)
    push!(stats.param_stats, s.v, s.weight)
    if s.logd > stats.logtf_stats.maximum
        stats.mode .= s.v
    end
    push!(stats.logtf_stats, s.logd, s.weight)
    stats
end


function Base.append!(stats::MCMCBasicStats, sv::DensitySampleVector)
    for i in eachindex(sv)
        p = sv.v[i]
        w = sv.weight[i]
        l = sv.logd[i]
        push!(stats.param_stats, p, w)  # Memory allocation (view)!
        if sv.logd[i] > stats.logtf_stats.maximum
            stats.mode .= p  # Memory allocation (view)!
        end
        push!(stats.logtf_stats, l, w)
        stats
    end
    stats
end


ValueShapes.totalndof(stats::MCMCBasicStats) = stats.param_stats.m

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


function reweight_relative!(stats::MCMCBasicStats, reweighting_factor::Real)
    reweight_relative!(stats.param_stats, reweighting_factor)
    reweight_relative!(stats.logtf_stats, reweighting_factor)

    stats
end


function _bat_stats(mcmc_stats::MCMCBasicStats)
    (
        mode = mcmc_stats.mode,
        mean = mcmc_stats.param_stats.mean,
        cov = mcmc_stats.param_stats.cov
    )
end

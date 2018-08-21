# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCStats end
export AbstractMCMCStats


Base.convert(::Type{AbstractMCMCCallback}, x::AbstractMCMCStats) = MCMCAppendCallback(x)

MCMCAppendCallback(x::AbstractMCMCStats, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_samples!, nonzero_weights)



struct MCMCNullStats <: AbstractMCMCStats end
export MCMCNullStats

Base.push!(stats::MCMCNullStats, sv::DensitySampleVector) = stats

Base.append!(stats::MCMCNullStats, sv::DensitySampleVector) = stats



struct MCMCBasicStats{L<:Real,P<:Real} <: AbstractMCMCStats
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{L,FrequencyWeights}
    mode::Vector{P}

    function MCMCBasicStats{L,P}(m::Integer) where {L<:Real,P<:Real}
        param_stats = BasicMvStatistics{P,FrequencyWeights}(m)
        logtf_stats = BasicUvStatistics{L,FrequencyWeights}()
        mode = Vector{P}(size(param_stats.mean, 1))

        new{L,P}(
            param_stats,
            logtf_stats,
            mode
        )
    end
end

export MCMCBasicStats

function MCMCBasicStats(::Type{S}, nparams::Integer) where {P,T,W,S<:DensitySample{P,T,W}}
    SL = promote_type(T, Float64)
    SP = promote_type(P, W, Float64)
    MCMCBasicStats{SL,SP}(nparams)
end

MCMCBasicStats(chain::MCMCIterator) = MCMCBasicStats(density_sample_type(chain.state), nparams(chain.state))

function MCMCBasicStats(sv::DensitySampleVector)
    stats = MCMCBasicStats(eltype(sv), size(sv.params, 1))
    append!(stats, sv)
end


function Base.push!(stats::MCMCBasicStats, s::DensitySample)
    push!(stats.param_stats, s.params, s.weight)
    if s.log_value > stats.logtf_stats.maximum
        stats.mode .= s.params
    end
    push!(stats.logtf_stats, s.log_value, s.weight)
    stats
end


function Base.append!(stats::MCMCBasicStats, sv::DensitySampleVector)
    for i in eachindex(sv)
        push!(stats.param_stats, view(sv.params, :, i), sv.weight[i])  # Memory allocation (view)!
        if sv.log_value[i] > stats.logtf_stats.maximum
            stats.mode .= view(sv.params, :, i)  # Memory allocation (view)!
        end
        push!(stats.logtf_stats, sv.log_value[i], sv.weight[i])
        stats
    end
    stats
end


nparams(stats::MCMCBasicStats) = stats.param_stats.m


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

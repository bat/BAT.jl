# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCStats end
export AbstractMCMCStats


function Base.push!(stats::AbstractMCMCStats, state::AbstractMCMCState)
    if sample_available(state)
        push!(stats, current_sample(state))
    end
    stats
end

function Base.push!(stats::AbstractMCMCStats, chain::MCMCChain)
    push!(stats, chain.state)
    chain
end

mcmc_callback(x::AbstractMCMCStats) = MCMCPushCallback(x)
mcmc_callback(max_level::Integer, x::AbstractMCMCStats) = MCMCPushCallback(max_level, x)



struct MCMCNullStats <: AbstractMCMCStats end
export MCMCNullStats

Base.push!(stats::MCMCNullStats, s::MCMCSample) = stats



struct MCMCBasicStats{L<:Real,P<:Real} <: AbstractMCMCStats
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{L,FrequencyWeights}
    mode::Vector{P}

    function MCMCBasicStats{L,P}(m::Integer) where {L<:Real,P<:Real}
        param_stats = BasicMvStatistics{P,FrequencyWeights}(m)
        logtf_stats = BasicUvStatistics{L,FrequencyWeights}()
        mode = Vector{P}(size(param_stats.mean, 1))

        new{L,P}(
            BasicMvStatistics{P,FrequencyWeights}(m),
            BasicUvStatistics{L,FrequencyWeights}(),
            fill(oob(P), m)
        )
    end
end

export MCMCBasicStats


MCMCBasicStats(chain::MCMCChain) = MCMCBasicStats(chain.state)


function Base.push!(stats::MCMCBasicStats, s::MCMCSample)
    push!(stats.param_stats, s.params, s.weight)
    if s.log_value > stats.logtf_stats.maximum
        stats.mode .= s.params
    end
    push!(stats.logtf_stats, s.log_value, s.weight)
    stats
end

nparams(stats::MCMCBasicStats) = stats.param_stats.m

# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MCMCConvergenceTest end

abstract type MCMCConvergenceTestResult end


function check_convergence!(
    ct::MCMCConvergenceTest,
    chains::AbstractVector{<:MCMCIterator},
    stats::AbstractVector{<:AbstractMCMCStats};
    ll::LogLevel = LOG_NONE
)
    result = check_convergence(ct, stats, ll = ll)
    for chain in chains
        chain.converged = result.converged
    end
    result
end



doc"""
    gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})

Gelman-Rubin $R^2$ for all parameters.
"""
function gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})
    m = nparams(first(stats))
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:m], 1)[:]
    B = var([cs.param_stats.mean[i] for cs in stats, i in 1:m], 1)[:]
    (W .+ B) ./ W
end



doc"""
    GRConvergence

Gelman-Rubin $maximum(R^2)$ convergence test.
"""
struct GRConvergence <: MCMCConvergenceTest
    threshold::Float64
end

GRConvergence() = GRConvergence(1.1)

export GRConvergence


struct GRConvergenceResult <: MCMCConvergenceTestResult
    converged::Bool
    max_Rsqr::Float64
end


function check_convergence(ct::GRConvergence, stats::AbstractVector{<:MCMCBasicStats}; ll::LogLevel = LOG_NONE)
    max_Rsqr = maximum(gr_Rsqr(stats))
    converged = max_Rsqr <= ct.threshold
    @log_msg ll begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(max_Rsqr), threshold = $(ct.threshold)"
    end
    GRConvergenceResult(converged, max_Rsqr)
end

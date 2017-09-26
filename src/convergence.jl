# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MCMCConvergenceTest end


doc"""
    gr_Rsqr(chains_stats::AbstractVector{<:MCMCChainStats})

Gelman-Rubin $R^2$ for all parameters.
"""
function gr_Rsqr(chains_stats::AbstractVector{<:MCMCChainStats})
    m = nparams(first(chains_stats))
    W = mean([cs.param_stats.cov[i,i] for cs in chains_stats, i in 1:m], 1)[:]
    B = var([cs.param_stats.mean[i] for cs in chains_stats, i in 1:m], 1)[:]
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


function (ct::GRConvergence)(chains_stats::AbstractVector{<:MCMCChainStats})
    maximum(gr_Rsqr(chains_stats)) <= ct.threshold
end

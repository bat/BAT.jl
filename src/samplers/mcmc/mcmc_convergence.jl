# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MCMCConvergenceTest end

abstract type MCMCConvergenceTestResult end


function check_convergence!(
    ct::MCMCConvergenceTest,
    chains::AbstractVector{<:MCMCIterator},
    samples::AbstractVector{<:DensitySampleVector}
)
    result = check_convergence(ct, samples)
    for chain in chains
        chain.info = MCMCIteratorInfo(chain.info, converged = result.converged)
    end
    result
end



@doc doc"""
    gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})
    gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})

*BAT-internal, not part of stable public API.*

Gelman-Rubin ``\$R^2\$`` for all DOF.
"""
function gr_Rsqr end

function gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})
    m = totalndof(first(stats))
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:m], dims=1)[:]
    B = var([cs.param_stats.mean[i] for cs in stats, i in 1:m], dims=1)[:]
    (W .+ B) ./ W
end

function gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})
    gr_Rsqr(MCMCBasicStats.(samples))
end



@doc doc"""
    GelmanRubinConvergence

Gelman-Rubin ``\$maximum(R^2)\$`` convergence test.
"""
@with_kw struct GelmanRubinConvergence <: MCMCConvergenceTest
    threshold::Float64 = 1.1
end

export GelmanRubinConvergence


struct GRConvergenceResult <: MCMCConvergenceTestResult
    converged::Bool
    max_Rsqr::Float64
end


function check_convergence(ct::GelmanRubinConvergence, samples::AbstractVector{<:DensitySampleVector})
    max_Rsqr = maximum(gr_Rsqr(samples))
    converged = max_Rsqr <= ct.threshold
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(max_Rsqr), threshold = $(ct.threshold)"
    end
    GRConvergenceResult(converged, max_Rsqr)
end



@doc doc"""
    bg_R_2sqr(stats::AbstractVector{<:MCMCBasicStats}; corrected::Bool = false)
    bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)

*BAT-internal, not part of stable public API.*

Brooks-Gelman $R_2^2$ for all DOF.
If normality is assumed, 'corrected' should be set to true to account for the sampling variability.
"""
function bg_R_2sqr(stats::AbstractVector{<:MCMCBasicStats}; corrected::Bool = false)
    p = totalndof(first(stats))
    m = length(stats)
    n = mean(Float64.(nsamples.(stats)))
    
    σ_W = var([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]
    B  = var([cs.param_stats.mean[i] for cs in stats, i in 1:p], dims = 1)[:]
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]

    σ_sq = m * (n - 1) / (m*n - 1) * W + n * (m - 1) / (m*n - 1) * B
    
    R_unc = σ_sq ./ W

    if corrected == false
        return R_unc
    end

    σ_ij = [cs.param_stats.cov[i,i] for cs in stats, i in 1:p]
    x_ij = [cs.param_stats.mean[i] for cs in stats, i in 1:p]
    
    cov_σx = [cov(σ_ij[:,j], x_ij[:,j]) for j in 1:p]
    cov_σx_sq = [cov(σ_ij[:,j], x_ij[:,j].^2) for j in 1:p]
    
    N = (n-1)/n
    M = (m-1)/m
    V = N*σ_sq + M*B

    σ_V = N^2/m*σ_W + 2*M/(m-1)*B.^2 + 2*M*N/m*(cov_σx_sq - 2*B.*cov_σx)
    d = 2 * V.^2 ./ σ_V

    R_unc.*(d.+3)./(d.+1)
end

function bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)
    bg_R_2sqr(MCMCBasicStats.(samples), corrected = corrected)
end



@doc doc"""
    BrooksGelmanConvergence

Brooks-Gelman $maximum(R^2)$ convergence test.
"""
@with_kw struct BrooksGelmanConvergence <: MCMCConvergenceTest
    threshold::Float64 = 1.1
    corrected::Bool = false
end

export BrooksGelmanConvergence


struct BGConvergenceResult <: MCMCConvergenceTestResult
    converged::Bool
    max_Rsqr::Float64
end


function check_convergence(ct::BrooksGelmanConvergence, samples::AbstractVector{<:DensitySampleVector})
    max_Rsqr = maximum(bg_R_2sqr(samples, corrected = ct.corrected))
    converged = max_Rsqr <= ct.threshold
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(max_Rsqr), threshold = $(ct.threshold)"
    end
    GRConvergenceResult(converged, max_Rsqr)
end

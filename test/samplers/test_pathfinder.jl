# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using StableRNGs
import ForwardDiff
import Optim

using BAT: pathfinder_gaussian_fit

@testset "pathfinder" begin
    context = BATContext(rng = StableRNG(996770566), ad = ForwardDiff)
    rng = StableRNG(996770566)

    d = 2
    Cinv = Diagonal([1.0, 0.5])
    m_true = randn(rng, d)
    f_logd = x -> -dot(x - m_true, Cinv, x - m_true) / 2

    lbfgs_alg(; kwargs...) = OptimAlg(optalg = Optim.LBFGS(); kwargs...)

    fit = pathfinder_gaussian_fit(
        f_logd,
        fill(4.0, d),
        lbfgs_alg(),
        context,
        history_length = 8,
        ndraws_elbo = 16,
    )
    @test fit !== nothing && norm(fit.μ - m_true) < 0.5
end

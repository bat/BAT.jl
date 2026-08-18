# This file is a part of BAT.jl, licensed under the MIT License (MIT).
#
# HMC geometry-adaptation benchmark: compares transform tuners and
# structures on difficult target geometries. Run manually, e.g.
#
#     julia --project=<env-with-BAT> benchmark/hmc_geometry_benchmark.jl
#
# Metrics per (target, tuner) cell: moment errors of the samples, total
# divergent trajectories, total leapfrog steps (gradient evaluations),
# effective sample size per 1000 gradient evaluations, final step sizes
# and wall time.

using BAT
using LinearAlgebra, Random, Statistics, StatsBase, Printf
using Distributions, ValueShapes, DensityInterface
import ForwardDiff

context = BATContext(ad = ForwardDiff)

# ================= targets =================

function _corr_gauss(d, ρ)
    Σ = Matrix(Symmetric([ρ^abs(i - j) * 1.0 for i in 1:d, j in 1:d]))
    (measure = batmeasure(MvNormal(fill(2.0, d), Σ)), ref = MvNormal(fill(2.0, d), Σ))
end

function _illcond_gauss(d)
    σs = exp10.(range(-1, 2, length = d))
    dist = MvNormal(zeros(d), Diagonal(σs .^ 2))
    (measure = batmeasure(dist), ref = dist)
end

function _funnel(dx)
    prior = distprod(v = Normal(0.0, 1.5), x = MvNormal(zeros(dx), I))
    loglik = logfuncdensity(
        p -> sum(logpdf.(Normal.(0.0, exp(p.v / 2)), p.x)) - sum(logpdf.(Normal(0.0, 1.0), p.x))
    )
    (measure = PosteriorMeasure(loglik, prior), ref = nothing)
end

function _banana(; b = 0.5)
    prior = distprod(a = Normal(0, 2), c = Normal(0, 2))
    loglik = logfuncdensity(p -> logpdf(Normal(p.a^2 * b, 0.5), p.c) - logpdf(Normal(0, 2), p.c))
    (measure = PosteriorMeasure(loglik, prior), ref = nothing)
end

targets = [
    "corr-gauss(6, 0.95)" => _corr_gauss(6, 0.95),
    "ill-cond(8, 1e3)" => _illcond_gauss(8),
    "funnel(3)" => _funnel(3),
    "banana" => _banana(),
]

# ================= tuner configurations =================

tuners = [
    "Fisher-dense" => (BAT.TriangularAffineTransform(), nothing),
    "Fisher-diag" => (BAT.DiagonalAffineTransform(), nothing),
    "Fisher-lowrank" => (BAT.LowRankAffineTransform(), nothing),
    "StanLike" => (BAT.TriangularAffineTransform(), BAT.StanLikeTuning()),
    "RAM" => (BAT.TriangularAffineTransform(), RAMTuning()),
]

# ================= run =================

function run_cell(target, tuner_cfg; nsteps = 10^4)
    at, tt = tuner_cfg
    alg = isnothing(tt) ?
        TransformedMCMC(proposal = HamiltonianMC(), pretransform = DoNotTransform(), adaptive_transform = at, nsteps = nsteps, strict = false) :
        TransformedMCMC(proposal = HamiltonianMC(), pretransform = DoNotTransform(), adaptive_transform = at, transform_tuning = tt, nsteps = nsteps, strict = false)
    t0 = time()
    em = evalmeasure(target.measure, alg, deepcopy(context))
    walltime = time() - t0

    smpls = BAT.samplesof(em)
    diags = BAT.evalinfo(em).result.chain_diagnostics
    n_div = sum(d.n_divergent for d in diags)
    n_leapfrog = sum(d.n_leapfrog for d in diags)
    ess = BAT.getess(BAT.empiricalof(em))

    moment_err = if !isnothing(target.ref)
        us = unshaped.(smpls)
        w = Weights(us.weight)
        m_est = mean(us.v, w)
        s_est = std(us.v, w, corrected = false)
        m_ref, s_ref = mean(target.ref), sqrt.(var(target.ref))
        max(maximum(abs.(m_est .- m_ref) ./ s_ref), maximum(abs.(s_est .- s_ref) ./ s_ref))
    else
        NaN
    end

    (
        moment_err = moment_err,
        n_divergent = n_div,
        ess_per_kgrad = 1000 * ess / n_leapfrog,
        n_leapfrog = n_leapfrog,
        walltime = walltime,
    )
end

println("\n=== HMC geometry-adaptation benchmark ===\n")
results = Dict()
for (tname, target) in targets, (aname, cfg) in tuners
    r = try
        run_cell(target, cfg)
    catch err
        @warn "cell failed" target = tname tuner = aname exception = (err, catch_backtrace())
        nothing
    end
    results[(tname, aname)] = r
    if !isnothing(r)
        @printf(
            "%-22s %-15s momerr %-8.3f div %-6d ess/kgrad %-8.1f leapfrog %-10d t %6.1fs\n",
            tname, aname, r.moment_err, r.n_divergent, r.ess_per_kgrad, r.n_leapfrog, r.walltime
        )
    end
end

println("\n(momerr = worst |mean/std error| in units of target std; NaN = no exact reference)")

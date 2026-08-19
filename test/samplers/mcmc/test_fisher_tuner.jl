# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface, InverseFunctions
using StableRNGs
import ForwardDiff

using Accessors: @set

using BAT: DenseFisherEstimator, DiagonalFisherEstimator, LowRankFisherEstimator,
    DriftCommitSchedule, FisherTransformTuning, DiagonalAffineTransform,
    LowRankAffineTransform, _new_moments, _moments_update!, _fisher_geometry,
    _spd_riccati_solve, _transform_drift, _fisher_A

# A pathological transform-tuner finalizer that changes the transform
# type, used to test the finalization contract:
struct _TypeChangingTrafoFinalizer <: BAT.MCMCTransformTunerState end
BAT.mcmc_trafo_tuning_finalize!!(f_transform::Function, tuner::_TypeChangingTrafoFinalizer, chain_state::BAT.MCMCIterator) = (identity, tuner, chain_state)

@testset "fisher_tuner" begin
    rng = StableRNG(438621057)

    @testset "Fisher geometry recovery" begin
        # For a Gaussian N(μ, Σ) the score is α = -Σ⁻¹(x - μ), and the
        # Fisher-optimal affine geometry is exactly G = Σ, μ* = μ:
        d = 4
        A_true = LowerTriangular(Matrix(I(d)) + 0.4 * randn(rng, d, d))
        Σ = Matrix(Symmetric(A_true * A_true' + 0.1 * I))
        Σinv = inv(Σ)
        μ_true = randn(rng, d)

        acc_dense = _new_moments(DenseFisherEstimator(), d)
        acc_diag = _new_moments(DiagonalFisherEstimator(), d)
        for _ in 1:10^4
            x = μ_true .+ cholesky(Σ).L * randn(rng, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc_dense, x, α)
            _moments_update!(acc_diag, x, α)
        end

        G, μ = _fisher_geometry(DenseFisherEstimator(), acc_dense, 1e-5)
        @test opnorm(Matrix(G) - Σ) / opnorm(Σ) < 0.1
        @test isapprox(μ, μ_true, atol = 0.2)

        # The diagonal variant recovers sqrt(Var(x) / Var(α)) per dimension,
        # with the regularization relative to the mean variance scales:
        G_diag, _ = _fisher_geometry(DiagonalFisherEstimator(), acc_diag, 1e-5)
        var_x = acc_diag.M2_x ./ (acc_diag.n - 1)
        var_g = acc_diag.M2_g ./ (acc_diag.n - 1)
        γ_x = 1e-5 * sum(var_x) / d
        γ_g = 1e-5 * sum(var_g) / d
        @test diag(G_diag) ≈ sqrt.((var_x .+ γ_x) ./ (var_g .+ γ_g))

        # Affine equivariance of the regularized geometry: rescaling the
        # target by c scales positions by c and scores by 1/c, so the
        # learned geometry must scale by c² even for extreme scales where
        # an absolute regularization floor would swamp the score covariance:
        c = 1e6
        acc_scaled = _new_moments(DenseFisherEstimator(), d)
        rng_eq = StableRNG(438621057)
        for _ in 1:10^3
            x = μ_true .+ cholesky(Σ).L * randn(rng_eq, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc_scaled, c .* x, α ./ c)
        end
        acc_unit = _new_moments(DenseFisherEstimator(), d)
        rng_eq = StableRNG(438621057)
        for _ in 1:10^3
            x = μ_true .+ cholesky(Σ).L * randn(rng_eq, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc_unit, x, α)
        end
        G_scaled, _ = _fisher_geometry(DenseFisherEstimator(), acc_scaled, 1e-5)
        G_unit, _ = _fisher_geometry(DenseFisherEstimator(), acc_unit, 1e-5)
        @test Matrix(G_scaled) ≈ c^2 .* Matrix(G_unit) rtol = 1e-6
    end

    @testset "Riccati solve and drift metric" begin
        d = 5
        R1, R2 = randn(rng, d, d), randn(rng, d, d)
        C_x = Symmetric(R1 * R1' + 0.1 * I)
        C_g = Symmetric(R2 * R2' + 0.1 * I)
        G = _spd_riccati_solve(C_x, C_g)
        @test Matrix(G * C_g * G) ≈ Matrix(C_x)

        A = LowerTriangular(Matrix(cholesky(Symmetric(Matrix(G))).L))
        # The installed geometry itself has zero drift:
        @test _transform_drift(A, G) < 1e-8
        # A pure rescaling G -> c² G has drift |log(c²)| √d:
        c2 = 4.0
        @test _transform_drift(A, Symmetric(c2 * Matrix(G))) ≈ log(c2) * sqrt(d)
    end

    @testset "low-rank geometry recovery" begin
        # A diagonal base geometry with a strong rank-1 correction: the
        # low-rank estimator must identify the correction direction and
        # reproduce the full geometry as G = D + W S Wᵀ:
        d = 6
        u = normalize(randn(rng, d))
        base = Diagonal(collect(range(0.5, 4.0, length = d)))
        Σ = Matrix(Symmetric(base + 12.0 * u * u'))
        Σinv = inv(Σ)
        μ_true = randn(rng, d)

        est = LowRankFisherEstimator(1.5, 0)
        acc = _new_moments(est, d)
        L = cholesky(Symmetric(Σ)).L
        for _ in 1:10^4
            x = μ_true .+ L * randn(rng, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc, x, α)
        end

        G, μ = _fisher_geometry(est, acc, 1e-5)
        @test opnorm(Matrix(G) - Σ) / opnorm(Σ) < 0.15
        @test isapprox(μ, μ_true, atol = 0.4)

        # The committed matrix part is a Gram factor of G:
        A = _fisher_A(est, G)
        @test Matrix(A) * Matrix(A)' ≈ Matrix(G)

        # A hard rank cap is respected:
        est_capped = LowRankFisherEstimator(1.5, 1)
        G1, _ = _fisher_geometry(est_capped, acc, 1e-5)
        @test size(G1.B, 2) <= 1  # W has at most max_rank columns in the Woodbury representation
    end

    @testset "guards" begin
        context = BATContext(ad = ForwardDiff)
        target = unshaped(batmeasure(NamedTupleDist(a = Normal(), b = Normal())))
        # Fisher tuning requires a gradient-based proposal:
        alg_rw = TransformedMCMC(
            proposal = RandomWalk(), transform_tuning = FisherTransformTuning(),
            nchains = 1, nsteps = 100
        )
        @test_throws ArgumentError bat_sample(target, alg_rw, context)
    end

    @testset "end-to-end geometry learning" begin
        context = BATContext(ad = ForwardDiff)
        Σ = [4.0 1.2 0.0; 1.2 2.0 -0.5; 0.0 -0.5 1.0]
        objective = MvNormal([1.0, -2.0, 0.5], Σ)
        target = batmeasure(objective)

        # No pretransform: the Fisher tuner has to learn the full geometry:
        alg = TransformedMCMC(
            proposal = HamiltonianMC(),
            pretransform = DoNotTransform(),
            nchains = 2,
            nsteps = 10^4
        )
        @test alg.transform_tuning isa FisherTransformTuning

        em = evalmeasure(target, alg, context)
        smpls = BAT.samplesof(em)
        @test BAT.test_dist_samples(objective, smpls, context)

        # The learned affine transform reproduces the target geometry:
        gen = BAT.samplegenof(em)
        f = gen.chain_states[1].f_transform
        G_learned = Matrix(f.A * f.A')
        @test opnorm(G_learned - Σ) / opnorm(Σ) < 0.35
        @test isapprox(f.b, mean(objective), atol = 0.5)

        # Trajectory diagnostics are recorded in the evaluation info,
        # split into warmup and retained sampling:
        diags = BAT.evalinfo(em).result.chain_diagnostics
        @test length(diags) == 2
        @test all(d -> d.n_transitions > 0, diags)
        @test all(d -> 0 < d.mean_p_accept <= 1, diags)
        @test all(d -> d.n_leapfrog > 0, diags)
        @test all(d -> d.warmup.n_transitions > 0, diags)
        @test all(d -> d.sampling.n_transitions >= alg.nsteps, diags)
        @test all(d -> d.warmup.n_transitions + d.sampling.n_transitions == d.n_transitions, diags)
        @test all(d -> 0 < d.sampling.mean_p_accept <= 1, diags)
    end

    @testset "structure selection end-to-end" begin
        context = BATContext(ad = ForwardDiff)

        # Independent scales: the diagonal structure suffices:
        objective_diag = MvNormal([0.5, -1.0, 2.0], Diagonal([0.04, 4.0, 25.0]))
        alg_diag = TransformedMCMC(
            proposal = HamiltonianMC(),
            adaptive_transform = DiagonalAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 2,
            nsteps = 6000
        )
        em_diag = evalmeasure(batmeasure(objective_diag), alg_diag, context)
        @test BAT.test_dist_samples(objective_diag, BAT.samplesof(em_diag), context)
        f_diag = BAT.samplegenof(em_diag).chain_states[1].f_transform
        @test f_diag.A isa Diagonal
        @test isapprox(diag(f_diag.A * f_diag.A'), [0.04, 4.0, 25.0], rtol = 0.6)

        # Diagonal base plus one strong correlation direction: low-rank
        # picks it up while keeping the correction small:
        u = normalize(fill(1.0, 4))
        Σ_lr = Matrix(Symmetric(Diagonal([1.0, 2.0, 0.5, 1.5]) + 8.0 * u * u'))
        objective_lr = MvNormal(zeros(4), Σ_lr)
        alg_lr = TransformedMCMC(
            proposal = HamiltonianMC(),
            adaptive_transform = LowRankAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 2,
            nsteps = 6000
        )
        em_lr = evalmeasure(batmeasure(objective_lr), alg_lr, context)
        @test BAT.test_dist_samples(objective_lr, BAT.samplesof(em_lr), context)
        f_lr = BAT.samplegenof(em_lr).chain_states[1].f_transform
        G_lr = Matrix(f_lr.A * Matrix(f_lr.A)')
        @test opnorm(G_lr - Σ_lr) / opnorm(Σ_lr) < 0.5
    end

    @testset "tuning freeze" begin
        context = BATContext(ad = ForwardDiff)
        # After mcmc_tuning_finalize!! the transition kernel must be fixed:
        # no transform commits, no step-size adaptation, no Fisher moment
        # accumulation during post-tuning stabilization or retained
        # sampling:
        prior = distprod(a = Normal(0.0, 10.0), b = Normal(0.0, 10.0))
        loglik = logfuncdensity(p -> logpdf(Normal(3.0, 0.5), p.a) + logpdf(Normal(-2.0, 0.5), p.b))
        target = unshaped(PosteriorMeasure(loglik, prior))
        alg = TransformedMCMC(
            proposal = HamiltonianMC(),
            adaptive_transform = DiagonalAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 1, nwalkers = 1, nsteps = 1000
        )
        mcmc_state = BAT.MCMCState(alg, target, 1, [randn(rng, 2)], deepcopy(context))
        BAT.mcmc_tuning_init!!(mcmc_state, 400)
        BAT.mcmc_tuning_reinit!!(mcmc_state, 400)
        mcmc_state = BAT.mcmc_iterate!!(nothing, mcmc_state; max_nsteps = 400, nonzero_weights = false)

        # A finalizer must not change the transform type (geometry changes
        # are only valid through the transform-commit path):
        mcmc_state_bad = @set mcmc_state.trafo_tuner_state = _TypeChangingTrafoFinalizer()
        @test_throws ErrorException BAT.mcmc_tuning_finalize!!(mcmc_state_bad)

        mcmc_state = BAT.mcmc_tuning_finalize!!(mcmc_state)

        @test mcmc_state.trafo_tuner_state isa BAT.FrozenMCMCTransformTunerState
        @test mcmc_state.proposal_tuner_state isa BAT.FrozenMCMCProposalTunerState

        f_frozen = mcmc_state.chain_state.f_transform
        step_frozen = BAT.get_active_proposal(mcmc_state.chain_state.proposal).step_size

        # Tuning finalization does not yet mark the warmup end in the
        # diagnostics, that only happens after post-tuning stabilization:
        d_fin = BAT._proposal_diagnostics(BAT.get_active_proposal(mcmc_state.chain_state.proposal))
        @test d_fin.n_transitions > 0
        @test d_fin.warmup.n_transitions == 0

        mcmc_state = BAT.mcmc_iterate!!(nothing, mcmc_state; max_nsteps = 100, nonzero_weights = false)
        BAT.mcmc_mark_warmup_end!(mcmc_state.chain_state.proposal)
        d_marked = BAT._proposal_diagnostics(BAT.get_active_proposal(mcmc_state.chain_state.proposal))
        @test d_marked.warmup.n_transitions == d_marked.n_transitions
        @test d_marked.sampling.n_transitions == 0

        mcmc_state = BAT.mcmc_iterate!!(nothing, mcmc_state; max_nsteps = 600, nonzero_weights = false)
        cs = mcmc_state.chain_state
        @test cs.f_transform === f_frozen
        @test BAT.get_active_proposal(cs.proposal).step_size == step_frozen
        @test mcmc_state.trafo_tuner_state isa BAT.FrozenMCMCTransformTunerState

        # Everything after the warmup mark counts as retained sampling:
        d_end = BAT._proposal_diagnostics(BAT.get_active_proposal(cs.proposal))
        @test d_end.sampling.n_transitions == d_end.n_transitions - d_marked.warmup.n_transitions
        @test d_end.sampling.n_transitions >= 600
    end

    @testset "forced geometry commit" begin
        context = BATContext(ad = ForwardDiff)
        # The prior-based initial geometry is badly mismatched to this
        # sharp posterior, forcing geometry commits; the commit protocol
        # must leave positions consistent and the step size readapted:
        prior = distprod(a = Normal(0.0, 10.0), b = Normal(0.0, 10.0))
        loglik = logfuncdensity(p -> logpdf(Normal(3.0, 0.5), p.a) + logpdf(Normal(-2.0, 0.5), p.b))
        target = unshaped(PosteriorMeasure(loglik, prior))
        alg = TransformedMCMC(
            proposal = HamiltonianMC(),
            adaptive_transform = DiagonalAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 1, nwalkers = 1, nsteps = 1000
        )
        v_init = [randn(rng, 2)]
        mcmc_state = BAT.MCMCState(alg, target, 1, v_init, deepcopy(context))
        f_initial = mcmc_state.chain_state.f_transform
        @test f_initial.A isa Diagonal
        # The initial geometry comes from the prior (scale 10), far from
        # the posterior scale 0.5:
        @test all(diag(f_initial.A) .> 5)
        BAT.mcmc_tuning_init!!(mcmc_state, 500)
        BAT.mcmc_tuning_reinit!!(mcmc_state, 500)
        mcmc_state = BAT.mcmc_iterate!!(nothing, mcmc_state; max_nsteps = 500, nonzero_weights = false)

        cs = mcmc_state.chain_state
        # A commit happened (the transform is a new object) and the learned
        # geometry contracted towards the posterior scale:
        @test cs.f_transform !== f_initial
        @test all(diag(cs.f_transform.A * cs.f_transform.A') .< 4)
        # The step size was re-searched and adapted in the new geometry:
        prop = BAT.get_active_proposal(cs.proposal)
        @test isfinite(prop.step_size) && prop.step_size > 0
        # Walker positions stayed consistent across the geometry changes:
        @test cs.current.z.v[1] ≈ inverse(cs.f_transform)(cs.current.x.v[1])
        @test isapprox(cs.current.x.logd[1], logdensityof(target, cs.current.x.v[1]), rtol = 1e-6, atol = 1e-6)
    end
end

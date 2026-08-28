# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface, InverseFunctions
using StableRNGs
using Random123
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

function _fixed_lowrank_campaign(draw_score, d::Int, seed::Int)
    rng = StableRNG(seed)
    estimator = LowRankFisherEstimator(1.5, 1)
    campaign = BAT._LowRankCampaign(d, 1000, 1)
    acc = _new_moments(estimator, d, 1)
    for _ in 1:(campaign.fit_start - 1)
        x, score = draw_score(rng)
        _moments_update!(acc, x, score)
    end

    G0, mu0 = BAT._fisher_diagonal_geometry(acc, 1e-5)
    for k in axes(campaign.fit.X, 2)
        x, score = draw_score(rng)
        campaign.fit.X[:, k] .= x
        campaign.fit.G[:, k] .= score
    end
    candidate = BAT._fit_lowrank_candidate(
        estimator, diag(G0), campaign.fit.X, campaign.fit.G, 1e-5,
    )
    G1 = BAT._lowrank_geometry(diag(G0), candidate)
    A0 = _fisher_A(estimator, G0)
    A1 = _fisher_A(estimator, G1)
    A1_diag = _fisher_A(
        estimator,
        Diagonal(BAT._lowrank_geometry_diagonal(diag(G0), candidate)),
    )

    for _ in 1:campaign.guard_steps
        draw_score(rng)
    end
    delta_baseline = zeros(1, campaign.validation_steps)
    delta_offdiag = zeros(1, campaign.validation_steps)
    for k in axes(delta_baseline, 2)
        x, score = draw_score(rng)
        loss1 = BAT._fisher_loss(A1, mu0, x, score)
        delta_baseline[k] = BAT._fisher_loss(A0, mu0, x, score) - loss1
        delta_offdiag[k] = BAT._fisher_loss(A1_diag, mu0, x, score) - loss1
    end
    return (
        accepted = BAT._lowrank_validation_accepts(
            candidate,
            delta_baseline,
            delta_offdiag,
        ),
        G0,
        G1,
    )
end

function _lowrank_fisher_state(; cutoff = 1.5, max_nsteps = 1000, reinit = true)
    d = 16
    alg = TransformedMCMC(
        proposal = HamiltonianMC(),
        adaptive_transform = LowRankAffineTransform(cutoff = cutoff),
        pretransform = DoNotTransform(),
        nchains = 1,
        nwalkers = 1,
        nsteps = 100,
    )
    state = BAT.MCMCState(
        alg,
        batmeasure(product_distribution(fill(Normal(), d))),
        1,
        [zeros(d)],
        BATContext(rng = Philox4x((564, 1)), ad = ForwardDiff),
    )
    BAT.mcmc_tuning_init!!(state, max_nsteps)
    reinit && BAT.mcmc_tuning_reinit!!(state, max_nsteps)
    return state
end

function _typed_fisher_commit(::Type{T}) where {T<:AbstractFloat}
    schedule = DriftCommitSchedule(
        commit_threshold = 0,
        check_interval = 1,
        memory_length = 100,
        min_observations = 8,
    )
    algorithm = TransformedMCMC(
        proposal = MALAProposal(),
        adaptive_transform = DiagonalAffineTransform(init = BAT.UnitTransformInit()),
        transform_tuning = FisherTransformTuning(schedule = schedule),
        pretransform = DoNotTransform(),
        nchains = 1,
        nwalkers = 1,
    )
    target = batmeasure(MvNormal(zeros(T, 2), Diagonal(ones(T, 2))))
    state = BAT.MCMCState(
        algorithm,
        target,
        1,
        [zeros(T, 2)],
        BATContext(precision = T, rng = Philox4x((564, 2)), ad = ForwardDiff),
    )
    chain_state = state.chain_state
    tuner = state.trafo_tuner_state
    proposal = BAT.get_active_proposal(chain_state.proposal)
    f_transform = chain_state.f_transform
    grad_storage = zeros(T, 4)
    committed = false

    for step = 1:8
        signs = (isodd(step) ? one(T) : -one(T), step % 4 < 2 ? one(T) : -one(T))
        x = T[10 * signs[1], 5 * signs[2]]
        score = T[-0.1 * signs[1], -0.2 * signs[2]]
        chain_state.current.x.v[1] .= x
        z_grad = view(grad_storage, 2:3)
        z_grad .= f_transform.A' * score
        f_new, tuner, _ = BAT.mcmc_tune_trafo_post_step!!(
            f_transform,
            tuner,
            chain_state,
            proposal,
            chain_state.current,
            chain_state.proposed,
            BAT.MCMCStepInfo(T[one(T)], [z_grad], nothing, nothing, nothing),
        )
        committed |= f_new !== f_transform
        f_transform = f_new
        chain_state.f_transform = f_transform
    end

    return (; committed, f_transform, tuner)
end

@testset "fisher_tuner" begin
    rng = StableRNG(438621057)

    @testset "numeric type preservation" begin
        for T in (Float32, BigFloat)
            result = _typed_fisher_commit(T)
            @test result.committed
            @test eltype(result.f_transform.A) === T
            @test eltype(result.f_transform.b) === T
            @test eltype(result.tuner.acc_a.mean_x) === T
            @test eltype(result.tuner.acc_a.mean_g) === T
            @test eltype(result.tuner.acc_a.lag1.prev) === T

            campaign = BAT._LowRankCampaign(T, 2, 1000, 1)
            @test eltype(campaign.fit.X) === T
            @test eltype(campaign.validation_loss) === T
        end

        x = BigFloat[big"0.5" + big"1e-30"]
        g = BigFloat[big"-1.0" + big"1e-30"]
        acc = _new_moments(DenseFisherEstimator(), x)
        _moments_update!(acc, view(x, :), view(g, :))
        @test only(acc.mean_x) == only(x)
        @test only(acc.mean_g) == only(g)

        validation_stats = BAT._lowrank_validation_stats(reshape(BigFloat.(1:64), 2, :))
        @test validation_stats.valid
        @test validation_stats.mean isa BigFloat
    end

    @testset "Fisher geometry recovery" begin
        # For a Gaussian N(μ, Σ) the score is α = -Σ⁻¹(x - μ), so
        # Cov(x) = Σ but Cov(α) = Σ⁻¹ (inverses, not equal!), and the
        # Fisher-optimal affine geometry G Cov(α) G = Cov(x) is exactly
        # G = Σ, μ* = μ:
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

        # Global scale equivariance of the regularized geometry (the
        # scalar ridge preserves only scalar and orthogonal, not
        # arbitrary affine, equivariance): rescaling the target by c
        # scales positions by c and scores by 1/c, so the learned
        # geometry must scale by c² even for extreme scales where an
        # absolute regularization floor would swamp the score covariance:
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

        # The (essentially) unregularized Fisher geometry is fully affine
        # equivariant: for x -> B x the scores transform as α -> B⁻ᵀ α
        # and the learned geometry as G -> B G Bᵀ:
        B = [1.2 0.4 0.0 0.1; -0.3 0.9 0.2 0.0; 0.0 0.5 1.5 -0.2; 0.1 0.0 -0.4 0.8]
        Binv_t = inv(B)'
        acc_B = _new_moments(DenseFisherEstimator(), d)
        rng_eq = StableRNG(438621057)
        for _ in 1:10^3
            x = μ_true .+ cholesky(Σ).L * randn(rng_eq, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc_B, B * x, Binv_t * α)
        end
        G_B, _ = _fisher_geometry(DenseFisherEstimator(), acc_B, 1e-12)
        G_ref, _ = _fisher_geometry(DenseFisherEstimator(), acc_unit, 1e-12)
        @test Matrix(G_B) ≈ B * Matrix(G_ref) * B' rtol = 1e-5
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

        # Window translation invariance: the low-rank correction is a
        # covariance property, so a window whose mean drifted away from
        # the longer history must not produce a spurious correction
        # direction (it would with history-mean centering):
        d_tr = 4
        est_tr = LowRankFisherEstimator(1.5, 0)
        acc_tr = _new_moments(est_tr, d_tr)
        Σ_tr = Diagonal([1.0, 2.0, 0.5, 1.5])
        X_tr = zeros(d_tr, 32)
        G_tr = zeros(d_tr, 32)
        for k in 1:500
            μ_k = k <= 460 ? zeros(d_tr) : fill(1.0, d_tr)
            x = μ_k .+ sqrt.(diag(Σ_tr)) .* randn(rng, d_tr)
            α = -Σ_tr \ (x .- μ_k)
            _moments_update!(acc_tr, x, α)
            if k > 468
                X_tr[:, k - 468] .= x
                G_tr[:, k - 468] .= α
            end
        end
        G_diag_tr, _ = BAT._fisher_diagonal_geometry(acc_tr, 1e-5)
        candidate_tr = BAT._fit_lowrank_candidate(
            est_tr,
            diag(G_diag_tr),
            X_tr,
            G_tr,
            1e-5,
        )
        @test isempty(candidate_tr.lambda)

        # Rank-deficient windows (repeated draws) must not create
        # spurious correction directions:
        acc_rd = _new_moments(est_tr, d_tr)
        xs_rd = [sqrt.(diag(Σ_tr)) .* randn(rng, d_tr) for _ in 1:8]
        X_rd = zeros(d_tr, 32)
        G_rd = zeros(d_tr, 32)
        for k in 1:200
            x = xs_rd[mod1(k, 8)]
            α = -Σ_tr \ x
            _moments_update!(acc_rd, x, α)
            if k > 168
                X_rd[:, k - 168] .= x
                G_rd[:, k - 168] .= α
            end
        end
        G_diag_rd, μ_rd = BAT._fisher_diagonal_geometry(acc_rd, 1e-5)
        candidate_rd = BAT._fit_lowrank_candidate(
            est_tr,
            diag(G_diag_rd),
            X_rd,
            G_rd,
            1e-5,
        )
        @test all(isfinite, μ_rd)
        @test isempty(candidate_rd.lambda)

        # AR(1)-corrected effective observation count:
        acc_ar = _new_moments(DiagonalFisherEstimator(), 2)
        acc_iid = _new_moments(DiagonalFisherEstimator(), 2)
        x_ar = zeros(2)
        for _ in 1:5000
            x_ar = 0.9 .* x_ar .+ randn(rng, 2)
            _moments_update!(acc_ar, x_ar, randn(rng, 2))
            _moments_update!(acc_iid, randn(rng, 2), randn(rng, 2))
        end
        @test BAT._effective_nobs(acc_ar) < 0.15 * acc_ar.n
        @test BAT._effective_nobs(acc_iid) > 0.8 * acc_iid.n
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
        Xfit = zeros(d, 2d)
        Gfit = zeros(d, 2d)
        for k in 1:10^4
            x = μ_true .+ L * randn(rng, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc, x, α)
            if k <= 2d
                Xfit[:, k] .= x
                Gfit[:, k] .= α
            end
        end

        G_diag, μ = BAT._fisher_diagonal_geometry(acc, 1e-5)
        candidate = BAT._fit_lowrank_candidate(
            est,
            diag(G_diag),
            Xfit,
            Gfit,
            1e-5,
        )
        G = BAT._lowrank_geometry(diag(G_diag), candidate)
        @test opnorm(Matrix(G) - Σ) / opnorm(Σ) < 0.15
        @test isapprox(μ, μ_true, atol = 0.4)

        # The committed matrix part is a Gram factor of G:
        A = _fisher_A(est, G)
        @test Matrix(A) * Matrix(A)' ≈ Matrix(G)

        # A hard rank cap is respected:
        est_capped = LowRankFisherEstimator(1.5, 1)
        candidate_capped = BAT._fit_lowrank_candidate(
            est_capped,
            diag(G_diag),
            Xfit,
            Gfit,
            1e-5,
        )
        @test length(candidate_capped.lambda) <= 1
    end

    @testset "low-rank correction campaign schedule" begin
        campaign = BAT._LowRankCampaign(16, 1000, 1)
        @test campaign.phase == BAT._LRWaiting

        BAT._advance_lowrank_campaign!(campaign, campaign.fit_start - 1)
        @test campaign.phase == BAT._LRWaiting

        BAT._advance_lowrank_campaign!(campaign, campaign.fit_start)
        @test campaign.phase == BAT._LRFit
        @test campaign.fit_steps == 64
        @test campaign.guard_steps == 16
        @test campaign.validation_steps == 256
        @test campaign.final_steps >= 150

        @test isnothing(BAT._LowRankCampaign(33, 1000, 1))
        @test isnothing(BAT._LowRankCampaign(16, 150, 1))
    end

    @testset "low-rank correction campaign lifecycle" begin
        state = _lowrank_fisher_state(reinit = false)
        @test isnothing(state.trafo_tuner_state.campaign)

        BAT.mcmc_tuning_reinit!!(state, 1000)
        campaign = state.trafo_tuner_state.campaign
        @test !isnothing(campaign)

        BAT.mcmc_tuning_reinit!!(state, 2000)
        @test state.trafo_tuner_state.campaign === campaign

        state_low_cutoff = _lowrank_fisher_state(cutoff = 1.2)
        diagonal_campaign = state_low_cutoff.trafo_tuner_state.campaign
        @test !isnothing(diagonal_campaign)
        @test diagonal_campaign.fit_steps == 0
        BAT._advance_lowrank_campaign!(diagonal_campaign, 850)
        @test diagonal_campaign.phase == BAT._LRWaiting
        BAT._advance_lowrank_campaign!(diagonal_campaign, 851)
        @test diagonal_campaign.phase == BAT._LRFrozen

        short_state = _lowrank_fisher_state(max_nsteps = 150)
        short_campaign = short_state.trafo_tuner_state.campaign
        @test !isnothing(short_campaign)
        @test short_campaign.fit_steps == 0
        BAT._advance_lowrank_campaign!(short_campaign, 128)
        @test short_campaign.phase == BAT._LRFrozen

        BAT.mcmc_tuning_reinit!!(short_state, 150)
        @test short_state.trafo_tuner_state.campaign === short_campaign

        BAT.mcmc_tuning_reinit!!(short_state, 1000)
        retry_campaign = short_state.trafo_tuner_state.campaign
        @test retry_campaign !== short_campaign
        @test retry_campaign.fit_steps == 64
        @test retry_campaign.phase == BAT._LRWaiting
    end

    @testset "failed low-rank baseline stays frozen" begin
        state = _lowrank_fisher_state()
        tuner = state.trafo_tuner_state
        campaign = tuner.campaign
        tuner.acc_a.n = 2
        fill!(tuner.acc_a.M2_x, NaN)
        fill!(tuner.acc_a.M2_g, 1.0)
        BAT._advance_lowrank_campaign!(campaign, campaign.fit_start - 2)

        state = BAT.mcmc_step!!(state)
        @test campaign.phase == BAT._LRFrozen
        @test campaign.attempted
        @test isnothing(campaign.baseline_dvec)

        state = BAT.mcmc_step!!(state)
        @test campaign.phase == BAT._LRFrozen
        @test campaign.attempted
        @test isnothing(campaign.baseline_dvec)
    end

    @testset "low-rank campaign pauses step-size tuning" begin
        base = _lowrank_fisher_state()
        campaign = base.trafo_tuner_state.campaign
        fit_end = campaign.fit_start + campaign.fit_steps - 1
        guard_end = fit_end + campaign.guard_steps
        validation_end = guard_end + campaign.validation_steps

        tuner_snapshot(state) = begin
            tuner = state.proposal_tuner_state
            proposal = BAT.get_active_proposal(state.chain_state.proposal)
            (tuner.m, tuner.log_mu, tuner.log_stepsize_bar, tuner.H_bar,
                tuner.run_nobs, tuner.run_accept_sum, tuner.run_accept_sqsum,
                tuner.run_ndivergent, tuner.run_skip, proposal.step_size)
        end

        observed_pauses = Bool[]
        expected_pauses = Bool[]
        for (cycle_step, paused) in (
            (0, false),
            (campaign.fit_start - 1, false),
            (fit_end, false),
            (guard_end, true),
            (validation_end - 1, true),
        )
            state = deepcopy(base)
            BAT._advance_lowrank_campaign!(
                state.trafo_tuner_state.campaign,
                cycle_step,
            )
            before = tuner_snapshot(state)
            state = BAT.mcmc_step!!(state)
            after = tuner_snapshot(state)
            push!(observed_pauses, after == before)
            push!(expected_pauses, paused)
        end
        @test observed_pauses == expected_pauses

        BAT._advance_lowrank_campaign!(campaign, validation_end + 1)
        @test !BAT.transform_tuning_pauses_proposal(
            base.trafo_tuner_state,
        )
    end

    @testset "low-rank correction fit separation" begin
        d = 16
        gamma = 1e-5
        est = LowRankFisherEstimator(1.5, 1)
        acc_diag = _new_moments(DiagonalFisherEstimator(), d)
        acc_lr = _new_moments(est, d)
        rng_fit = StableRNG(91042)

        for _ in 1:200
            x = randn(rng_fit, d)
            g = -x
            _moments_update!(acc_diag, x, g)
            _moments_update!(acc_lr, x, g)
        end

        G_diag, mu_diag = BAT._fisher_diagonal_geometry(acc_diag, gamma)
        G_lr, mu_lr = BAT._fisher_diagonal_geometry(acc_lr, gamma)
        @test diag(G_lr) ≈ diag(G_diag)
        @test mu_lr ≈ mu_diag

        u = normalize(ones(d))
        Sigma = Symmetric(Matrix(I, d, d) + 16.0 * u * u')
        Sigma_inv = inv(Sigma)
        L = cholesky(Sigma).L
        Xfit = reduce(hcat, (L * randn(rng_fit, d) for _ in 1:(2d)))
        Gfit = reduce(hcat, (-Sigma_inv * x for x in eachcol(Xfit)))
        dvec = fill(sqrt(17 / 8), d)

        candidate = BAT._fit_lowrank_candidate(est, dvec, Xfit, Gfit, gamma)
        @test length(candidate.lambda) == 1
        G_candidate = BAT._lowrank_geometry(dvec, candidate)
        @test opnorm(Matrix(G_candidate) - Sigma) / opnorm(Sigma) < 0.1
        @test BAT._lowrank_geometry_diagonal(dvec, candidate) ≈
            diag(Matrix(G_candidate))

        axis_candidate = BAT._LowRankCandidate(
            [2.0],
            reshape([1.0; zeros(d - 1)], d, 1),
            reshape([1.0; zeros(d - 1)], d, 1),
            Symmetric(ones(1, 1)),
        )
        @test BAT._valid_lowrank_candidate(axis_candidate, ones(d), zeros(d))

        hub_direction = normalize([0.8, 0.6, zeros(d - 2)...])
        W_hub, S_hub, lambda_hub, vectors_hub = BAT._lowrank_correction(
            ones(d),
            [3.0],
            reshape(hub_direction, d, 1),
            1.5,
            1,
        )
        hub_candidate = BAT._LowRankCandidate(
            lambda_hub,
            vectors_hub,
            W_hub,
            S_hub,
        )
        @test BAT._valid_lowrank_candidate(hub_candidate, ones(d), zeros(d))
        @test BAT._valid_lowrank_candidate(candidate, dvec, zeros(d))

        candidate_nonfinite = BAT._fit_lowrank_candidate(
            est,
            dvec,
            fill(NaN, size(Xfit)),
            Gfit,
            gamma,
        )
        @test isempty(candidate_nonfinite.lambda)
        candidate_bad_base = BAT._fit_lowrank_candidate(
            est,
            zeros(d),
            Xfit,
            Gfit,
            gamma,
        )
        @test isempty(candidate_bad_base.lambda)
    end

    @testset "low-rank held-out admission" begin
        d = 4
        mu = collect(range(-0.4, 0.5, length = d))
        x = [0.2, -1.1, 0.7, 1.4]
        alpha = [-0.3, 0.8, -0.2, 0.5]
        dvec = [0.7, 1.2, 2.0, 0.9]
        A_diag = Diagonal(sqrt.(dvec))

        loss_diag = BAT._fisher_loss(A_diag, mu, x, alpha)
        residual = x - mu
        loss_diag_dense = sum(abs2, A_diag' * alpha + A_diag \ residual)
        @test loss_diag ≈ loss_diag_dense

        direction = normalize([1.0, -2.0, 0.5, 1.5])
        lambda = [3.0]
        W, S, lambda_kept, vectors_kept = BAT._lowrank_correction(
            sqrt.(dvec),
            lambda,
            reshape(direction, d, 1),
            1.5,
            1,
        )
        candidate = BAT._LowRankCandidate(
            lambda_kept,
            vectors_kept,
            W,
            S,
        )
        G_lr = BAT._lowrank_geometry(dvec, candidate)
        A_lr = _fisher_A(LowRankFisherEstimator(1.5, 1), G_lr)
        loss_lr = BAT._fisher_loss(A_lr, mu, x, alpha)
        A_lr_dense = Matrix(A_lr)
        loss_lr_dense = sum(abs2, A_lr_dense' * alpha +
            A_lr_dense \ residual)
        @test loss_lr ≈ loss_lr_dense

        candidate0 = BAT._LowRankCandidate(
            Float64[],
            zeros(d, 0),
            zeros(d, 0),
            Symmetric(zeros(0, 0)),
        )
        @test BAT._fisher_loss(
            _fisher_A(
                LowRankFisherEstimator(1.5, 1),
                BAT._lowrank_geometry(dvec, candidate0),
            ),
            mu,
            x,
            alpha,
        ) ≈ loss_diag

        t = collect(1:256)
        positive = reshape(1 .+ 0.1 .* sin.(t), 1, :)
        negative = -positive
        nonfinite = copy(positive)
        nonfinite[1, 20] = NaN
        short = reshape(1 .+ 0.1 .* sin.(1:40), 1, :)
        moderate = reshape(1 .+ 0.1 .* (-1.0).^(1:40), 1, :)

        @test BAT._lowrank_validation_accepts(candidate, positive, positive)
        candidate_15 = BAT._LowRankCandidate(
            [15.0], candidate.vectors, candidate.W, candidate.S,
        )
        candidate_21 = BAT._LowRankCandidate(
            [21.0], candidate.vectors, candidate.W, candidate.S,
        )
        @test BAT._lowrank_validation_accepts(candidate_15, positive, positive)
        @test !BAT._lowrank_validation_accepts(candidate_21, positive, positive)
        @test !BAT._lowrank_validation_accepts(candidate, negative, positive)
        @test !BAT._lowrank_validation_accepts(candidate, positive, negative)
        @test !BAT._lowrank_validation_accepts(candidate, nonfinite, positive)
        @test !BAT._lowrank_validation_accepts(candidate, positive, nonfinite)
        @test !BAT._lowrank_validation_accepts(
            candidate,
            ones(1, 256),
            positive,
        )
        @test !BAT._lowrank_validation_accepts(candidate, short, positive)
        @test BAT._lowrank_validation_accepts(candidate, moderate, moderate)
        @test !BAT._lowrank_validation_accepts(candidate0, positive, positive)
        @test !BAT._lowrank_validation_accepts(
            candidate,
            positive,
            zeros(size(positive)),
        )

        sparse_direction = normalize([0.8, 0.6, zeros(d - 2)...])
        W_sparse, S_sparse, lambda_sparse, vectors_sparse =
            BAT._lowrank_correction(
                ones(d),
                [3.0],
                reshape(sparse_direction, d, 1),
                1.5,
                1,
            )
        sparse_candidate = BAT._LowRankCandidate(
            lambda_sparse,
            vectors_sparse,
            W_sparse,
            S_sparse,
        )
        G_sparse = BAT._lowrank_geometry(ones(d), sparse_candidate)
        A_sparse = _fisher_A(LowRankFisherEstimator(1.5, 1), G_sparse)
        A_sparse_diag = _fisher_A(
            LowRankFisherEstimator(1.5, 1),
            Diagonal(BAT._lowrank_geometry_diagonal(ones(d), sparse_candidate)),
        )
        sparse_target = MvNormal(zeros(d), Symmetric(Matrix(G_sparse)))
        sparse_baseline_delta = zeros(1, 256)
        sparse_offdiag_delta = similar(sparse_baseline_delta)
        sparse_rng = StableRNG(826_564_004)
        for k in axes(sparse_baseline_delta, 2)
            x_sparse = rand(sparse_rng, sparse_target)
            score_sparse = -(G_sparse \ x_sparse)
            loss_sparse = BAT._fisher_loss(
                A_sparse, zeros(d), x_sparse, score_sparse,
            )
            sparse_baseline_delta[k] = BAT._fisher_loss(
                Diagonal(ones(d)), zeros(d), x_sparse, score_sparse,
            ) - loss_sparse
            sparse_offdiag_delta[k] = BAT._fisher_loss(
                A_sparse_diag, zeros(d), x_sparse, score_sparse,
            ) - loss_sparse
        end
        @test BAT._lowrank_validation_accepts(
            sparse_candidate,
            sparse_baseline_delta,
            sparse_offdiag_delta,
        )

        disagreeing = vcat(
            reshape(0.1 .* sin.(t), 1, :),
            reshape(2 .+ 0.1 .* sin.(t), 1, :),
        )
        stats = BAT._lowrank_validation_stats(disagreeing)
        @test stats.se == max(stats.se_within, stats.se_between)
        @test stats.se_between > stats.se_within
    end

    @testset "low-rank null and power controls" begin
        independent_draw_score(dist, d) = begin
            nu = dist isa TDist ? first(Distributions.params(dist)) : NaN
            score = dist isa TDist ?
                (x -> -(nu + 1) * x / (nu + x^2)) :
                dist isa Logistic ? (x -> -tanh(x / 2)) :
                (x -> -sign(x))
            rng -> begin
                x = rand(rng, dist, d)
                x, score.(x)
            end
        end

        for dist in (TDist(5), TDist(10), Logistic(), Laplace()), d in (16, 32)
            result = _fixed_lowrank_campaign(
                independent_draw_score(dist, d), d, 101,
            )
            @test !result.accepted
        end

        d = 32
        nu = 5.0
        scale = sqrt((nu - 2) / nu)
        u = normalize(ones(d))
        Sigma = Symmetric(Matrix(I, d, d) + 16.0 * u * u')
        L = cholesky(Sigma).L
        draw_score = rng -> begin
            y = scale .* rand(rng, TDist(nu), d)
            score_y = @. -(nu + 1) * y / (nu * scale^2 + y^2)
            L * y, L' \ score_y
        end
        result = _fixed_lowrank_campaign(draw_score, d, 101)
        fisher_scale = inv(sqrt(nu * (nu + 1) / ((nu - 2) * (nu + 3))))
        G_oracle = fisher_scale .* Sigma
        baseline_error = BAT._transform_drift(
            Diagonal(sqrt.(diag(result.G0))), G_oracle,
        )
        candidate_error = BAT._transform_drift(
            BAT._fisher_A(LowRankFisherEstimator(1.5, 1), result.G1),
            G_oracle,
        )
        @test result.accepted
        @test candidate_error < 0.8 * baseline_error
    end

    @testset "low-rank product-t HMC lifecycle" begin
        objective = product_distribution(fill(TDist(3), 32))
        alg = TransformedMCMC(
            proposal = HamiltonianMC(max_depth = 4),
            adaptive_transform = LowRankAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
            nsteps = 20,
            init = MCMCChainPoolInit(nsteps_init = 10),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = 1000,
                max_ncycles = 1,
                nsteps_final = 0,
            ),
            convergence = AssumeConvergence(),
        )
        result = evalmeasure(
            batmeasure(objective),
            alg,
            BATContext(rng = Philox4x((42, 43)), ad = ForwardDiff),
        )
        @test only(BAT.samplegenof(result).chain_states).info.tuned
    end

    @testset "guards" begin
        # The low-rank transform owns its parameter invariants:
        M_id = Matrix(1.0 * I, 3, 3)
        @test_throws ArgumentError BAT._affine_init_A(LowRankAffineTransform(cutoff = 1.0), M_id)
        @test_throws ArgumentError BAT._affine_init_A(LowRankAffineTransform(cutoff = 0.5), M_id)
        @test_throws ArgumentError BAT._affine_init_A(LowRankAffineTransform(max_rank = -1), M_id)

        context = BATContext(rng = Philox4x((564, 3)), ad = ForwardDiff)
        target = unshaped(batmeasure(NamedTupleDist(a = Normal(), b = Normal())))
        # Fisher tuning requires a gradient-based proposal (nchains = 2 so
        # the single-chain convergence guard can't fire first):
        alg_rw = TransformedMCMC(
            proposal = RandomWalk(), transform_tuning = FisherTransformTuning(),
            nchains = 2, nsteps = 100
        )
        @test_throws ArgumentError bat_sample(target, alg_rw, context)
    end

    @testset "end-to-end geometry learning" begin
        context = BATContext(rng = Philox4x((564, 4)), ad = ForwardDiff)
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
        context = BATContext(rng = Philox4x((1, 0)), ad = ForwardDiff)

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
        context = BATContext(rng = Philox4x((564, 5)), ad = ForwardDiff)
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
        context = BATContext(rng = Philox4x((564, 6)), ad = ForwardDiff)
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

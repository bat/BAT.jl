# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using Distributions, Random123
import ForwardDiff

@testset "Student-t affine initialization moments" begin
    function affine_init_error(d)
        try
            BAT.init_adaptive_transform(BAT.TriangularAffineTransform(), batmeasure(d), BATContext())
        catch err
            return err
        end
    end

    @testset "finite moments" begin
        d = TDist(3.0)
        target = batmeasure(d)

        @test BAT._approx_mean(target, 1) == [mean(d)]
        @test Matrix(BAT._approx_cov(target, 1)) == fill(var(d), 1, 1)
    end

    @testset "undefined moments" begin
        d_undefined_mean = TDist(1.0)
        @test all(isnan, BAT._approx_mean(batmeasure(d_undefined_mean), 1))
        @test all(isnan, BAT._approx_cov(batmeasure(d_undefined_mean), 1))
        err_mean = affine_init_error(d_undefined_mean)
        @test err_mean isa DomainError && occursin("affine transform requires finite mean", sprint(showerror, err_mean))

        d_undefined_cov = TDist(2.0)
        @test BAT._approx_mean(batmeasure(d_undefined_cov), 1) == [mean(d_undefined_cov)]
        @test all(isinf, BAT._approx_cov(batmeasure(d_undefined_cov), 1))
        err_cov = affine_init_error(d_undefined_cov)
        @test err_cov isa DomainError && occursin("affine transform requires finite covariance", sprint(showerror, err_cov))
    end

    @testset "HMC and MALA preserve state scalar precision" begin
        function test_sampler_state(::Type{T}, target, proposal, context_precision, seed) where {T}
            algorithm = TransformedMCMC(
                proposal = proposal,
                pretransform = DoNotTransform(),
                nchains = 1,
                nwalkers = 1,
            )
            state = BAT.MCMCState(
                algorithm,
                batmeasure(target),
                1,
                [T[0.25, -0.5]],
                BATContext(precision = context_precision, rng = Philox4x(seed), ad = ForwardDiff),
            )
            function assert_scalar_type(state)
                chain_state = state.chain_state
                @test eltype(chain_state.f_transform.A) === T
                @test eltype(chain_state.f_transform.b) === T
                @test eltype(only(chain_state.current.x.v)) === T
                @test eltype(only(chain_state.current.z.v)) === T
            end
            assert_scalar_type(state)
            state = BAT.mcmc_iterate!!(nothing, state; max_nsteps = 1, nonzero_weights = false)
            assert_scalar_type(state)
        end

        for (proposal, seed) in ((HamiltonianMC(), (564, 89)), (MALAProposal(), (564, 90))), T in (Float32, BigFloat)
            ν = T === BigFloat ? big"3.1" : T(3)
            target = product_distribution(fill(TDist(ν), 2))
            moments_target = batmeasure(target)
            context_precision = T === Float32 ? BigFloat : Float32
            μ, Σ = BAT._approx_mean(moments_target, 2), BAT._approx_cov(moments_target, 2)
            @test eltype(μ) === T
            @test eltype(Σ) === T
            @test diag(Σ) == fill(ν / (ν - 2), 2)
            test_sampler_state(T, target, proposal, context_precision, seed)
        end

        test_sampler_state(
            Float32,
            MvNormal(zeros(Float32, 2), Diagonal(ones(Float32, 2))),
            MALAProposal(),
            BigFloat,
            (564, 91),
        )
    end
end

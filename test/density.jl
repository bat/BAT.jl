# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

using Distributions, PDMats, StatsBase

struct test_density <: AbstractDensity
end

BAT.nparams(td::test_density) = Int(3)
BAT.sampler(td::test_density) = BAT.sampler(MvNormal(ones(3), PDMat(eye(3))))

@testset "density" begin

    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)

    density = @inferred GenericDensity(params -> logpdf(mvnorm, params), 2)

    econtext = @inferred ExecContext()

    params = [0.0 -0.3; 0.0 0.3]

    @testset "rand" begin
        td = test_density()
        @test rand(MersenneTwister(7002), td, Float64) ≈
            [-2.415270938, 0.7070171342, 1.0224848653]
        @test rand(MersenneTwister(7002), td, Float64, 2) ≈
            [-2.415270938 1.2951273090;
            0.7070171342  0.8896838714;
            1.0224848653  0.8824274590]
        x = ones(3,2)
        rand!(MersenneTwister(7002), td, x)
        @test x ≈
            [-2.415270938 1.2951273090;
            0.7070171342  0.8896838714;
            1.0224848653  0.8824274590]
    end

    @testset "param_bounds" begin
        pbounds = @inferred param_bounds(density)
        @test typeof(pbounds) == NoParamBounds
        @test pbounds.ndims == 2
    end

    @testset "convert" begin
        cdensity = @inferred convert(
            AbstractDensity, HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds))
        @test typeof(cdensity) <: ConstDensity
    end

    @testset "density_logval!" begin
        r = zeros(Float64, 2)

        density_logval!(r, density, params, econtext)
        @test r ≈ logpdf(mvnorm, params)
        BAT.unsafe_density_logval!(r, density, params, econtext)
        @test r ≈ logpdf(mvnorm, params)
    end

    @testset "density_logval" begin
        @test density_logval(density, params[:, 1], econtext) ≈ logpdf(mvnorm, params[:, 1])
        @test BAT.unsafe_density_logval(density, params[:, 1], econtext) ≈ logpdf(mvnorm, params[:, 1])
    end

    @testset "exec_capabilities" begin
        ecap = @inferred BAT.exec_capabilities(
            density_logval!, similar(params[1, :]), density, params)
        @test ecap.nthreads == 1
        @test ecap.threadsafe == false
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(
            BAT.unsafe_density_logval!, similar(params[1, :]), density, params)
        @test ecap.nthreads == 1
        @test ecap.threadsafe == false
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(density_logval, density, params[:, 1])
        @test ecap.nthreads == 1
        @test ecap.threadsafe == true
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(
            BAT.unsafe_density_logval, density, params[:, 1])
        @test ecap.nthreads == 1
        @test ecap.threadsafe == true
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(
            BAT.unsafe_density_logval, test_density())
        @test ecap.nthreads == 1
        @test ecap.threadsafe == false
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

    end

    @testset "parent" begin
        @test @inferred parent(density) == density.log_f
    end

    @testset "nparams" begin
        @test @inferred nparams(density) == 2
    end
end

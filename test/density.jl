# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

using Distributions, PDMats, StatsBase

struct DummyDF <: AbstractDensity end

@testset "target_density" begin
    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)
    density = @inferred MvDistDensity(mvnorm)

    econtext = @inferred ExecContext()

    params = [0.0 -0.3; 0.0 0.3]
      
    @testset "density_logval!" begin
        r = zeros(Float64, 2)

        density_logval!(r, density, params, econtext)
        @test r ≈ logpdf(mvnorm, params)
    end

    @testset "density_logval" begin
        @test density_logval(density, params[:, 1], econtext) ≈ logpdf(mvnorm, params[:, 1])
    end

    @testset "exec_capabilities" begin
        td = DummyDF()
        ecap = @inferred BAT.exec_capabilities(density_logval!, similar(params[1, :]), td, params)
        @test ecap.nthreads == 1
        @test ecap.threadsafe == false
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(density_logval, td, params[:, 1])
        @test ecap.nthreads == 1
        @test ecap.threadsafe == false
        @test ecap.nprocs == 1
        @test ecap.remotesafe == true
    end
end

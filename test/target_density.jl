# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

using Distributions, PDMats, StatsBase

struct atd_wrap <: AbstractTargetDensity end

@testset "target_density" begin
    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)
    tdensity = @inferred MvDistTargetDensity(mvnorm)

    econtext = @inferred ExecContext()

    params = [0.0 -0.3; 0.0 0.3]
      
    @testset "target_logval!" begin
        r = zeros(Float64, 2)

        target_logval!(r, tdensity, params, econtext)
        @test r ≈ logpdf(mvnorm, params)
    end

    @testset "target_logval" begin
        @test target_logval(tdensity, params[:, 1], econtext) ≈ logpdf(mvnorm, params[:, 1])
    end

    @testset "exec_capabilities" begin
        td = @inferred atd_wrap()
        ecap = @inferred BAT.exec_capabilities(target_logval!, td, params)
        @test ecap.nthreads == 0
        @test ecap.threadsafe == false
        @test ecap.nprocs == 0
        @test ecap.remotesafe == true

        ecap = @inferred BAT.exec_capabilities(target_logval, td, params[:, 1])
        @test ecap.nthreads == 0
        @test ecap.threadsafe == false
        @test ecap.nprocs == 0
        @test ecap.remotesafe == true
    end
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Distributions, PDMats

@testset "mvdist_density" begin
    mvt = @inferred MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))
    mvdd = @inferred MvDistDensity(mvt)

    @testset "mvdist_density" begin
        @test typeof(mvdd) <: AbstractDensity
        @test parent(mvdd) == mvt
        @test nparams(mvdd) == 2

        @test typeof(sampler(mvdd)) <: BATMvTDistSampler
    end

    @testset "unsafe_density_logval" begin
        BAT.unsafe_density_logval(mvdd, [0.0, 0.0]) ≈
            -2.64259602
        param = zeros(2)
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval, mvdd, param)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true

        ret = Array{Float64}(undef, 2)
        param = VectorOfSimilarVectors([0.0 0.5; 0.0 -0.5])
        BAT.unsafe_density_logval!(ret, mvdd, param)
        @test ret ≈ [-2.64259602, -3.00960695]
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval!, ret,
            mvdd, param)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true
    end
end

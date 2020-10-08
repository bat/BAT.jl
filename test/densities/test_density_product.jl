# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, PDMats


@testset "density_product" begin
    mvec = [-0.3, 0.3, 0.1]
    cmat = [1.0 1.5 0.0; 1.5 4.0 0.0; 0.0 0.0 1.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)
    mvn_density = @inferred BAT.GenericDensity(params -> LogDVal(logpdf(mvnorm, params)))

    cmat = [3.76748 0.446731 0.625418; 0.446731 3.9317 0.237361; 0.625418 0.237361 3.43867]
    mvec = [1., 2, 1.]
    Σ = @inferred PDMat(cmat)
    mvt = MvTDist(3, mvec, Σ)
    mvt_density = @inferred BAT.GenericDensity(params -> LogDVal(logpdf(mvt, params)))

    params = VectorOfSimilarVectors([0.0 -0.3; 0.0 0.3; 0.0 1.0])
    
    pb = BAT.HyperRectBounds([-2.0, -1.0, -0.5], [2.0, 3.0, 1.0],
                                BAT.reflective_bounds)
    pb2 = BAT.HyperRectBounds([-1.5, -2.0, -0.5], [2.0, 2.5, 1.5],
                                BAT.reflective_bounds)
    dp = @inferred BAT.DensityProduct((mvt_density, mvn_density), pb)
    dp1 = @inferred BAT.DensityProduct((mvt_density,), pb)
    dp2 = @inferred BAT.DensityProduct((mvn_density,), pb2)

    res = Array{Float64}(undef, size(params, 1))
    
    @testset "DensityProduct" begin
        @test typeof(dp) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.f)},
                  BAT.GenericDensity{typeof(mvn_density.f)}},BAT.HyperRectBounds{Float64}}
        
        @test parent(dp)[1] == mvt_density        
        @test BAT.var_bounds(dp) == pb
        @test totalndof(dp) == 3        
    end

    @testset "unsafe_pod" begin
        up = @inferred BAT._unsafe_prod(mvt_density, mvn_density, pb)
        @test parent(up)[2] == mvn_density
        @test BAT.var_bounds(up) == pb

        prd = @inferred dp1*dp2
        @test typeof(prd) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.f)},
                  BAT.GenericDensity{typeof(mvn_density.f)}},BAT.HyperRectBounds{Float64}}
        @test parent(prd)[2] == mvn_density
        prd_pb = @inferred BAT.var_bounds(prd)
        cut_pb = @inferred pb ∩ pb2 
        @test prd_pb.vol.lo ≈ cut_pb.vol.lo
        @test prd_pb.vol.hi ≈ cut_pb.vol.hi
        
        up = @inferred BAT._unsafe_prod(dp1, mvn_density, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test BAT.var_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(mvt_density, dp2, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test BAT.var_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(dp1, dp2, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test BAT.var_bounds(up) == pb
    end

    @testset "BAT.eval_logval_unchecked" begin
        @test BAT.eval_logval_unchecked(dp, params[1]) ≈ -8.8547305        
    end
end

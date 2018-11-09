# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, PDMats


@testset "density_product" begin
    econtext = @inferred ExecContext()
    mvec = [-0.3, 0.3, 0.1]
    cmat = [1.0 1.5 0.0; 1.5 4.0 0.0; 0.0 0.0 1.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)
    mvn_density = @inferred GenericDensity(params -> logpdf(mvnorm, params), 3)

    cmat = [3.76748 0.446731 0.625418; 0.446731 3.9317 0.237361; 0.625418 0.237361 3.43867]
    mvec = [1., 2, 1.]
    Σ = @inferred PDMat(cmat)
    mvt = MvTDist(3, mvec, Σ)
    mvt_density = @inferred GenericDensity(params -> logpdf(mvt, params), 3)

    params = VectorOfSimilarVectors([0.0 -0.3; 0.0 0.3; 0.0 1.0])
    
    pb = BAT.HyperRectBounds([-2.0, -1.0, -0.5], [2.0, 3.0, 1.0],
                                reflective_bounds)
    pb2 = BAT.HyperRectBounds([-1.5, -2.0, -0.5], [2.0, 2.5, 1.5],
                                reflective_bounds)
    dp = @inferred BAT.DensityProduct((mvt_density, mvn_density), pb)
    dp1 = @inferred BAT.DensityProduct((mvt_density,), pb)
    dp2 = @inferred BAT.DensityProduct((mvn_density,), pb2)

    res = Array{Float64}(undef, size(params, 1))
    
    @testset "DensityProduct" begin
        @test typeof(dp) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.log_f)},
                  BAT.GenericDensity{typeof(mvn_density.log_f)}},BAT.HyperRectBounds{Float64}}
        
        @test parent(dp)[1] == mvt_density        
        @test param_bounds(dp) == pb
        @test nparams(dp) == 3        
    end

    @testset "unsafe_pod" begin
        up = @inferred BAT._unsafe_prod(mvt_density, mvn_density, pb)
        @test parent(up)[2] == mvn_density
        @test param_bounds(up) == pb

        prd = @inferred dp1*dp2
        @test typeof(prd) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.log_f)},
                  BAT.GenericDensity{typeof(mvn_density.log_f)}},BAT.HyperRectBounds{Float64}}
        @test parent(prd)[2] == mvn_density
        prd_pb = @inferred param_bounds(prd)
        cut_pb = @inferred pb ∩ pb2 
        @test prd_pb.vol.lo ≈ cut_pb.vol.lo
        @test prd_pb.vol.hi ≈ cut_pb.vol.hi
        
        up = @inferred BAT._unsafe_prod(dp1, mvn_density, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(mvt_density, dp2, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(dp1, dp2, pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == pb
    end

    @testset "unsafe_density_logval" begin
        @test BAT.unsafe_density_logval(dp, params[1], econtext) ≈ -8.8547305
        
        @test_throws ArgumentError BAT.unsafe_density_logval(BAT.DensityProduct(Tuple([]), pb),
                                                             params[1], econtext)

        BAT.unsafe_density_logval!(res, dp, params, econtext) 
        
        @test res ≈ [-8.8547305, -8.8634491]
    end

    @testset "ExecCapabilities" begin
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval, dp, params[1])
        ec_tocmp = @inferred ∩(
            BAT.exec_capabilities(BAT.unsafe_density_logval, mvt_density, params[1]),
            BAT.exec_capabilities(BAT.unsafe_density_logval, mvn_density, params[1]))

        @test ec.nthreads == ec_tocmp.nthreads
        @test ec.threadsafe == ec_tocmp.threadsafe
        @test ec.nprocs == ec_tocmp.nprocs
        @test ec.remotesafe == ec_tocmp.remotesafe

        
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval!, res, dp, params)
        ec_tocmp = @inferred ∩(
            BAT.exec_capabilities(BAT.unsafe_density_logval!, res, mvt_density, params),
            BAT.exec_capabilities(BAT.unsafe_density_logval!, res, mvn_density, params))

        @test ec.nthreads == ec_tocmp.nthreads
        @test ec.threadsafe == ec_tocmp.threadsafe
        @test ec.nprocs == ec_tocmp.nprocs
        @test ec.remotesafe == ec_tocmp.remotesafe
                
    end
end

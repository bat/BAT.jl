using BAT
using Compat.Test

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

    params = [0.0 -0.3; 0.0 0.3; 0.0 1.0]
    
    gd1 = GenericDensity(params -> logpdf.(Normal(1,2), params), 1)
    gd2 = GenericDensity(params -> logpdf.(Normal(0,1), params), 1)
    pb = BAT.HyperRectBounds([-2.0], [2.0],
        [reflective_bounds])
    pb2 = BAT.HyperRectBounds([-1.0], [2.5],
        [reflective_bounds])

    mv_pb = BAT.HyperRectBounds([-2.0, -1.0, -0.5], [2.0, 3.0, 1.0],
                                reflective_bounds)
    mv_pb2 = BAT.HyperRectBounds([-1.5, -2.0, -0.5], [2.0, 2.5, 1.5],
                                reflective_bounds)
        
    dp = @inferred BAT.DensityProduct((gd1,gd2), pb)

    dp1 = @inferred BAT.DensityProduct((gd1,), pb)
    dp2 = @inferred BAT.DensityProduct((gd2,), pb2)

    mv_dp = @inferred BAT.DensityProduct((mvt_density, mvn_density), mv_pb)
    mv_dp1 = @inferred BAT.DensityProduct((mvt_density,), mv_pb)
    mv_dp2 = @inferred BAT.DensityProduct((mvn_density,), mv_pb2)

    
    @testset "DensityProduct" begin
        @test typeof(mv_dp) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.log_f)},
                  BAT.GenericDensity{typeof(mvn_density.log_f)}},BAT.HyperRectBounds{Float64}}
        
        @test parent(mv_dp)[1] == mvt_density        
        @test param_bounds(mv_dp) == mv_pb
        @test nparams(mv_dp) == 3        
    end

    @testset "unsafe_pod" begin
        mv_up = @inferred BAT._unsafe_prod(mvt_density, mvn_density, mv_pb)
        @test parent(mv_up)[2] == mvn_density
        @test param_bounds(mv_up) == mv_pb

        mv_prd = @inferred mv_dp1*mv_dp2
        @test typeof(mv_prd) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(mvt_density.log_f)},
                  BAT.GenericDensity{typeof(mvn_density.log_f)}},BAT.HyperRectBounds{Float64}}
        @test parent(mv_prd)[2] == mvn_density
        prd_pb = @inferred param_bounds(mv_prd)
        cut_pb = @inferred mv_pb ∩ mv_pb2 
        @test prd_pb.vol.lo ≈ cut_pb.vol.lo
        @test prd_pb.vol.hi ≈ cut_pb.vol.hi
        
        up = @inferred BAT._unsafe_prod(mv_dp1, mvn_density, mv_pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == mv_pb

        up = @inferred BAT._unsafe_prod(mvt_density, mv_dp2, mv_pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == mv_pb

        up = @inferred BAT._unsafe_prod(mv_dp1, mv_dp2, mv_pb)
        @test parent(up)[1] == mvt_density
        @test parent(up)[2] == mvn_density        
        @test param_bounds(up) == mv_pb
    end

    @testset "unsafe_density_logval" begin
        @test BAT.unsafe_density_logval(mv_dp, params[:,1]) ≈ -8.8547305
        
        @test_throws ArgumentError BAT.unsafe_density_logval(BAT.DensityProduct(Tuple([]), mv_pb),
                                                             params[:,1])
        res = Array{Float64}(size(params, 2))

        #print(BAT.unsafe_density_logval!(res, mv_dp, params, ExecContext()))
        
        #print(BAT.unsafe_density_logval!(res, mvt_density, params, ExecContext()) +
        #    BAT.unsafe_density_logval!(res, mvn_density, params, ExecContext()) )
    end

    @testset "ExecCapabilities" begin
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval, dp, [0.0])
        ec_tocmp = @inferred ∩(
            BAT.exec_capabilities(BAT.unsafe_density_logval, gd1, [0.0]),
            BAT.exec_capabilities(BAT.unsafe_density_logval, gd2, [0.0]))
        @test ec.nthreads == ec_tocmp.nthreads
        @test ec.threadsafe == ec_tocmp.threadsafe
        @test ec.nprocs == ec_tocmp.nprocs
        @test ec.remotesafe == ec_tocmp.remotesafe
    end
end

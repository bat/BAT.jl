using BAT
using Compat.Test
using Distributions


@testset "density_product" begin
    gd1 = GenericDensity(params -> logpdf.(Normal(1,2), params), 1)
    gd2 = GenericDensity(params -> logpdf.(Normal(0,1), params), 1)
    pb = BAT.HyperRectBounds([-2.0], [2.0],
        [reflective_bounds])
    pb2 = BAT.HyperRectBounds([-1.0], [2.5],
        [reflective_bounds])

    dp = @inferred BAT.DensityProduct((gd1,gd2), pb)

    dp1 = @inferred BAT.DensityProduct((gd1,), pb)
    dp2 = @inferred BAT.DensityProduct((gd2,), pb2)

    @testset "DensityProduct" begin
        @test typeof(dp) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(gd1.log_f)},
            BAT.GenericDensity{typeof(gd2.log_f)}},BAT.HyperRectBounds{Float64}}
        @test parent(dp)[1] == gd1
        @test param_bounds(dp) == pb
        @test nparams(dp) == 1
    end

    @testset "unsafe_pod" begin
        up = @inferred BAT._unsafe_prod(gd1, gd2, pb)
        @test parent(up)[2] == gd2
        @test param_bounds(up) == pb

        prd = @inferred dp1*dp2
        @test typeof(prd) <: BAT.DensityProduct{2,
            Tuple{BAT.GenericDensity{typeof(gd1.log_f)},
            BAT.GenericDensity{typeof(gd2.log_f)}},BAT.HyperRectBounds{Float64}}
        @test parent(prd)[2] == gd2
        @test param_bounds(prd).vol.lo[1] ≈ -1.0
        @test param_bounds(prd).vol.hi[1] ≈ 2.0

        up = @inferred BAT._unsafe_prod(gd1, dp2, pb)
        @test parent(up)[2] == gd2
        @test param_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(dp1, gd2, pb)
        @test parent(up)[1] == gd1
        @test param_bounds(up) == pb

        up = @inferred BAT._unsafe_prod(dp1, dp2, pb)
        @test parent(up)[1] == gd1
        @test parent(up)[2] == gd2
        @test param_bounds(up) == pb
    end

    @testset "unsafe_density_logval" begin
        @test BAT.unsafe_density_logval(dp, [0.0]) ≈
            [-2.656024246969]
        @test_throws ArgumentError BAT.unsafe_density_logval(BAT.DensityProduct(Tuple([]), pb),
            [0.0])
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

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

using IntervalSets

@testset "parameter bounds" begin
    @testset "BAT.oob" begin
        @test BAT.isoob(BAT.oob(1.))
        @test BAT.isoob(1.) == false
        @test BAT.isoob(BAT.oob(1))
        @test BAT.isoob(1) == false
        @test BAT.isoob(BAT.oob(ones(Float64,2,2)))
        @test BAT.isoob(ones(Float64,2,2)) == false
    end
    
    @testset "BAT.apply_bounds" begin
        @test BAT.apply_bounds(+0.3, -1, 2, hard_bounds) ≈ +0.3
        @test BAT.apply_bounds(-0.3, -1, 2, reflective_bounds) ≈ -0.3
        @test BAT.apply_bounds(+1.3, -1, 2, cyclic_bounds) ≈ +1.3

        @test isnan(BAT.apply_bounds(-1.3, -1, 2, hard_bounds))
        @test isnan(BAT.apply_bounds(2.3, -1, 2, hard_bounds))

        @test BAT.apply_bounds(-1.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(-4.3, -1, 2, reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+2.3, -1, 2, reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(+5.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, reflective_bounds) ≈ +1.7

        @test BAT.apply_bounds(-1.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-4.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(+2.3, -1, 2, cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+5.3, -1, 2, cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, cyclic_bounds) ≈ -0.7

        @test typeof(@inferred BAT.apply_bounds(+0.3, -1, 2, hard_bounds)) == Float64
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2, hard_bounds)) == Float32
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2.0, hard_bounds)) == Float64

        @test BAT.apply_bounds(+5.3, ClosedInterval(-1, 2), reflective_bounds) ≈ -0.7
    end

    @testset "BAT.UnboundedParams" begin
        n = 2
        @test typeof(@inferred BAT.UnboundedParams{Float32}(n)) == BAT.UnboundedParams{Float32}

        params = [-1000., 1000]
        uparams = BAT.UnboundedParams{Float32}(n)
        @test length(uparams) == n
        @test params in uparams
        @test in( hcat(params, params), uparams, 1)
        
        @test BAT.apply_bounds!(params, uparams) == params
        @test eltype(uparams) == Float32
    end

    @testset "BAT.HyperCubeBounds" begin
        @test typeof(@inferred BAT.HyperCubeBounds{Float64}([-1, 0.5], [2.,1], [hard_bounds, reflective_bounds])) == HyperCubeBounds{Float64}
        @test_throws ArgumentError BAT.HyperCubeBounds{Float64}([-1.], [2.,1], [hard_bounds, reflective_bounds])
        @test_throws ArgumentError BAT.HyperCubeBounds{Float64}([-1., 2.], [2.,1], [hard_bounds, reflective_bounds])

        hcb = BAT.HyperCubeBounds{Float64}([-1., -1.], [0.5,1], [hard_bounds, hard_bounds])
        @test length(hcb) == 2
        @test [0.0, 0.0] in hcb
        @test ([0.5, 2] in hcb) == false

        @test in([0.0 0.0; 0 2], hcb, 1)
        @test in([0.0 0.0; 0 2], hcb, 2) == false

        @test BAT.apply_bounds!([0.3, -4.3, -7.3], BAT.HyperCubeBounds{Float64}([-1.,-1,-1], [2.,2,2], [hard_bounds, reflective_bounds, cyclic_bounds])) ≈ [+0.3, 1.7, 1.7]
        
        @test BAT.apply_bounds!([0.3 0.3 0.3; 0.3 -7.3 +8.3; 0.3 -7.3 +8.3], BAT.HyperCubeBounds{Float64}([-1., -1., -1], [2., 2.,2.], [hard_bounds, reflective_bounds, cyclic_bounds])) ≈ [+0.3 0.3 0.3;0.3 -0.7 1.7;0.3 1.7 -0.7]

        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002), zeros(Float64, 2, 2)), hcb))
        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002), hcb, zeros(Float64, 2, 2)), hcb)) == false
    end
end

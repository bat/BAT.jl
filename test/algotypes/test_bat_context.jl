# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using HeterogeneousComputing: CPUnit, get_precision, get_rng, get_compute_unit
using ScopedSettings: with, default_value

@testset "bat_context" begin
    # The default context must never leak from one test into the next, the
    # global override is process-wide:
    reset_default_context() = (BAT.default_batcontext[] = default_value)

    @testset "default context" begin
        reset_default_context()
        try
            # Without an override, every access yields a fresh context, so
            # the default rng is seeded from Random.default_rng() each time
            # and BAT results are reproducible via Random.seed!:
            Random.seed!(4711)
            r1 = rand(get_rng(get_batcontext()))
            Random.seed!(4711)
            r2 = rand(get_rng(get_batcontext()))
            @test r1 == r2
        finally
            reset_default_context()
        end
    end

    @testset "set_batcontext" begin
        reset_default_context()
        try
            ctx = BATContext(ad = :ForwardDiff)
            @test set_batcontext(ctx) === ctx
            @test get_batcontext() === ctx

            # Fields not given are taken from the current context:
            set_batcontext(precision = Float32)
            @test get_precision(get_batcontext()) == Float32
            set_batcontext(cunit = CPUnit())
            @test get_compute_unit(get_batcontext()) == CPUnit()
            @test get_precision(get_batcontext()) == Float32
        finally
            reset_default_context()
        end
    end

    @testset "scoped default context" begin
        reset_default_context()
        try
            global_ctx = BATContext()
            set_batcontext(global_ctx)
            ctx = BATContext(ad = :ForwardDiff)
            with(BAT.default_batcontext => ctx) do
                @test get_batcontext() === ctx
            end
            @test get_batcontext() === global_ctx
        finally
            reset_default_context()
        end
    end
end

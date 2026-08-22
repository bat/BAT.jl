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
            @test get_batcontext() isa BATContext

            # Without an override, every access yields a fresh context, so
            # the default rng is seeded from Random.default_rng() each time
            # and BAT results are reproducible via Random.seed!:
            Random.seed!(4711)
            r1 = rand(get_rng(get_batcontext()))
            Random.seed!(4711)
            r2 = rand(get_rng(get_batcontext()))
            @test r1 == r2
            @test rand(get_rng(get_batcontext())) != rand(get_rng(get_batcontext()))
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
            # The override is process-wide, not task-local:
            @test fetch(Threads.@spawn get_batcontext()) === ctx

            # Fields not given are taken from the current context:
            set_batcontext(precision = Float32)
            @test get_precision(get_batcontext()) == Float32
            @test BAT.get_adselector(get_batcontext()) == BAT.get_adselector(ctx)
            set_batcontext(cunit = CPUnit())
            @test get_compute_unit(get_batcontext()) == CPUnit()
            @test get_precision(get_batcontext()) == Float32

            reset_default_context()
            @test BAT.get_adselector(get_batcontext()) isa BAT.NoAutoDiff
        finally
            reset_default_context()
        end
    end

    @testset "scoped default context" begin
        reset_default_context()
        try
            ctx = BATContext(ad = :ForwardDiff)
            with(BAT.default_batcontext => ctx) do
                @test get_batcontext() === ctx
                # Scoped bindings are inherited by tasks started in the scope:
                @test fetch(Threads.@spawn get_batcontext()) === ctx
                # A scoped binding would shadow a global assignment, so
                # setting the global default inside the scope is rejected
                # instead of silently having no effect:
                @test_throws ErrorException set_batcontext(BATContext())
            end
            @test BAT.get_adselector(get_batcontext()) isa BAT.NoAutoDiff
        finally
            reset_default_context()
        end
    end
end

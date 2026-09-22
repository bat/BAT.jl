# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, Test

@testset "Makie buffering configuration" begin
    cfg = BATMakieVisualization()
    positional(; n_batch = cfg.n_batch, max_buffered = cfg.max_buffered,
        rate = cfg.batch_growth_rate) = BATMakieVisualization(
        cfg.recipes, cfg.vsel, cfg.N_max, n_batch, max_buffered,
        cfg.adaptive_batching, rate, cfg.poll_interval, cfg.dark,
        cfg.triagonal_config, cfg.diagonal_config,
    )
    @test cfg.n_batch == 50
    @test cfg.max_buffered == 200
    for limit in (-1, 0, 1, 49)
        @test_throws ArgumentError BATMakieVisualization(max_buffered = limit)
        @test_throws ArgumentError positional(max_buffered = limit)
    end
    for limit in (50, 200, typemax(Int))
        @test BATMakieVisualization(max_buffered = limit).max_buffered == limit
    end
    for batch in (-1, 0, 201)
        @test_throws ArgumentError positional(n_batch = batch)
    end
    @test positional(n_batch = 1, max_buffered = 1).n_batch == 1
    for rate in (NaN, Inf, -Inf, -1.0, 0.0, 1.0), adaptive in (false, true)
        @test_throws ArgumentError BATMakieVisualization(batch_growth_rate = rate, adaptive_batching = adaptive)
    end
    for rate in (NaN, Inf, -Inf, 1.0)
        @test_throws ArgumentError positional(rate = rate)
    end
    for rate in (nextfloat(1.0), 1.2, floatmax(Float64))
        @test BATMakieVisualization(batch_growth_rate = rate).batch_growth_rate == rate
    end
    @test_throws ArgumentError positional(rate = BigFloat(floatmax(Float64)) * 2)
    @test_throws ArgumentError positional(rate = nextfloat(BigFloat(1)))
    @test positional(n_batch = big(50), max_buffered = big(200), rate = big"1.2").batch_growth_rate === 1.2
    @test_throws InexactError positional(max_buffered = big(typemax(Int)) + 1)
end

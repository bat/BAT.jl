# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using StatsBase, StableRNGs
using Distributions


@testset "test_showhist" begin
    showhist_str(h) = (io = IOBuffer(); BAT.showhist_unicode(io, h); String(take!(io)))

    h_exp = fit(Histogram, rand(StableRNG(789990641), Exponential(), 10^4), 1:0.1:2, closed = :left)
    @test showhist_str(h_exp) == "⠀⠀⠀⠀⠀⠀⠀1[██▇▆▆▅▄▄▄▃[2⠀⠀⠀⠀⠀⠀⠀"
    h_normal = fit(Histogram, rand(StableRNG(789990641), Normal(), 10^4), -2:0.2:2)
    @test showhist_str(h_normal) == "⠀⠀⠀⠀⠀⠀-2[▁▂▃▃▄▆▇▇████▇▇▆▅▃▃▂▁[2⠀⠀⠀⠀⠀⠀⠀"
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test
using Plots
using StatsBase


@testset "bin centers & local modes" begin

    data1 = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10]
    hist_1d = fit(Histogram, data1, nbins = 10, closed = :left)
    bathist_1d = BATHistogram(hist_1d)

    @testset "1D BATHistogram" begin
        @test BAT.get_bin_centers(bathist_1d) == [collect(1.5:1:10.5)]
        @test BAT.find_marginalmodes(bathist_1d) == [[5.5], [9.5]]
    end

    data2 = [10, 20, 30, 40, 50, 50, 60, 70, 80, 90, 90, 100]
    hist_2d = fit(Histogram, (data1, data2), nbins = 10, closed = :left)
    bathist_2d = BATHistogram(hist_2d)

    @testset "2D BATHistogram" begin
        @test BAT.get_bin_centers(bathist_2d) == [collect(1.5:1:10.5), collect(15:10:105)]
        @test BAT.get_bin_centers(bathist_2d.h) == [collect(1.5:1:10.5), collect(15:10:105)]
        @test BAT.find_marginalmodes(bathist_2d) == [[5.5, 55.0], [9.5, 95.0]]
    end
end

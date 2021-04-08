using BAT
using Distributions, ArraysOfArrays, ValueShapes
using StatsBase
using Test

@testset "mcmc_stats" begin
    mvnorm = @inferred(MultivariateNormal(2, 1))
    xs = [rand(mvnorm) for i in 1:10^5]
    ys = [logpdf(mvnorm, xi) for xi in xs]
    weight = ones(length(xs))

    x1 = [-2.0, -2.0]; x2 = [0.0, 0.0]; x3 = [2.0, 2.0]; x = hcat(x1, x2, x3)
    density_sample_vector = DensitySampleVector(xs, ys, weight=weight)
    density_sample_1 = DensitySample(x1, logpdf(mvnorm, x1), 1.0, nothing, nothing)
    density_sample_2 = DensitySample(x2, logpdf(mvnorm, x2), 1.0, nothing, nothing)
    density_sample_3 = DensitySample(x3, logpdf(mvnorm, x3), 1.0, nothing, nothing)

    basic_stats = @inferred(BAT.MCMCBasicStats(density_sample_vector))
    moments = @inferred(BAT._bat_stats(basic_stats))

    @inferred(append!(basic_stats, density_sample_vector))
    moments_appended = @inferred(BAT._bat_stats(basic_stats))

    for i in eachindex(moments)
        @test isapprox(moments[i], moments_appended[i])
    end
    
    @inferred(empty!(basic_stats))
    moments_empty = BAT._bat_stats(basic_stats)
    for i in eachindex(moments_empty)
        @test isnan.(moments_empty[i]) == fill(1, size(moments[i]))
    end

    @inferred(push!(basic_stats, density_sample_1))
    @inferred(push!(basic_stats, density_sample_2))
    @inferred(push!(basic_stats, density_sample_3))
end 
# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Random
using JLArrays
import GPUArraysCore

@testset "mcmc_state" begin
    @testset "sample selection" begin
        n, k = 3, 6
        function samples(seed)
            rng = Xoshiro(seed)
            DensitySampleVector(
                v = nestedview(randn(rng, n, k)), logd = randn(rng, k), weight = rand(rng, 1:5, k),
                info = [BAT.MCMCSampleID(1, i, seed, 0, 1, true) for i in 1:k], aux = fill(nothing, k),
            )
        end
        a, b = samples(1), samples(2)
        mask = [true, false, false, true, true, false]
        selected = deepcopy(b)
        selected[findall(mask)] = a[findall(mask)]

        @test BAT._select_samples!(samples(3), mask, a, b) == selected
        # The destination may be one of the sources:
        @test BAT._select_samples!(deepcopy(a), mask, a, b) == selected
        a2, b2 = deepcopy(a), deepcopy(b)
        @test BAT._select_samples!(a2, mask, a2, b) == selected
        @test BAT._select_samples!(b2, mask, a, b2) == selected

        # Batches of variates on a device are selected as a whole:
        on_device(samples) = nestedview(JLArray(flatview(samples.v)))
        GPUArraysCore.allowscalar(false)
        try
            V = BAT._select_variates!(on_device(samples(3)), JLArray(mask), on_device(a), on_device(b))
            @test flatview(V) isa JLArray
            @test Array(flatview(V)) == flatview(selected.v)
        finally
            GPUArraysCore.allowscalar()
        end
    end
end

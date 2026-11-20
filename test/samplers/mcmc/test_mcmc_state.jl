# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Random
using JLArrays
import GPUArraysCore
using HeterogeneousComputing: get_compute_unit, get_gencontext

# Operations on device arrays must not fall back to scalar indexing:
GPUArraysCore.allowscalar(false)

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
        V = BAT._select_variates!(on_device(samples(3)), JLArray(mask), on_device(a), on_device(b))
        @test flatview(V) isa JLArray
        @test Array(flatview(V)) == flatview(selected.v)
    end

    @testset "walker draws on devices" begin
        # The random draws of a step are arrays on the compute unit of the
        # context, in its precision:
        cunit = get_compute_unit(JLArray(Float32[0]))
        genctx = get_gencontext(BATContext(precision = Float32, cunit = cunit))
        n, k = 3, 8
        draws = BAT._BatchedWalkerDraws(genctx, collect(1:k))
        for innovations in (BAT._NormalInnovations(0.5), BAT._TInnovations(1.0, 0.5), BAT._TInnovations(3.0, 0.5))
            E = BAT._draw_innovations(draws, innovations, n)
            @test E isa JLArray{Float32,2} && size(E) == (n, k)
        end
        @test BAT._walker_uniforms(draws) isa JLArray{Float32,1}
    end
end

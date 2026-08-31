# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Distributions, LinearAlgebra, Random123


@testset "StretchMove" begin
    check_ensemble_gaussian_moments(StretchMove(); seed = 25201)

    @testset "is affine equivariant" begin
        matrix = [1.3 0.4; -0.2 0.8]
        shift = [-0.6, 1.1]
        initial = elliptic_ensemble(8)
        transformed_initial = [matrix * x + shift for x in initial]
        transformed_state = ensemble_move_state(
            StretchMove(), transformed_initial;
            target = batmeasure(MvNormal(
                matrix * ENSEMBLE_TARGET_MEAN + shift,
                Symmetric(matrix * ENSEMBLE_TARGET_COVARIANCE * matrix'),
            )),
            seed = 4313,
        )
        reference = ensemble_move_samples(
            StretchMove(), initial; seed = 4313, nwarmup = 0, nsweeps = 32,
        )
        outputs = BAT._empty_chain_outputs(transformed_state)
        transformed_state = BAT.mcmc_iterate!!(outputs, transformed_state; max_nsteps = 32)
        transformed = BAT._merge_chain_outputs(transformed_state, [outputs])

        @test all(
            isapprox(x, matrix \ (y - shift); atol = 1e-10, rtol = 0)
            for (x, y) in zip(reference.v, transformed.v)
        )
    end
end

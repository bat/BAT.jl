# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "tjelmeland" begin
    @test BAT._tjl_multipropT2([135, 288, 64] / 487) ≈ [0, 359, 64] / 423
    @test BAT._tjl_multipropT2([0.2, 0.5, 0.3]) ≈ [0.0, 0.625, 0.375]
    @test sum(BAT._tjl_multipropT2([123, 12, 324, 3, 31] / 493)) ≈ 1.

    #pdist = GenericProposalDist(MvNormal([0., 1.], [1. .5; .5 2.]))
    #target = BAT.MvDistDensity(MvNormal([2., 1.], [1. .6; .6 8.]))
    #rng = MersenneTwister(8477)
    #params_current = [1. , 2.]
    #nproposals = 99
    #all_params = zeros(size(params_current, 1), nproposals + 1)
    #all_params[:, 1] = params_current[:]
    #is_inbounds = BitVector(size(all_params, 2))
    #P_T = zeros(size(all_params, 2))
    #BAT.multi_propose!(rng, pdist, target, all_params, is_inbounds)
    #all_logdensity_values = zeros(size(all_params, 2))
    #BAT.density_logval!(view(all_logdensity_values, 1), target, view(all_params, :, 1))
    #BAT._tjl_multipropT1!(rng, pdist, target, all_params, all_logdensity_values, is_inbounds, P_T)
    #@test sum(P_T) ≈ 1.
    #@test !any(x -> x < 0, P_T)
    #@test all_params[:,1] ≈ params_current

    #params_new1 = zeros(2,190)
    #params_new2 = zeros(3,100)
    #P_T1_1 = zeros(100)
    #@test_throws ArgumentError BAT._tjl_multipropT1!(rng, pdist, target, params_old, params_new1, P_T1_1)
    #@test_throws ArgumentError BAT._tjl_multipropT1!(rng, pdist, target, params_old, params_new2, P_T1_1)
    #@test_throws ArgumentError BAT.multiprop_transition!(P_T1, params_new1, params_old)
    #@test_throws ArgumentError BAT.multiprop_transition!(P_T1, params_new, zeros(4))

    #pdist = GenericProposalDist(MvNormal([0.], [1.]))
    #target = BAT.MvDistDensity(MvNormal([2.4], [.5]))
    #rng = MersenneTwister(8477)
    #params_old = [1.]
    #num_prop = 99


    #@test_throws ArgumentError BAT._tjl_multipropT2([-0.1, 0.1])
    #@test_throws ArgumentError BAT._tjl_multipropT2([0.1, 0.8])
end

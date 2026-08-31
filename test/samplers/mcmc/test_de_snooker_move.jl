# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "DESnookerMove" begin
    check_ensemble_gaussian_moments(DESnookerMove(); seed = 27201)

    @testset "is invariant to ensemble storage order" begin
        initial = elliptic_ensemble(8)
        reference = BAT.mcmc_step!!(ensemble_move_state(DESnookerMove(), initial; seed = 33))
        permutation = Int32[5, 1, 7, 3, 8, 4, 2, 6]
        reordered = ensemble_move_state(DESnookerMove(), initial[permutation]; seed = 33)
        chain_state = reordered.chain_state
        # Update every per-walker buffer.
        for samples in (
            chain_state.current.x, chain_state.current.z,
            chain_state.proposed.x, chain_state.proposed.z, chain_state.output,
        )
            # Rewrite each walker ID.
            for i in eachindex(samples.info)
                id = samples.info[i]
                samples.info[i] = BAT.MCMCSampleID(
                    id.chainid, permutation[i], id.chaincycle,
                    id.stepno, id.proposalid, id.sampletype,
                )
            end
        end
        chain_state.walker_order = sortperm(permutation)
        reordered = BAT.mcmc_step!!(reordered)
        reference_order = sortperm(getproperty.(reference.chain_state.current.x.info, :walkerid))
        reordered_order = sortperm(getproperty.(reordered.chain_state.current.x.info, :walkerid))

        @test reference.chain_state.current.z.v[reference_order] ==
            reordered.chain_state.current.z.v[reordered_order]
        @test reference.chain_state.accepted[reference_order] ==
            reordered.chain_state.accepted[reordered_order]
    end
end

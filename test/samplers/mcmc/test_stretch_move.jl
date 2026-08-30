# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform, NoMCMCTempering
using Distributions, LinearAlgebra, Random123


function _stretch_move_state(
    v_init;
    nwalkers = length(v_init),
    scale = 2,
    adaptive_transform = NoAdaptiveTransform(),
    sample_weighting = RepetitionWeighting(),
    proposal_tuning = NoMCMCProposalTuning(),
    transform_tuning = NoMCMCTransformTuning(),
)
    algorithm = TransformedMCMC(
        proposal = StretchMove(scale = scale),
        pretransform = DoNotTransform(),
        adaptive_transform = adaptive_transform,
        convergence = AssumeConvergence(),
        nwalkers = nwalkers,
        sample_weighting = sample_weighting,
        proposal_tuning = proposal_tuning,
        transform_tuning = transform_tuning,
    )
    target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
    return BAT.MCMCState(algorithm, target, 1, v_init, BATContext(rng = Philox4x((564, 80))))
end

function _capture_error(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end


@testset "StretchMove" begin
    @testset "constructor and defaults" begin
        proposal = StretchMove()
        @test proposal.scale == 2
        @test StretchMove(scale = 3f0).scale === 3f0

        for scale in (1, 0, -1, Inf, NaN)
            err = _capture_error(() -> StretchMove(scale = scale))
            @test err isa ArgumentError
            @test occursin("scale", sprint(showerror, err))
        end

        algorithm = TransformedMCMC(proposal = proposal, nwalkers = 4)
        @test algorithm.proposal_tuning isa NoMCMCProposalTuning
        @test algorithm.adaptive_transform isa NoAdaptiveTransform
        @test algorithm.transform_tuning isa NoMCMCTransformTuning
        @test algorithm.tempering isa NoMCMCTempering
        @test algorithm.init isa MCMCRetryInit
    end

    @testset "requires no adaptive transform" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(
            full_rank; adaptive_transform = BAT.TriangularAffineTransform(),
        ))
        @test err isa ArgumentError
        @test occursin("NoAdaptiveTransform", sprint(showerror, err))
    end

    @testset "requires explicit walker count" begin
        err = _capture_error(() -> TransformedMCMC(proposal = StretchMove()))
        @test err isa ArgumentError
        @test occursin("nwalkers", sprint(showerror, err))
    end

    @testset "validates initialized ensemble" begin
        too_few = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(too_few))
        @test err isa ArgumentError
        @test occursin("at least 2 * d", sprint(showerror, err))

        nearly_collinear = [[0.0, 0.0], [1.0, 1e-18], [2.0, 2e-18], [3.0, 4e-18]]
        err = _capture_error(() -> _stretch_move_state(nearly_collinear))
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("4 walkers", message)
        @test occursin("dimension 2", message)
        @test occursin("rank 1", message)

        nonfinite = [[0.0, 0.0], [1.0, 0.0], [0.0, NaN], [1.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(nonfinite))
        @test err isa ArgumentError
        @test occursin("finite transformed coordinates", sprint(showerror, err))

        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        @test _stretch_move_state(full_rank) isa BAT.MCMCState
        state32 = _stretch_move_state(map(v -> Float32.(v), full_rank); scale = 2.0)
        @test state32 isa BAT.MCMCState
        @test state32.chain_state.proposal.scale === 2f0

        err = _capture_error(() -> _stretch_move_state(
            map(v -> Float32.(v), full_rank); scale = nextfloat(1.0),
        ))
        @test err isa ArgumentError
        @test occursin("after conversion", sprint(showerror, err))
    end

    @testset "rejects unsupported configurations" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        err = _capture_error(() -> _stretch_move_state(full_rank; sample_weighting = ARPWeighting()))
        @test err isa ArgumentError
        @test occursin("RepetitionWeighting", sprint(showerror, err))

        err = _capture_error(() -> _stretch_move_state(
            full_rank; proposal_tuning = BAT.StepSizeAdaptor(),
        ))
        @test err isa ArgumentError
        @test occursin("NoMCMCProposalTuning", sprint(showerror, err))

        err = _capture_error(() -> _stretch_move_state(
            full_rank; transform_tuning = BAT.StanLikeTuning(),
        ))
        @test err isa ArgumentError
        @test occursin("NoMCMCTransformTuning", sprint(showerror, err))

        multi = MCMCMultiProposal(proposals = BAT.MCMCProposal[StretchMove(), RandomWalk()])
        algorithm = TransformedMCMC(
            proposal = multi,
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
        )
        target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
        err = _capture_error(() -> BAT.MCMCState(
            algorithm, target, 1, full_rank, BATContext(rng = Philox4x((564, 81))),
        ))
        @test err isa ArgumentError
        @test occursin("StretchMove", sprint(showerror, err))
    end
end

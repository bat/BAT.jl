# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions
using Random123
import ForwardDiff

using BAT: batmeasure, TriangularAffineTransform, NoMCMCTransformTuning

@testset "multitrafo_tuner" begin
    context = BATContext(rng = Philox4x((564, 85)), ad = ForwardDiff)

    objective = MvNormal([1.0, -1.0], [2.0 1.2; 1.2 1.5])
    target = batmeasure(objective)

    # An untuned outer component (prior-based init) plus a RAM-tuned
    # inner component starting from the identity: the composite geometry
    # has to be learned by the inner component alone:
    at = AdaptiveTransformChain((
        TriangularAffineTransform(init = BAT.UnitTransformInit()),
        TriangularAffineTransform(),
    ))
    alg = TransformedMCMC(
        proposal = RandomWalk(),
        adaptive_transform = at,
        transform_tuning = MultiTrafoTuning((RAMTuning(), NoMCMCTransformTuning())),
        pretransform = DoNotTransform(),
        nchains = 2,
        nsteps = 5 * 10^4
    )

    smplres = BAT.sample_and_verify(target, alg, objective, context, max_retries = 0)
    @test smplres.verified

end

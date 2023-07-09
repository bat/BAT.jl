using BAT
using Test

import Cuba

using Base.Threads
using Distributions, ValueShapes


@testset "cuba_integration" begin
    function test_integration(target, algorithm::BAT.CubaIntegration)
        r = bat_integrate(target, algorithm).result
        val_expected = 1
        @test isapprox(r.val, val_expected, rtol = 20 * algorithm.rtol)
        # @test r.err < 100 * abs(r.val - val_expected)
    end

    uvprior = truncated(Normal(), -1, 3)

    mvprior = NamedTupleDist(
        a = Beta(0.5, 0.5),
        b = truncated(Normal(), -1, 3),
        c = Beta(5, 1),
        e = Uniform(-4, 7)
    )

    mvprior_simple = NamedTupleDist(
        a = truncated(Normal(), -1, 3),
        b = truncated(Normal(), 1, 3),
    )

    # ToDo: Use more compex test targets:

    test_integration(mvprior, VEGASIntegration(nthreads=2))
    test_integration(mvprior, VEGASIntegration())

    test_integration(mvprior, SuaveIntegration(rtol = 1e-3))

    test_integration(mvprior_simple, DivonneIntegration(rtol = 1e-3))

    test_integration(mvprior, CuhreIntegration(rtol = 1e-3))
    
    @test_throws ErrorException test_integration(mvprior, CuhreIntegration(maxevals=4))
    test_integration(mvprior, CuhreIntegration(maxevals=4, strict=false))
end

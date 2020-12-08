using BAT
using Test

import Cuba

using Base.Threads
using Distributions, ValueShapes


@testset "cuba_integration" begin
    function test_integration(target, algorithm::BAT.CubaIntegration)
        r = bat_integrate(target, algorithm).result
        val_expected = 1
        @test isapprox(r.val, val_expected, rtol = 100 * algorithm.rtol)
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

    test_integration(uvprior, VEGASIntegration())
    test_integration(uvprior, VEGASIntegration())
    test_integration(mvprior, VEGASIntegration(nthreads = 1))
    test_integration(mvprior, VEGASIntegration(nthreads = nthreads()))

    test_integration(mvprior, SuaveIntegration())

    test_integration(mvprior_simple, DivonneIntegration(rtol = 1e-4))

    test_integration(mvprior, CuhreIntegration())

    test_integration(mvprior, CuhreIntegration(nthreads=2))
end

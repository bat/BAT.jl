# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, ValueShapes, DensityInterface
using MeasureBase: getdof, massof
using BAT: BispacedMeasure, batmeasure, empiricalof

# A minimal sample generator for testing the samplegen carry-over rules:
struct _BSTestSampleGen <: BAT.AbstractSampleGenerator end
BAT.getproposal(::_BSTestSampleGen) = nothing

@testset "bispaced_measure" begin
    context = BATContext()
    prior = distprod(a = Normal(2.0, 1.0), b = Exponential(0.7))
    m = batmeasure(prior)
    vs = varshape(m)

    n = 100
    xs = rand(Random.default_rng(), m^n)
    smpls = DensitySampleVector(v = xs, logd = logdensityof(m).(xs))
    dsm = DensitySampleMeasure(smpls, dof = getdof(m))

    f = BAT.transform_function(NormalBased(), m)
    zs = f.(xs)
    smpls_z = DensitySampleVector(v = zs, logd = fill(NaN, n))
    dsm_z = DensitySampleMeasure(smpls_z, dof = getdof(m))

    p = BispacedMeasure(dsm, dsm_z, hash(f))
    @test p.main === dsm
    @test p.transformed === dsm_z
    @test p.f_hash == hash(f)
    # Canonical form without a transformed representation:
    @test BispacedMeasure(dsm) === BispacedMeasure(dsm, nothing, UInt(0))

    # Self-building form: the transformed side is the transform result by
    # construction, stamped with the hash of the transformation used:
    pf = BispacedMeasure(f, dsm, context)
    @test pf.main === dsm
    @test pf.f_hash == hash(f)
    @test BAT.samplesof(pf.transformed).v == zs
    # A measure is not a transformation, adopting an existing transformed
    # representation requires the explicit three-argument form:
    @test_throws ArgumentError BispacedMeasure(dsm, dsm_z)

    @testset "measure interface delegates to main" begin
        @test varshape(p) == varshape(dsm)
        @test getdof(p) == getdof(dsm)
        @test massof(p) == massof(dsm)
        @test BAT.samplesof(p) === BAT.samplesof(dsm)
        @test empiricalof(p) === dsm

        pm = BispacedMeasure(m)
        x = first(xs)
        @test logdensityof(pm, x) == logdensityof(m, x)

        up = unshaped(p, vs)
        @test up.main == unshaped(dsm, vs)
        # The transformed representation is already unshaped and stays as-is:
        @test up.transformed === dsm_z
    end

    @testset "in EvaluatedMeasure" begin
        em = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = p)
        @test em.transform_intent === NormalBased()
        @test em.f_transform === f
        @test em.empirical === p
        @test empiricalof(em) === dsm
        @test BAT.samplesof(em) === BAT.samplesof(dsm)

        # Pair replacement and clearing stay atomic, the view is kept:
        em2 = EvaluatedMeasure(em, empirical = dsm)
        @test em2.empirical === BispacedMeasure(dsm)
        @test em2.transform_intent === NormalBased()
        em3 = EvaluatedMeasure(em, empirical = nothing)
        @test isnothing(em3.empirical)

        # The transformed keyword updates the measure cache with a bare
        # measure, labeled by the transform intent it belongs to:
        m_z = unshaped(m)
        em4 = EvaluatedMeasure(em, transform_intent = NormalBased(), transformed = m_z)
        @test em4.unevaluated === BispacedMeasure(m, m_z, hash(f))
        @test BAT.unevaluated(em4) === m
        @test EvaluatedMeasure(em4, empirical = nothing).unevaluated === em4.unevaluated
        em5 = EvaluatedMeasure(em4, transformed = nothing)
        @test em5.unevaluated === BispacedMeasure(m)
        @test em5.transform_intent === NormalBased()

        # The constructor rejects unlabeled or ill-formed transformed-space
        # content:
        @test_throws ArgumentError EvaluatedMeasure(em, transformed = m_z)
        @test_throws ArgumentError EvaluatedMeasure(m, empirical = p)
        @test_throws ArgumentError EvaluatedMeasure(m, transform_intent = DoNotTransform(), transformed = m_z)

        # Pairs from another EvaluatedMeasure of the same measure are
        # adopted based on their transformation-hash witness:
        em_b = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = p)
        em_c = EvaluatedMeasure(em, empirical = em_b.empirical)
        @test em_c.empirical === em_b.empirical

        # approx is stored as a pair as well, atomically:
        em6 = EvaluatedMeasure(em4, approx = m)
        @test em6.approx === BispacedMeasure(m)
        @test BAT.approxof(em6) === m
        em7 = EvaluatedMeasure(em6, transform_intent = NormalBased(), approx = BispacedMeasure(m, m_z, hash(f)))
        @test em7.approx === BispacedMeasure(m, m_z, hash(f))
        @test BAT.approxof(em7) === m
        @test isnothing(EvaluatedMeasure(em7, approx = nothing).approx)

        # Switching the transformed-space view strips the transformed sides
        # off all kept entries, their main sides stay, and the cached
        # transformation function is dropped with them:
        em8 = EvaluatedMeasure(em7, transform_intent = UniformBased())
        @test em8.transform_intent === UniformBased()
        @test isnothing(em8.f_transform)
        @test em8.unevaluated === BispacedMeasure(m)
        @test em8.empirical === BispacedMeasure(dsm)
        @test em8.approx === BispacedMeasure(m)

        # Updates that don't touch the view leave the pairs untouched:
        em9 = EvaluatedMeasure(em7, mass = 1)
        @test em9.unevaluated === em7.unevaluated
        @test em9.empirical === em7.empirical
        @test em9.approx === em7.approx
        @test em9.transform_intent === em7.transform_intent
        @test em9.f_transform === f
    end

    @testset "cached views keep live empirical weights" begin
        for intent in (NormalBased(), UniformBased()), seed in (7, 4711, 892374)
            m_z, f_view = BAT.transform_and_unshape(intent, m, context)
            view_xs = rand(Xoshiro(seed), m^2)
            view_smpls = DensitySampleVector(
                v = view_xs,
                logd = logdensityof(m).(view_xs),
                weight = [1.0, 1.0],
            )
            view_dsm = DensitySampleMeasure(view_smpls, dof = getdof(m))
            view_smpls_z = BAT.transform_samples(f_view, view_smpls)
            @test view_smpls_z.weight !== view_smpls.weight

            shared_pair = BAT._viewrep_empirical(
                view_dsm, view_smpls_z, f_view, intent, getdof(m), nothing,
            )
            @test BAT.samplesof(shared_pair.main).weight === BAT.samplesof(shared_pair.transformed).weight
            shared_em = EvaluatedMeasure(
                m,
                transform_intent = intent,
                f_transform = f_view,
                empirical = shared_pair,
                transformed = m_z,
            )
            BAT.samplesof(shared_em).weight .= [0.0, 2.0]
            shared_z, _ = BAT.transform_and_unshape(intent, shared_em, context)
            @test BAT.samplesof(shared_z).weight == [0.0, 2.0]
            @test empiricalof(shared_z) === shared_pair.transformed
            @test BAT.validate_evalmeasure(shared_em, context = context) === shared_em
            shared_direct = bat_transform(intent, shared_em, PriorSubstitution(), context)
            @test BAT.samplesof(shared_direct.result).weight == [0.0, 2.0]

            external_smpls = DensitySampleVector(view_dsm)
            external_smpls.weight .= [1.0, 1.0]
            external_dsm = DensitySampleMeasure(external_smpls, dof = getdof(m))
            external_smpls_z = BAT.transform_samples(f_view, external_smpls)
            external_pair = BispacedMeasure(
                external_dsm,
                DensitySampleMeasure(external_smpls_z, dof = getdof(m)),
                hash(f_view),
            )
            external_em = EvaluatedMeasure(
                m,
                transform_intent = intent,
                f_transform = f_view,
                empirical = external_pair,
                transformed = m_z,
            )
            BAT.samplesof(external_em).weight .= [0.0, 2.0]
            external_z, _ = BAT.transform_and_unshape(intent, external_em, context)
            @test BAT.samplesof(external_z).weight == [0.0, 2.0]
            @test empiricalof(external_z) !== external_pair.transformed
            external_direct = bat_transform(intent, external_em, PriorSubstitution(), context)
            @test BAT.samplesof(external_direct.result).weight == [0.0, 2.0]
            @test empiricalof(external_direct.result) !== external_pair.transformed
        end
    end

    @testset "unshaped transports the pairs" begin
        em = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = p)
        uem = unshaped(em, vs)
        @test uem.transform_intent === NormalBased()
        @test uem.empirical isa BispacedMeasure
        @test uem.empirical.main == unshaped(dsm, vs)
        # The transformed representation is already unshaped and stays as-is,
        # re-stamped for the correspondingly composed transformation:
        @test uem.empirical.transformed === dsm_z
        @test uem.empirical.f_hash == hash(uem.f_transform)

        shared_p = BAT._viewrep_empirical(dsm, smpls_z, f, NormalBased(), getdof(m), nothing)
        shared_em = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = shared_p)
        shared_uem = unshaped(shared_em, vs)
        @test BAT.samplesof(shared_uem.empirical.main).weight === BAT.samplesof(shared_uem.empirical.transformed).weight
    end

    @testset "resampling keeps the pair coherent" begin
        em = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = p)
        em_r = evalmeasure(em, RandResampling(nsamples = 50), context)
        pr = em_r.empirical
        @test pr isa BispacedMeasure
        @test em_r.transform_intent === NormalBased()
        @test length(BAT.samplesof(pr.main)) == 50
        @test length(BAT.samplesof(pr.transformed)) == 50
        @test BAT.samplesof(pr.main).weight === BAT.samplesof(pr.transformed).weight
        # Shared indices: the transformed samples are the transforms of the
        # main samples, in the same order:
        @test f.(BAT.samplesof(pr.main).v) == BAT.samplesof(pr.transformed).v
    end

    @testset "PriorSubstitution carries matching representations" begin
        post = PosteriorMeasure(logfuncdensity(v -> -0.5 * (v.a - 1)^2), prior)
        q_z = batmeasure(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]))
        gen = _BSTestSampleGen()
        _, f_post = BAT.transform_and_unshape(NormalBased(), post, context)
        em = EvaluatedMeasure(
            post,
            transform_intent = NormalBased(),
            f_transform = f_post,
            approx = BispacedMeasure(m, q_z, hash(f_post)),
            samplegen = gen
        )

        # On a matching intent the transformed-space sides carry over by
        # identity, the sample generator is native to that space:
        em_z, _ = BAT.transform_and_unshape(NormalBased(), em, context)
        @test BAT.approxof(em_z) === q_z
        @test BAT.samplegenof(em_z) === gen
        @test em_z.transform_intent === DoNotTransform()
        @test em_z.f_transform === identity

        # On a different intent there is no safe way to recover them:
        em_u, _ = BAT.transform_and_unshape(UniformBased(), em, context)
        @test isnothing(BAT.approxof(em_u))
        @test isnothing(BAT.samplegenof(em_u))

        # Switching the view also strips the sample generator, it has no
        # meaning in another view:
        em_sw = EvaluatedMeasure(em, transform_intent = UniformBased())
        @test isnothing(em_sw.samplegen)
    end
end

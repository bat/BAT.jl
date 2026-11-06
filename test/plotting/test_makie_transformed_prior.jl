# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieTransformedPrior
using BAT, Makie, Test, Distributions, ValueShapes, Random, Statistics, Logging, InverseFunctions

include(joinpath(pkgdir(BAT), "test", "plotting", "makie_rng_testutils.jl"))
const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct FigureCapture <: AbstractDisplay
    figures::Vector{Makie.Figure}
end
Base.displayable(::FigureCapture, ::MIME"image/png") = true
Base.display(c::FigureCapture, fig::Makie.Figure) = (push!(c.figures, fig); nothing)

fields(s) = (v=deepcopy(s.v), logd=copy(s.logd), weight=copy(s.weight),
    info=deepcopy(s.info), aux=deepcopy(s.aux))

function fixture()
    prior = MvNormal(zeros(2), ones(2))
    calls = Ref(0)
    likelihood = BAT.DensityInterface.logfuncdensity() do x
        calls[] += 1
        -sum(abs2, x) / 2
    end
    posterior = PosteriorMeasure(likelihood, prior)
    f = BAT.DistributionTransform(MvNormal(fill(2.0, 2), fill(3.0, 2)), prior)
    target = bat_transform(f, posterior, BATContext(rng=BAT.Philox4x((91, 23)))).result
    return (; prior, posterior, f, target, calls)
end

function algorithm()
    TransformedMCMC(nchains=1, nwalkers=1, nsteps=12, strict=false,
        convergence=AssumeConvergence(), init=MCMCRetryInit(nsteps_init=10),
        adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(), proposal_tuning=NoMCMCProposalTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1, nsteps_final=0))
end

@testset "Transformed posterior display domain" begin
    (; prior, posterior, f, target, calls) = fixture()
    n, tail = 101, 0.1
    rng = BAT.Philox4x((128, 91))
    original_rng = copy(rng)
    oracle_rng = copy(rng)
    # Independent affine-coordinate oracle from the original prior draws.
    original = reduce(hcat, [rand(oracle_rng, prior) for _ in 1:n])
    displayed = 2 .+ 3 .* original
    expected = ([quantile(row, tail) for row in eachrow(displayed)],
        [quantile(row, 1-tail) for row in eachrow(displayed)])
    original_bounds = ([quantile(row, tail) for row in eachrow(original)],
        [quantile(row, 1-tail) for row in eachrow(original)])
    calls_before = calls[]
    for input in (target, BAT.unevaluated(target))
        private = copy(rng)
        default_before = copy(Random.default_rng())
        lo, hi = EXT._estimate_prior_domain(private, input, 2;
            n_prior_samples=n, tail_prob=tail, display_shape=varshape(target))
        @test lo ≈ expected[1] && hi ≈ expected[2]
        @test private == oracle_rng
        @test rng == original_rng && Random.default_rng() == default_before
        @test calls[] == calls_before
        lo, hi = EXT._estimate_prior_domain(copy(rng), input, 2;
            n_prior_samples=n, tail_prob=tail, display_shape=varshape(prior),
            f_display=inverse(f))
        @test lo ≈ original_bounds[1] && hi ≈ original_bounds[2]
    end

    # An already sampleable pushforward keeps the same direct-draw sequence.
    iid = BAT.MeasureBase.pushfwd(f, batmeasure(prior))
    lo, hi = EXT._estimate_prior_domain(copy(rng), iid, 2;
        n_prior_samples=n, tail_prob=tail)
    @test lo ≈ expected[1] && hi ≈ expected[2]
    @test calls[] == calls_before
end

@testset "Live transformed posterior preserves sampling" begin
    @test _prepare_makie_rng_test!() === nothing
    (; posterior, target) = fixture()
    capture = FigureCapture(Makie.Figure[])
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        with_logger(NullLogger()) do
            for input in (posterior, target, BAT.unevaluated(target))
                vis = BATVisualizer(BATMakieVisualization())
                plain_context = BATContext(rng=BAT.Philox4x((42, 17)))
                live_context = BATContext(rng=BAT.Philox4x((42, 17)), visualizer=vis)
                try
                    Random.seed!(713)
                    plain = BAT.evalmeasure(input, algorithm(), plain_context)
                    default_plain = copy(Random.default_rng())
                    Random.seed!(713)
                    observed = BAT.evalmeasure(input, algorithm(), live_context)
                    default_live = copy(Random.default_rng())
                    @test isequal(fields(samplesof(observed)), fields(samplesof(plain)))
                    @test sum(samplesof(observed).weight) == 12
                    @test BAT.get_rng(live_context) == BAT.get_rng(plain_context)
                    @test rand(copy(BAT.get_rng(live_context)), UInt64, 16) ==
                        rand(copy(BAT.get_rng(plain_context)), UInt64, 16)
                    @test default_live == default_plain
                    status = BAT.visualizer_status(vis)
                    @test status.outcome === :complete && status.data_complete && status.presentation_valid
                    @test status.quiesced && !status.cleanup_pending && isnothing(status.error)
                    @test BAT.visualizer_figure(vis) === last(capture.figures)
                    graph = EXT._live_graph(vis)
                    @test length(graph[:domain_lo][]) == length(graph[:domain_hi][]) == 2
                    @test all(graph[:domain_lo][] .< graph[:domain_hi][])
                    history = only(only(graph[:samples][]))
                    expected = samplesof(plain)
                    # Live history includes warmup; spans identify main-run mass.
                    timing = only(only(graph[:sample_timing][]))
                    spans = filter(s -> s.cycle == first(expected.info).chaincycle, timing)
                    rows = unique(s.row for s in spans)
                    @test length(rows) == length(expected)
                    @test all(isapprox(a, b) for (a, b) in zip(history.v[rows], expected.v))
                    @test history.logd[rows] ≈ expected.logd
                    @test [sum(s.last-s.first+1 for s in spans if s.row == row) for row in rows] == expected.weight
                    @test history.aux[rows] == expected.aux
                finally
                    BAT.dispose_visualizer!(vis)
                    @test timedwait(() -> !BAT.visualizer_status(vis).cleanup_pending, 10.0) == :ok
                    status = BAT.visualizer_status(vis)
                    @test status.quiesced && status.history_released && !status.cleanup_pending
                end
            end
        end
        @test length(capture.figures) == 3
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end
end

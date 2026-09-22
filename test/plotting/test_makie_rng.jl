# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional Makie regression entry point. Run with Makie available; a loaded
# rendering backend additionally exercises shader warmup. The application suite
# runs this file with CairoMakie, without adding Makie to BAT's core test target.
module TestMakieRNG

using BAT, Test
using Distributions, LinearAlgebra, Makie, Random, ValueShapes
using BAT: Philox4x, Threefry4x

const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct FigureCapture <: AbstractDisplay
    count::Int
end
Base.displayable(::FigureCapture, ::MIME"image/png") = true
Base.display(d::FigureCapture, ::Makie.Figure) = (d.count += 1; nothing)

function with_capture(f)
    capture = FigureCapture(0)
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        result = f()
        @test capture.count == 1
        return result
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end

function cleanup(vis)
    # Assert normal shutdown before this fallback; failed tests must not leak tasks.
    vis.content.is_live[] = false
    lock(vis.content.buffer_lock) do
        notify(vis.content.buffer_cond, all=true)
    end
    task = vis.content.listener_task[]
    isnothing(task) || wait(task)
end

function algorithm(; init = MCMCChainPoolInit(nsteps_init = 30), nwalkers = 1, store_burnin = false)
    TransformedMCMC(
        nchains = 2, nwalkers = nwalkers, nsteps = 120, strict = false,
        store_burnin = store_burnin, init = init,
        adaptive_transform = BAT.NoAdaptiveTransform(),
        transform_tuning = NoMCMCTransformTuning(),
        convergence = GelmanRubinConvergence(),
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 30, max_ncycles = 1, nsteps_final = 5),
    )
end

const RNGS = (Philox4x((564, 84)), Threefry4x((564, 84, 19, 71)))

# Fail only the visualization's domain source, after consuming private randomness.
struct FailingDomainTarget end
struct FailingDomainSource <: ContinuousUnivariateDistribution end
const DOMAIN_FAILURE = ErrorException("Intentional failure after a private domain draw")
const FAILED_DRAWS = Ref(0)
BAT.get_initsrc_from_target(::FailingDomainTarget) = FailingDomainSource()
function Random.rand(rng::AbstractRNG, ::FailingDomainSource)
    rand(rng, UInt64)
    FAILED_DRAWS[] += 1
    throw(DOMAIN_FAILURE)
end

# No visualization must not require RNG copying, even for another backend's RNG.
struct UncopyableRNG <: AbstractRNG end
Base.copy(::UncopyableRNG) = error("Unexpected RNG copy")

@testset "Makie RNG isolation" begin
    @test BAT.init_visualizer!(BATVisualizer(); rng = UncopyableRNG()) === nothing

    @testset "Visualizer construction leaves the default RNG untouched" begin
        Random.seed!(771)
        before = copy(Random.default_rng())
        vis = BATVisualizer(BATMakieVisualization())
        @test isnothing(vis.content.listener_task[])
        @test rand(UInt64, 16) == rand(before, UInt64, 16)
    end

    @testset "Explicit domain randomness" begin
        prior = NamedTupleDist(a = Normal(100, 2), b = Uniform(20, 30))
        targets = (BAT.BATDistMeasure(Normal(100, 2)), BAT.batmeasure(prior),
                   BAT.lbqintegral(x -> -abs2(x.a - 100), prior))
        for base in RNGS, partitioned in (false, true), offset in (0, 1, 3, 9)
            rng = partitioned ? AbstractRNG(BAT.RNGPartition(copy(base), 1:3), 2) : copy(base)
            rand(rng, UInt64, offset)
            before = copy(rng)
            for target in targets, budget in (17, 201)
                ndof = totalndof(varshape(target))
                private = copy(rng)
                Random.seed!(771)
                default_before = copy(Random.default_rng())
                lo, hi = EXT._estimate_prior_domain(private, target, ndof; n_prior_samples = budget)
                @test rand(UInt64, 16) == rand(default_before, UInt64, 16)
                @test rng == before
                @test private != before
                @test length(lo) == length(hi) == ndof
                @test all(isfinite, lo) && all(isfinite, hi) && all(lo .< hi)
                @test lo[1] < 100 < hi[1]
                ndof == 2 && @test 20 <= lo[2] < hi[2] <= 30
                Random.seed!(992)
                @test (lo, hi) == EXT._estimate_prior_domain(copy(rng), target, ndof; n_prior_samples = budget)
            end
        end
    end

    @testset "Initializer copies the supplied RNG" begin
        for base in RNGS, fail in (false, true)
            vis = BATVisualizer(BATMakieVisualization(max_buffered = 10^6, adaptive_batching = false))
            ctx = BATContext(rng = copy(base), visualizer = vis)
            target = BAT.BATDistMeasure(MvNormal([100.0, 25.0], Matrix{Float64}(I, 2, 2)))
            state = BAT.MCMCState(algorithm(), target, Int32(1), [[100.0, 25.0]], ctx)
            state.chain_state.current.x.weight .= 1
            outputs = BAT._empty_chain_outputs(state)
            push!(only(outputs), deepcopy(first(state.chain_state.current.x)))
            rng = BAT.get_rng(ctx)
            rand(rng, UInt64, 3)
            before = copy(rng)
            expected_domain = EXT._estimate_prior_domain(copy(rng), target, 2)
            Random.seed!(771)
            default_before = copy(Random.default_rng())
            try
                if fail
                    old_count = FAILED_DRAWS[]
                    caught = try
                        BAT.init_visualizer!(vis; rng = rng, mcmc_states = [state], outputs = [outputs],
                            f_pretransform = identity, target = FailingDomainTarget())
                        nothing
                    catch err
                        err
                    end
                    @test caught === DOMAIN_FAILURE
                    @test FAILED_DRAWS[] == old_count + 1
                    @test isnothing(vis.content.listener_task[])
                    @test isempty(vis.content.chain_ids)
                else
                    with_capture() do
                        BAT.init_visualizer!(vis; rng = rng, mcmc_states = [state], outputs = [outputs],
                            f_pretransform = identity, target = target)
                    end
                    @test (vis.content.graph[:domain_lo][], vis.content.graph[:domain_hi][]) == expected_domain
                    @test !isnothing(vis.content.listener_task[])
                end
                @test rng == before
                @test rand(copy(rng), UInt64, 16) == rand(before, UInt64, 16)
                @test rand(UInt64, 16) == rand(default_before, UInt64, 16)
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Sampling is identical with visualization" begin
        target = NamedTupleDist(a = Normal(-2, 0.7), b = Normal(3, 1.1))
        for base in RNGS, retry in (false, true), retained in (false, true)
            init = retry ? MCMCRetryInit(nsteps_init = 30) : MCMCChainPoolInit(nsteps_init = 30)
            alg = algorithm(; init = init, nwalkers = retry ? 2 : 1, store_burnin = retained)
            # Construct both contexts before resetting the default RNG. Both use
            # explicit RNGs; the paired runs differ only in visualization.
            vis = BATVisualizer(BATMakieVisualization(max_buffered = 10^6, adaptive_batching = false))
            plain_ctx = BATContext(rng = copy(base))
            live_ctx = BATContext(rng = copy(base), visualizer = vis)
            Random.seed!(882)
            expected = BAT.evalmeasure(target, alg, plain_ctx)
            default_after_plain = copy(Random.default_rng())
            Random.seed!(882)
            try
                observed = with_capture() do
                    BAT.evalmeasure(target, alg, live_ctx)
                end
                @test !vis.content.is_live[]
                @test istaskdone(vis.content.listener_task[])
                @test vis.content.n_buffer_steps[] == 0
                @test rand(UInt64, 16) == rand(default_after_plain, UInt64, 16)
                plain_samples, live_samples = BAT.samplesof(expected), BAT.samplesof(observed)
                @test !isempty(live_samples)
                for field in (:v, :weight, :logd, :info)
                    @test getproperty(live_samples, field) == getproperty(plain_samples, field)
                end
                @test BAT.get_rng(live_ctx) == BAT.get_rng(plain_ctx)
                @test rand(copy(BAT.get_rng(live_ctx)), UInt64, 16) == rand(copy(BAT.get_rng(plain_ctx)), UInt64, 16)
                plain_chains = BAT.samplegenof(expected).chain_states
                live_chains = BAT.samplegenof(observed).chain_states
                @test length(live_chains) == length(plain_chains) == 2
                for (live, plain) in zip(live_chains, plain_chains)
                    @test BAT.get_rng(live.context) == BAT.get_rng(plain.context)
                    @test rand(copy(BAT.get_rng(live.context)), UInt64, 16) == rand(copy(BAT.get_rng(plain.context)), UInt64, 16)
                    @test length(live.walker_genctxs) == length(plain.walker_genctxs) == alg.nwalkers
                    for (live_walker, plain_walker) in zip(live.walker_genctxs, plain.walker_genctxs)
                        @test BAT.get_rng(live_walker) == BAT.get_rng(plain_walker)
                    end
                end
            finally
                cleanup(vis)
            end
        end
    end
end
end # module

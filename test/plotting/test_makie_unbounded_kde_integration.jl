# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run as a process-bounded optional Makie case. Numerical admission must reach
# both public entry points and the existing live failure owner before rendering.
module TestMakieUnboundedKDEIntegration

using BAT, Test, Makie, CairoMakie, Distributions, Random, Logging

const EXT = Base.get_extension(BAT, :BATMakieExt)
const DATA = [-2.0 -1.0 0.0 1.0 2.0; 1.0 -2.0 2.0 0.0 -1.0]
const DIAG = BAT._default_makie_diagonal_config()
const PAIR = BAT._default_makie_triagonal_config()
const RECIPES = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)
const QUANTILES = (upper=QuantileKDE2D, diagonal=QuantileKDE1D, lower=QuantileKDE2D)

samples(data=DATA; weights=ones(size(data, 2))) = DensitySampleVector(
    v=EXT.VectorOfSimilarVectors(copy(data)), logd=zeros(size(data, 2)), weight=copy(weights))

function primitives(data, recipe; weights=ones(size(data, 2)), levels=nothing, support=nothing)
    config = recipe isa Union{KDE1D,QuantileKDE1D} ? DIAG : PAIR
    config = merge(config, (; support))
    isnothing(levels) || (config = merge(config, (; levels)))
    return EXT.compute_plotting_primitives(view(data, :, :), view(weights, :), recipe,
        EXT.LiveRecipe(), EXT.LiveCell(), config)
end

function caught(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

function check_kde_error(f; direct=false, message="KDE:")
    err = caught(f)
    @test !isnothing(err)
    direct && @test err isa ArgumentError
    isnothing(err) || @test occursin(message, sprint(showerror, err))
    return nothing
end

cell(fig, i, j) = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))

function descendants(axis, T)
    result = Any[]
    function visit(plot)
        plot isa T && push!(result, plot)
        foreach(visit, plot.plots)
    end
    foreach(visit, axis.scene.plots)
    return result
end

function heat(axis)
    plot = only(descendants(axis, Makie.Heatmap))
    return (x=copy(plot[1][]), y=copy(plot[2][]), z=copy(plot[3][]))
end

@testset "Unbounded KDE public numerical admission" begin
    @testset "All four recipes reject before masking or region construction" begin
        for recipe in (KDE1D(), QuantileKDE1D(), KDE2D(), QuantileKDE2D())
            one_dimensional = recipe isa Union{KDE1D,QuantileKDE1D}
            fixtures = one_dimensional ? (reshape(1e14 .+ DATA[1, :], 1, :),) :
                (1e-100 .* DATA, 1e100 .* DATA)
            for data in fixtures, infinite_support in (false, true)
                support = infinite_support ? (one_dimensional ? (-Inf, Inf) :
                    ((-Inf, Inf), (-Inf, Inf))) : nothing
                # Empty quantile levels must not bypass numerical admission.
                for levels in (nothing, Float64[])
                    check_kde_error(() -> primitives(data, recipe; support, levels); direct=true)
                end
                source = samples(data)
                before = deepcopy(source)
                recipes = one_dimensional ?
                    (upper=Scatter2D, diagonal=typeof(recipe), lower=Scatter2D) :
                    (upper=typeof(recipe), diagonal=Mean1D, lower=typeof(recipe))
                n = size(data, 1)
                public_support = infinite_support ? fill((-Inf, Inf), n) : nothing
                check_kde_error(() -> Makie.convert_arguments(Makie.Plot{Makie.plot}, source;
                    recipes, vsel=collect(1:n), N_max=n, support=public_support))
                check_kde_error(() -> bat_makie_plot(source, recipes, collect(1:n), n;
                    support=public_support, display=false))
                @test source == before
            end
        end
    end

    @testset "Empty and inactive recipes retain their sentinels" begin
        for recipe in (KDE1D(), QuantileKDE1D(), KDE2D(), QuantileKDE2D())
            n = recipe isa Union{KDE1D,QuantileKDE1D} ? 1 : 2
            bad = fill(NaN, n, 2)
            config = n == 1 ? DIAG : PAIR
            @test all(isempty, values(primitives(bad, recipe; weights=zeros(2))))
            @test all(isempty, values(primitives(bad[:, 1:0], recipe)))
            for (rs, cs) in ((EXT.DeadRecipe(), EXT.LiveCell()),
                    (EXT.LiveRecipe(), EXT.DeadCell()))
                result = EXT.compute_plotting_primitives(view(bad, :, :), view(ones(2), :),
                    recipe, rs, cs, config)
                @test all(isempty, values(result))
            end
        end
        q1 = primitives(DATA[1:1, :], QuantileKDE1D(); levels=Float64[])
        q2 = primitives(DATA, QuantileKDE2D(); levels=Float64[])
        @test isempty(q1.polys) && length(q1.full_line) == 2048
        @test size(q2.color_grid) == (256, 256) && all(c -> iszero(c.alpha), q2.color_grid)
    end

    @testset "Native geometry and transpose preserve admitted estimates" begin
        source = samples()
        before = deepcopy(source)
        for recipes in (RECIPES, QUANTILES), ordinary in (false, true),
                support in (nothing, fill((-Inf, Inf), 2))
            fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [2, 1], 2;
                support, display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source; recipes, vsel=[2, 1], N_max=2, support)
                upper, lower = heat(cell(fig, 1, 2)), heat(cell(fig, 2, 1))
                expected = primitives(DATA, recipes.upper())
                @test upper.x == expected.xedges && upper.y == expected.yedges
                @test upper.x == lower.y && upper.y == lower.x
                @test isequal(upper.z, permutedims(lower.z))
                @test length(upper.x) == size(upper.z, 1) + 1 == 257
                @test length(upper.y) == size(upper.z, 2) + 1 == 257
                @test all(isfinite, upper.x) && all(>(0), diff(upper.x))
                @test all(isfinite, upper.y) && all(>(0), diff(upper.y))
                @test first(upper.x) < minimum(DATA[1, :]) < maximum(DATA[1, :]) < last(upper.x)
                @test first(upper.y) < minimum(DATA[2, :]) < maximum(DATA[2, :]) < last(upper.y)
                @test !isempty(descendants(cell(fig, 1, 1), Makie.Lines))
                if recipes === RECIPES
                    @test isequal(upper.z, Float32.(expected.density))
                    @test any(isfinite, upper.z) && any(isnan, upper.z)
                else
                    @test upper.z == expected.color_grid
                    @test any(c -> c.alpha > 0, upper.z)
                end
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        @test source == before
    end

    @testset "Fixed domains remain viewports" begin
        for recipes in (RECIPES, QUANTILES)
            graph, _ = EXT._setup_static_graph(samples(), recipes, [1, 2], 2, PAIR, DIAG)
            node = EXT.primitive_symbol(recipes.upper(), (2, 1))
            before = deepcopy(graph[node][])
            Makie.update!(graph; domain_lo=[-0.2, -0.3], domain_hi=[0.2, 0.3])
            @test isequal(graph[node][], before)
            fig = Figure()
            try
                EXT._bind_interactive_grid!(fig, EXT._init_gridlayout(graph, 2))
                rendered = heat(cell(fig, 1, 2))
                xlimits, ylimits = cell(fig, 1, 2).limits[]
                @test first(rendered.x) < first(xlimits) < last(xlimits) < last(rendered.x)
                @test first(rendered.y) < first(ylimits) < last(ylimits) < last(rendered.y)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
    end

    @testset "Numeric colors require safe native normalization; RGBA keeps its units" begin
        for density in (fill(1.0, 2, 2), [1.0 NaN; 2.0 3.0])
            @test_nowarn EXT._kde_check_density_colors(density)
        end
        for density in (fill(0.0, 2, 2), fill(Inf, 2, 2), fill(-1.0, 2, 2),
                fill(Float64(nextfloat(0.0f0)), 2, 2), fill(2.0^30, 2, 2),
                reshape(Float64[floatmin(Float32), floatmax(Float32)], 1, 2),
                reshape(Float64[floatmin(Float32), nextfloat(floatmin(Float32))], 1, 2))
            check_kde_error(() -> EXT._kde_check_density_colors(density);
                direct=true, message="Float32 color values")
        end

        data = 1e20 .* DATA
        numeric = primitives(data, KDE2D())
        visible = filter(isfinite, vec(numeric.density))
        @test !isempty(visible) && all(>(0), visible)
        # This is the previous guard's blind spot: positive finite Float32
        # values whose subnormal range cannot be normalized safely by Makie.
        @test all(d -> 0 < Float32(d) < Inf, visible)
        @test all(d -> Float32(d) < floatmin(Float32), visible)
        for transposed in (false, true)
            check_kde_error(() -> EXT.compose_plotspecs(numeric, KDE2D(), PAIR; transposed);
                direct=true, message="Float32 color values")
        end
        numeric_recipes = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)
        check_kde_error(() -> Makie.convert_arguments(Makie.Plot{Makie.plot}, samples(data);
            recipes=numeric_recipes, vsel=[1, 2], N_max=2); message="Float32 color values")
        check_kde_error(() -> bat_makie_plot(samples(data), numeric_recipes, [1, 2], 2;
            display=false); message="Float32 color values")
        rgba = primitives(data, QuantileKDE2D())
        @test rgba.xedges == numeric.xedges && rgba.yedges == numeric.yedges
        for ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(samples(data), QUANTILES, [2, 1], 2;
                display=false)
            try
                ordinary && Makie.plot(fig[1, 1], samples(data); recipes=QUANTILES, vsel=[2, 1], N_max=2)
                rendered = heat(cell(fig, 1, 2))
                @test rendered.x == rgba.xedges && rendered.y == rgba.yedges
                @test rendered.z == rgba.color_grid
                @test all(c -> all(isfinite, (c.r, c.g, c.b, c.alpha)), rendered.z)
                @test any(c -> c.alpha > 0, rendered.z)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
    end

    @testset "Bounded single cells share only the numeric-color admission" begin
        # Explicit bounded edges remain valid despite a single conceptual
        # center. It is the uniform Float32 color range that collapses here.
        data = [0.0 0.0; 10.0 10.0]
        support = ((-1e-7, 2e-7), (10.0 - 3e-7, 10.0 + 4e-7))
        numeric = primitives(data, KDE2D(); support)
        @test size(numeric.density) == (1, 1)
        @test isfinite(only(numeric.density)) && only(numeric.density) > 0
        check_kde_error(() -> EXT.compose_plotspecs(numeric, KDE2D(), PAIR);
            direct=true, message="Float32 color values")
        rgba = primitives(data, QuantileKDE2D(); support)
        @test rgba.xedges == numeric.xedges && rgba.yedges == numeric.yedges
        @test size(rgba.color_grid) == (1, 1) && only(rgba.color_grid).alpha > 0
        spec = only(EXT.compose_plotspecs(rgba, QuantileKDE2D(), PAIR))
        fig = Figure()
        try
            plot = heatmap!(Axis(fig[1, 1]), spec.args...; spec.kwargs...)
            @test (first(plot[1][]), last(plot[1][])) == support[1]
            @test (first(plot[2][]), last(plot[2][])) == support[2]
            @test only(plot[3][]).alpha > 0
            @test !isempty(Makie.colorbuffer(fig))
        finally
            empty!(fig)
        end
    end
end

mutable struct Capture <: AbstractDisplay
    figures::Vector{Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(capture::Capture, figure::Figure) = (push!(capture.figures, figure); nothing)

function with_capture(f)
    capture = Capture(Figure[])
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        return with_logger(() -> f(capture), NullLogger())
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end

function wait_done(task; seconds=30.0)
    @test task isa Task
    task isa Task || error("Missing owned task in unbounded KDE failure fixture")
    done = timedwait(() -> istaskdone(task), seconds; pollint=0.005) === :ok
    @test done
    done || error("Unbounded KDE failure fixture did not terminate its owned task")
    @test !istaskfailed(task)
    wait(task)
end

function chain_snapshot(chain)
    (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted),
        order=copy(chain.walker_order), info=chain.info, stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(context)) for context in chain.walker_genctxs])
end

function configuration(recipes)
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization(recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark, PAIR, DIAG; observer_timeout=60.0)
end

function algorithm(callback)
    TransformedMCMC(nchains=2, nsteps=24, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

@testset "Unbounded KDE live rejection preserves sampling and releases its owner" begin
    for mode in (:publication, :control)
        @testset "$mode" begin
            initial_recipes = mode === :publication ? RECIPES :
                (upper=QuantileKDE2D, diagonal=KDE1D, lower=QuantileKDE2D)
            vis = BATVisualizer(configuration(initial_recipes))
            plain_context = BATContext(rng=BAT.Philox4x((144, 72)))
            live_context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
            target = MvNormal(zeros(2), fill(mode === :publication ? 1.0 : 1e20, 2))
            plain_callbacks, live_callbacks = Any[], Any[]
            injected = Ref(false)
            listener = Ref{Union{Nothing,Task}}(nothing)
            releaser = Ref{Union{Nothing,Task}}(nothing)
            callback_steps = Ref{Vector{Int64}}(Int64[])
            plain_callback = function(tag, states)
                push!(plain_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
                return nothing
            end
            live_callback = function(tag, states)
                push!(live_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
                if !injected[]
                    graph = EXT._live_graph(vis)
                    @test !isempty(graph[:flat_weights][])
                    good = graph[EXT.primitive_symbol(initial_recipes.upper(), (2, 1))][]
                    @test !isempty(good.x)
                    listener[] = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
                    callback_steps[] = [state.chain_state.stepno for state in states]
                    injected[] = true
                    if mode === :publication
                        # Change only an owned copy of the display payload. The
                        # sampler's chains, outputs, weights and RNGs are intact.
                        payload = deepcopy(graph[:samples][])
                        for walkers in payload, walker in walkers
                            walker.v.data .*= 1e100
                        end
                        flat = graph[:flat_samples][]
                        check_kde_error(() -> EXT._weighted_kde2d(
                            1e100 .* flat.v.data[1, :], 1e100 .* flat.v.data[2, :], flat.weight);
                            direct=true)
                        @test !EXT._live_graph_update!(vis; publication=true, samples=payload)
                    else
                        # The same admitted density is valid as RGBA regions,
                        # but cannot become physical numeric Float32 colors.
                        @test any(c -> c.alpha > 0, good.color_grid)
                        flat = graph[:flat_samples][]
                        bad = primitives(flat.v.data, KDE2D(); weights=flat.weight)
                        check_kde_error(() -> EXT.compose_plotspecs(bad, KDE2D(), PAIR);
                            direct=true, message="Float32 color values")
                        @test !EXT._live_graph_update!(vis; upper_recipe=KDE2D)
                    end
                    releaser[] = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
                    @test !vis.content.lifecycle.accepting[]
                end
                return nothing
            end

            try
                Random.seed!(8921)
                plain = with_logger(() -> BAT.evalmeasure(target, algorithm(plain_callback), plain_context), NullLogger())
                default_after_plain = copy(Random.default_rng())
                Random.seed!(8921)
                observed = with_capture() do capture
                    result = BAT.evalmeasure(target, algorithm(live_callback), live_context)
                    @test length(capture.figures) == 1
                    result
                end
                @test injected[]
                @test length(plain_callbacks) == length(live_callbacks) == 1
                @test live_callbacks == plain_callbacks
                @test BAT.samplesof(observed) == BAT.samplesof(plain)
                live_chains = BAT.samplegenof(observed).chain_states
                plain_chains = BAT.samplegenof(plain).chain_states
                @test chain_snapshot.(live_chains) == chain_snapshot.(plain_chains)
                @test all(chain -> chain.stepno == 24, live_chains)
                @test callback_steps[] == fill(12, 2)
                @test BAT.get_rng(live_context) == BAT.get_rng(plain_context)
                @test rand(UInt64, 16) == rand(default_after_plain, UInt64, 16)

                # Wait on the actual owners; no status poll or rescue disposal
                # drives this release, and sampling's successful finish cannot
                # overwrite the observation failure with false completion.
                wait_done(listener[])
                wait_done(releaser[])
                @test isnothing(vis.content.graph[])
                status = BAT.visualizer_status(vis)
                @test status.outcome === :failed
                @test !status.data_complete && !status.presentation_valid
                @test status.quiesced && status.history_released && status.view_disposed
                @test !status.cleanup_pending && isnothing(status.cleanup_error)
                @test status.error.type == "ResolveException" && status.error.phase === :view
                @test status.error.message == "Exception details are not retained"
                @test isnothing(BAT.visualizer_figure(vis))
            finally
                BAT.request_visualizer_stop!(vis; reason=:disabled)
                BAT.finish_visualizer!(vis; success=false)
                BAT.dispose_visualizer!(vis)
                coordinator = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
                isnothing(coordinator) || wait_done(coordinator)
            end
        end
    end
end

end # module

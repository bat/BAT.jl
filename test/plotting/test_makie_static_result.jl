# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieStaticResult
using BAT, Test, Makie, CairoMakie

const EXT = Base.get_extension(BAT, :BATMakieExt)
const RECIPES = (upper=Hist2D, diagonal=Hist1D, lower=Hist2D)
const DISPLAY_FAILURE = ErrorException("intentional static display failure")

mutable struct Capture <: AbstractDisplay
    figures::Vector{Figure}
    fail::Bool
end
Base.displayable(::Capture, ::MIME"image/png") = true
function Base.display(c::Capture, fig::Figure)
    push!(c.figures, fig)
    c.fail && throw(DISPLAY_FAILURE)
    return nothing
end

figure_axes(fig) = filter(x -> x isa Axis, fig.content)
function plots_of_type(fig, T)
    result = Any[]
    function visit(plot)
        plot isa T && push!(result, plot)
        foreach(visit, plot.plots)
    end
    for ax in figure_axes(fig), plot in ax.scene.plots
        visit(plot)
    end
    return result
end

# Use the rendered bin edges, but independently count sample mass in each bin.
# This oracle does not call BAT histogram helpers or StatsBase.fit/normalize.
function check_histogram(fig, values, weights)
    @test length(figure_axes(fig)) == 1
    @test occursin("2", string(only(figure_axes(fig)).xlabel[]))
    bar = only(plots_of_type(fig, Makie.BarPlot))
    stairs = only(plots_of_type(fig, Makie.Stairs))
    points = bar[1][]
    edges = Float64[p[1] for p in stairs[1][]]
    @test length(edges) == length(points) + 1
    mass = [sum((weights[j] for j in eachindex(values)
        if edges[i] <= values[j] < edges[i+1]); init=0.0) for i in 1:length(edges)-1]
    @test sum(mass) == sum(weights)
    expected = mass ./ (sum(weights) .* diff(edges))
    @test Float64[p[2] for p in points] ≈ expected rtol=3e-5 atol=1e-6
    @test Float64[p[1] for p in points] ≈ (edges[1:end-1] .+ edges[2:end]) ./ 2 rtol=1e-6
    return copy(points)
end

@testset "Public static figure result" begin
    values = [0.137, 0.473, 0.819]
    weights = [1.0, 2.0, 5.0]
    samples = DensitySampleVector(v=[[100.0*i, values[i]] for i in eachindex(values)],
        logd=zeros(3), weight=weights)
    owned = Figure[]
    capture = Capture(Figure[], false)
    inline_before = Makie.ALWAYS_INLINE_PLOTS[]
    backend_before = Makie.current_backend()
    suppression_before = EXT._SUPPRESS_DISPLAY[]
    ambient = Theme(fontsize=37, Axis=(xticklabelsize=31,))
    Makie.inline!(true)
    pushdisplay(capture)
    try
        # First call in this fresh subprocess: the false branch must not warm
        # shaders, explicitly display, or change the display gate.
        @test isempty(EXT._WARMED_BACKENDS)
        deferred = with_theme(ambient) do
            fig = bat_makie_plot(samples, RECIPES, [2], 1; dark=true, display=false)
            push!(owned, fig)
            @test fig isa Figure
            @test Makie.theme(:fontsize)[] == 37
            @test Makie.theme(:Axis)[:xticklabelsize][] == 31
            fig
        end
        @test isempty(capture.figures)
        @test isempty(EXT._WARMED_BACKENDS)
        @test EXT._SUPPRESS_DISPLAY[] === suppression_before
        @test Makie.current_backend() === backend_before
        @test Makie.ALWAYS_INLINE_PLOTS[]
        @test only(figure_axes(deferred)).xticklabelsize[] == 10
        @test deferred.scene.backgroundcolor[] == Makie.to_color(:gray10)
        original_points = check_histogram(deferred, values, weights)

        # Rendering happens later, under a different ambient theme. The BAT
        # properties already installed on this figure must remain in force.
        mktempdir() do dir
            with_theme(Theme(fontsize=43, backgroundcolor=:pink, Axis=(xticklabelsize=29,))) do
                path = joinpath(dir, "weighted-samples.png")
                save(path, deferred)
                @test isfile(path) && filesize(path) > 0
                @test only(figure_axes(deferred)).xticklabelsize[] == 10
                @test deferred.scene.backgroundcolor[] == Makie.to_color(:gray10)
                @test Makie.theme(:fontsize)[] == 43
                @test check_histogram(deferred, values, weights) == original_points
            end
        end

        displayed = with_theme(ambient) do
            fig = bat_makie_plot(samples, RECIPES, [2], 1)
            push!(owned, fig)
            @test Makie.theme(:fontsize)[] == 37
            @test Makie.theme(:Axis)[:xticklabelsize][] == 31
            fig
        end
        @test backend_before in EXT._WARMED_BACKENDS
        @test displayed isa Figure
        @test length(capture.figures) == 1
        @test only(capture.figures) === displayed
        @test displayed !== deferred
        @test only(figure_axes(displayed)).xticklabelsize[] == 10
        check_histogram(displayed, values, weights)

        # Drive the actual returned figure's Apply/All controls. Selecting row
        # two changes the visible density, while the second figure is untouched.
        slider = only(filter(x -> x isa IntervalSlider, deferred.content))
        apply = only(filter(x -> x isa Button && x.label[] == "Apply", deferred.content))
        all_button = only(filter(x -> x isa Button && x.label[] == "All", deferred.content))
        slider.interval[] = (2, 2)
        @test check_histogram(deferred, values, weights) == original_points
        Makie.notify(apply.clicks)
        check_histogram(deferred, values[2:2], weights[2:2])
        @test check_histogram(displayed, values, weights) == original_points
        @test !isempty(Makie.colorbuffer(displayed))
        Makie.notify(all_button.clicks)
        @test check_histogram(deferred, values, weights) == original_points

        # Errors retain their identity and do not leak BAT's scoped theme.
        capture.fail = true
        with_theme(ambient) do
            caught = try
                bat_makie_plot(samples, RECIPES, [2], 1)
                nothing
            catch err
                err
            end
            @test caught === DISPLAY_FAILURE
            @test Makie.theme(:fontsize)[] == 37
            @test Makie.theme(:Axis)[:xticklabelsize][] == 31
        end
        @test length(capture.figures) == 2
        capture.fail = false

        # Ordinary Makie conversion and layout composition keep caller theming
        # and do not acquire the BAT convenience method's display keyword.
        @test :display ∉ Makie.used_attributes(Makie.Plot{Makie.plot}, samples)
        with_theme(ambient) do
            spec = Makie.convert_arguments(Makie.Plot{Makie.plot}, samples;
                recipes=RECIPES, vsel=[2], N_max=1)
            @test spec isa Makie.GridLayoutSpec
            fig = Figure()
            push!(owned, fig)
            Makie.plot(fig[1, 1], samples; recipes=RECIPES, vsel=[2], N_max=1)
            @test only(figure_axes(fig)).xticklabelsize[] == 31
            check_histogram(fig, values, weights)
            @test !isempty(Makie.colorbuffer(fig))
            @test Makie.theme(:fontsize)[] == 37
        end
        @test length(capture.figures) == 2
    finally
        popdisplay(capture)
        Makie.inline!(inline_before)
        for fig in unique(vcat(owned, capture.figures))
            empty!(fig)
        end
    end
end
end

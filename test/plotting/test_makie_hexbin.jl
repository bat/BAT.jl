# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieHexbin
using BAT, Test, Makie, CairoMakie, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)
const CONFIG = merge(BAT._default_makie_triagonal_config(), (nbins=(5, 5), colormap=:viridis))
const XY = [0.0 0.5 1.0; 0.0 0.8 1.0]

function specs(weights; xy=XY, threshold=nothing, rev=false, transposed=false)
    config = merge(CONFIG, (; threshold, rev))
    raw = EXT.compute_plotting_primitives(view(xy, :, :), view(weights, :), Hexbin2D(),
        EXT.LiveRecipe(), EXT.LiveCell(), config)
    return EXT.compose_plotspecs(raw, Hexbin2D(), config; transposed)
end

function make(weights; guarded=true, kwargs...)
    spec = only(specs(weights; kwargs...))
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot = if guarded
        EXT.batcheckedhexbin!(ax, spec.args...; spec.kwargs...)
    else
        hexbin!(ax, spec.args...; spec.kwargs...)
    end
    return fig, guarded ? plot.plots[1] : plot, guarded ? plot.plots[2] : only(plot.plots)
end

colors(scatter) = Makie.to_color(Makie.get_colormapping(scatter))
count_ordered_colors(wrapper) = colors(wrapper.plots[2])[sortperm(wrapper.plots[1].count_hex[])]
function rejected(f)
    err = try f(); nothing catch e; e end
    @test err !== nothing
    @test err !== nothing && occursin("Hexbin2D displayed counts", sprint(showerror, err))
end

# Independent wider-precision oracle, including native uniform-range semantics.
function check_coordinates(counts)
    @test EXT._check_hexbin_count_colors(counts) === counts
    isempty(counts) && return
    lo, hi = BigFloat.(extrema(counts))
    lo == hi && (lo = big"0"; iszero(hi) && (hi = big"1"))
    reference = (BigFloat.(counts) .- lo) ./ (hi - lo)
    flo, fhi = Float32.((lo, hi))
    cpu = (Float64.(Float32.(counts)) .- Float64(flo)) ./ (Float64(fhi) - Float64(flo))
    numerators = Float32.(counts) .- flo
    gpu = numerators ./ (fhi - flo)
    reciprocal = numerators .* inv(fhi - flo)
    flushed = map(x -> x < floatmin(Float32) ? 0f0 : x, numerators) ./ (fhi - flo)
    for actual in (cpu, gpu, reciprocal, flushed)
        @test all(abs.(BigFloat.(actual) .- reference) .<= BigFloat(8eps(Float32)))
    end
end

function descendants(root, T)
    result = Any[]
    function visit(p)
        p isa T && push!(result, p)
        foreach(visit, p.plots)
    end
    if root isa Figure
        for axis in filter(x -> x isa Axis, root.content)
            foreach(visit, axis.scene.plots)
        end
    else
        foreach(visit, root.scene.plots)
    end
    return result
end

@testset "Hexbin absolute-count color admission" begin
    @testset "Native counts, orientation, palette and colorbar" begin
        for scale in (1e-30, 1.0, 1e30), rev in (false, true), transposed in (false, true)
            weights = scale .* [1.0, 2.0, 3.0]
            before = copy(weights)
            fig, native, visible = make(weights; rev, transposed)
            original, reference, scatter = make(weights; rev, transposed, guarded=false)
            try
                @test native.count_hex[] == reference.count_hex[]
                @test sort(native.count_hex[]) == weights
                @test native.points[] == reference.points[]
                @test native.threshold[] == minimum(weights)
                @test isequal(visible.scaled_color[], scatter.scaled_color[])
                @test isequal(visible.scaled_colorrange[], scatter.scaled_colorrange[])
                @test colors(visible) == colors(scatter)
                @test weights == before
                check_coordinates(native.count_hex[])
                Colorbar(fig[1, 2], visible)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig); empty!(original)
            end
        end
    end
    @testset "Adversarial ranges and aggregation boundary" begin
        for weights in (1e100 .* [1, 2, 3], 1e-100 .* [1, 2, 3], 1e10 .+ [0, 512, 1025],
                1.0 .+ [0, eps(), 2eps()], Float64(floatmin(Float32)) .* [1, 1.5, 2])
            rejected() do
                fig, native, visible = make(weights)
                try
                    colors(visible)
                finally
                    empty!(fig)
                end
            end
        end
        for weights in (ones(3), zeros(3), fill(1e30, 3), [1e10, 1e10 + 1024, 1e10 + 1024],
                Float64[floatmin(Float32), nextfloat(floatmin(Float32)), 2floatmin(Float32)])
            fig, native, visible = make(weights; threshold=all(iszero, weights) ? 0.0 : nothing)
            try
                check_coordinates(native.count_hex[])
                @test all(isfinite, visible.scaled_colorrange[])
                Colorbar(fig[1, 2], visible)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        for weights in (1e100 .* [1, 2, 3], 1e-100 .* [1, 2, 3])
            fig, native, visible = make(weights; threshold=maximum(weights) * 2)
            try
                @test isempty(native.count_hex[])
                @test isempty(visible.scaled_color[])
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
            # The hidden renderer must itself stay safe even for rejected colors.
            fig, native, visible = make(weights)
            try
                hidden = only(native.plots)
                @test !hidden.visible[] && !hidden.inspectable[]
                @test all(isfinite, hidden.scaled_color[])
                @test hidden.scaled_colorrange[] == Makie.Vec2f(0, 1)
                @test all(c -> all(isfinite, (c.r, c.g, c.b, c.alpha)), colors(hidden))
            finally
                empty!(fig)
            end
        end
        tiny = Float64(floatmin(Float32)) / 2
        fig, native, visible = make(repeat(tiny .* [1, 2, 3]; inner=4); xy=repeat(XY; inner=(1, 4)))
        @test sort(native.count_hex[]) == 4tiny .* [1, 2, 3]
        check_coordinates(native.count_hex[])
        empty!(fig)
    end
    @testset "Independent numerical families" begin
        # Loss below the coordinate tolerance is intentional; substantial
        # contrast loss is rejected, even when every count remains normal.
        check_coordinates([1e-30, 1e-30 + 1e-39, 2e-30])
        check_coordinates([1.0, 1.0 + 2.0^-17 + 3eps(Float32) * 2.0^-16, 1.0 + 2.0^-16])
        @test_throws ArgumentError EXT._check_hexbin_count_colors(
            [1.0, 1.0 + 2.0^-17 + 5eps(Float32) * 2.0^-16, 1.0 + 2.0^-16])
        @test_throws ArgumentError EXT._check_hexbin_count_colors([1e8, 1e8 + 1, 1e8 + 8])
        rng = Xoshiro(823)
        for _ in 1:1000
            counts = 10.0^rand(rng, -36:36) .* rand(rng, 0:1000, rand(rng, 1:16))
            before = copy(counts)
            actual = try EXT._check_hexbin_count_colors(counts) catch e; e end
            if actual isa Exception
                @test actual isa ArgumentError
            else
                check_coordinates(counts)
            end
            @test counts == before
        end
        # All of these exact binary inputs are representable: rejecting every
        # generated input must not be sufficient to pass this campaign.
        for _ in 1:1000
            counts = ldexp.(Float64.(rand(rng, UInt16, rand(rng, 1:16))), rand(rng, -100:100))
            before = copy(counts)
            check_coordinates(counts)
            @test counts == before
        end
        for counts in (Float64[], [0.0], [1.0], [Float64(floatmin(Float32))])
            check_coordinates(counts)
        end
        for counts in ([NaN], [Inf], [-1.0], [nextfloat(0.0)],
                [Float64(floatmax(Float32))], 1e38 .* [1, 2, 3])
            @test_throws ArgumentError EXT._check_hexbin_count_colors(counts)
        end
        limit = Float64(inv(2floatmin(Float32)))
        check_coordinates([limit])
        @test_throws ArgumentError EXT._check_hexbin_count_colors([2limit])
    end
    @testset "Native geometry, transforms, sparse/full grids" begin
        rng = Xoshiro(1129)
        xy = rand(rng, 2, 200) .+ 1
        weights = Float64.(rand(rng, 0:20, 200))
        before = copy(weights)
        for threshold in (0.0, 1.0, 7.0, 10000.0), bins in ((4, 9), (9, 4)), scale in (identity, log10)
            fig = Figure()
            ax = Axis(fig[1, 1]; xscale=scale, yscale=scale)
            p = EXT.batcheckedhexbin!(ax, xy[1, :], xy[2, :]; weights, bins, threshold, colormap=:viridis)
            reference = hexbin!(ax, xy[1, :], xy[2, :]; weights, bins, threshold, colormap=:viridis)
            native, visible = p.plots
            try
                @test native.count_hex[] == reference.count_hex[]
                @test native.points[] == reference.points[]
                @test visible.markersize[] == only(reference.plots).markersize[]
                @test isequal(Makie.data_limits(p), Makie.data_limits(reference))
                for translated in (false, true)
                    if translated
                        Makie.translate!(p, 3.0, 4.0, 0.0)
                        Makie.translate!(reference, 3.0, 4.0, 0.0)
                    end
                    bp, bn = Makie.boundingbox(p), Makie.boundingbox(reference)
                    @test isequal(Tuple(bp.origin), Tuple(bn.origin)) && isequal(Tuple(bp.widths), Tuple(bn.widths))
                    @test visible.model[] == only(reference.plots).model[]
                end
                @test colors(visible) == colors(only(reference.plots))
                @test !native.visible[] && !only(native.plots).visible[] && visible.visible[]
                @test visible.markerspace[] === :data
                Colorbar(fig[1, 2], p)
                @test !isempty(Makie.colorbuffer(fig))
                p.visible[] = false
                @test !visible.visible[] && !native.visible[]
            finally
                empty!(fig)
            end
        end
        @test weights == before
        # A rectangular bin configuration must swap with the plotted axes.
        config = merge(CONFIG, (nbins=(4, 9),))
        raw = EXT.compute_plotting_primitives(view(XY, :, :), view([1.0, 2.0, 3.0], :),
            Hexbin2D(), EXT.LiveRecipe(), EXT.LiveCell(), config)
        spec = only(EXT.compose_plotspecs(raw, Hexbin2D(), config; transposed=true))
        @test spec.kwargs[:bins] == (9, 4)
        @test spec.args == Any[raw.y, raw.x]
    end
    @testset "Real SpecApi updates and fail-stop disposal" begin
        fig = Figure()
        ax = Axis(fig[1, 1])
        obs = Observable(specs([1.0, 2.0, 3.0]))
        Makie.plot!(ax, obs)
        wrapper = only(descendants(ax, EXT.BATCheckedHexbin))
        Colorbar(fig[1, 2], wrapper)
        reference = count_ordered_colors(wrapper)
        for scale in (1e-30, 1e30, 1.0), transposed in (true, false)
            weights = scale .* [1, 2, 3]
            obs[] = specs(weights; transposed)
            p = only(descendants(ax, EXT.BATCheckedHexbin))
            @test sort(p.plots[1].count_hex[]) == weights
            @test all(zip(count_ordered_colors(p), reference)) do pair
                a, b = pair
                max(abs(a.r-b.r), abs(a.g-b.g), abs(a.b-b.b), abs(a.alpha-b.alpha)) <= 2e-7
            end
            @test !isempty(Makie.colorbuffer(fig))
        end
        obs[] = Makie.PlotSpec[]
        @test isempty(descendants(ax, EXT.BATCheckedHexbin))
        obs[] = specs([1.0, 2.0, 3.0])
        @test count_ordered_colors(only(descendants(ax, EXT.BATCheckedHexbin))) == reference
        rejected() do
            obs[] = specs(1e100 .* [1, 2, 3])
            Makie.colorbuffer(fig)
        end
        empty!(fig)
        @test isempty(fig.content)
        @test isempty(Makie.Observables.listeners(obs))
        fresh = Figure()
        freshax = Axis(fresh[1, 1])
        Makie.plot!(freshax, specs([1.0, 2.0, 3.0]))
        @test count_ordered_colors(only(descendants(freshax, EXT.BATCheckedHexbin))) == reference
        @test !isempty(Makie.colorbuffer(fresh))
        empty!(fresh)
    end
    @testset "Public figures, Apply and color themes" begin
        recipes = (upper=Hexbin2D, lower=Hexbin2D, diagonal=Hist1D)
        source = DensitySampleVector(v=[collect(v) for v in eachcol(XY)], logd=zeros(3), weight=[1.0, 2.0, 3.0])
        before = deepcopy(source)
        for ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [1, 2], 2; display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source; recipes, vsel=[1, 2], N_max=2)
                hexes = descendants(fig, EXT.BATCheckedHexbin)
                @test length(hexes) == 2
                @test all(h -> sort(h.plots[1].count_hex[]) == [1, 2, 3], hexes)
                @test all(h -> length(unique(colors(h.plots[2]))) == 3, hexes)
                @test !isempty(Makie.colorbuffer(fig))
                if !ordinary
                    slider = only(filter(x -> x isa IntervalSlider, fig.content))
                    button = only(filter(x -> x isa Button && x.label[] == "Apply", fig.content))
                    slider.interval[] = (2, 3)
                    Makie.notify(button.clicks)
                    @test all(h -> sort(h.plots[1].count_hex[]) == [2, 3], descendants(fig, EXT.BATCheckedHexbin))
                    @test !isempty(Makie.colorbuffer(fig))
                end
            finally
                empty!(fig)
            end
        end
        @test source == before
        Makie.with_theme(Hexbin=(colorscale=log10, colorrange=(1e100, 2e100)),
                BATCheckedHexbin=(colorscale=log10, colorrange=(1e100, 2e100))) do
            spec = only(specs([1.0, 2.0, 3.0]))
            @test spec.kwargs[:colorrange] === Makie.automatic
            @test spec.kwargs[:colorscale] === identity
            fig = Figure()
            ax = Axis(fig[1, 1])
            Makie.plot!(ax, [spec])
            p = only(descendants(ax, EXT.BATCheckedHexbin))
            @test sort(p.plots[1].count_hex[]) == [1, 2, 3]
            @test p.plots[2].scaled_color[] == Float32.(p.plots[1].count_hex[])
            @test p.plots[2].scaled_colorrange[] == Makie.Vec2f(1, 3)
            Colorbar(fig[1, 2], p)
            @test !isempty(Makie.colorbuffer(fig))
            empty!(fig)
        end
        Makie.with_theme(Hexbin=(strokewidth=3.0, strokecolor=:red, alpha=0.4)) do
            for explicit in (false, true)
                fig = Figure()
                ax = Axis(fig[1, 1])
                kwargs = explicit ? (strokewidth=1.0, strokecolor=:blue, alpha=0.8) : NamedTuple()
                spec = only(specs([1.0, 2.0, 3.0]))
                p = EXT.batcheckedhexbin!(ax, spec.args...; spec.kwargs..., kwargs...)
                native = hexbin!(ax, spec.args...; spec.kwargs..., kwargs...)
                try
                    for key in (:strokewidth, :strokecolor, :alpha)
                        @test p[key][] == native[key][]
                        @test p.plots[2][key][] == only(native.plots)[key][]
                    end
                    @test !isempty(Makie.colorbuffer(fig))
                finally
                    empty!(fig)
                end
            end
        end
    end
    @testset "Validator allocation and RNG" begin
        rng = copy(Random.default_rng())
        counts = repeat([1.0, 2.0, 3.0], 10000)
        @test EXT._check_hexbin_count_colors(counts) === counts
        bytes = @allocated EXT._check_hexbin_count_colors(counts)
        @test bytes < 4096
        @test rand(copy(Random.default_rng()), UInt64, 8) == rand(rng, UInt64, 8)
        @info "Hexbin validator warmed allocations" cells=length(counts) bytes
    end
end
end # module

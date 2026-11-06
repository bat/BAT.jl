# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with an external deadline and a working native GL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
const EXT = Base.get_extension(BAT, :BATMakieExt)
const CONFIG = merge(BAT._default_makie_triagonal_config(), (colormap=:viridis,))
GLMakie.activate!()
Makie.inline!(false)

@testset "Native Hist2D color admission" begin
    other = Figure(size=(240,160))
    lines!(Axis(other[1,1]), [0.0, 1.0], [1.0, 0.0])
    other_screen = nothing
    try
        other_screen = GLMakie.Screen(; visible=false, start_renderloop=false)
        display(other_screen, other; start_renderloop=false)
        other_pixels = copy(Makie.colorbuffer(other_screen))
        for values in ([1.0, 2.0], [2.0, 2.0], fill(Float64(nextfloat(0f0)), 2),
                [6666.666666, 6666.666667], [Float64(2f0^23)], [Float64(2f0^23 + 1f0)],
                [1.0, Float64(2f0^125)])
            raw = (xedges=Float64.(0:length(values)), yedges=[0.0, 1.0],
                weights=reshape(copy(values), :, 1))
            before = deepcopy(raw)
            spec = only(EXT.compose_plotspecs(raw, Hist2D(), CONFIG))
            fig = Figure(size=(300,240))
            screen = nothing
            try
                axis = Axis(fig[1,1])
                hidedecorations!(axis)
                hidespines!(axis)
                plot = heatmap!(axis, spec.args...; spec.kwargs...)
                lo, hi = plot.scaled_colorrange[]
                @test isfinite(lo) && isfinite(hi) && lo < hi
                span = hi - lo
                @test isfinite(span) && span >= floatmin(Float32)
                @test isfinite(inv(span)) && inv(span) >= 2floatmin(Float32)
                screen = GLMakie.Screen(; visible=false, start_renderloop=false)
                display(screen, fig; start_renderloop=false)
                pixels = Makie.colorbuffer(screen)
                # No axes or colorbar can supply these chromatic pixels.
                @test count(pixels) do color
                    rgb = Makie.RGBf(color)
                    max(rgb.r, rgb.g, rgb.b) - min(rgb.r, rgb.g, rgb.b) > 0.1
                end > 100
                @test isequal(raw, before)
            finally
                isnothing(screen) || close(screen)
                empty!(fig)
            end
        end
        for regions in (false, true), scale in (1.0, 1e-5)
            data = scale .* [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
            sample = DensitySampleVector([collect(v) for v in eachcol(data)], zeros(4))
            before = deepcopy(sample)
            recipes = (upper=regions ? QuantileHist2D : Hist2D,
                diagonal=Hist1D, lower=QuantileHist2D)
            fig = nothing
            screen = nothing
            try
                if scale == 1e-5 && !regions
                    err = try
                        bat_makie_plot(sample, recipes, [1,2], 2; display=false)
                        nothing
                    catch e
                        e
                    end
                    @test err !== nothing && occursin("Hist2D visible heights", sprint(showerror, err))
                else
                    fig = bat_makie_plot(sample, recipes, [1,2], 2; display=false)
                    screen = GLMakie.Screen(; visible=false, start_renderloop=false)
                    display(screen, fig; start_renderloop=false)
                    @test !isempty(Makie.colorbuffer(screen))
                end
                @test sample == before
                @test isopen(other_screen) && other_screen.scene === other.scene
                @test Makie.colorbuffer(other_screen) == other_pixels
            finally
                isnothing(screen) || close(screen)
                isnothing(fig) || empty!(fig)
            end
        end
    finally
        isnothing(other_screen) || close(other_screen)
        empty!(other)
    end
end

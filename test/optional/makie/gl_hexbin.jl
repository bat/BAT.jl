# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run in a separate process with an external deadline and a native GL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
const EXT = Base.get_extension(BAT, :BATMakieExt)
GLMakie.activate!()
Makie.inline!(false)

function specs(weights)
    config = merge(BAT._default_makie_triagonal_config(), (nbins=(5, 5),))
    xy = [0.0 0.5 1.0; 0.0 0.8 1.0]
    raw = EXT.compute_plotting_primitives(view(xy, :, :), view(weights, :),
        Hexbin2D(), EXT.LiveRecipe(), EXT.LiveCell(), config)
    EXT.compose_plotspecs(raw, Hexbin2D(), config)
end

@testset "Native GL checked hexbin counts and scoped failure disposal" begin
    other = Figure()
    lines!(Axis(other[1, 1]), [0, 1], [1, 0])
    independent = GLMakie.Screen(; visible=false, start_renderloop=false)
    try
        display(independent, other; start_renderloop=false)
        for specapi in (false, true), bad in (1e-100 .* [1, 2, 3], 1e100 .* [1, 2, 3],
                1e10 .+ [0, 512, 1025], 1e38 .* [1, 2, 3], fill(Float64(floatmax(Float32)), 3))
            fig = Figure()
            ax = Axis(fig[1, 1])
            obs = Observable(specs([1.0, 2.0, 3.0]))
            p = if specapi
                only(Makie.plot!(ax, obs).plots)
            else
                spec = only(obs[])
                EXT.batcheckedhexbin!(ax, spec.args...; spec.kwargs...)
            end
            Colorbar(fig[1, 2], p)
            screen = GLMakie.Screen(; visible=false, start_renderloop=false)
            try
                display(screen, fig; start_renderloop=false)
                for scale in (1e-30, 1e30, 1.0)
                    weights = scale .* [1, 2, 3]
                    if specapi
                        obs[] = specs(weights)
                    else
                        Makie.update!(p; weights, threshold=scale)
                    end
                    @test sort(p.plots[1].count_hex[]) == weights
                    @test length(unique(Makie.to_color(Makie.get_colormapping(p.plots[2])))) == 3
                    @test !isempty(Makie.colorbuffer(screen))
                end
                err = try
                    if specapi
                        obs[] = specs(bad)
                    else
                        Makie.update!(p; weights=bad, threshold=minimum(bad))
                    end
                    Makie.colorbuffer(screen)
                    nothing
                catch e
                    e
                end
                @test err !== nothing && occursin("Hexbin2D displayed counts", sprint(showerror, err))
            finally
                close(screen)
                empty!(fig)
            end
            # Each rejected figure is disposed, with a fresh one next iteration.
            @test isempty(fig.content)
            @test isempty(Makie.Observables.listeners(obs))
            @test isopen(independent)
            @test !isempty(Makie.colorbuffer(independent))
        end
    finally
        close(independent)
        empty!(other)
    end
end

@testset "Actual GL pixels retain admitted count colors" begin
    scene = Scene(; size=(600, 400), camera=campixel!, backgroundcolor=:white)
    p = EXT.batcheckedhexbin!(scene, [100.0, 300.0, 500.0], [100.0, 250.0, 100.0];
        weights=[1.0, 2.0, 3.0], bins=(5, 5), threshold=1.0, colormap=:viridis)
    screen = GLMakie.Screen(; visible=false, start_renderloop=false)
    try
        display(screen, scene; start_renderloop=false)
        for weights in ([1.0, 2.0, 3.0], ones(3))
            Makie.update!(p; weights, threshold=minimum(weights))
            reference = copy(Makie.colorbuffer(screen))
            @test any(c -> c.r < 0.9 || c.g < 0.9 || c.b < 0.9, reference)
            # The last scale puts the reciprocal at its admitted lower bound.
            span = all(==(1.0), weights) ? 1.0 : 2.0
            scales = (1e-30, 1e30, 2Float64(floatmin(Float32)),
                Float64(inv(2floatmin(Float32))) / span)
            for scale in scales
                counts = scale .* weights
                Makie.update!(p; weights=counts, threshold=minimum(counts))
                pixels = Makie.colorbuffer(screen)
                error = maximum(map(pixels, reference) do a, b
                    max(abs(Float64(a.r) - Float64(b.r)), abs(Float64(a.g) - Float64(b.g)),
                        abs(Float64(a.b) - Float64(b.b)))
                end)
                @test error <= 2/255
            end
        end
    finally
        close(screen)
        empty!(scene)
    end
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Isolated optional-runner child: observe the real warmup Figure and inject
# faults at rendering/cleanup boundaries without changing production methods.
module TestMakieWarmup
using BAT, Makie, CairoMakie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
const FIGURES = Makie.Figure[]
const BACKENDS = Module[]
const FAIL_RENDER = Ref(false)
const FAIL_CLOSE = Ref(false)
const PRIMARY = ErrorException("intentional warmup render failure")
const CLOSE_ATTEMPTS = Ref(0)

# A distinct identity for the bookkeeping test only. Render through real Cairo;
# native cross-renderer coverage lives in optional/makie/gl_warmup.jl.
module AlternateRenderer end
function Makie.colorbuffer(fig::Makie.Figure,
        format::Makie.ImageStorageFormat=Makie.JuliaNative; backend, kwargs...)
    push!(FIGURES, fig)
    push!(BACKENDS, backend)
    FAIL_RENDER[] && throw(PRIMARY)
    actual_backend = backend === AlternateRenderer ? CairoMakie : backend
    return invoke(Makie.colorbuffer, Tuple{Makie.FigureLike,Makie.ImageStorageFormat},
        fig, format; backend=actual_backend, kwargs...)
end
function EXT._close_owned_live_screens!(fig::Makie.Figure)
    CLOSE_ATTEMPTS[] += 1
    invoke(EXT._close_owned_live_screens!, Tuple{Any}, fig)
    FAIL_CLOSE[] && error("intentional warmup screen cleanup failure")
    return nothing
end

@testset "Renderer warmup completion, retry and ownership" begin
    original_backend = Makie.current_backend()
    @test isempty(EXT._WARMED_BACKENDS)
    try
        Makie.set_active_backend!(missing)
        @test EXT.warmup_makie_shaders() === nothing
        @test isempty(FIGURES)
        @test isempty(EXT._WARMED_BACKENDS)

        CairoMakie.activate!()
        FAIL_RENDER[] = true
        FAIL_CLOSE[] = true
        caught = try
            EXT.warmup_makie_shaders()
            nothing
        catch err
            err
        end
        @test caught === PRIMARY
        @test CLOSE_ATTEMPTS[] == 1
        @test isempty(only(FIGURES).content)
        @test isempty(only(FIGURES).scene.plots)
        @test isempty(only(FIGURES).scene.children)
        @test isempty(EXT._WARMED_BACKENDS)

        FAIL_RENDER[] = false
        FAIL_CLOSE[] = false
        @test EXT.warmup_makie_shaders() === nothing
        @test EXT._WARMED_BACKENDS == Set([CairoMakie])
        @test length(FIGURES) == 2
        @test EXT.warmup_makie_shaders() === nothing
        @test length(FIGURES) == 2

        Makie.set_active_backend!(AlternateRenderer)
        @test EXT.warmup_makie_shaders() === nothing
        @test EXT._WARMED_BACKENDS == Set([CairoMakie, AlternateRenderer])
        @test BACKENDS == [CairoMakie, CairoMakie, AlternateRenderer]
        @test Makie.current_backend() === AlternateRenderer
        CairoMakie.activate!()
        @test EXT.warmup_makie_shaders() === nothing
        @test length(FIGURES) == 3
        @test CLOSE_ATTEMPTS[] == 3
        @test all(fig -> isempty(fig.content) && isempty(fig.scene.plots) &&
            isempty(fig.scene.children), FIGURES)

        # Module initialization resets only bookkeeping; it does not render or
        # activate a backend. A separate fresh-load probe checks the real loader.
        EXT.__init__()
        @test isempty(EXT._WARMED_BACKENDS)
        @test length(FIGURES) == 3
        @test Makie.current_backend() === CairoMakie
    finally
        Makie.set_active_backend!(original_backend)
        foreach(empty!, FIGURES)
    end
end
end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native supplement to the headless warmup case. Run alone with an external
# timeout, this checkout, CairoMakie, GLMakie and a working OpenGL display.
using BAT, Makie, CairoMakie, GLMakie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
realpath(dirname(dirname(pathof(BAT)))) == realpath(joinpath(@__DIR__, "../../..")) ||
    error("Native warmup test loaded a different BAT checkout")
const WARMUP_FIGURES = Figure[]
const WARMUP_BACKENDS = Module[]
const WARMUP_SCREENS = Any[]

function EXT._render_warmup_figure(fig::Figure, backend::Module)
    push!(WARMUP_FIGURES, fig)
    push!(WARMUP_BACKENDS, backend)
    return invoke(EXT._render_warmup_figure, Tuple{Any,Any}, fig, backend)
end

function EXT._with_gl_screen(f::Function, fig::Figure, backend::Module; temporary=false, on_failure=nothing)
    wrapped = function(screen)
        push!(WARMUP_SCREENS, screen)
        result = f(screen)
        @test temporary
        @test !isempty(result)
        @test screen.rendertask === nothing
        return result
    end
    return invoke(EXT._with_gl_screen, Tuple{Any,Any,Any}, wrapped, fig, backend;
        temporary=temporary, on_failure=on_failure)
end

@testset "Cairo/GL warmup round trip preserves unrelated screen" begin
    @test isempty(EXT._WARMED_BACKENDS)
    GLMakie.activate!(visible=false)
    Makie.inline!(false)
    other = Figure(size=(240, 160))
    lines!(Axis(other[1, 1]), [0, 1], [1, 0])
    existing = filter(s -> s isa GLMakie.Screen, other.scene.current_screens)
    screen = isempty(existing) ? GLMakie.Screen(; visible=false) : only(existing)
    try
        display(screen, other)
        CairoMakie.activate!()
        inline_before = Makie.ALWAYS_INLINE_PLOTS[]
        @test EXT.warmup_makie_shaders() === nothing
        @test Makie.ALWAYS_INLINE_PLOTS[] === inline_before
        GLMakie.activate!(visible=false)
        inline_before = Makie.ALWAYS_INLINE_PLOTS[]
        @test EXT.warmup_makie_shaders() === nothing
        @test Makie.ALWAYS_INLINE_PLOTS[] === inline_before
        CairoMakie.activate!()
        inline_before = Makie.ALWAYS_INLINE_PLOTS[]
        @test EXT.warmup_makie_shaders() === nothing
        @test Makie.ALWAYS_INLINE_PLOTS[] === inline_before
        @test WARMUP_BACKENDS == [CairoMakie, GLMakie]
        @test EXT._WARMED_BACKENDS == Set([CairoMakie, GLMakie])
        @test length(WARMUP_FIGURES) == 2
        @test all(fig -> isempty(fig.content) && isempty(fig.scene.children) &&
            isempty(fig.scene.plots), WARMUP_FIGURES)
        @test !isempty(WARMUP_SCREENS)
        @test all(s -> !isopen(s), WARMUP_SCREENS)
        @test isopen(screen)
        @test screen.scene === other.scene
        @test !isempty(other.content)
        @test !isempty(Makie.colorbuffer(screen))
        @test Makie.current_backend() === CairoMakie
    finally
        close(screen)
        empty!(other)
    end
end

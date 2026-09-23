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

function Makie.colorbuffer(fig::Figure,
        format::Makie.ImageStorageFormat=Makie.JuliaNative; backend, kwargs...)
    push!(WARMUP_FIGURES, fig)
    push!(WARMUP_BACKENDS, backend)
    result = invoke(Makie.colorbuffer, Tuple{Makie.FigureLike,Makie.ImageStorageFormat},
        fig, format; backend=backend, kwargs...)
    @test !isempty(result)
    append!(WARMUP_SCREENS, filter(s -> s isa GLMakie.Screen, fig.scene.current_screens))
    return result
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

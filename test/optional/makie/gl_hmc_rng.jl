# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native counterpart of the maintained Cairo HMC/MultiTrafo observation gate.
# Run in its own externally bounded process with GLMakie and a native display.
using BAT, GLMakie, Makie, Test
realpath(dirname(dirname(pathof(BAT)))) == realpath(joinpath(@__DIR__, "../../..")) ||
    error("Native HMC RNG test loaded a different BAT checkout")
const RENDER_CALLS = Ref(0)
renderloop = screen -> begin
    # This callback needs Makie's global theme lock. BAT's construction scope
    # must have released it before any possible renderer join.
    with_theme() do
        RENDER_CALLS[] += 1
    end
    GLMakie.renderloop(screen)
end
GLMakie.activate!(renderloop=renderloop, visible=false)
Makie.inline!(false)
@test isempty(Base.get_extension(BAT, :BATMakieExt)._WARMED_BACKENDS)
ENV["BAT_MAKIE_HMC_NATIVE"] = "true"
include("../../plotting/test_makie_hmc_rng.jl")
@test RENDER_CALLS[] == 2

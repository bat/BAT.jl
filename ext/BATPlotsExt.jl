# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATPlotsExt

@static if isdefined(Base, :get_extension)
    using Plots
else
    using ..Plots
end

using BAT

BAT._Plots_backend(args...; kwargs...) = Plots.backend(args...; kwargs...)
BAT._Plots_cgrad(args...; kwargs...) = Plots.cgrad(args...; kwargs...)
BAT._Plots_grid(args...; kwargs...) = Plots.grid(args...; kwargs...)
BAT._Plots_Shape(args...; kwargs...) = Plots.Shape(args...; kwargs...)
BAT._Plots_Surface(args...; kwargs...) = Plots.Surface(args...; kwargs...)

BAT._Plots_backend_is_pyplot() = Plots.backend() isa Plots.PyPlotBackend

end # module BATPlotsExt

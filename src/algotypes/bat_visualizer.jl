# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type BATVisBackend end

struct BATNoVisBackend <: BATVisBackend end

struct BATVisualizer{B<:BATVisBackend}
        backend::B
        content::Any
end
export BATVisualizer

function BATVisualizer()
        return BATVisualizer(BATNoVisBackend(), nothing)
end

function init_visualizer!(vis::BATVisualizer; kwargs...) end

function register_state_for_vis!() end

function update_visualizer!(vis::BATVisualizer; kwargs...)
        update_visualizer_impl!(vis; kwargs...)
end

function update_visualizer_impl!(vis::BATVisualizer; kwargs...) end

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type BATVisBackend end

struct BATNoVisBackend <: BATVisBackend end

struct BATVisualizer
        vislock::ReentrantLock
        backend::BATVisBackend
        content::Any
end

function BATVisualizer()
        vislock = ReentrantLock()
        return BATVisualizer(vislock, BATNoVisBackend(), nothing)
end

function init_visualizer(vis::BATVisualizer; kwargs) end

function update_visualizer(vis::BATVisualizer; kwargs) end

function update_visualizer_impl(vis::BATVisualizer; kwargs) end

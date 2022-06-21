# This file is a part of BAT.jl, licensed under the MIT License (MIT).

Base.@deprecate MaxDensityNelderMead(args...; kwargs...) NelderMeadOpt(args...; kwargs...)
export MaxDensityNelderMead

Base.@deprecate MaxDensityLBFGS(args...; kwargs...) LBFGSOpt(args...; kwargs...)
export MaxDensityLBFGS

Base.@deprecate MaxDensitySampleSearch(args...; kwargs...) MaxDensitySearch(args...; kwargs...)
export MaxDensitySampleSearch

Base.@deprecate NoDensityTransform(args...; kwargs...) DoNotTransform(args...; kwargs...)
export NoDensityTransform

Base.@deprecate PosteriorDensity(args...) PosteriorMeasure(args...)
export PosteriorDensity

Base.@deprecate SampledDensity(args...; kwargs...) SampledMeasure(args...; kwargs...)
export SampledMeasure

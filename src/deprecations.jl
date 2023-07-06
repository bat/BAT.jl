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


#=
@deprecate bat_sample(rng::AbstractRNG, target::AnySampleable, algorithm::AbstractSamplingAlgorithm) bat_sample(target, algorithm, BAT.set_rng(BAT.default_context(), rng))
@deprecate bat_sample(rng::AbstractRNG, target::AnySampleable) bat_sample(target, BAT.set_rng(BAT.default_context(), rng))

@deprecate bat_findmode(rng::AbstractRNG, target::AnySampleable, algorithm) bat_findmode(target, algorithm, BAT.set_rng(BAT.default_context(), rng))
@deprecate bat_findmode(rng::AbstractRNG, target::AnySampleable) bat_findmode(target, BAT.set_rng(BAT.default_context(), rng))

@deprecate bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm) = bat_initval(target, algorithm, BAT.set_rng(BAT.default_context(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity) = bat_initval(target, BAT.set_rng(BAT.default_context(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm) = bat_initval(target, n, algorithm, BAT.set_rng(BAT.default_context(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, n::Integer) = bat_initval(target, n, BAT.set_rng(BAT.default_context(), rng))
=#

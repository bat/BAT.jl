# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@noinline function MaxDensityNelderMead(; kwargs...)
    Base.depwarn("`MaxDensityNelderMead(;kwargs...)` is deprecated, use `OptimAlg(;optalg = Optim.NelderMead, kwargs...)` instead.", :MaxDensityNelderMead)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    OptimAlg(; optalg=optalg, kwargs...)
end
export MaxDensityNelderMead

@noinline function MaxDensityLBFGS(; kwargs...)
    Base.depwarn("`MaxDensityLBFGS(;kwargs...)` is deprecated, use `OptimAlg(;optalg = Optim.LBFGS, kwargs...)` instead.", :MaxDensityLBFGS)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    OptimAlg(; optalg=optalg, kwargs...)
end
export MaxDensityLBFGS

@noinline function NelderMeadOpt(; kwargs...)
    Base.depwarn("`NelderMeadOpt(;kwargs...)` is deprecated, use `OptimAlg(;optalg = Optim.NelderMead, kwargs...)` instead.", :NelderMeadOpt)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    OptimAlg(; optalg=optalg, kwargs...)
end
export NelderMeadOpt

@noinline function LBFGSOpt(; kwargs...)
    Base.depwarn("`LBFGSOpt(;kwargs...)` is deprecated, use `OptimAlg(;optalg = Optim.LBFGS, kwargs...)` instead.", :LBFGSOpt)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    OptimAlg(; optalg=optalg, kwargs...)
end
export LBFGSOpt


Base.@deprecate MaxDensitySampleSearch(args...; kwargs...) MaxDensitySearch(args...; kwargs...)
export MaxDensitySampleSearch

Base.@deprecate NoDensityTransform(args...; kwargs...) DoNotTransform(args...; kwargs...)
export NoDensityTransform

Base.@deprecate PosteriorDensity(args...) PosteriorMeasure(args...)
export PosteriorDensity

#=
@deprecate bat_sample(rng::AbstractRNG, target::AnySampleable, algorithm::AbstractSamplingAlgorithm) bat_sample(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_sample(rng::AbstractRNG, target::AnySampleable) bat_sample(target, BAT.set_rng(BAT.get_batcontext(), rng))

@deprecate bat_findmode(rng::AbstractRNG, target::AnySampleable, algorithm) bat_findmode(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_findmode(rng::AbstractRNG, target::AnySampleable) bat_findmode(target, BAT.set_rng(BAT.get_batcontext(), rng))

@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, algorithm::InitvalAlgorithm) = bat_initval(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike) = bat_initval(target, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, n::Integer, algorithm::InitvalAlgorithm) = bat_initval(target, n, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, n::Integer) = bat_initval(target, n, BAT.set_rng(BAT.get_batcontext(), rng))
=#

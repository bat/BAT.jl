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


Base.@deprecate MaxDensitySampleSearch(args...; kwargs...) EmpiricalMode(args...; kwargs...)
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


Base.@deprecate MetropolisHastings() RandomWalk()

Base.@deprecate MCMCSampling(;
    mcalg::MCMCProposal = RandomWalk(),
    trafo::AbstractTransformTarget = bat_default(TransformedMCMC, Val(:pretransform), mcalg),
    nchains::Int = 4,
    nsteps::Int = bat_default(TransformedMCMC, Val(:nsteps), mcalg, trafo, nchains),
    init::MCMCInitAlgorithm = bat_default(TransformedMCMC, Val(:init), mcalg, trafo, nchains, nsteps),
    burnin::MCMCBurninAlgorithm = bat_default(TransformedMCMC, Val(:burnin), mcalg, trafo, nchains, nsteps),
    convergence::ConvergenceTest = BrooksGelmanConvergence(),
    strict::Bool = true,
    store_burnin::Bool = false,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
) TransformedMCMC(
    proposal = mcalg,
    pretransform = trafo,
    nchains = nchains,
    nsteps = nsteps,
    init = init,
    burnin = burnin,
    convergence = convergence,
    strict = strict,
    store_burnin = store_burnin,
    nonzero_weights = nonzero_weights,
    callback = callback
)
export MCMCSampling


@deprecate PriorToGaussian() PriorToNormal()


@deprecate ModeAsDefined() PredefinedMode()
export ModeAsDefined

@deprecate MaxDensitySearch() EmpiricalMode()
export MaxDensitySearch

@deprecate BinnedModeEstimator() BinnedMode()
export BinnedModeEstimator

Base.@deprecate OptimAlg(;
    optalg::ALG = ext_default(pkgext(Val(:Optim)), Val(:DEFAULT_OPTALG))
    pretransform::AbstractTransformTarget = PriorToNormal()
    init::InitvalAlgorithm = InitFromTarget()
    maxiters::Int = 1_000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    kwargs::NamedTuple = (;)
) ModeOptimization(;
    optimizer = OptimOpt(;
        optalg = optalg,
        maxiters = maxiters,
        maxtime = maxtime,
        abstol = abstol,
        reltol = reltol,
        optargs = kwargs
    )
    pretransform = pretransform,
    init = init,
    optalg = optalg,
)
export OptimAlg


Base.@deprecate OptimizationAlg(;
    optalg::ALG = ext_default(pkgext(Val(:Optim)), Val(:DEFAULT_OPTALG))
    pretransform::AbstractTransformTarget = PriorToNormal()
    init::InitvalAlgorithm = InitFromTarget()
    maxiters::Int = 1000
    maxtime::Float64 = NaN
    abstol::Float64 = NaN
    reltol::Float64 = 0.0
    kwargs::NamedTuple = (;)
) ModeOptimization(;
    optimizer = OptimizationOpt(;
        optalg = ext_default(pkgext(Val(:Optimization)), Val(:DEFAULT_OPTALG)),
        maxiters = maxiters,
        maxtime = maxtime,
        abstol = abstol,
        reltol = reltol,
        optargs = kwargs
    )
    pretransform = pretransform,
    init = init,
    optalg = optalg,
)
export OptimizationAlg

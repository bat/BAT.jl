# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@deprecate bat_report(obj...) lazyreport(obj...)

@noinline function bat_report(smplv::DensitySampleVector; kwargs...)
    Base.depwarn("`bat_report` is deprecated, use `lazyreport` instead.", :bat_report)
    lazyreport(smplv; kwargs...)
end


function _deprecated_optim_alg(
    name::Symbol,
    optalg,
    optalg_name::Symbol;
    pretransform = NormalBased(),
    init = InitFromTarget(),
    kwargs...
)
    Base.depwarn(
        "`$name(; kwargs...)` is deprecated. Use `TransformedMaxDensity(optalg = OptimAlg(optalg = Optim.$optalg_name()))`; pass `pretransform` and `init` to `TransformedMaxDensity`, and optimizer options to `OptimAlg`.",
        name
    )
    TransformedMaxDensity(
        optalg = OptimAlg(; optalg, kwargs...),
        pretransform = pretransform,
        init = init
    )
end

@noinline function MaxDensityNelderMead(; kwargs...)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    _deprecated_optim_alg(:MaxDensityNelderMead, optalg, :NelderMead; kwargs...)
end
export MaxDensityNelderMead

@noinline function MaxDensityLBFGS(; kwargs...)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    _deprecated_optim_alg(:MaxDensityLBFGS, optalg, :LBFGS; kwargs...)
end
export MaxDensityLBFGS

@noinline function NelderMeadOpt(; kwargs...)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    _deprecated_optim_alg(:NelderMeadOpt, optalg, :NelderMead; kwargs...)
end
export NelderMeadOpt

@noinline function LBFGSOpt(; kwargs...)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    _deprecated_optim_alg(:LBFGSOpt, optalg, :LBFGS; kwargs...)
end
export LBFGSOpt


Base.@deprecate MaxDensitySampleSearch(args...; kwargs...) MaxDensitySearch(args...; kwargs...)
export MaxDensitySampleSearch

Base.@deprecate NoDensityTransform(args...; kwargs...) DoNotTransform(args...; kwargs...)
export NoDensityTransform

Base.@deprecate PosteriorDensity(args...) PosteriorMeasure(args...)
export PosteriorDensity

#=
@deprecate bat_sample(rng::AbstractRNG, target::MeasureLike, algorithm::AbstractSamplingAlgorithm) bat_sample(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_sample(rng::AbstractRNG, target::MeasureLike) bat_sample(target, BAT.set_rng(BAT.get_batcontext(), rng))

@deprecate bat_findmode(rng::AbstractRNG, target::MeasureLike, algorithm) bat_findmode(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_findmode(rng::AbstractRNG, target::MeasureLike) bat_findmode(target, BAT.set_rng(BAT.get_batcontext(), rng))

@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, algorithm::InitvalAlgorithm) = bat_initval(target, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike) = bat_initval(target, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, n::Integer, algorithm::InitvalAlgorithm) = bat_initval(target, n, algorithm, BAT.set_rng(BAT.get_batcontext(), rng))
@deprecate bat_initval(rng::AbstractRNG, target::MeasureLike, n::Integer) = bat_initval(target, n, BAT.set_rng(BAT.get_batcontext(), rng))
=#


Base.@deprecate MetropolisHastings() RandomWalk()

Base.@deprecate MCMCSampling(;
    mcalg::MCMCProposal = RandomWalk(),
    trafo::TransformIntent = bat_default(TransformedMCMC, Val(:pretransform), mcalg),
    nchains::Int = 4,
    # Defaults are taken from the actual TransformedMCMC constructor, so they
    # can never drift from it:
    nsteps::Int = TransformedMCMC(proposal = mcalg, pretransform = trafo, nchains = nchains).nsteps,
    init::MCMCInitAlgorithm = TransformedMCMC(proposal = mcalg, pretransform = trafo, nchains = nchains, nsteps = nsteps).init,
    burnin::MCMCBurninAlgorithm = TransformedMCMC(proposal = mcalg, pretransform = trafo, nchains = nchains, nsteps = nsteps).burnin,
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


@deprecate PriorToGaussian() NormalBased()

Base.@deprecate_binding AbstractTransformTarget TransformIntent
Base.@deprecate_binding PriorToNormal NormalBased
Base.@deprecate_binding PriorToUniform UniformBased
Base.@deprecate_binding OrderedResampling SystematicResampling


@deprecate DensitySampleVector(
    v::AbstractVector,
    logd::AbstractVector{<:Real};
    weight::Union{AbstractVector{<:Real}, Symbol} = fill(1, length(eachindex(v))),
    info::AbstractVector = fill(nothing, length(eachindex(v))),
    aux::AbstractVector = fill(nothing, length(eachindex(v)))
) DensitySampleVector(v = v, logd = logd, weight = weight, info = info, aux = aux)

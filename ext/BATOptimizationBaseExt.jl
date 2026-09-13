# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationBaseExt

import OptimizationBase

using BAT
BAT.pkgext(::Val{:OptimizationBase}) = BAT.PackageExtension{:OptimizationBase}()

using BAT: get_adselector

using FunctionChains
using AutoDiffOperators: AbstractADType, NoAutoDiff, reverse_adtype


BAT.ext_default(::BAT.PackageExtension{:OptimizationBase}, ::Val{:DEFAULT_OPTALG}) = nothing #Optim.NelderMead()


struct _OptimizationTargetFunc{F} <: Function
    f::F
end
_OptimizationTargetFunc(::Type{F}) where F = _OptimizationTargetFunc{Type{F}}(F)

(ft::_OptimizationTargetFunc)(x, ::Any) = ft.f(x)


build_optimizationfunction(f, ad::AbstractADType) = OptimizationBase.OptimizationFunction(f, ad)
build_optimizationfunction(f, ::NoAutoDiff) = OptimizationBase.OptimizationFunction(f)


function BAT.maximize_density(f_logdensity, x_init::AbstractVector{<:Real}, algorithm::OptimizationAlg, context::BATContext)
    f = fchain(f_logdensity, -)

    f_target = _OptimizationTargetFunc(f)
    ad = reverse_adtype(get_adselector(context))
    optimization_function = build_optimizationfunction(f_target, ad)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    # Not all algorithms support abstol, just filter all NaN-valued opts out:
    filtered_algopts = NamedTuple(filter(p -> !isnan(p[2]), pairs(algopts)))

    if algorithm.store_trace
        haskey(algorithm.kwargs, :callback) && throw(ArgumentError(
            "store_trace = true of OptimizationAlg is not compatible with a user-supplied callback"
        ))
        recorder = _TraceRecorder(x_init)
        optimization_result = OptimizationBase.solve(
            optimization_problem, algorithm.optalg;
            filtered_algopts..., algorithm.kwargs..., callback = recorder
        )
        trace = _recorded_trace(recorder)
    else
        optimization_result = OptimizationBase.solve(optimization_problem, algorithm.optalg; filtered_algopts..., algorithm.kwargs...)
        trace = nothing
    end

    ret_a = (result = optimization_result.u,)
    # Abstractly typed trace and info fields keep the return type
    # inferrable despite the solver-dependent result type:
    ret_b = @NamedTuple{trace::Union{Nothing,NamedTuple}, info::Any}((trace, optimization_result))
    return merge(ret_a, ret_b)
end


# Records the solver state per iteration in the sign convention of the
# minimized objective; sign flips to log-density convention happen in
# `_recorded_trace`. Solvers reuse their state buffers, so entries must
# be copied:
struct _TraceRecorder{T<:Real}
    v::Vector{Vector{T}}
    negld::Vector{T}
    neggrad::Vector{Union{Nothing,Vector{T}}}
end

_TraceRecorder(x_init::AbstractVector{<:Real}) = _TraceRecorder{float(eltype(x_init))}([], [], [])

function (recorder::_TraceRecorder{T})(state, loss) where T
    # Vector{T}(x) instead of convert: convert is a no-op for matching
    # types, which would store a reference into the solver's mutated
    # state buffer instead of a snapshot:
    push!(recorder.v, Vector{T}(state.u))
    push!(recorder.negld, T(loss))
    grad = hasproperty(state, :grad) ? state.grad : nothing
    push!(recorder.neggrad, isnothing(grad) ? nothing : Vector{T}(grad))
    return false
end

# Gradient availability in the callback state depends on the solver, so
# `grad_logd` is only part of the trace if gradients were recorded for
# every iteration:
function _recorded_trace(recorder::_TraceRecorder)
    base = (v = recorder.v, logd = -recorder.negld)
    if !isempty(recorder.neggrad) && all(!isnothing, recorder.neggrad)
        return (; base..., grad_logd = .-recorder.neggrad)
    else
        return base
    end
end


end # module BATOptimizationBaseExt

# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    bat_default(f::Base.Callable, argname::Symbol, objectives...)
    bat_default(f::Base.Callable, argname::Val, objectives...)

Get the default value for argument `argname` of function `f` to use
for `objective`(s).

`objective`(s) are mandatory arguments of function `f` that semantically
constitute it's main objective(s), and that that a good default choice of
optional arguments (e.g. choice of algorithm(s), etc.) may depend on.
Which arguments are considered to be objectives is function-specific.

For example:

```julia
bat_default(bat_sample, :algorithm, density::PosteriorMeasure) == RandomWalk()
bat_default(bat_sample, Val(:algorithm), samples::DensitySampleVector) == OrderedResampling()
```
"""
function bat_default end
export bat_default

@inline bat_default(f::Base.Callable, argname::Symbol, objectives...) = bat_default(Val{argname}(), objectives...)


"""
    argchoice_msg(f::Base.Callable, argname::Val, x)

*BAT-internal, not part of stable public API.*

Generates an information message regarding the choice of value `x` for
argument `argname` of function `f`.

The value `x` will often be the result of [`bat_default`](@ref).
"""
function argchoice_msg end

function bat_default_withinfo(f::Base.Callable, argname::Val, objective...)
    default = bat_default(f::Base.Callable, argname::Val, objective...)
    @info argchoice_msg(f, argname::Val, default)
    default
end

function bat_default_withdebug(f::Base.Callable, argname::Val, objective...)
    default = bat_default(f::Base.Callable, argname::Val, objective...)
    @debug argchoice_msg(f, argname::Val, default)
    default
end



result_with_args(r::NamedTuple) = merge(r, (optargs = NamedTuple(),))

result_with_args(r::NamedTuple, optargs::NamedTuple) = merge(r, (optargs = optargs,))

function result_with_args(::Val, ::Any, r::NamedTuple, optargs::NamedTuple)
    return result_with_args(r, optargs)
end

function result_with_args(::Val{resultkind}, target::Union{AbstractMeasure,Distribution}, r::NamedTuple, optargs::NamedTuple) where resultkind
    measure = batmeasure(target)
    augmented_result = _augment_bat_retval(Val(resultkind), measure, r)
    result_with_args(augmented_result, optargs)
end

function _augment_bat_retval(::Val{resultkind}, measure, r::R) where {resultkind,R}
    if hasfield(R, :evaluated)
        return r
    else
        if resultkind == :samples
            empirical = DensitySampleMeasure(r.result, getdof(measure))
        elseif hasfield(R, :samples)
            empirical = DensitySampleMeasure(r.samples, getdof(measure))
        else
            empirical = maybe_empiricalof(measure)
        end

        if resultkind == :approx
            approx = r.result
        elseif hasfield(R, :approx)
            approx = r.approx
        else
            approx = maybe_approxof(measure)
        end

        if resultkind == :mass
            mass = r.result
        elseif hasfield(R, :mass)
            mass = r.mass
        else
            mass = massof(measure)
        end

        if resultkind == :modes
            modes = r.result
        elseif resultkind == :mode
            modes = [r.result]
        elseif hasfield(R, :modes)
            modes = r.modes
        elseif hasfield(R, :mode)
            modes = [r.mode]
        else
            modes = maybe_modesof(measure)
        end

        if resultkind == :samplegen
            samplegen = r.result
        elseif hasfield(R, :samplegen)
            samplegen = r.samplegen
        else
            samplegen = maybe_samplegen(measure)
        end
        global g_state = (unevaluated(measure), empirical, approx, mass, modes, samplegen)
        evaluated = EvaluatedMeasure(unevaluated(measure), empirical, approx, mass, modes, samplegen)
        r_add = (result = r.result, evaluated = evaluated)
        return merge(r_add, r)
    end
end

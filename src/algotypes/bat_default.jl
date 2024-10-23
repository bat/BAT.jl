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

function result_with_args(::Val{resultname}, target::Union{AbstractMeasure,Distribution}, r::NamedTuple, optargs::NamedTuple) where resultname
    measure = batmeasure(target)
    augmented_result = _augment_bat_retval(Val(resultname), measure, r)
    result_with_args(augmented_result, optargs)
end

function _augment_bat_retval(::Val{resultname}, measure, r::R) where {resultname,R}
    if hasfield(R, :evaluated)
        return r
    else
        if resultname == :samples
            samples = r.result
        elseif hasfield(R, :samples)
            samples = r.samples
        else
            samples = maybe_samplesof(measure)
        end

        if resultname == :approx
            approx = r.result
        elseif hasfield(R, :approx)
            approx = r.approx
        else
            approx = maybe_approxof(measure)
        end

        if resultname == :mass
            mass = r.result
        elseif hasfield(R, :mass)
            mass = r.mass
        else
            mass = massof(measure)
        end

        if resultname == :modes
            modes = r.result
        elseif resultname == :mode
            modes = [r.result]
        elseif hasfield(R, :modes)
            modes = r.modes
        elseif hasfield(R, :mode)
            modes = [r.mode]
        else
            modes = maybe_modesof(measure)
        end

        if resultname == :generator
            generator = r.result
        elseif hasfield(R, :generator)
            generator = r.generator
        else
            generator = maybe_generator(measure)
        end
        evaluated = EvaluatedMeasure(unevaluated(measure), samples, approx, mass, modes, generator)
        r_add = (result = r.result, evaluated = evaluated)
        return merge(r_add, r)
    end
end

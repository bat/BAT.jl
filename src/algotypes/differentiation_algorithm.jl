# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type DifferentiationAlgorithm

*Experimental feature, not part of stable public API.*

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type DifferentiationAlgorithm end
export DifferentiationAlgorithm



"""
    bat_valgrad(f::Function, [algorithm::DifferentiationAlgorithm])

*Experimental feature, not part of stable public API.*

Generated a function that calculates both value and gradient of `f` at given
points.

The function `f` must support `ValueShapes.varshape(f)` and
`ValueShapes.unshaped(f)`.

Returns a NamedTuple of the shape

```julia
(result = valgrad_f, ...)
```

with

```julia
f_at_x, grad_of_f_at_x = valgrad_f(x)

grad_of_f_at_x = zero(x)
f_at_x = valgrad_f(!, grad_of_f_at_x, x)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_valgrad`, add methods to
    `bat_valgrad_impl` instead.
"""
function bat_valgrad end
export bat_valgrad

function bat_valgrad_impl end


function bat_valgrad(
    f::Function,
    algorithm = bat_default_withdebug(bat_valgrad, Val(:algorithm), f)
)
    r = bat_valgrad_impl(f, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_valgrad), ::Val{:algorithm}, x::DifferentiationAlgorithm)
    "Using automiatic differentiation algorithm $x"
end



gradient_shape(vs::AbstractValueShape) = replace_const_shapes(ValueShapes.const_zero_shape, vs)


abstract type GradientFunction end


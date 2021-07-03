# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type DifferentiationAlgorithm

*Experimental feature, not part of stable public API.*

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type DifferentiationAlgorithm end
export DifferentiationAlgorithm


function vjp_algorithm end
vjp_algorithm(f::Function) = ZygoteAD()


function jvp_algorithm end
jvp_algorithm(f::Function) = ForwardDiffAD()



"""
    valgradof(f::Function, algorithm::DifferentiationAlgorithm  = vjp_algorithm(f))

*Experimental feature, not part of stable public API.*

Generates a function that calculates both value and gradient of `f` at given
points. The differentiation algorithm used depends on `f`.

The function `f` must support `ValueShapes.varshape(f)` and
`ValueShapes.unshaped(f)`.

Returns function `valgrad_f` with

```julia
f_at_x, grad_of_f_at_x = valgrad_f(x)

grad_of_f_at_x = zero(x)
f_at_x = valgrad_f(!, grad_of_f_at_x, x)
```
"""
function valgradof end
export valgradof


abstract type GradientFunction <: Function end

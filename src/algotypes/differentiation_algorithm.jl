# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type DifferentiationAlgorithm

*Experimental feature, not part of stable public API.*

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type DifferentiationAlgorithm end
export DifferentiationAlgorithm

MaybeDiffAlg = Union{DifferentiationAlgorithm,Missing}

function vjp_algorithm end
vjp_algorithm(::Any) = missing

function promote_vjp_algorithm end
promote_vjp_algorithm() = missing
promote_vjp_algorithm(a::MaybeDiffAlg) = a
promote_vjp_algorithm(a::DifferentiationAlgorithm, b::DifferentiationAlgorithm) = a
promote_vjp_algorithm(a::DifferentiationAlgorithm, b::Missing) = a
promote_vjp_algorithm(a::Missing, b::DifferentiationAlgorithm) = b
promote_vjp_algorithm(a::Missing, b::Missing) = b
function promote_vjp_algorithm(a::MaybeDiffAlg, b::MaybeDiffAlg, cs::Vararg{MaybeDiffAlg,N}) where {N}
    promote_vjp_algorithm(promote_vjp_algorithm(a, b), cs...)
end

generic_vjp_algorithm(obj::T) where {T} = promote_vjp_algorithm(map(vjp_algorithm, object_contents(obj))...)
vjp_algorithm(f::F) where {F<:Function} = generic_vjp_algorithm(f)
vjp_algorithm(d::AbstractMeasureOrDensity) = generic_vjp_algorithm(d)

# ToDo: Switch default from ForwardDiff to Zygote:
default_vjp_algorithm() = ForwardDiffAD()
#default_vjp_algorithm() = ZygoteAD()

@inline function some_vjp_algorithm(x)
    alg = vjp_algorithm(x)
    ismissing(alg) ? default_vjp_algorithm() : alg
end


function jvp_algorithm end
jvp_algorithm(::Any) = missing

function promote_jvp_algorithm end
promote_jvp_algorithm() = missing
promote_jvp_algorithm(a::MaybeDiffAlg) = a
promote_jvp_algorithm(a::DifferentiationAlgorithm, b::DifferentiationAlgorithm) = a
promote_jvp_algorithm(a::DifferentiationAlgorithm, b::Missing) = a
promote_jvp_algorithm(a::Missing, b::DifferentiationAlgorithm) = b
promote_jvp_algorithm(a::Missing, b::Missing) = b
function promote_jvp_algorithm(a::MaybeDiffAlg, b::MaybeDiffAlg, cs::Vararg{MaybeDiffAlg,N}) where {N}
    promote_jvp_algorithm(promote_jvp_algorithm(a, b), cs...)
end

generic_jvp_algorithm(obj::T) where {T} = promote_jvp_algorithm(map(jvp_algorithm, object_contents(obj))...)
jvp_algorithm(f::F) where {F<:Function} = generic_jvp_algorithm(f)

default_jvp_algorithm() = ForwardDiffAD()

function some_jvp_algorithm(x)
    alg = jvp_algorithm(x)
    ismissing(alg) ? default_jvp_algorithm() : alg
end


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

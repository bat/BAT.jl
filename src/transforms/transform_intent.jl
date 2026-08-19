# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type TransformIntent

Abstract type for variate space transformation intents.

A `TransformIntent`, together with an object to be transformed, implies a
concrete transformation function; the same intent and object always yield
the same transformation. Implementations must derive the transformation
from the intent and the object alone, and must support meaningful equality
comparison (value-carrying intent types must specialize `Base.:(==)`
accordingly; singleton intent types get this for free).
"""
abstract type TransformIntent end
export TransformIntent

TransformIntent(::Type{Vector}) = ToRealVector()
Base.convert(::Type{TransformIntent}, x::Type) = TransformIntent(x)


# Intents of different types never match, intents of equal type compare by
# value (trivial for singleton intents):
_intents_match(::Any, ::TransformIntent) = false
_intents_match(a::TR, b::TR) where {TR<:TransformIntent} = a == b


"""
    struct DoNotTransform <: TransformIntent

The identity density transformation target, specifies that densities
should not be transformed.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct DoNotTransform <: TransformIntent end
export DoNotTransform


"""
    struct ToRealVector <: TransformIntent

Specifies that the input should be transformed into a measure over the space
of real-valued flat vectors.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct ToRealVector <: TransformIntent end
export ToRealVector


"""
    struct UniformBased <: TransformIntent

Specifies that the target measure of an operation should be transformed
so that it is based on a uniform distribution over the unit hypercube:
the prior — descending through nested posteriors to the innermost prior —
becomes standard uniform. Applies to any measure with such a
transformable base, not just posteriors.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct UniformBased <: TransformIntent end
export UniformBased


"""
    struct NormalBased <: TransformIntent

Specifies that the target measure of an operation should be transformed
so that it is based on a standard multivariate normal distribution:
the prior — descending through nested posteriors to the innermost prior —
becomes standard normal. Applies to any measure with such a
transformable base, not just posteriors.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct NormalBased <: TransformIntent end
export NormalBased

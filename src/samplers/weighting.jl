# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AbstractWeightingScheme{T<:Real}

Abstract class for sample weighting schemes.

Weight values will have type `T`.
"""
abstract type AbstractWeightingScheme{T<:Real} end
export AbstractWeightingScheme


"""
    struct RepetitionWeighting{T<:AbstractFloat} <: AbstractWeightingScheme{T}

Sample weighting scheme suitable for sampling algorithms which may repeated
samples multiple times in direct succession (e.g.
[`MetropolisHastings`](@ref)). The repeated sample is stored only once,
with a weight equal to the number of times it has been repeated (e.g.
because a Markov chain has not moved during a sampling step).

Constructors:

```julia
RepetitionWeighting()

RepetitionWeighting{T}()
```
"""
struct RepetitionWeighting{T<:Real} <: AbstractWeightingScheme{T} end
export RepetitionWeighting
RepetitionWeighting() = RepetitionWeighting{Int}()


"""
    struct ARPWeighting{T<:AbstractFloat} <: AbstractWeightingScheme{T}

Sample weighting scheme suitable for accept/reject-based sampling algorithms
(e.g. [`MetropolisHastings`](@ref)). Both accepted and rejected samples
become part of the output, with a weight proportional to their original
acceptance probability.

Constructors:

```julia
ARPWeighting()

ARPWeighting{T}()
```
"""
struct ARPWeighting{T<:AbstractFloat} <: AbstractWeightingScheme{T} end
export ARPWeighting
ARPWeighting() = ARPWeighting{Float64}()

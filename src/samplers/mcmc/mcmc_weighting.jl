# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AbstractMCMCWeightingScheme{T<:Real}

Abstract class for weighting schemes for MCMC samples.

Weight values will have type `T`.
"""
abstract type AbstractMCMCWeightingScheme{T<:Real} end
export AbstractMCMCWeightingScheme


sample_weight_type(::Type{<:AbstractMCMCWeightingScheme{T}}) where {T} = T



"""
    struct RepetitionWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T}

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
struct RepetitionWeighting{T<:Real} <: AbstractMCMCWeightingScheme{T} end
export RepetitionWeighting
RepetitionWeighting() = RepetitionWeighting{Int}()


"""
    struct ARPWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T}

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
struct ARPWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T} end
export ARPWeighting
ARPWeighting() = ARPWeighting{Float64}()
